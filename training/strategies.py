"""Inner and outer training strategies (SimpleAccum and Diloco)."""

from abc import ABC, abstractmethod
import logging

import torch
import torch.distributed as dist

from methods import method_handles_own_communication
from models import unwrap_model
from .synchronization import SynchronousSynchronizer, offload_model_params

logger = logging.getLogger("loco")


class InnerOuterStrategy(ABC):
    def _init_loader_tracking(self) -> None:
        """Initialize persistent per-DataLoader iterator and RNG state tracking."""
        self._loader_states: dict[int, dict] = {}

    def _create_epoch_iterator(self, loader, epoch_index: int = 0) -> dict:
        """Create a fresh epoch iterator while snapshotting the pre-epoch generator RNG state."""
        gen = getattr(loader, "generator", None)
        epoch_start_rng_state = (
            gen.get_state().clone() if isinstance(gen, torch.Generator) else None
        )
        iterator = iter(loader)
        state = {
            "loader": loader,
            "iterator": iterator,
            "epoch_index": int(epoch_index),
            "batches_consumed_in_epoch": 0,
            "epoch_start_rng_state": epoch_start_rng_state,
        }
        self._loader_states[id(loader)] = state
        return state

    def _next_batch(self, loader):
        """
        Fetch the next batch from the persistent iterator for `loader`.
        Reshuffles (creates a new iterator) only when the current epoch is exhausted.
        """
        if not hasattr(self, "_loader_states"):
            self._init_loader_tracking()

        lid = id(loader)
        state = self._loader_states.get(lid)
        if state is None or state.get("loader") is not loader:
            state = self._create_epoch_iterator(loader, epoch_index=0)

        try:
            batch = next(state["iterator"])
            state["batches_consumed_in_epoch"] += 1
            return batch
        except StopIteration:
            next_epoch = int(state["epoch_index"]) + 1
            state = self._create_epoch_iterator(loader, epoch_index=next_epoch)
            batch = next(state["iterator"])
            state["batches_consumed_in_epoch"] += 1
            return batch

    def get_dataloader_state(self, loaders) -> list[dict]:
        """Export serializable per-worker DataLoader traversal state for checkpointing."""
        if not hasattr(self, "_loader_states"):
            self._init_loader_tracking()

        exported: list[dict] = []
        for idx, loader in enumerate(loaders):
            state = self._loader_states.get(id(loader))
            gen = getattr(loader, "generator", None)
            current_rng = (
                gen.get_state().clone() if isinstance(gen, torch.Generator) else None
            )
            if state is None or state.get("loader") is not loader:
                exported.append(
                    {
                        "worker_idx": idx,
                        "initialized": False,
                        "epoch_index": 0,
                        "batches_consumed_in_epoch": 0,
                        "epoch_start_rng_state": current_rng,
                        "current_rng_state": current_rng,
                    }
                )
            else:
                exported.append(
                    {
                        "worker_idx": idx,
                        "initialized": True,
                        "epoch_index": int(state["epoch_index"]),
                        "batches_consumed_in_epoch": int(
                            state["batches_consumed_in_epoch"]
                        ),
                        "epoch_start_rng_state": (
                            state["epoch_start_rng_state"].clone()
                            if isinstance(state["epoch_start_rng_state"], torch.Tensor)
                            else None
                        ),
                        "current_rng_state": current_rng,
                    }
                )
        return exported

    def load_dataloader_state(
        self,
        loaders,
        saved_state: list[dict] | None,
        global_step: int = 0,
        batches_per_step: int = 0,
    ) -> None:
        """
        Restore per-worker DataLoader iterators to the exact epoch and batch offset
        at checkpoint time so resuming never repeats or skips batches.
        """
        if not hasattr(self, "_loader_states"):
            self._init_loader_tracking()
        self._loader_states.clear()

        if saved_state and isinstance(saved_state, list) and len(saved_state) == len(loaders):
            for loader, entry in zip(loaders, saved_state):
                if not entry.get("initialized", True):
                    gen = getattr(loader, "generator", None)
                    saved_rng = entry.get("current_rng_state")
                    if isinstance(gen, torch.Generator) and isinstance(
                        saved_rng, torch.Tensor
                    ):
                        gen.set_state(saved_rng.cpu().to(dtype=torch.uint8))
                    continue

                gen = getattr(loader, "generator", None)
                epoch_rng = entry.get("epoch_start_rng_state")
                if isinstance(gen, torch.Generator) and isinstance(
                    epoch_rng, torch.Tensor
                ):
                    gen.set_state(epoch_rng.cpu().to(dtype=torch.uint8))

                epoch_idx = int(entry.get("epoch_index", 0))
                consumed = int(entry.get("batches_consumed_in_epoch", 0))
                state = self._create_epoch_iterator(loader, epoch_index=epoch_idx)

                for _ in range(consumed):
                    try:
                        next(state["iterator"])
                        state["batches_consumed_in_epoch"] += 1
                    except StopIteration:
                        epoch_idx += 1
                        state = self._create_epoch_iterator(
                            loader, epoch_index=epoch_idx
                        )
                        next(state["iterator"])
                        state["batches_consumed_in_epoch"] += 1
        elif global_step > 0 and batches_per_step > 0:
            total_batches_to_advance = int(global_step) * int(batches_per_step)
            for loader in loaders:
                for _ in range(total_batches_to_advance):
                    try:
                        self._next_batch(loader)
                    except StopIteration:
                        break

    @abstractmethod
    def inner_step(self, model, loader, inner_optimizer=None, inner_scheduler=None):
        """Execute inner optimization step."""
        pass

    @abstractmethod
    def outer_step(self, models, optimizer, scheduler=None):
        """Execute outer optimization step."""
        pass


class SimpleAccum(InnerOuterStrategy):
    def __init__(self, device, world_size, global_rank, tokenizer, config):
        self._init_loader_tracking()
        self.device = device
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.config = config
        self.handles_own_communication = method_handles_own_communication(
            self.config.outer_optimizer
        )

    def inner_step(self, model, loader, inner_optimizer=None, inner_scheduler=None):
        total_loss = 0
        batch_tokens = 0
        batch_count = 0
        accum_batch_size = 0

        i = 0
        while accum_batch_size < self.config.batch_size:
            try:
                batch = self._next_batch(loader)
            except StopIteration:
                break

            input_ids = batch.to(self.device, non_blocking=True)
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            accum_batch_size += len(batch)

            with torch.amp.autocast(
                device_type=self.device.type, dtype=torch.bfloat16
            ):
                outputs = model(input_ids=input_ids, labels=labels)

            loss = outputs.loss / (
                self.config.batch_size / self.config.micro_batch_size
            )
            loss.backward()

            total_loss += loss.item()
            batch_count += 1
            batch_tokens += (labels != -100).sum().item()

            if self.global_rank == 0 and i % 20 == 0:
                logger.info(
                    f"Batch {i}, loss: {outputs.loss.item():.4f}, accum: {accum_batch_size}/{self.config.batch_size}"
                )

            i += 1
            if accum_batch_size >= self.config.batch_size:
                break

        return {
            "total_loss": total_loss,
            "loss_after_gather": total_loss,
            "batch_count": batch_count,
            "batch_tokens": batch_tokens,
        }

    def outer_step(self, models, optimizer, scheduler):
        model = models[0]
        actual_model = model.module if hasattr(model, "module") else model

        if len(models) > 1:
            master_params = list(actual_model.parameters())
            for param_idx, param in enumerate(master_params):
                if param.grad is not None:
                    for vmodel in models[1:]:
                        vmodel_actual = (
                            vmodel.module if hasattr(vmodel, "module") else vmodel
                        )
                        vparam = list(vmodel_actual.parameters())[param_idx]
                        if vparam.grad is not None:
                            param.grad.add_(vparam.grad)
                    param.grad.div_(len(models))

        if self.world_size > 1 and not self.handles_own_communication:
            for param in actual_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        torch.nn.utils.clip_grad_norm_(actual_model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if len(models) > 1 and not self.handles_own_communication:
            with torch.no_grad():
                master_params = list(actual_model.parameters())
                for vmodel in models[1:]:
                    vmodel_actual = (
                        vmodel.module if hasattr(vmodel, "module") else vmodel
                    )
                    for master_p, v_p in zip(
                        master_params, vmodel_actual.parameters()
                    ):
                        v_p.copy_(master_p)



class Diloco(InnerOuterStrategy):
    def __init__(
        self,
        device,
        world_size,
        global_rank,
        tokenizer,
        config,
        synchronizer=None,
        outer_optimizer=None,
    ):
        self._init_loader_tracking()
        self.device = device
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.config = config
        self.params_offloaded = None
        self.outer_optimizer = outer_optimizer
        self.handles_own_communication = method_handles_own_communication(
            self.config.outer_optimizer
        )
        self.synchronizer = synchronizer or SynchronousSynchronizer(
            device=self.device, world_size=self.world_size
        )

    def inner_step(self, model, loader, inner_optimizer=None, inner_scheduler=None):
        total_loss = 0
        batch_tokens = 0
        batch_count = 0
        loss_after_gather = 0.0

        if self.params_offloaded is None:
            if self.outer_optimizer is not None and hasattr(
                self.outer_optimizer, "prepare_lookahead"
            ):
                self.outer_optimizer.prepare_lookahead()
            self.params_offloaded = self._get_offloaded_param(model)

        inner_step_count = 0
        accum_batch_size = 0
        running_loss_this_accum = 0.0

        while inner_step_count < self.config.inner_steps:
            try:
                batch = self._next_batch(loader)
            except StopIteration:
                break

            input_ids = batch.to(self.device, non_blocking=True)
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            accum_batch_size += len(batch)

            with torch.amp.autocast(
                device_type=self.device.type, dtype=torch.bfloat16
            ):
                outputs = model(input_ids=input_ids, labels=labels)

            loss = outputs.loss / (
                self.config.batch_size / self.config.micro_batch_size
            )
            loss.backward()

            total_loss += loss.item()
            running_loss_this_accum += loss.item()
            batch_count += 1
            batch_tokens += (labels != -100).sum().item()

            if accum_batch_size >= self.config.batch_size:
                if inner_step_count == 0:
                    loss_after_gather = running_loss_this_accum
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                inner_optimizer.step()
                inner_scheduler.step()
                inner_optimizer.zero_grad(set_to_none=True)

                if self.global_rank == 0 and inner_step_count % 5 == 0:
                    logger.info(
                        f"Inner Step {inner_step_count + 1}/{self.config.inner_steps}, loss: {outputs.loss.item():.4f}"
                    )

                inner_step_count += 1
                accum_batch_size = 0
                running_loss_this_accum = 0.0

                if inner_step_count >= self.config.inner_steps:
                    break

        return {
            "total_loss": total_loss / max(1, inner_step_count),
            "loss_after_gather": loss_after_gather,
            "batch_count": batch_count,
            "batch_tokens": batch_tokens,
        }

    def outer_step(self, models, optimizer, scheduler=None):
        if self.params_offloaded is None:
            raise ValueError("No offloaded parameters found. Run inner_step first.")

        # 1 & 2. Compute pseudo-gradients, synchronize if needed, and restore base params
        self.synchronizer.prepare_pseudo_gradients(
            models=models,
            params_offloaded=self.params_offloaded,
            handles_own_communication=self.handles_own_communication,
        )

        # 3. Outer optimizer / aggregation step
        optimizer.step()

        # 4. Broadcast updated parameters of vmodel[0] to virtual models on same GPU if needed
        self.synchronizer.sync_virtual_models(
            models=models,
            handles_own_communication=self.handles_own_communication,
        )

        # 5. Clear offloaded parameters so they are captured at the start of the next window
        self.params_offloaded = None

    def _get_offloaded_param(self, model):
        return offload_model_params(model)

    def _unwrap(self, m):
        return unwrap_model(m)
