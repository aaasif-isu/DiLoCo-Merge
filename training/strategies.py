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

        for i, batch in enumerate(loader):
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
    ):
        self.device = device
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.config = config
        self.params_offloaded = None
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
            self.params_offloaded = self._get_offloaded_param(model)

        inner_step_count = 0
        accum_batch_size = 0
        running_loss_this_accum = 0.0

        for i, batch in enumerate(loader):
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
