"""Distributed model trainer and timing utilities."""

from collections import defaultdict
from datetime import datetime
import logging
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
import wandb

from data import (
    PartitionManifest,
    ShardedGPUDataset,
    TokenSpan,
    WorkerPartition,
    build_partitioner,
    discover_language_shards,
    get_dataloader,
    load_partition_manifest,
    save_partition_manifest,
    validate_partition_manifest,
    verify_frozen_dataset,
)
from evaluation import (
    ExperimentTracker,
    evaluate_global,
    evaluate_multilingual,
    evaluate_per_worker,
)
from methods import apply_strategy_preset
from models import (
    build_model,
    build_tokenizer,
    init_torchtitan_weights,
    load_model_hparams,
)
from .config import load_experiment_config
from .optimizers import (
    build_inner_optimizers_and_schedulers,
    build_outer_optimizer_and_scheduler,
)
from .strategies import Diloco, SimpleAccum

logger = logging.getLogger("loco")


class Timer:
    """Context manager for timing code blocks."""

    _timings = defaultdict(list)
    _active_timers = {}
    _disable = False

    def __init__(self, name, logger_obj=None, disabled=False, enabled=False):
        self.name = name
        self.logger = logger_obj
        self.start_time = None
        self.disabled = disabled or (Timer._disable and not enabled)

    def __enter__(self):
        if self.disabled:
            return self

        self.start_time = time.perf_counter()
        Timer._active_timers[self.name] = self.start_time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disabled or self.start_time is None:
            return

        end_time = time.perf_counter()
        duration = end_time - self.start_time
        Timer._timings[self.name].append(duration)

        if self.logger and self.name in Timer._active_timers:
            self.logger.debug(f"{self.name}: {duration:.6f}s")

        if self.name in Timer._active_timers:
            del Timer._active_timers[self.name]

    @classmethod
    def get_stats(cls, name=None):
        """Get timing statistics for a specific timer or all timers."""
        if name is not None:
            times = cls._timings.get(name, [])
            if not times:
                return {}
            return {
                "total": sum(times),
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "last": times[-1],
            }
        return {name: cls.get_stats(name) for name in cls._timings.keys()}

    @classmethod
    def reset(cls):
        """Reset all timings."""
        cls._timings = defaultdict(list)
        cls._active_timers = {}

    @classmethod
    def disable(cls, disabled=True):
        """Disable all timers."""
        cls._disable = disabled

    @classmethod
    def summarize(cls, logger_obj=None):
        """Summarize all timings."""
        result = {}
        for name, times in cls._timings.items():
            if not times:
                continue

            stats = cls.get_stats(name)
            msg = (
                f"{name} - total: {stats['total']:.3f}s, "
                f"mean: {stats['mean']:.3f}s, "
                f"max: {stats['max']:.3f}s"
            )
            result[name] = stats
            if logger_obj:
                logger_obj.info(msg)

        return result


class DistributedModelTrainer:
    """Orchestrates synchronous distributed language model training and aggregation."""

    def __init__(self, config: Optional[Any] = None):
        self.process_start_time = time.perf_counter()
        self.config = config if config is not None else load_experiment_config()

        if getattr(self.config, "debug", False):
            logger.setLevel(logging.DEBUG)

        if not getattr(self.config, "debug", False):
            Timer.disable(True)

        apply_strategy_preset(self.config)

        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))

        if (
            getattr(self.config, "num_workers", None) is not None
            and self.config.num_workers > 0
            and self.config.inner_outer_strategy != "simple_accum"
        ):
            if self.config.num_workers % self.world_size != 0:
                raise ValueError(
                    f"num_workers ({self.config.num_workers}) must be divisible by "
                    f"distributed world_size ({self.world_size})."
                )
            self.virtual_workers = self.config.num_workers // self.world_size
            self.config.virtual_workers = self.virtual_workers
        else:
            self.virtual_workers = self.config.virtual_workers
            self.config.num_workers = self.world_size * self.virtual_workers

        if self.config.max_steps == -1:
            if self.config.inner_outer_strategy == "diloco":
                divisor = (
                    self.config.inner_steps
                    * self.config.batch_size
                    * self.config.sequence_length
                    * self.world_size
                    * self.virtual_workers
                )
            else:
                divisor = (
                    self.config.batch_size
                    * self.config.sequence_length
                    * self.world_size
                    * self.virtual_workers
                )
            self.config.max_steps = self.config.token_budget // divisor
            logger.info(f"Calculated max_steps: {self.config.max_steps}")

        if self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            init_process_group(backend="nccl")
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

        self.seed = getattr(self.config, "seed", 42)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.model_hparams = load_model_hparams(
            hparams_file=getattr(self.config, "hparams_file", None),
            model_size=getattr(self.config, "model_size", "178M"),
            overrides=getattr(self.config, "model_hparams", None),
        )
        tokenizer_name = (
            getattr(self.config, "tokenizer_name", None)
            or self.model_hparams["tokenizer_name"]
        )
        self.tokenizer = build_tokenizer(
            tokenizer_name,
            revision=getattr(self.config, "tokenizer_revision", None),
        )
        self.config.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)

        # Prevent accidental overwriting of completed runs
        save_dir = Path(self.config.save_path)
        if (
            not getattr(self.config, "resume_from_checkpoint", False)
            and (save_dir / "metrics_summary.json").is_file()
        ):
            raise FileExistsError(
                f"Output directory '{save_dir}' already contains a completed run "
                f"(metrics_summary.json exists). Specify a new --run_id / --run_name "
                f"or --save_path, or pass --resume_from_checkpoint."
            )

        self.wandb = None
        if self.global_rank == 0 and self.config.log_wandb:
            self.wandb = wandb.init(
                project=getattr(self.config, "project", None),
                name=self.config.run_name,
                config=vars(self.config),
                group=self.config.outer_optimizer,
            )

        self.dataset_metadata: Dict[str, Any] = {}
        self.train_manifest: Optional[PartitionManifest] = None

        self.tracker = ExperimentTracker(
            output_dir=self.config.save_path,
            config_dict={
                **vars(self.config),
                "resolved_model_hparams": self.model_hparams,
                "world_size": self.world_size,
                "virtual_workers": self.virtual_workers,
                "total_workers": self.world_size * self.virtual_workers,
            },
            global_rank=self.global_rank,
            wandb_run=self.wandb,
        )

        self.virtual_models = []
        if self.global_rank == 0:
            logger.info(f"Initializing {self.virtual_workers} virtual model(s)...")

        for i in range(self.virtual_workers):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            arch = getattr(self.config, "model_architecture", None)
            if not arch or (
                arch == "llama" and "architecture" in self.model_hparams
            ):
                arch = self.model_hparams.get("architecture", "llama")

            vm = build_model(
                model_hparams=self.model_hparams,
                vocab_size=len(self.tokenizer),
                architecture=arch,
                torchtitan_init=getattr(self.config, "torchtitan_init", False),
                device=self.device,
                use_compile=bool(self.config.use_compile),
            )
            self.virtual_models.append(vm)

            worker_seed = self.seed + self.global_rank + i * self.world_size + 1
            torch.manual_seed(worker_seed)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.inner_optimizers, self.inner_schedulers = (
            build_inner_optimizers_and_schedulers(
                virtual_models=self.virtual_models,
                config=self.config,
                global_rank=self.global_rank,
            )
        )
        self.outer_optimizer, self.outer_scheduler = (
            build_outer_optimizer_and_scheduler(
                virtual_models=self.virtual_models,
                config=self.config,
                virtual_workers=self.virtual_workers,
                process_group=dist.group.WORLD if self.world_size > 1 else None,
                global_rank=self.global_rank,
            )
        )

        if self.config.inner_outer_strategy == "simple_accum":
            self.strategy = SimpleAccum(
                self.device,
                self.world_size,
                self.global_rank,
                self.tokenizer,
                self.config,
            )
        elif self.config.inner_outer_strategy == "diloco":
            self.strategy = Diloco(
                self.device,
                self.world_size,
                self.global_rank,
                self.tokenizer,
                self.config,
            )
        else:
            raise ValueError(
                f"Unknown inner_outer_strategy: {self.config.inner_outer_strategy}"
            )

        self.train_loaders = []
        self.val_loader = None
        self.lang_val_loaders: Dict[str, Any] = {}
        self.worker_val_loaders = []
        self._setup_dataloaders()

        # Update saved experiment config with resolved dataset & partition metadata
        if self.global_rank == 0:
            self.tracker.config_dict["dataset_metadata"] = {
                k: v
                for k, v in self.dataset_metadata.items()
                if k in ("dataset", "tokenizer", "reproducibility", "tokens", "partition_manifests")
            }
            if self.train_manifest is not None:
                self.tracker.config_dict["resolved_partition_info"] = {
                    "strategy": self.train_manifest.strategy,
                    "num_workers": self.train_manifest.num_workers,
                    "token_budget": self.train_manifest.token_budget,
                    "sequence_length": self.train_manifest.sequence_length,
                    "metadata": self.train_manifest.metadata,
                }
            self.tracker.save_config()

        self.global_step = 0
        self.best_loss = float("inf")
        self.best_round: Optional[int] = None
        if self.config.resume_from_checkpoint and self.config.load_checkpoint:
            self._load_checkpoint(self.config.load_checkpoint)

        self._setup_timing_logger()

        if self.global_rank == 0:
            os.makedirs(self.config.save_path, exist_ok=True)

    def _init_torchtitan_weights(self, model, std: float = 0.02):
        init_torchtitan_weights(model, std=std)

    def _setup_dataloaders(self):
        """Set up training and validation dataloaders and validate dataset & partition compatibility."""
        self.dataset_metadata = verify_frozen_dataset(self.config.shards_path)

        effective_world_size = self.world_size * self.virtual_workers
        partition_strategy = getattr(self.config, "partition_strategy", "iid").lower()
        partition_file = getattr(self.config, "partition_file", None)

        train_manifest: Optional[PartitionManifest] = None
        if partition_file:
            p_path = Path(partition_file)
            if p_path.is_file():
                train_manifest = load_partition_manifest(p_path)
            else:
                partitioner = build_partitioner(partition_strategy)
                train_manifest = partitioner.create_partitions(
                    shards_path=self.config.shards_path,
                    split="train",
                    num_workers=effective_world_size,
                    token_budget=self.config.token_budget,
                    sequence_length=self.config.sequence_length,
                    shard_token_size=self.config.shard_token_size,
                    seed=self.seed,
                )
                if self.global_rank == 0:
                    try:
                        save_partition_manifest(train_manifest, p_path)
                        logger.info(f"Created and saved training partition manifest: {p_path}")
                    except Exception as e:
                        logger.warning(f"Could not save partition manifest to {p_path}: {e}")
        else:
            auto_manifest = (
                Path(self.config.shards_path)
                / f"partitions_train_{partition_strategy}_w{effective_world_size}.json"
            )
            if auto_manifest.is_file():
                candidate = load_partition_manifest(auto_manifest)
                if (
                    candidate.token_budget == self.config.token_budget
                    and candidate.sequence_length == self.config.sequence_length
                    and candidate.num_workers == effective_world_size
                ):
                    train_manifest = candidate
                    self.config.partition_file = str(auto_manifest)
                    if self.global_rank == 0:
                        logger.info(
                            f"Using saved training partition manifest: {auto_manifest}"
                        )
            if train_manifest is None:
                partitioner = build_partitioner(partition_strategy)
                train_manifest = partitioner.create_partitions(
                    shards_path=self.config.shards_path,
                    split="train",
                    num_workers=effective_world_size,
                    token_budget=self.config.token_budget,
                    sequence_length=self.config.sequence_length,
                    shard_token_size=self.config.shard_token_size,
                    seed=self.seed,
                )

        validate_partition_manifest(
            manifest=train_manifest,
            shards_path=self.config.shards_path,
            expected_workers=effective_world_size,
            expected_token_budget=self.config.token_budget,
            expected_sequence_length=self.config.sequence_length,
            expected_strategy=partition_strategy,
        )
        self.train_manifest = train_manifest

        for i in range(self.virtual_workers):
            effective_rank = self.global_rank * self.virtual_workers + i
            worker_part = (
                train_manifest.workers.get(effective_rank)
                if train_manifest is not None
                else None
            )

            train_dataset = ShardedGPUDataset(
                shards_path=self.config.shards_path,
                token_budget=self.config.token_budget,
                sequence_length=self.config.sequence_length,
                rank=effective_rank,
                world_size=effective_world_size,
                device=self.device,
                shard_token_size=self.config.shard_token_size,
                split="train",
                pin_to_gpu=self.config.pin_data_to_gpu,
                worker_partition=worker_part,
            )

            worker_loader_gen = torch.Generator()
            worker_loader_gen.manual_seed(self.seed + effective_rank + 1)

            train_loader = get_dataloader(
                dataset=train_dataset,
                batch_size=self.config.micro_batch_size,
                shuffle=True,
                generator=worker_loader_gen,
            )
            self.train_loaders.append(train_loader)

        if self.global_rank == 0 and getattr(self.config, "save_partitions", True):
            try:
                manifest_to_save = train_manifest or build_partitioner(
                    partition_strategy
                ).create_partitions(
                    shards_path=self.config.shards_path,
                    split="train",
                    num_workers=effective_world_size,
                    token_budget=self.config.token_budget,
                    sequence_length=self.config.sequence_length,
                    shard_token_size=self.config.shard_token_size,
                    seed=self.seed,
                )
                save_partition_manifest(
                    manifest_to_save,
                    Path(self.config.save_path)
                    / f"partitions_train_{partition_strategy}_w{effective_world_size}.json",
                )
            except Exception as e:
                logger.warning(f"Could not save training partition manifest: {e}")




        val_shards_exist = (
            len(list(Path(self.config.shards_path).glob("valid_*.npy"))) > 0
        )

        if val_shards_exist and self.config.eval_frequency > 0:
            val_token_budget = getattr(self.config, "val_token_budget", None) or min(
                16_777_216, int(self.config.token_budget * 0.1)
            )
            lang_val_shards = discover_language_shards(
                self.config.shards_path, "valid"
            )
            if lang_val_shards:
                num_langs = len(lang_val_shards)
                per_lang_val_budget = val_token_budget // num_langs
                for lang, shard_name, shard_tokens in lang_val_shards:
                    eff_lang_val = min(shard_tokens, per_lang_val_budget)
                    tok_per_gpu = eff_lang_val // self.world_size
                    l_start = self.global_rank * tok_per_gpu
                    l_end = (
                        eff_lang_val
                        if self.global_rank == self.world_size - 1
                        else l_start + tok_per_gpu
                    )
                    lang_wp = WorkerPartition(
                        worker_id=self.global_rank,
                        spans=[
                            TokenSpan(
                                shard=shard_name,
                                local_start=int(l_start),
                                local_end=int(l_end),
                                global_start=int(l_start),
                                global_end=int(l_end),
                            )
                        ],
                        total_tokens=int(l_end - l_start),
                        num_sequences=int(l_end - l_start)
                        // self.config.sequence_length,
                    )
                    lang_val_ds = ShardedGPUDataset(
                        shards_path=self.config.shards_path,
                        token_budget=eff_lang_val,
                        sequence_length=self.config.sequence_length,
                        rank=self.global_rank,
                        world_size=self.world_size,
                        device=self.device,
                        shard_token_size=self.config.shard_token_size,
                        split="valid",
                        pin_to_gpu=self.config.pin_data_to_gpu,
                        worker_partition=lang_wp,
                    )
                    self.lang_val_loaders[lang] = get_dataloader(
                        dataset=lang_val_ds,
                        batch_size=self.config.micro_batch_size,
                        shuffle=False,
                    )
                if self.global_rank == 0:
                    logger.info(
                        f"Loaded multilingual validation loaders for languages: {list(self.lang_val_loaders.keys())}"
                    )
            else:
                val_dataset = ShardedGPUDataset(
                    shards_path=self.config.shards_path,
                    token_budget=val_token_budget,
                    sequence_length=self.config.sequence_length,
                    rank=self.global_rank,
                    world_size=self.world_size,
                    device=self.device,
                    shard_token_size=self.config.shard_token_size,
                    split="valid",
                    pin_to_gpu=self.config.pin_data_to_gpu,
                )
                self.val_loader = get_dataloader(
                    dataset=val_dataset,
                    batch_size=self.config.micro_batch_size,
                    shuffle=False,
                )
                if self.global_rank == 0:
                    logger.info(
                        f"Validation dataset loaded with {len(val_dataset)} samples per GPU"
                    )

            if getattr(self.config, "per_worker_eval", False):
                eval_manifest = None
                if getattr(self.config, "eval_partition_file", None):
                    ep_path = Path(self.config.eval_partition_file)
                    if ep_path.is_file():
                        eval_manifest = load_partition_manifest(ep_path)
                        if eval_manifest.num_workers != effective_world_size:
                            raise ValueError(
                                f"Eval partition manifest '{self.config.eval_partition_file}' was "
                                f"generated for {eval_manifest.num_workers} workers, but current "
                                f"run uses {effective_world_size} workers."
                            )
                    else:
                        eval_manifest = build_partitioner(
                            partition_strategy
                        ).create_partitions(
                            shards_path=self.config.shards_path,
                            split="valid",
                            num_workers=effective_world_size,
                            token_budget=val_token_budget,
                            sequence_length=self.config.sequence_length,
                            shard_token_size=self.config.shard_token_size,
                            seed=self.seed,
                        )
                        if self.global_rank == 0:
                            try:
                                save_partition_manifest(eval_manifest, ep_path)
                                logger.info(
                                    f"Created and saved eval partition manifest: {ep_path}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Could not save eval partition manifest to {ep_path}: {e}"
                                )
                else:
                    auto_eval_manifest = (
                        Path(self.config.shards_path)
                        / f"partitions_valid_{partition_strategy}_w{effective_world_size}.json"
                    )
                    if auto_eval_manifest.is_file():
                        candidate_eval = load_partition_manifest(auto_eval_manifest)
                        if (
                            candidate_eval.token_budget == val_token_budget
                            and candidate_eval.sequence_length
                            == self.config.sequence_length
                        ):
                            eval_manifest = candidate_eval
                    if eval_manifest is None:
                        eval_manifest = build_partitioner(
                            partition_strategy
                        ).create_partitions(
                            shards_path=self.config.shards_path,
                            split="valid",
                            num_workers=effective_world_size,
                            token_budget=val_token_budget,
                            sequence_length=self.config.sequence_length,
                            shard_token_size=self.config.shard_token_size,
                            seed=self.seed,
                        )
                for i in range(self.virtual_workers):
                    effective_rank = self.global_rank * self.virtual_workers + i
                    w_eval_part = (
                        eval_manifest.workers.get(effective_rank)
                        if eval_manifest is not None
                        else None
                    )
                    w_val_ds = ShardedGPUDataset(
                        shards_path=self.config.shards_path,
                        token_budget=val_token_budget,
                        sequence_length=self.config.sequence_length,
                        rank=effective_rank,
                        world_size=effective_world_size,
                        device=self.device,
                        shard_token_size=self.config.shard_token_size,
                        split="valid",
                        pin_to_gpu=self.config.pin_data_to_gpu,
                        worker_partition=w_eval_part,
                    )
                    self.worker_val_loaders.append(
                        get_dataloader(
                            dataset=w_val_ds,
                            batch_size=self.config.micro_batch_size,
                            shuffle=False,
                        )
                    )
        else:
            if self.global_rank == 0 and self.config.eval_frequency > 0:
                logger.warning(
                    f"No validation shards found in {self.config.shards_path}. Skipping validation."
                )

    def evaluate(self):
        """Run global (and per-language if multilingual) validation evaluation."""
        if self.lang_val_loaders:
            return evaluate_multilingual(
                model=self.virtual_models[0],
                lang_val_loaders=self.lang_val_loaders,
                tokenizer=self.tokenizer,
                device=self.device,
                world_size=self.world_size,
                global_rank=self.global_rank,
            )
        return evaluate_global(
            model=self.virtual_models[0],
            val_loader=self.val_loader,
            tokenizer=self.tokenizer,
            device=self.device,
            world_size=self.world_size,
            global_rank=self.global_rank,
        )

    def _setup_timing_logger(self):
        if self.global_rank != 0 or not getattr(self.config, "debug", False):
            self.timing_logger = None
            return

        self.timing_logger = logging.getLogger("timing")
        self.timing_logger.setLevel(logging.INFO)
        self.timing_logger.propagate = False
        self.timing_logger.handlers.clear()

        log_dir = os.path.dirname(self.config.save_path)
        if not log_dir:
            log_dir = "."
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            log_dir, f"timing_{self.config.outer_optimizer}_{timestamp}.log"
        )

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.timing_logger.addHandler(file_handler)

        self.timing_logger.info(
            f"Timing log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.timing_logger.info(
            f"Configuration: optimizer={self.config.outer_optimizer}, lr={self.config.outer_learning_rate}, "
            f"world_size={self.world_size}, batch_size={self.config.batch_size}"
        )
        self.timing_logger.info("-" * 80)

    def log_timing(self, message):
        if self.global_rank == 0 and self.timing_logger is not None:
            self.timing_logger.info(message)


    def run(self):
        """Main synchronous distributed training loop."""
        for window in range(self.global_step, self.config.max_steps):
            if self.global_rank == 0:
                logger.info(
                    f"\n{'-' * 40} Window: {window}/{self.config.max_steps} {'-' * 40}"
                )
                if self.config.debug:
                    self.log_timing(f"Window {window} - Starting gradient accumulation")
                Timer.reset()

            with Timer("window_total", enabled=True):
                if self.global_rank == 0:
                    logger.info("Start accumulating gradients...")

                for vm in self.virtual_models:
                    vm.zero_grad(set_to_none=True)

                with Timer("inner_step"):
                    per_gpu_metrics = {
                        "total_loss": 0,
                        "loss_after_gather": 0,
                        "batch_count": 0,
                        "batch_tokens": 0,
                    }
                    for vmodel, voptim, vsched, vloader in zip(
                        self.virtual_models,
                        self.inner_optimizers,
                        self.inner_schedulers,
                        self.train_loaders,
                    ):
                        virtual_metrics = self.strategy.inner_step(
                            vmodel, vloader, voptim, vsched
                        )

                        per_gpu_metrics["total_loss"] += (
                            virtual_metrics["total_loss"] / self.virtual_workers
                        )
                        per_gpu_metrics["loss_after_gather"] += (
                            virtual_metrics["loss_after_gather"] / self.virtual_workers
                        )
                        per_gpu_metrics["batch_count"] += virtual_metrics["batch_count"]
                        per_gpu_metrics["batch_tokens"] += virtual_metrics[
                            "batch_tokens"
                        ]

                with Timer("reduce_metrics"):
                    metrics_to_reduce = torch.tensor(
                        [
                            per_gpu_metrics["total_loss"],
                            per_gpu_metrics["batch_count"],
                            per_gpu_metrics["batch_tokens"],
                            per_gpu_metrics["loss_after_gather"],
                        ],
                        device=self.device,
                    )
                    if self.world_size > 1:
                        dist.all_reduce(metrics_to_reduce, op=dist.ReduceOp.SUM)

                    loss_after_inner = metrics_to_reduce[0].item() / self.world_size
                    batch_count = int(metrics_to_reduce[1].item())
                    batch_tokens = int(metrics_to_reduce[2].item())
                    loss_after_gather = metrics_to_reduce[3].item() / self.world_size

                is_eval_window = self.config.eval_frequency > 0 and (
                    (window + 1) % self.config.eval_frequency == 0
                    or window == self.config.max_steps - 1
                )

                local_worker_val_metrics: Dict[str, Any] = {}
                if (
                    is_eval_window
                    and getattr(self.config, "per_worker_eval", False)
                    and self.worker_val_loaders
                ):
                    with Timer("local_worker_validation"):
                        local_worker_val_metrics = evaluate_per_worker(
                            virtual_models=self.virtual_models,
                            worker_val_loaders=self.worker_val_loaders,
                            tokenizer=self.tokenizer,
                            device=self.device,
                            world_size=self.world_size,
                            global_rank=self.global_rank,
                            process_group=dist.group.WORLD
                            if self.world_size > 1
                            else None,
                        )

                with Timer("outer_step"):
                    self.strategy.outer_step(
                        self.virtual_models,
                        self.outer_optimizer,
                        self.outer_scheduler,
                    )

            val_metrics = None
            worker_val_metrics: Dict[str, Any] = {}
            eval_elapsed_sec: Optional[float] = None
            if is_eval_window:
                with Timer("validation"):
                    val_metrics = self.evaluate()
                    if (
                        getattr(self.config, "per_worker_eval", False)
                        and self.worker_val_loaders
                    ):
                        worker_val_metrics = evaluate_per_worker(
                            virtual_models=self.virtual_models,
                            worker_val_loaders=self.worker_val_loaders,
                            tokenizer=self.tokenizer,
                            device=self.device,
                            world_size=self.world_size,
                            global_rank=self.global_rank,
                            process_group=dist.group.WORLD
                            if self.world_size > 1
                            else None,
                        )
                eval_elapsed_sec = time.perf_counter() - self.process_start_time

            if self.global_rank == 0:
                if self.config.debug:
                    Timer.summarize(self.timing_logger)

                window_stats = Timer.get_stats("window_total")
                tokens_per_sec = (
                    batch_tokens / window_stats["last"] if window_stats else 0
                )
                logger.info(
                    f"Window {window}: Processing rate: {tokens_per_sec:.1f} tokens/sec"
                )
                logger.info(
                    f"Window {window} completed: loss: {loss_after_inner:.4f}, "
                    f"tokens: {batch_tokens}, lr: {self.outer_optimizer.param_groups[0]['lr']:.6f}"
                )

                step_log_dict: Dict[str, Any] = {
                    "train/loss_after_inner": loss_after_inner,
                    "train/perplexity": float(np.exp(min(loss_after_inner, 20))),
                    "train/loss_after_gather": loss_after_gather,
                    "train/perplexity_after_gather": float(
                        np.exp(min(loss_after_gather, 20))
                    ),
                    "train/total_tokens": batch_tokens,
                    "train/batch_count": batch_count,
                    "train/outer_learning_rate": self.outer_optimizer.param_groups[0][
                        "lr"
                    ],
                    "train/inner_learning_rate": self.inner_optimizers[0].param_groups[
                        0
                    ]["lr"]
                    if self.inner_optimizers[0] is not None
                    else 0,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/window": window,
                }

                if hasattr(self.outer_optimizer, "get_diagnostics"):
                    opt_diagnostics = self.outer_optimizer.get_diagnostics()
                    if isinstance(opt_diagnostics, dict):
                        for diag_key, diag_val in opt_diagnostics.items():
                            step_log_dict[diag_key] = diag_val
                            step_log_dict[f"train/{diag_key}"] = diag_val

                if val_metrics is not None:
                    if val_metrics["val_loss"] < self.best_loss:
                        self.best_loss = val_metrics["val_loss"]
                        self.best_round = window

                    elapsed_s = (
                        float(eval_elapsed_sec)
                        if eval_elapsed_sec is not None
                        else float(time.perf_counter() - self.process_start_time)
                    )
                    elapsed_m = elapsed_s / 60.0

                    step_log_dict.update(
                        {
                            "eval/valid_loss": val_metrics["val_loss"],
                            "eval/valid_perplexity": val_metrics["val_perplexity"],
                            "eval/valid_tokens": val_metrics["val_tokens"],
                            "eval/valid_batches": val_metrics["val_batches"],
                            "eval/best_valid_loss": self.best_loss,
                            "eval/best_valid_round": self.best_round,
                            "eval/elapsed_sec": elapsed_s,
                            "eval/elapsed_min": elapsed_m,
                            "eval/elapsed_time_sec": elapsed_s,
                            "eval/elapsed_time_min": elapsed_m,
                        }
                    )
                    for k, v in val_metrics.items():
                        if k.startswith("valid_loss_") or k.startswith(
                            "valid_perplexity_"
                        ) or k.startswith("valid_tokens_") or k == "val_loss_equal_language":
                            step_log_dict[f"eval/{k}"] = v

                if local_worker_val_metrics:
                    for k, v in local_worker_val_metrics.items():
                        step_log_dict[f"eval/local_{k}"] = v

                if worker_val_metrics:
                    for k, v in worker_val_metrics.items():
                        step_log_dict[f"eval/{k}"] = v

                self.tracker.log_metrics(step=window, metrics=step_log_dict)

                if (
                    self.config.checkpoint_frequency > 0
                    and (window + 1) % self.config.checkpoint_frequency == 0
                ):
                    ckpt_loss = (
                        val_metrics["val_loss"]
                        if val_metrics is not None
                        else loss_after_inner
                    )
                    self._save_checkpoint(window, ckpt_loss)

                if window == self.config.max_steps - 1 and getattr(
                    self.config, "save_final_diagnostics", False
                ):
                    self._save_final_diagnostics(window)

            self.global_step += 1

        if self.global_rank == 0:
            total_elapsed_s = float(time.perf_counter() - self.process_start_time)
            self.tracker.save_summary(
                extra_summary={
                    "best_valid_loss": self.best_loss
                    if self.best_loss < float("inf")
                    else None,
                    "best_valid_round": self.best_round,
                    "total_elapsed_sec": total_elapsed_s,
                    "total_elapsed_min": total_elapsed_s / 60.0,
                }
            )


    def _save_final_diagnostics(self, window: int):
        """Optional diagnostic export of final parameters and momentum buffers (disabled by default)."""
        model_to_save = self.virtual_models[0]
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        params_and_grads = {}
        for idx, (name, param) in enumerate(model_to_save.named_parameters()):
            param_dict = {
                "data": param.data.cpu().clone(),
                "grad": param.grad.cpu().clone() if param.grad is not None else None,
            }
            if hasattr(self.outer_optimizer, "momentum_buffers") and idx < len(
                self.outer_optimizer.momentum_buffers
            ):
                buf = self.outer_optimizer.momentum_buffers[idx]
                param_dict["momentum_buffer"] = (
                    buf.cpu().clone() if isinstance(buf, torch.Tensor) else None
                )
            elif (
                param in self.outer_optimizer.state
                and "momentum_buffer" in self.outer_optimizer.state[param]
            ):
                param_dict["momentum_buffer"] = (
                    self.outer_optimizer.state[param]["momentum_buffer"].cpu().clone()
                )
            else:
                param_dict["momentum_buffer"] = None
            params_and_grads[name] = param_dict

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.run_name}_{timestamp}_final_model_params_and_grads.pt"
        save_file = os.path.join(self.config.save_path, filename)
        torch.save(params_and_grads, save_file)
        logger.info(f"Saved final model parameters and gradients to {save_file}")

    def _save_checkpoint(self, window, loss):
        """Save model checkpoint."""
        is_best = loss <= self.best_loss and (
            self.best_round == window or loss < self.best_loss
        )
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_round = window

        model_to_save = self.virtual_models[0]
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "outer_optimizer_state_dict": self.outer_optimizer.state_dict(),
            "inner_optimizer_state_dict": self.inner_optimizers[0].state_dict()
            if self.inner_optimizers[0] is not None
            else None,
            "outer_scheduler_state_dict": self.outer_scheduler.state_dict()
            if self.outer_scheduler is not None
            else None,
            "inner_scheduler_state_dict": self.inner_schedulers[0].state_dict()
            if self.inner_schedulers[0] is not None
            else None,
            "window": window,
            "global_step": self.global_step,
            "loss": loss,
            "best_loss": self.best_loss,
            "config": vars(self.config),
        }

        checkpoint_path = os.path.join(
            self.config.save_path, f"checkpoint_window_{window}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        latest_path = os.path.join(self.config.save_path, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.config.save_path, "checkpoint_best.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with loss: {self.best_loss:.4f}")

    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Checkpoint not found at {checkpoint_path}, starting from scratch"
            )
            return

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        for vm in self.virtual_models:
            if hasattr(vm, "module"):
                vm.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                vm.load_state_dict(checkpoint["model_state_dict"])

        self.outer_optimizer.load_state_dict(checkpoint["outer_optimizer_state_dict"])
        for inner_opt in self.inner_optimizers:
            if (
                inner_opt is not None
                and checkpoint["inner_optimizer_state_dict"] is not None
            ):
                inner_opt.load_state_dict(checkpoint["inner_optimizer_state_dict"])

        if (
            self.outer_scheduler is not None
            and checkpoint.get("outer_scheduler_state_dict") is not None
        ):
            self.outer_scheduler.load_state_dict(
                checkpoint["outer_scheduler_state_dict"]
            )
        for inner_sched in self.inner_schedulers:
            if (
                inner_sched is not None
                and checkpoint.get("inner_scheduler_state_dict") is not None
            ):
                inner_sched.load_state_dict(checkpoint["inner_scheduler_state_dict"])

        self.global_step = checkpoint.get(
            "global_step", checkpoint.get("window", 0) + 1
        )
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        logger.info(
            f"Resumed from step {self.global_step} with best loss {self.best_loss:.4f}"
        )

    def cleanup(self):
        """Clean up distributed resources."""
        if self.world_size > 1 and dist.is_initialized():
            destroy_process_group()

        if self.wandb is not None:
            self.wandb.finish()



