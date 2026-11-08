# Modular Research Framework for Model-Merging Aggregation in DiLoCo (`diloco_merge`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)

A clean, lightweight, modular research codebase for experimenting with model-merging-based outer aggregation in distributed language-model training. Refactored from the official [IsoLoCo](https://github.com/shoroi/isoloco) repository accompanying **"Can Model Merging Improve Aggregation in DiLoCo?"** (Stefan Horoi, Benjamin Thérien, Guy Wolf, Eugene Belilovsky), which builds on the [SparseLoCo](https://github.com/tplr-ai/SparseLoCo) distributed training scaffolding.

All baseline algorithms — **DiLoCo**, **IsoLoCo** (Iso-C with Nesterov momentum), **Iso-C**, **Iso-CTS**, **TSVLoCo**, **MergeLoCo** (TIES / DARE), and the **AdamW** data-parallel baseline — are preserved mathematically unchanged.

---

## 1. Project Structure

```text
diloco_merge/
├── configs/
│   ├── default.yaml                  # Main configuration file with all defaults
│   ├── diloco_178m.yaml              # DiLoCo baseline (178M, 8 workers)
│   ├── isoloco_178m.yaml             # IsoLoCo: Iso-C + Nesterov (178M, 8 workers)
│   ├── isoc_178m.yaml                # Iso-C without outer momentum (178M, 8 workers)
│   ├── isocts_178m.yaml              # Iso-CTS (178M, 8 workers)
│   ├── tsv_178m.yaml                 # TSVLoCo (178M, 8 workers)
│   ├── ties_178m.yaml                # MergeLoCo TIES (178M, 8 workers)
│   ├── dare_178m.yaml                # MergeLoCo DARE (178M, 8 workers)
│   ├── adamw_178m.yaml               # AdamW data-parallel baseline (178M)
│   ├── diloco_512m.yaml              # DiLoCo baseline (512M, 8 workers)
│   └── isoloco_512m.yaml             # IsoLoCo (512M, 8 workers)
├── models/
│   ├── __init__.py
│   ├── registry.py                   # Model & tokenizer builder, registry, Torchtitan init
│   └── configs/
│       ├── 178M_model_hparams.json   # 178M LLaMA architecture hyperparameters
│       └── 512M_model_hparams.json   # 512M LLaMA architecture hyperparameters
├── data/
│   ├── __init__.py
│   ├── preprocess.py                 # Deterministic DCLM preprocessing & metadata recording
│   ├── partitioning.py               # IID / Non-IID partitioning interfaces & manifest I/O
│   └── dataset.py                    # ShardedGPUDataset & DataLoader builders
├── methods/
│   ├── __init__.py
│   ├── base.py                       # Aggregation method registry, presets, & factory
│   ├── diloco.py                     # DiLoCo outer optimizers (Nesterov SGD, SGD, AdamW)
│   ├── isocloco.py                   # Official ISOCLoCo (Iso-C & IsoLoCo)
│   ├── isoctsloco.py                 # Official ISOCTSLoCo (Iso-CTS)
│   ├── tsvloco.py                    # Official TSVLoCo
│   └── mergeloco.py                  # Official MergeLoCo (TIES / DARE)
├── training/
│   ├── __init__.py
│   ├── config.py                     # YAML + CLI configuration loader & normalizer
│   ├── optimizers.py                 # Inner & outer optimizer and scheduler management
│   ├── synchronization.py            # Pseudo-gradient computation & synchronous worker sync
│   ├── strategies.py                 # InnerOuterStrategy, SimpleAccum, Diloco
│   └── trainer.py                    # DistributedModelTrainer & Timer
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py                  # Global & per-worker validation loss evaluation
│   └── metrics.py                    # Config & metrics persistence (JSON/JSONL/W&B)
├── src/tplr/                         # Backward-compatible package re-exports
├── hparams/                          # Original model JSONs and W&B sweep YAMLs
├── scripts/                          # Original SLURM sweep launch scripts
├── pretokenize_data.py               # Entrypoint for dataset preparation & partitioning
├── train.py                          # Entrypoint for distributed training
└── pyproject.toml                    # Package metadata and dependencies
```

---

## 2. Environment Setup

### Prerequisites
- Python 3.11+
- PyTorch with CUDA / NCCL support
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

### Installation
```bash
cd /lustre/hdd/LAS/jannesar-lab/aaasif/diloco_merge
uv sync
source .venv/bin/activate
```
Or with `pip` in an existing virtual/conda environment:
```bash
pip install -e .
```


---

## 3. Dataset Preparation & Reproducibility

Data preparation is strictly separated from training via `pretokenize_data.py` (`data/preprocess.py`). By default, it processes DCLM (`mlfoundations/dclm-baseline-1.0-parquet`) using the `togethercomputer/LLaMA-2-7B-32K` tokenizer.

### Tokenizing & Sharding DCLM
```bash
export DATA_DIR="/path/to/datasets"
python pretokenize_data.py \
  --dataset mlfoundations/dclm-baseline-1.0-parquet \
  --dataset_revision main \
  --tokenizer togethercomputer/LLaMA-2-7B-32K \
  --tokenizer_revision main \
  --output_dir $DATA_DIR/dclm_tokenized-train_valid_test \
  --total_tokens 10e9 \
  --val_tokens 1e9 \
  --test_tokens 1e9 \
  --seed 42 \
  --num_workers 8 \
  --partition_strategy iid
```

### Reproducibility Artifacts Recorded in `--output_dir`
- `train_*.npy`, `valid_000000.npy`, `test_000000.npy`: Pre-tokenized `uint16` token shards.
- `train_files.txt`, `valid_files.txt`, `test_files.txt`, and `source_splits.json`: Exact source parquet file splits. Pass `--split_manifest /path/to/source_splits.json` to reuse fixed file splits.
- `metadata.json`: Records dataset name & commit SHA, tokenizer name & commit SHA, random seeds, shuffle buffer size, target vs. actual token counts, shard manifests, and generated partition manifests.
- `partitions_train_iid_w8.json` and `partitions_valid_iid_w8.json`: Saved worker partition assignments (`TokenSpan` slices per worker) so all methods can be evaluated on the exact same worker data partitions.

### Generating Worker Partition Manifests from Existing Shards
To create or export worker partition manifests for a different worker count (e.g., 16 workers) without re-tokenizing:
```bash
python pretokenize_data.py \
  --create_partitions_only \
  --output_dir $DATA_DIR/dclm_tokenized-train_valid_test \
  --total_tokens 3552440320 \
  --num_workers 16 \
  --partition_strategy iid \
  --seed 42
```

---

## 4. Configuration Options

Experiments are controlled through a simple YAML configuration system (`configs/default.yaml` + optional experiment YAML passed via `--config`). Any CLI flag passed to `train.py` overrides the YAML configuration.

| Section | Key | Description |
| :--- | :--- | :--- |
| `experiment` | `run_name`, `project`, `seed`, `log_wandb`, `debug` | Run identifier, W&B project, global seed, and logging flags. |
| `dataset` | `name`, `shards_path`, `shard_token_size`, `token_budget`, `sequence_length`, `pin_data_to_gpu` | Dataset identifier, tokenized shard directory, sequence length, and total training token budget. |
| `partitioning` | `strategy`, `partition_file`, `save_partitions`, `eval_partition_file` | Partitioning strategy (`iid` baseline or `non_iid`), and paths to load/save deterministic worker partition manifests. |
| `model` | `architecture`, `size`, `hparams_file`, `hparams`, `use_compile`, `torchtitan_init` | Model architecture (`llama`), preset size (`178M`, `512M`), hyperparameter JSON path, and `torch.compile`. |
| `workers` | `num_workers`, `virtual_workers` | Total workers $M$. When `num_workers` is set, `virtual_workers = num_workers // world_size` is computed automatically. |
| `training` | `inner_outer_strategy`, `inner_steps`, `batch_size`, `micro_batch_size`, `max_steps` | Local training steps ($H$), per-worker batch size and micro-batch size, and total outer windows. |
| `inner_optimizer` | `type`, `learning_rate`, `beta2`, `weight_decay`, `warmup_steps` | Inner optimizer (`adamw` or `sgd`) and cosine schedule warmup. |
| `method` | `strategy`, `outer_optimizer`, `learning_rate`, `momentum`, `nesterov`, method-specific keys | Aggregation strategy (`diloco_baseline`, `isoloco`, `isocloco`, `isoctsloco`, `tsvloco`, `mergeloco`, `diloco_sgd_noMom`, `adam_baseline`) and outer hyperparameters. |
| `evaluation` | `eval_frequency`, `val_token_budget`, `per_worker_eval` | Global validation frequency, token budget, and optional per-worker validation loss on fixed partitions. |
| `checkpointing` | `save_path`, `checkpoint_frequency`, `load_checkpoint`, `resume_from_checkpoint`, `save_final_diagnostics` | Checkpoint directory, frequency, resume options, and optional final SVD/gradient diagnostic dump (disabled by default). |

Each run automatically saves `experiment_config.json`, `experiment_config.yaml`, `metrics.jsonl`, `metrics_summary.json`, and `partitions_train_*.json` inside `save_path`.




---

## 5. Example Experiment Commands

Set environment paths before launching:
```bash
export DATA_DIR="/path/to/datasets"
export SAVE_DIR="./outputs/run_1"
```

### Run IsoLoCo (178M, 8 workers across 4 GPUs)
```bash
torchrun --nproc_per_node=4 train.py --config configs/isoloco_178m.yaml
```

### Run DiLoCo Baseline (178M, 8 workers across 4 GPUs)
```bash
torchrun --nproc_per_node=4 train.py --config configs/diloco_178m.yaml
```

### Run Iso-CTS (178M, 8 workers across 4 GPUs)
```bash
torchrun --nproc_per_node=4 train.py --config configs/isocts_178m.yaml
```

### Override Worker Count, Seed, or Hyperparameters via CLI
Changing the number of workers or GPUs does not require any source code changes:
```bash
torchrun --nproc_per_node=4 train.py \
  --config configs/isoloco_178m.yaml \
  --num_workers 16 \
  --inner_steps 50 \
  --seed 123 \
  --per_worker_eval true \
  --partition_file $DATA_DIR/dclm_tokenized-train_valid_test/partitions_train_iid_w16.json
```

---

## 6. Extending the Framework

Adding a new model architecture, dataset/partitioning strategy, or aggregation method does not require modifying `training/trainer.py`.

### Adding a New Model Architecture
Register a builder function in `models/registry.py` using `@register_model`:
```python
from models.registry import register_model

@register_model("my_arch")
def build_my_model(model_hparams: dict, vocab_size: int):
    return MyCausalLM(vocab_size=vocab_size, **model_hparams)
```
Then set `model.architecture: my_arch` and `model.hparams_file: path/to/hparams.json` in your YAML config.

### Adding a New Dataset or Non-IID Partitioning Strategy
- **Dataset**: Pre-tokenize any text corpus into `train_*.npy`, `valid_000000.npy`, and `test_000000.npy` shards via `pretokenize_data.py --dataset <hf_dataset_name> --output_dir <dir>` and point `dataset.shards_path` to `<dir>`.
- **Non-IID Partitioning**: Subclass `NonIIDPartitioner` (or `DataPartitioner`) in `data/partitioning.py` and decorate with `@register_partitioner("my_non_iid")`:
```python
from data.partitioning import NonIIDPartitioner, PartitionManifest, register_partitioner

@register_partitioner("my_non_iid")
class DomainNonIIDPartitioner(NonIIDPartitioner):
    name = "my_non_iid"

    def create_partitions(
        self, shards_path, split, num_workers, token_budget, sequence_length, shard_token_size=1024**3, seed=42, **kwargs
    ) -> PartitionManifest:
        # Return a PartitionManifest mapping worker_id -> WorkerPartition(spans=[TokenSpan(...)])
        ...
```
Then set `partitioning.strategy: my_non_iid` in your YAML config.

### Adding a New Aggregation Algorithm
1. Implement your `torch.optim.Optimizer` subclass in `methods/<your_method>.py`.
2. Register it in `methods/base.py` using `@register_method("<your_method>", handles_own_communication=True)`:
```python
from methods.base import register_method

@register_method("my_merge_loco", handles_own_communication=True)
def _build_my_merge_loco(virtual_models, config, virtual_workers=1, process_group=None):
    return MyMergeLoCo(
        [vm.parameters() for vm in virtual_models],
        lr=config.outer_learning_rate,
        virtual_workers=virtual_workers,
        process_group=process_group,
    )
```
3. Select it in YAML with `method.strategy: my_merge_loco` (or `method.outer_optimizer: my_merge_loco`).

---

## Citation

```bibtex
@article{horoi2026isoloco,
  title  = {Can Model Merging Improve Aggregation in DiLoCo?},
  author = {Horoi, Stefan and Thérien, Benjamin and Wolf, Guy and Belilovsky, Eugene},
  year   = {2026},
  note   = {Preprint. arXiv link forthcoming.}
}
```
