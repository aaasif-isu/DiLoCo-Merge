# Spectral & Model-Merging Aggregation in DiLoCo (`diloco_merge`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/3110/)

A modular research framework for spectral outer aggregation in distributed low-communication language model training (DiLoCo). Built upon the [IsoLoCo](https://github.com/shoroi/isoloco) and [SparseLoCo](https://github.com/tplr-ai/SparseLoCo) distributed training foundations, this repository implements and benchmarks four active outer aggregation methods across 15M TinyGPT (mC4 IID & Non-IID) and 178M Llama (DCLM Pilot, 300M-token, and 3.55B Chinchilla-budget) scales:

1. **DiLoCo (`diloco`)**: Standard pseudo-gradient averaging with outer Nesterov SGD (`methods/diloco.py`).
2. **IsoLoCo (`isoloco`)**: Isotropic Covariance (`Iso-C`) spectral flattening ($\tilde{\sigma}_i = \bar{\sigma}$) with outer Nesterov momentum (`methods/isocloco.py`).
3. **RankLoCo V1 (`erank_iso`)**: $L_1$ Shannon Effective-Rank Tikhonov spectral shaping ($\psi_i = \sigma_i / \sqrt{\sigma_i^2 + \tau_1^2}$, $\tau_1 = \sigma_{\lceil r_{\text{eff}} \rceil}$) with IsoLoCo Frobenius-norm matching ($\tilde{\sigma}_i = \psi_i \frac{\sqrt{R}\bar{\sigma}}{\|\psi\|_2}$) (`methods/erank_iso.py`).
4. **RankLoCo V2 (`rankiso_v2`)**: $L_1$ Shannon Effective-Rank Tikhonov spectral shaping with Spectral-Ceiling L1 normalization ($\tilde{\sigma}_i = \bar{\sigma} \frac{\psi_i}{\psi_1} \le \bar{\sigma}$), preserving IsoLoCo's exact operator-norm ceiling while attenuating low-rank tail noise (`methods/rankiso_v2.py`).

---

## 1. Repository Structure

```text
diloco_merge/
├── README.md                                  # Documentation, methods, configs, & commands
├── LICENSE
├── pyproject.toml                             # Package metadata and dependencies
├── uv.lock
├── train.py                                   # Main distributed training entrypoint
├── pretokenize_data.py                        # Dataset pre-tokenization & partition manifest builder
├── run_178m_dclm_chinchilla.sbatch            # Slurm script: 178M Llama 3.55B Chinchilla benchmark
│
├── methods/                                   # Active outer aggregation optimizers
│   ├── __init__.py
│   ├── base.py                                # Method registry, strategy presets, & factory
│   ├── diloco.py                              # DiLoCo (Nesterov SGD, SGD, AdamW)
│   ├── isocloco.py                            # IsoLoCo (Iso-C + Nesterov)
│   ├── erank_iso.py                           # RankLoCo V1 (Effective-Rank Tikhonov + Frobenius match)
│   └── rankiso_v2.py                          # RankLoCo V2 (Effective-Rank Tikhonov + Spectral-Ceiling L1)
│
├── configs/                                   # Flat YAML experiment configurations
│   ├── default.yaml                           # Base defaults & model/dataset/method presets
│   ├── experiment.yaml                        # Default 15M mC4 experiment configuration
│   ├── diloco_178m.yaml                       # 178M DCLM Chinchilla (3.55B tokens, 225r) — DiLoCo
│   ├── isoloco_178m.yaml                      # 178M DCLM Chinchilla (3.55B tokens, 225r) — IsoLoCo
│   ├── erank_iso_178m.yaml                    # 178M DCLM Chinchilla (3.55B tokens, 225r) — RankLoCo V1
│   ├── pilot_178m_diloco_50r.yaml             # 178M DCLM Pilot (24.6M tokens, 50r) — DiLoCo
│   ├── pilot_178m_isoloco_50r.yaml            # 178M DCLM Pilot (24.6M tokens, 50r) — IsoLoCo
│   ├── pilot_178m_erank_iso_50r.yaml          # 178M DCLM Pilot (24.6M tokens, 50r) — RankLoCo V1
│   ├── pilot_178m_rankiso_v2_50r.yaml         # 178M DCLM Pilot (24.6M tokens, 50r) — RankLoCo V2
│   ├── dclm_178m_diloco_300m.yaml             # 178M DCLM 300M-Token (610r) — DiLoCo
│   ├── dclm_178m_isoloco_300m.yaml            # 178M DCLM 300M-Token (610r) — IsoLoCo
│   ├── dclm_178m_erank_iso_300m.yaml          # 178M DCLM 300M-Token (610r) — RankLoCo V1
│   ├── diloco_iid_300m.yaml                   # 15M mC4 300M-Token (2442r) — DiLoCo
│   ├── isoloco_iid_300m.yaml                  # 15M mC4 300M-Token (2442r) — IsoLoCo
│   ├── erank_iso_iid_300m.yaml                # 15M mC4 300M-Token (2442r) — RankLoCo V1
│   ├── isoloco_iid_50r.yaml                   # 15M mC4 IID (50r) — IsoLoCo
│   ├── erank_iso_iid_50r.yaml                 # 15M mC4 IID (50r) — RankLoCo V1
│   ├── erank_iso_iid_100r.yaml                # 15M mC4 IID (100r) — RankLoCo V1
│   └── erank_iso_noniid_language_100r.yaml    # 15M mC4 Non-IID Language (100r) — RankLoCo V1
│
├── models/                                    # Model registry, TinyGPT & Llama builders, hparams JSONs
│   ├── __init__.py
│   ├── registry.py
│   ├── tinygpt.py
│   └── configs/{15M,178M,512M}_model_hparams.json
├── data/                                      # Dataset loading, streaming sharding, & IID/Non-IID partitioning
│   ├── __init__.py
│   ├── dataset.py
│   ├── partitioning.py
│   ├── preprocess.py
│   └── mc4_preprocess.py
├── training/                                  # Trainer, inner/outer strategies, sync, & config loader
│   ├── __init__.py
│   ├── config.py
│   ├── optimizers.py
│   ├── strategies.py
│   ├── synchronization.py
│   └── trainer.py
├── evaluation/                                # Global/per-worker evaluation & JSONL/W&B metrics tracker
│   ├── __init__.py
│   ├── evaluator.py
│   └── metrics.py
├── scripts/                                   # Comparison & plotting utilities
│   ├── compare_methods.py
│   └── logs/.gitkeep
├── tests/                                     # Mathematical & integration unit tests for active methods
│   ├── test_erank_iso.py
│   └── test_rankiso_v2.py
└── archive/                                   # Archived exploratory methods, configs, tests, & legacy files
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

## 5. Reproducible Experiment Commands

### A. 178M Llama DCLM Chinchilla-Budget Benchmark (3.55B Tokens, 225 Rounds)

Submit the full sequential Slurm benchmark (DiLoCo $\to$ IsoLoCo $\to$ RankLoCo V1 `erank_iso` + post-run summary comparison):
```bash
sbatch run_178m_dclm_chinchilla.sbatch
```
Or run individual methods directly:
```bash
python -u train.py --config configs/diloco_178m.yaml
python -u train.py --config configs/isoloco_178m.yaml
python -u train.py --config configs/erank_iso_178m.yaml
```

### B. 178M Llama DCLM 50-Round Pilot Experiments (24.58M Tokens, $W=4, H=30, \text{bs}=2$)

```bash
python -u train.py --config configs/pilot_178m_diloco_50r.yaml
python -u train.py --config configs/pilot_178m_isoloco_50r.yaml
python -u train.py --config configs/pilot_178m_erank_iso_50r.yaml
python -u train.py --config configs/pilot_178m_rankiso_v2_50r.yaml
```

### C. 178M Llama DCLM 300M-Token Experiments (610 Rounds)

```bash
python -u train.py --config configs/dclm_178m_diloco_300m.yaml
python -u train.py --config configs/dclm_178m_isoloco_300m.yaml
python -u train.py --config configs/dclm_178m_erank_iso_300m.yaml
```

### D. 15M TinyGPT mC4 IID & Non-IID Experiments

```bash
# 50-round and 100-round IID benchmarks
python -u train.py --config configs/isoloco_iid_50r.yaml
python -u train.py --config configs/erank_iso_iid_50r.yaml
python -u train.py --config configs/erank_iso_iid_100r.yaml

# 100-round Non-IID language-partitioned benchmark
python -u train.py --config configs/erank_iso_noniid_language_100r.yaml

# 300M-token (2,442-round) IID benchmarks
python -u train.py --config configs/diloco_iid_300m.yaml
python -u train.py --config configs/isoloco_iid_300m.yaml
python -u train.py --config configs/erank_iso_iid_300m.yaml
```

### E. Post-Run Comparison Summary & Plots

Compare any set of completed experiment directories and generate `comparison.csv`, `convergence_rounds.png`, `runtime_comparison.png`, and `convergence_time.png`:
```bash
python -u scripts/compare_methods.py \
  --dirs \
    outputs/dclm_178m_diloco_iid_w4_h30_r225_s42 \
    outputs/dclm_178m_isoloco_iid_w4_h30_r225_s42 \
    outputs/dclm_178m_erank_iso_iid_w4_h30_r225_s42 \
  --output_dir outputs/comparison_chinchilla_178m
```

### F. Running Unit Tests

```bash
python -m unittest discover -s tests -v
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
