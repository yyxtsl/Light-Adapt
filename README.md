# Light-Adapt

Code accompanying **Bypass Before Adaptation: Efficient Document-Level Relation
Extraction from Memory-Augmented Models**.

Light-Adapt bypasses the entity-pair TTM pathway before target adaptation. The
study compares it with full-path retention using the same Stage-1 source
checkpoint, target objective and paired target seeds. The repository contains
the recorded implementations, benchmark configuration, reference results and
portable Re-DocRED, DWIE and Dev diagnostic commands.

## Get the code and check the download

```bash
git clone https://github.com/yyxtsl/Light-Adapt.git
cd Light-Adapt
python verify_repository.py
```

You can also use GitHub's **Code > Download ZIP** and run the same check from
the extracted directory. The repository root includes `scripts/`, `meta/`,
`metadata/`, `docs/`, `results/` and `refine-logs/`. The check verifies the full
file list, source checksums and archived metric counts using the standard
library. It runs without loading models or benchmark text.

## Environment

The recorded environment uses Python 3.10.22, PyTorch 2.7.1+cu128,
Transformers 4.34.0 and NumPy 1.24.3. Timing in the paper uses an NVIDIA
RTX 5090 and FP32. Use a project-specific virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

`metadata/runtime.json` records package versions read from the research
environment. GPU memory requirements depend on the selected model and batch
size. Checkpoint averaging requires enough CPU RAM to hold the accumulated
state and one input state.

## Data and model preparation

Prepare the benchmark files and generic RoBERTa-large model using the official
sources and file identities in [Data and models](docs/DATA_AND_MODELS.md).
The document also describes the shared Stage-1 checkpoint and the recorded
target checkpoints.

```text
assets/
  roberta-large/
  redocred/
    train_revised.json
    dev_revised.json
    test_revised.json
  checkpoints/
    source.pth
```

`metadata/dwie/` contains the study's target split and relation vocabulary.
The adapted DWIE evaluation retains all concept IDs, including unanchored
concepts, and uses 602 training, 100 development and 100 test documents with
62 positive relations plus TH.

The exact source checkpoint has been retained by the authors. Its distribution
location is a release-preparation item recorded in `docs/DATA_AND_MODELS.md`.
`metadata/checkpoints.json` lists the source and target checkpoint identities,
their roles and their distribution status.

## Portable Re-DocRED commands

Inspect the command-line interface and validate paths before starting a run.

```bash
python reproduce_redocred.py --help
python -B check_reproduce_redocred_cli.py
python reproduce_redocred.py --mode validate --validate-for train --variant B \
  --model-dir assets/roberta-large --data-dir assets/redocred \
  --checkpoint assets/checkpoints/source.pth
```

The executable modes cover C/B/D/G/U target training, explicit checkpoint evaluation,
C-off evaluation and uniform checkpoint averaging. Each mode accepts explicit
asset and output paths. Training uses train/Dev selection. Evaluation uses an
explicit split, with Dev as the default. See `--help` for the exact arguments.

```bash
# Train one target seed. Use the same source checkpoint for the C/B pair.
python reproduce_redocred.py --mode train --variant B --seed 75 \
  --model-dir assets/roberta-large --data-dir assets/redocred \
  --checkpoint assets/checkpoints/source.pth --output-dir runs/B75

# Evaluate an adapted B checkpoint on Dev.
python reproduce_redocred.py --mode evaluate --variant B --split dev \
  --model-dir assets/roberta-large --data-dir assets/redocred \
  --checkpoint runs/B75/best_model.pth --output-dir runs/B75-dev

# Evaluate post-hoc bypass using the corresponding trained C checkpoint.
python reproduce_redocred.py --mode evaluate --variant C-off --split dev \
  --model-dir assets/roberta-large --data-dir assets/redocred \
  --checkpoint assets/checkpoints/redocred-C75.pth --output-dir runs/C75-off-dev

# Average exactly three Dev-selected checkpoints.
python reproduce_redocred.py --mode average \
  --checkpoints runs/B75/best_model.pth runs/B76/best_model.pth runs/B77/best_model.pth \
  --output-dir runs/B-average
```

Use `--variant C` with the same source for full-path target training. Evaluate
the averaged B checkpoint with `--variant B` and
`--checkpoint runs/B-average/averaged_model.pth`. Test scoring requires the
explicit `--split test` argument. Each output directory is created as a new run.

For a Dev evaluation of the averaged B checkpoint, use

```bash
python reproduce_redocred.py --mode evaluate --variant B --split dev \
  --model-dir assets/roberta-large --data-dir assets/redocred \
  --checkpoint runs/B-average/averaged_model.pth --output-dir runs/B-average-dev
```

The target protocol uses ATGL with alpha 16.5, FP32, batch size 4, 30 epochs,
backbone/head learning rates 3e-5/1e-5, AdamW epsilon 1e-6, weight decay 0.01,
6% warmup and gradient clipping at 1.0. Dev evaluation occurs after the
penultimate update of each epoch, following the recorded implementation.
The earliest checkpoint wins tied Dev scores. Pair-independent memory reads,
canonical entity-pair order, memory-token freezing and bypass placement follow
the recorded C/B implementations.

## Recorded implementation map

| Study component | Source |
|---|---|
| C target adaptation | `scripts/train_ttmre_atgl_human.py` |
| B target adaptation | `scripts/train_ttmre_pair_replacement.py` |
| Re-DocRED paired target repeats | `scripts/train_ttmre_atgl_view_repeat.py`, `scripts/train_ttmre_bypass_repeat.py` |
| D/G/U controls | `scripts/train_ttmre_bypass_source_control.py`, `scripts/train_ttmre_redocred_generic_control.py`, `scripts/train_ttmre_untied_pair_control.py` |
| DWIE C/B/D and final G | `scripts/train_ttmre_dwie_adaptation.py`, `scripts/train_ttmre_dwie_generic_budget_replacement.py` |
| C-off | `scripts/evaluate_ttmre_c_off_control.py` |
| Entity-density latency | `scripts/evaluate_ttmre_entity_scale_latency.py` |
| C/B checkpoint averaging | `scripts/evaluate_ttmre_atgl_soup.py`, `scripts/evaluate_ttmre_lightweight_average.py` |
| Fixed final evaluations | `scripts/evaluate_ttmre_final_fixed_test.py`, `scripts/evaluate_ttmre_dwie_fixed_test.py` |
| Symmetric Dev calibration | `refine-logs/DEV_THRESHOLD_CALIBRATION_20261103/calibration_core.py` |

The recorded scripts preserve their historical run identities and audit
dependencies. Use the explicit-path entry points for new runs:

- `reproduce_redocred.py`: Re-DocRED target adaptation, controls, evaluation and averaging.
- `reproduce_dwie.py`: DWIE preparation, C/B/D/G adaptation, evaluation and Dev timing.
- `reproduce_diagnostics.py`: symmetric calibration from saved Dev logits and Re-DocRED Dev timing.

See [Portable workflows](docs/WORKFLOWS.md) for commands, data preparation and
timing boundaries. The source-pretraining runners are preserved for provenance;
the portable target commands start from the identified source checkpoint.
Run the command/data checks with `python -B check_reproduce_redocred_cli.py`
and `python -B check_portable_workflows.py`. Weight access and actual
model-execution verification are tracked in [Preparation status](docs/PREPARATION_STATUS.md).

## Reference results

`results/` contains machine-readable archived metrics and paired-document
intervals. Re-DocRED C/B three-seed mean Test F1 is 79.915786/80.059914.
Uniform checkpoint averaging gives 80.813853/80.895582.
The primary seed-76/77 B-minus-C mean is +0.074755 percentage points,
with a 95% paired-document interval of [-0.317786, +0.469193].

The measured 500-document Dev warm-forward time decreases by
43.61%–43.84% on Re-DocRED. The 2.03x ratio is the observed batch-1,
high-entity-density single-document forward speedup. The study also reports
the adapted DWIE coverage tradeoff and the exploratory status of averaging.

## Provenance and licensing

This work builds on [TTM-RE](https://github.com/chufangao/TTM-RE), initialized
from revision `6c86abdfbe38d1e12e991ea54cbeac8d057c2219`.
`SOURCE_PROVENANCE.json` records the copied source hashes.
The original upstream instructions are preserved in `README_UPSTREAM.md`.

See [Third-party notices](THIRD_PARTY_NOTICES.md) for the upstream MIT license,
the GPLv3 DWIE scorer and dataset provenance.
