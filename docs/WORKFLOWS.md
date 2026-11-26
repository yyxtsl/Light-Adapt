# Portable workflows

Run commands from the repository root in the recorded environment. Each command
creates a new output directory. Dataset preparation obtains article text through
the official providers. Model weights and benchmark inputs are separate assets.

## Re-DocRED controls and logit export

`reproduce_redocred.py` accepts C, B, D, G and U variants. C/B share `source.pth`.
D uses the separately trained `no-ttm-source.pth`. U copies the head projection
into an independent tail projection before adaptation. G uses only the generic
RoBERTa files and freshly initialized task parameters.

```bash
python reproduce_redocred.py --mode train --variant D --seed 75 \
  --model-dir assets/roberta-large --data-dir assets/redocred \
  --checkpoint assets/checkpoints/no-ttm-source.pth --output-dir runs/D75

python reproduce_redocred.py --mode train --variant G --seed 75 \
  --model-dir assets/roberta-large --data-dir assets/redocred --output-dir runs/G75

python reproduce_redocred.py --mode train --variant U --seed 75 \
  --model-dir assets/roberta-large --data-dir assets/redocred \
  --checkpoint assets/checkpoints/source.pth --output-dir runs/U75

python reproduce_redocred.py --mode evaluate --variant B --split dev --export-logits \
  --model-dir assets/roberta-large --data-dir assets/redocred \
  --checkpoint assets/checkpoints/redocred-B75.pth --output-dir runs/B75-dev-logits
```

The last command writes predictions, scores, `logits.npy` and `pair_order.json`.
Logit columns are TH followed by the 96 relation IDs. Pair order is constructed
from document inputs, and model inference receives no gold labels.

## DWIE data preparation

Use the official DWIE downloader to populate `DWIE/data/annos_with_content`.
Keep the matching official `data/article_id_to_url.json`. Prepare the fixed
602/100 train/Dev partition with the pinned tokenizer.

```bash
python reproduce_dwie.py --mode prepare \
  --annotations-dir ../DWIE/data/annos_with_content \
  --article-index ../DWIE/data/article_id_to_url.json \
  --model-dir assets/roberta-large --output-dir assets/dwie-prepared
```

Preparation checks article hashes, mention offsets, concept IDs and the training
relation vocabulary. All concepts and self-pairs are retained. Development gold
remains in a separate scoring file, including relations outside the training
vocabulary. The command writes `features/train.jsonl`, `features/dev.jsonl`,
`gold/train.json` and `gold/dev.json`. Add `--include-test` explicitly to prepare
the 100 official Test documents in a separate new output directory. Test content
is required only when reproducing the paper's final Test evaluations.

## DWIE target adaptation and evaluation

```bash
python reproduce_dwie.py --mode train --variant B --split dev \
  --model-dir assets/roberta-large --data-dir assets/dwie-prepared \
  --checkpoint assets/checkpoints/source.pth --output-dir runs/DWIE-B75

python reproduce_dwie.py --mode evaluate --variant B --split dev \
  --model-dir assets/roberta-large --data-dir assets/dwie-prepared \
  --checkpoint assets/checkpoints/dwie-B75.pth --output-dir runs/DWIE-B75-dev
```

For C use `--variant C` and the shared full source. For D use `--variant D` and
the no-TTM source. For G use `--variant G` and omit `--checkpoint` during
training. Evaluation always loads the complete trained target checkpoint.
DWIE uses target seed 75, 30 epochs, four-document logical groups, a one-document
microbatch and pair-count-weighted gradient accumulation. Selection uses full-ID
Dev micro-F1 after complete epochs. The official scorer supplies supplemental
soft and hard relation scores.

## Symmetric calibration from saved Dev logits

```bash
python reproduce_diagnostics.py calibrate \
  --logits runs/B75-dev-logits/logits.npy \
  --pair-order runs/B75-dev-logits/pair_order.json \
  --dev-gold assets/redocred/dev_revised.json --output-dir runs/B75-calibration
```

Run the same command separately for C75 and C75-off exports. The search uses the
fixed 4,001-point grid [-100,100] at step 0.05 and the original tie rule. This
stage reads saved arrays and Dev labels, with no model execution or training.
The selected scores describe Dev selection. The code performs no automatic
transfer of selected thresholds to Test.

## Dev latency measurements

```bash
python reproduce_diagnostics.py benchmark-redocred --batch-size 4 \
  --model-dir assets/roberta-large --data-dir assets/redocred \
  --checkpoint-c assets/checkpoints/redocred-C75.pth \
  --checkpoint-b assets/checkpoints/redocred-B75.pth --output-dir runs/latency-redocred

python reproduce_dwie.py --mode benchmark --split dev \
  --model-dir assets/roberta-large --data-dir assets/dwie-prepared \
  --checkpoint-c assets/checkpoints/dwie-C75.pth \
  --checkpoint-b assets/checkpoints/dwie-B75.pth --output-dir runs/latency-dwie
```

Use `--batch-size 1` for Re-DocRED per-document timings and the four entity-count
bins. The new commands record their timing boundaries and actual environment.
They measure C followed by B with three passes per model. The paper's archived
timing sessions used their documented alternating/rotating order. Treat new
timing measurements as new measurements, with their own machine and protocol.
Re-DocRED timing excludes batch preparation and loading. DWIE timing includes
window tensor construction and input transfer.

## Verification

```bash
python -B verify_repository.py
python -B check_reproduce_redocred_cli.py
python -B check_portable_workflows.py
```

These checks exercise file identity, source parsing, preserved counts, command
behavior and CPU data contracts. `docs/PREPARATION_STATUS.md` records whether
actual pretrained-model execution has been checked for the release.
