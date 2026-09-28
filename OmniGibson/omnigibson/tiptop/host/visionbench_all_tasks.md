# All-task vision benchmark capture

The `vision-bench` branch in BEHAVIOR-1K provides deterministic simulator queries and
approximate evaluation masks. Inference, reference prompting, calibration, scoring,
and the comparison website live in the **separate** `b1k-submission` repository's
`vision-bench` branch. Equal branch names do not mean equal contents.

Use an existing BEHAVIOR-1K/Isaac Sim environment with dataset access and the matching
TiPToP submodule. The original host uses the approved UV environment; another host can
use its existing `behavior` environment. Do not install the manipulation planner into
the simulator environment. No model weights are needed for capture.

## Frozen comprehensive cohort

`runs/vision/comprehensive_v1/selection_fullcoverage.json` declares **365 snapshots from
365 distinct episodes across all 100 official evaluation tasks**, with three camera
views per snapshot. Every task has at least three snapshots from independent episodes.
The base 300 queries cover the start, middle, and later part of **separate annotated
manipulation segments**, not the beginning, middle, and end of an entire task.

A further 65 metadata-selected queries cover categories absent from the base primary
target list. They bring planned primary coverage to all **220 annotation-eligible rigid
object categories**. This is planned annotation coverage; measured visibility is
reported separately. The normal supplement cap is three per task; one predeclared
fourth query for `sorting_household_items` includes toothpaste. Selection excludes
269 previous query/reference-source episodes. No images, visibility masks, detection
scores, or predictions were read to choose these cases.

Primary targets are annotated manipulation objects. Generic annotations resolve to
all matching task instances, so a generic toy/can annotation does not silently choose
one model. Evaluation labels also contain other task objects and loaded scene objects
matching the supplied task categories, including invisible instances. Particle/material
systems and future entities are explicitly listed in the selection's task records;
they are outside rigid-instance mask scoring. Floor/lawn/agent categories are excluded.
Coverage of all 100 tasks does not mean complete visibility of every object or full task
execution.

## Capture an existing selection

Set paths for your checkout, installed simulation interpreter, datasets, and output.
The original manifests contain source hash receipts and machine-specific paths; use the
`visionbench_all_tasks_rebase.py` tool and verify the companion artifact manifest before
replaying on another machine. Do not edit an archived manifest in place.

```bash
/path/to/sim/python OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_rebase.py \
  --selection /path/to/artifacts/selection_fullcoverage.json \
  --output /path/to/new-run/selection_replay.json \
  --checkout /path/to/BEHAVIOR-1K \
  --dataset /path/to/behavior1k-20k --sim-data /path/to/BEHAVIOR-1K/datasets \
  --catalog /path/to/artifacts/catalog.parquet \
  --inventory /path/to/artifacts/inventory.json \
  --snapshot-root /path/to/new-run/snapshots --capture-root /path/to/new-run/captures \
  --python /path/to/sim/python
```

The tool automatically remaps the original simulator checkout and demo dataset roots.
Use repeated `--path-map /original/root=/new/root` arguments for other shared artifact
roots. It verifies every required annotation, capture-source, catalog, and inventory
hash; refuses nonempty capture outputs or existing snapshot filenames; preserves the
interpreter's virtual-environment symlink; and creates a separate mapping receipt.
The original selection stays byte-identical. Missing optional historical provenance
documents are listed explicitly; they are not read by replay and are not silently
asserted to have been verified. The output includes the required environment and exact
validation/capture/audit command arrays. This validates portable deployment inputs; it
does not promise identical rendering on different simulator versions or hardware.

```bash
export OMNIGIBSON_HEADLESS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K/datasets
export B1K_DEMOS=/path/to/behavior1k-20k
export PYTHONPATH=/path/to/BEHAVIOR-1K/OmniGibson:/path/to/BEHAVIOR-1K/bddl3:/path/to/BEHAVIOR-1K/tiptop

/path/to/sim/python OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_capture.py \
  --selection /path/to/new-run/selection_replay.json \
  --out /path/to/new-run/captures --data "$OMNIGIBSON_DATA_PATH" --validate

# Inspect GPU ownership/headroom first. Maximum two simulator workers.
/path/to/sim/python OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_capture.py \
  --selection /path/to/new-run/selection_replay.json \
  --out /path/to/new-run/captures --data "$OMNIGIBSON_DATA_PATH" --gpus 1 3
```

`--tasks task_name ...` limits a smoke run; `--resume` skips valid completed task
receipts with exactly the same selection, Python, GPU list, and launch provenance.
Each new case independently replays from frame zero using recorded evaluator controls
and zero settle steps. Capture/render and geometry labeling take no physics steps.
The launcher never replaces difficult/invisible cases or terminates another process.
Infrastructure failures remain recorded; retry the same selected case after preserving
failed artifacts and fixing the failure cause.

Outputs per case are `input.json`, `input.npz`, three PNG camera images, `labels.json`,
`labels.npz`, and `labels_strict4mm.npz`, with independent materialization/case/task
receipts. The model receives only the input files. Runtime asset identity, geometry
bounds, target flags, and masks are evaluation-only labels.

## Audit and seal before inference

```bash
/path/to/sim/python OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_audit.py \
  --selection /path/to/new-run/selection_replay.json --captures /path/to/new-run/captures \
  --out /path/to/new-run/sealed --seal
```

Without `--seal`, the command produces a progress audit without pretending incomplete
captures are a final dataset. `--seal` requires every selected case and verifies array
shapes, native Boolean masks, strict4mm subset relations, object pixel counts, frozen
poses, snapshots, selection receipts, and absence of label identity fields in inputs.
Sealed outputs are `query_manifest.json`, `capture_audit.json`,
`capture_quality_summary.json`, and a hash-linked `seal_receipt.json`.

Masks are **geometry-proximity proxies** at 8mm, with a 4mm tolerance sensitivity;
they are not renderer instance ground truth. Saved-query render convergence flags live
in `labels.json["render_convergence"][view]`. Nonconverged views remain in the main
cohort and are explicitly marked for the fixed-threshold quality sensitivity. Recorded
head-depth replay fidelity is reported separately from integrity. `aabb_base` is an
orientation-dependent, base-frame axis-aligned box, not an orientation-invariant native
asset dimension or proof an object is graspable.

## Select a new version

`visionbench_all_tasks_select.py --help` takes explicit official task inventory,
annotated demo catalog, dataset root, repeated episode-exclusion documents, and a fresh
output filename. `visionbench_all_tasks_supplement.py --help` adds metadata-only global
category coverage with explicit per-task caps. Both refuse to overwrite their output.
Changing inputs or selection requires a new versioned run and new frozen receipts.

Metadata-only tests (no simulator launch):

```bash
OMNIGIBSON_HEADLESS=1 /path/to/sim/python -m pytest -q \
  OmniGibson/omnigibson/tiptop/host/test_visionbench_all_tasks.py \
  OmniGibson/omnigibson/tiptop/host/test_visionbench_all_tasks_rebase.py
```
