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

## Verify the capture provenance companion

The main inference data archive contains `plan.json` and the eight native files in
`captures/<case_id>/`. Its separate companion is
`vision-bench-comprehensive-capture-provenance.tar`. Extract them into separate roots:

```bash
export VISION_RUN=/path/to/comprehensive_v1
export VISION_CAPTURE_PROVENANCE=/path/to/capture-provenance
export SIM_PY=/path/to/sim/python
mkdir -p "$VISION_RUN" "$VISION_CAPTURE_PROVENANCE"
# Extract the main inference data archive into "$VISION_RUN" using its handoff guide.
tar -xf vision-bench-comprehensive-capture-provenance.tar -C "$VISION_CAPTURE_PROVENANCE"

"$SIM_PY" OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_package_v2.py verify \
  --provenance "$VISION_CAPTURE_PROVENANCE"
"$SIM_PY" OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_package_v2.py audit-overlay \
  --provenance "$VISION_CAPTURE_PROVENANCE" --main-captures "$VISION_RUN/captures" \
  --mount /path/to/fresh-capture-audit-mount --output /path/to/fresh-capture-audit.json
```

Run these from this simulator checkout. The companion has its own
`ARTIFACT_MANIFEST.json` verifier. `audit-overlay` first checks all 2,920 main capture
files against the original sealed hashes, then runs the independent capture auditor
using the companion's archived snapshots and receipts. The new mount consists of
symlinks; the original selection, case data, and provenance bytes are unchanged. Both
mount and audit output must be new paths. The audit requires CPU dependencies from the
existing simulator environment and does not launch Isaac Sim or run either model.

The companion contains these paths:

- `selection/selection_fullcoverage.json` and preserved initial 300-case selection.
- `metadata/catalog.parquet`, `metadata/inventory.json`, dataset metadata, 100 task
  templates, 100 BDDL definitions, the evaluator robot configuration, and 365 selected
  task-instance configuration JSON files under `metadata/sim_data/`.
- The 365 selected annotation JSON files in `annotations/`, frozen capture sources in
  `source_capture/`, and explicitly named selection ancestry and launch receipts.
  `source_geometry/` and `source_diagnostics/` preserve geometry dependencies and the
  additive audit/packaging sources; `diagnostic_amendment/` preserves the failed earlier
  audit and independent saved-array evidence. Geometry source receipts match the original
  committed simulator and TiPToP sources.
- `snapshots/`, `provenance/case_receipts/`, `provenance/materialization/`,
  `provenance/task_receipts/`, final execution receipts, and the four `sealed/` files.
- `replay_fidelity_summary.json`, comparing every final saved head-depth frame with
  the recorded dataset frame while retaining the separate preliminary depth probe.

Mesh and texture assets, video/action shards, model weights, reference images, and
unrelated experiments are outside this companion. Recapturing still requires the
complete installed simulation and demonstration datasets. Packaged instance/template
JSON files preserve the configurations for inspection and checksum comparison; the
package does not overwrite an installed dataset. The separate inference handoff covers
reference-bank and model dependencies. Rebuilding a newly selected reference bank is
not implied by reproducing this frozen capture cohort.

## Capture an existing selection

Set paths for your checkout, installed simulation interpreter, datasets, and output.
The original manifests contain source hash receipts and machine-specific paths; use the
`visionbench_all_tasks_rebase.py` tool and verify the companion artifact manifest before
replaying on another machine. Do not edit an archived manifest in place.

```bash
/path/to/sim/python OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_rebase.py \
  --selection "$VISION_CAPTURE_PROVENANCE/selection/selection_fullcoverage.json" \
  --output /path/to/new-run/selection_replay.json \
  --checkout /path/to/BEHAVIOR-1K \
  --dataset /path/to/behavior1k-20k --sim-data /path/to/BEHAVIOR-1K/datasets \
  --catalog "$VISION_CAPTURE_PROVENANCE/metadata/catalog.parquet" \
  --inventory "$VISION_CAPTURE_PROVENANCE/metadata/inventory.json" \
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
asserted to have been verified. The output includes the required environment and validation/capture commands. Its
archived audit-command field still names the original auditor; run the explicit v2
audit command below for this amended cohort. This validates portable deployment inputs; it
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

The additive v2 auditor records the pre-TEST diagnostic amendment. The first auditor
incorrectly required every object's 4mm mask to be a subset of its 8mm mask. Three
coffee-station cases exposed contact ownership transfers. The original masks and primary
8mm score remain unchanged. The v2 auditor requires that each tolerance has disjoint
instance masks and that the 4mm foreground union is contained in the 8mm union, and
reports every ownership transfer separately. A synthetic test of the frozen labeler
reproduces one possible NaN/degenerate-triangle mechanism; the cause in these real cases
is not proven because raw meshes and per-pixel distances were not archived. This is a
numerical limitation of approximate geometry labels, not a calibrated accuracy bound.


```bash
/path/to/sim/python OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_audit_v2.py \
  --selection /path/to/new-run/selection_replay.json --captures /path/to/new-run/captures \
  --out /path/to/new-run/sealed --seal
```

Without `--seal`, the command produces a progress audit without pretending incomplete
captures are a final dataset. `--seal` requires every selected case and verifies array
shapes, native Boolean masks, disjoint instances at each tolerance, nested foreground
unions, object pixel counts, frozen
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
  OmniGibson/omnigibson/tiptop/host/test_visionbench_all_tasks_rebase.py \
  OmniGibson/omnigibson/tiptop/host/test_visionbench_all_tasks_fidelity.py \
  OmniGibson/omnigibson/tiptop/host/test_visionbench_all_tasks_package_v2.py
```

## Check replay fidelity against the final saved query

The materialization receipt includes an early head-depth probe. A transient camera or
render settling issue can make that probe disagree with the final saved query even when
the final query converges. Preserve it as provenance, and compare the **saved input
depth** directly with the recorded dataset frame before making fidelity claims:

```bash
/path/to/sim/python OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_fidelity.py \
  --selection /path/to/new-run/selection_replay.json \
  --captures /path/to/new-run/captures --dataset "$B1K_DEMOS" \
  --output /path/to/new-run/replay_fidelity_summary.json --fresh
```

This is an independent diagnostic that can run after capture sealing. It preserves both
`saved_depth_*` and `materialization_depth_*` statistics, per-case input hashes, decoded
frame hashes, episode offsets and timestamps. It also reports joint and base-relative
end-effector differences. It never replaces cases, tunes models, or excludes examples
from the primary score. Base-versus-dead-reckoned differences are against integrated
odometry, not measured world-pose ground truth. The raw replay-depth aggregates in the
capture quality summary describe the preliminary probe; use the fidelity diagnostic's
`saved_depth_*` fields for conclusions about the final model inputs.

## Produce the companion artifact on the capture host

Stage only explicitly named immutable metadata while capture is running. Use a new
staging path. After capture sealing, run the fidelity command above with `--fresh`,
then finalize. Finalization requires complete execution, a valid seal, and all 365
fresh saved-query comparisons; it will not publish a partial capture as complete.

```bash
"$SIM_PY" OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_package_v2.py stage \
  --selection /path/to/comprehensive_v1/selection_fullcoverage.json \
  --sim-data "$OMNIGIBSON_DATA_PATH" --stage /path/to/comprehensive_v1/provenance-stage
"$SIM_PY" OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_package_v2.py finalize \
  --stage /path/to/comprehensive_v1/provenance-stage \
  --archive /path/to/comprehensive_v1/vision-bench-comprehensive-capture-provenance.tar
```

Staging and finalization copy original file bytes and record SHA-256 receipts. They
never edit the frozen selection or active capture sources. A staging directory is not
a publishable artifact. Finalization refuses to overwrite an existing final manifest
or archive. Validate the extracted result with `verify` and `audit-overlay` before
sharing it with a coworker.
