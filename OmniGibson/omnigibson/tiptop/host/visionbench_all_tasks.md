# All-task vision benchmark capture

The `vision-bench` branch in BEHAVIOR-1K provides frozen simulator queries and
approximate evaluation masks. Inference, reference prompting, calibration, scoring,
and the comparison website live in the **separate** `b1k-submission` repository's
`vision-bench` branch. Equal branch names do not mean equal contents.

For an offline checkout, follow the release's START_HERE.md: clone `BEHAVIOR-1K-vision-bench.bundle` on branch `vision-bench`, initialize its `tiptop` directory, fetch HEAD from `tiptop-b2968b7.bundle`, and check out `b2968b7182740487e5276b17b0bed82c4da09fb4` detached. Confirm that TiPToP HEAD equals the simulator checkout's `HEAD:tiptop` gitlink. These source bundles do not include the installed simulator environment or licensed assets/demonstrations.

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
export VISION_RELEASE=/absolute/path/to/vision-bench-release
export VISION_CAPTURE_CHECKOUT=/path/to/BEHAVIOR-1K
export VISION_RUN=/path/to/unpacked/comprehensive_v1
export VISION_RELEASE_RUN="$VISION_RUN"
export VISION_CAPTURE_PROVENANCE=/path/to/new/capture-provenance
export SIM_PY=/path/to/existing/simulator/environment/bin/python
cd "$VISION_CAPTURE_CHECKOUT"
test -f "$VISION_RELEASE_RUN/plan.json"
mkdir "$VISION_CAPTURE_PROVENANCE"
tar -xf "$VISION_RELEASE/vision-bench-comprehensive-capture-provenance.tar" -C "$VISION_CAPTURE_PROVENANCE"

"$SIM_PY" OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_package_v3.py verify \
  --provenance "$VISION_CAPTURE_PROVENANCE"
"$SIM_PY" OmniGibson/omnigibson/tiptop/host/visionbench_driveway_recovery_verify.py verify-portable \
  --provenance "$VISION_CAPTURE_PROVENANCE" --captures "$VISION_RUN/captures" \
  --output /path/to/fresh-recovery-audit.json
"$SIM_PY" \
  "$VISION_CAPTURE_PROVENANCE/driveway_recovery/prior_task_retry/tools/visionbench_task_retry_portable.py" \
  --provenance "$VISION_CAPTURE_PROVENANCE" --captures "$VISION_RELEASE_RUN/captures" \
  --mapping driveway_recovery/prior_task_retry/portable_path_map.json \
  --receipt driveway_recovery/prior_task_retry/publication.json \
  --output /path/to/fresh-task-retry-audit.json
"$SIM_PY" OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_package_v3.py audit-overlay \
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
For this run, `verify-portable` additionally checks the driveway recovery chain and
unchanged successful captures. Use the **complete main data artifact**, including its
original overlay PNGs; an input-only inference fork omits protected capture files and
cannot satisfy this audit. Published success receipts resolve from the companion, so
`$VISION_RUN/captures` does not need a separate `case_receipts` directory.

The exact-task retry verifier separately checks five relocated `wash_dog_toys` captures,
11 published metadata records, their snapshots and mapped source/crash/launch/audit
evidence. Require `ok: true` and `historical_bindings_resolved_from_portable_copies: true`.
Its `whole_original_corpus_reverified: false` is intentional: the 8,697-file preservation
proof remains a linked historical receipt, not a claim that the whole original corpus
was bundled. Its historical 358/365 audit does not replace the final 365-case audit.

The companion contains these paths:

- `selection/selection_fullcoverage.json` and preserved initial 300-case selection.
- `metadata/catalog.parquet`, `metadata/inventory.json`, dataset metadata, 100 task
  templates, 100 BDDL definitions, the evaluator robot configuration, and 365 selected
  task-instance configuration JSON files under `metadata/sim_data/`.
- The 365 selected annotation JSON files in `annotations/`, frozen capture sources in
  `source_capture/`, and explicitly named selection ancestry and launch receipts.
  `source_geometry/` and `source_diagnostics/` preserve geometry dependencies and the
  additive audit/packaging sources; `diagnostic_amendment/` preserves the failed earlier
  audit and independent saved-array evidence. `diagnostic_amendment_v3/` also preserves
  the failed foreground-union audit and its measured support changes. Geometry source receipts match the original
  committed simulator and TiPToP sources.
- `snapshots/`, `provenance/case_receipts/`, `provenance/materialization/`,
  `provenance/task_receipts/`, final execution receipts, and the four `sealed/` files.
- `replay_fidelity_summary.json`, comparing every final saved head-depth frame with
  the recorded dataset frame while retaining the separate preliminary depth probe.
- `driveway_recovery/`, `driveway_failure_originals/`, and `source_recovery/`, preserving
  the recovery amendment, original failures, snapshot bindings, successful-capture
  hash inventory, verification/publication receipts, and the additive implementation.
- `saved_query_outlier_diagnosis/`, with the saved-query replay diagnosis, previously
  extracted recorded-frame PNGs, and the robot pose representation source evidence.

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
archived audit-command field still names the original auditor; run the explicit v3
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

## Preserve the exact task-init retry

The original capture produced 353 successful snapshots. A single `wash_dog_toys`
initialization SIGSEGV (`returncode: -11`) occurred before any snapshots, task-success
receipt or per-case failure files. One identical frozen task attempt from frame zero
captured the same five selected snapshots, without case replacement or source changes.
Its publication audited 358/365 captures and preserved 8,697 protected files unchanged.
The remaining seven failures were subsequently recovered as described below.

Preserve the original task-level crash/execution, exact launch/completion, successful
outputs, five snapshots and publication evidence under
`driveway_recovery/prior_task_retry`, with its explicit portable path map and tools.
Do not synthesize per-case failures. The viewer's seven recovered flags correspond to
real driveway per-case failure files; disclose the five task-retry snapshots separately.
The final sealed audit confirms 353 + five + seven = 365 captured snapshots.

## Recover the declared driveway binding failures

The original bridge excludes driveway from `task_scope()` with floors and lawns, while
this benchmark's frozen selection excludes only floor/lawn. Seven predeclared cases
contain a driveway target. The additive recovery accepts only those exact IDs **after**
the original capture records the matching missing-driveway error and saves its immutable
snapshot. It restores that snapshot, exposes the existing fixed driveway's raw BDDL
binding, and calls the unchanged capture and geometry-labeling functions. It does not
replay actions, substitute a frame, or take physics steps after snapshot restoration.
Environment setup/reset occurs before restoration. Annotation targets include support
surfaces; the driveway-only `chopping_wood` case retains its original target scope.

On the capture host, `runs/vision/comprehensive_v1/driveway_recovery_v3/` records the
completed recovery, preserving the earlier failed v2 attempt and the exact-task retry.
Its `status.json` reports `phase: complete`. The reviewed
`tools/visionbench_shared_capture.py` wrapper ran the unchanged continuation with the
scheduling-only amendment `schedule_amendment.json`,
SHA256 `6124f02a403b336b6c72314dc78a054950a499b2969d3f77272a58b4edc01cf2`.
It ran one recovery worker at a time on GPU1 with checked shared occupancy and at least
40,000 MiB free. GPUs1/3 were not required to be empty, and other work was not terminated.
Frozen capture/label/model sources, selection and DEV lock remained unchanged. Do not
rerun the completed continuation or describe shared capture wall times as controlled latency.

Recovery writes into a separate directory and preserves original failed task receipts
and the original `execution.json` with status `incomplete`. A CPU verifier checks the
failure/materialization/snapshot chain, restored poses/joints, zero capture physics,
unchanged successful files, and recovered masks. Only absent case outputs are published.
The published outputs are verified again **before** task receipts and execution status
are reconciled to complete. The original failed attempt history remains in the reconciled
execution receipt; it is not a claim that every original attempt succeeded.

The continuation completed verified recovery, the V3 capture seal, the fresh saved-query
fidelity diagnostic, recovery-aware metadata staging, companion finalization, extraction,
portable recovery verification, and the independent overlay audit. The original
monitor/finalizer stopped at incomplete execution; this additive continuation completed
the recovery chain. Successful publication is recorded in `publication.json`.

The finalized companion covers capture provenance for all 365 cases across 100 tasks. Its SHA256 is
`ecf14e0184bc270d00318d0875182a28dd880796f05cce147d86b7cbab1a5516`
(275,466,240 bytes). The portable recovery and independent overlay audits both report
`ok: true`; the latter verified all 2,920 native main capture files.

The reviewed shared wrapper wrote `shared_schedule_completion.json` (schema
`visionbench-shared-capture-completion/1`) binding unchanged frozen sources and the final
`status.json`. These final receipts were produced after companion archive creation;
preserve their bytes and hashes separately in the release validation inventory. They
must not be assumed present inside that immutable companion.

Capture completion is separate from the main benchmark's timing and release validation.
Its controller state is `continuation_shared_v1/status.json` (schema remains
`visionbench-continuation/3`); require `state: complete` before release.
Its scheduling amendment SHA256 is
`aa8b2a39f4ab01db7ec84548910d211982c7f4e1726599d3472490bd87331bf9`.
Capture acceptance and bulk inference allow checked shared GPUs1/3 with at least 40,000
MiB free at launch, not a per-arm reservation. Bulk timings are not controlled runtime.
The original GPU1 exclusive latency checks and 30-second monitoring remain unchanged.
This capture handoff does not assert completion of those timing or release checks;
consult their separate final receipts before presenting controlled runtime or a final
model recommendation.

This recovery wrapper is bound to the original selection SHA and seven case IDs; it is
not a general retry command for a rebased or newly selected run.

## Audit and seal before inference

The additive v3 auditor records the pre-TEST diagnostic amendments. Coffee-station
cases exposed instance ownership transfers between the independently computed 4mm and
8mm masks; desk cases also exposed small foreground support additions at 4mm. Neither
per-instance nor foreground-union nesting is therefore assumed. Both captured masks and
the primary 8mm score remain unchanged.

The v3 auditor requires native Boolean masks, matching shapes, disjoint instances at
each tolerance, exact stored pixel counts, and per-object/foreground conservation. It
reports foreground additions, removals, and ownership transfers separately. Changed
pixels are counted once as additions + removals + transfers; Boolean instance-stack
XOR instead counts transfers twice. Synthetic tests of the unchanged labeler reproduce
NaN/degenerate-triangle mechanisms, including hit rejection before ownership assignment.
The causes in the actual coffee/laptop cases remain unproven because their raw mesh
arrays and per-pixel distances were not archived. These are numerical limitations of
approximate geometry labels, not calibrated accuracy bounds.

```bash
/path/to/sim/python OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_audit_v3.py \
  --selection /path/to/new-run/selection_replay.json --captures /path/to/new-run/captures \
  --out /path/to/new-run/sealed --seal
```

Without `--seal`, the command produces a progress audit without pretending incomplete
captures are a final dataset. `--seal` requires every selected case and verifies array
shapes, native Boolean masks, disjoint instances at each tolerance, foreground/instance
conservation, object pixel counts, frozen
poses, snapshots, selection receipts, and absence of label identity fields in inputs.
Sealed outputs are `query_manifest.json`, `capture_audit.json`,
`capture_quality_summary.json`, and a hash-linked `seal_receipt.json`.

Masks are **geometry-proximity proxies** at 8mm, with a 4mm tolerance sensitivity;
they are not renderer instance ground truth. Saved-query render convergence flags live
in `labels.json["render_convergence"][view]`. Nonconverged views remain in the main
cohort and are explicitly marked for the fixed-threshold quality sensitivity. Recorded
head-depth replay fidelity is reported separately from integrity. The completed cohort
has 1,095 saved views, including 41 nonconverged views in 34 snapshots; all were retained,
with zero case replacements. Passing integrity checks does not establish exact masks or
faithful replay of every recorded viewpoint. `aabb_base` is an
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
  OmniGibson/omnigibson/tiptop/host/test_visionbench_all_tasks_package_v3.py
```

## Check replay fidelity against the final saved query

The materialization receipt includes an early head-depth probe. A transient camera or
render settling issue can make that probe disagree with the final saved query even when
the final query converges. Preserve it as provenance, and compare the **saved input
depth** directly with the recorded dataset frame before making fidelity claims:

Rerunning this recorded-video diagnostic additionally requires the existing simulator environment to provide PyAV (`av`) and PyArrow (`pyarrow.parquet`), together with the original demonstration video/parquet shards. Reading its archived JSON summary does not require decoding videos. These are separate from cached inference rescoring dependencies.

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

All 365 final saved queries have fresh recorded-depth comparisons. Across cases, the
median of the per-case median absolute depth differences is 3.24mm, the 95th percentile
is 15.01mm, and the maximum is 889.38mm. No cases were excluded on fidelity grounds.
These aggregate diagnostics do not identify the cause of every replay mismatch.

One preserved outlier, `comprehensive.sorting_household_items.e5484.f5179`, has a
723.7mm median saved head-depth difference and a robot base tilted 123.91 degrees: the saved
query looks toward the ceiling while the exact recorded frame looks toward a basket.
The frozen capture itself did not move. The snapshot is consistent with that pose:
its upright `root_link` is a virtual root, and its six base joints encode the translation
and tilt. Reconstruction matches the saved base position within 0.64 micrometers and
quaternion components within 8.8e-8. A direct root-link/base-footprint comparison would
incorrectly suggest stale snapshot state. The diagnosis preserves the images, calculation,
and source evidence. The case is retained; its zero-visible-pixel targets remain ineligible
under the original 25-pixel recall rule, without a new exclusion or model-based replacement.

## Produce the companion artifact on the capture host

Stage only explicitly named immutable metadata while capture is running. Use a new
staging path. After capture sealing, run the fidelity command above with `--fresh`,
then finalize. Finalization requires complete execution, a valid seal, and all 365
fresh saved-query comparisons; it will not publish a partial capture as complete.

```bash
"$SIM_PY" OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_package_v3.py stage \
  --selection /path/to/comprehensive_v1/selection_fullcoverage.json \
  --sim-data "$OMNIGIBSON_DATA_PATH" --stage /path/to/comprehensive_v1/provenance-stage
"$SIM_PY" OmniGibson/omnigibson/tiptop/host/visionbench_all_tasks_package_v3.py finalize \
  --stage /path/to/comprehensive_v1/provenance-stage \
  --archive /path/to/comprehensive_v1/vision-bench-comprehensive-capture-provenance.tar
```

This completed comprehensive run used recovery-aware staging instead of the plain
`stage` command above. Its continuation performed this after verification. To create a
new artifact from a completed recovery, the equivalent manual staging command is below;
use a fresh staging path, and do not rerun finalization over the published companion:

```bash
"$SIM_PY" OmniGibson/omnigibson/tiptop/host/visionbench_driveway_recovery_verify.py stage \
  --selection /path/to/comprehensive_v1/selection_fullcoverage.json \
  --recovery /path/to/comprehensive_v1/driveway_recovery_v3 \
  --sim-data "$OMNIGIBSON_DATA_PATH" \
  --stage /path/to/comprehensive_v1/capture_provenance_stage_driveway_recovery
```

Before finalization, verify that the stage includes the `prior_task_retry` path map,
publication receipt, portable verifier and every mapped dependency. Pass that stage to
the unchanged V3 `finalize` command. It includes both recovery chains, the scheduling
amendment/wrapper and fidelity diagnosis. For this completed run, final
`shared_schedule_completion.json` and `status.json` were generated after archive creation
and must be preserved as separate hashed release receipts.

Staging and finalization copy original file bytes and record SHA-256 receipts. They
never edit the frozen selection or active capture sources. A staging directory is not
a publishable artifact. Finalization refuses to overwrite an existing final manifest
or archive. Validate the extracted result with `verify` and `audit-overlay` before
sharing it with a coworker.
