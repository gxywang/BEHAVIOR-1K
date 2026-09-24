# Collision fixes: cross-task validation — 2026-09-22

This batch tests the current collision repairs on three additional full tasks. It uses one public instance
per task (CLI index 0, public instance 301), with the normal strategy, retry policy and simulator-step budget.
No objects are prepared in the gripper, and no cached trajectories are injected.

## Protocol

- Tasks: `putting_away_toys`, `dispose_of_batteries`, `putting_dirty_dishes_in_sink`.
- GPUs 1, 2 and 3 respectively, with isolated planners on ports 8891, 8892 and 8893.
- Shared immutable source snapshot, including cuTAMP patches and the patched cuRobo Python and existing binaries.
- Oracle labels/masks, simulator physical collision map, sticky grasping, and normal base teleports.
- Three camera views; torso `[1.2, -1.7, -0.9, 0.0]`; 256 particles; planner seed 2300;
  40 s nominal planning budget; `PYTHONHASHSEED=2300`.
- Installed simulator and planner environments reused without installation or rebuilding.
- Normal final goal score is reported separately from executed motion and collision counts.

The generic observer retains raw contact pairs. Intended grasp contact requires the exact target and active
hand, or the object actually held by its own hand. Furniture, placement-container and opposite-hand contacts
are not category-wide exemptions. Supporting-surface contacts during pickup/release are reported separately.
Counts represent simulation steps containing contact, not distinct impacts or impulse severity. The observer
measures upper-body/environment, carried-object/environment, and carried-object/non-grasping-robot contacts;
base/wheel-only furniture impacts are outside these metrics.

## Results

All three runs launched at 14:05:27 UTC from a shared snapshot of 808 files and finished normally.
None solved its complete task. A final hash check found all 799 non-Markdown production files unchanged.
Only instrumentation and reports were edited during this batch.

| Task | New requested transfers | q_score | Task steps | Benchmark wall time | Status |
|---|---:|---:|---:|---:|---|
| Putting away toys | 0/8 | 0.00 | 2,839 | 1,877.3 s (31.3 min) | Strategy finished; task incomplete |
| Disposing of batteries | 1/3 | 0.25 | 2,815 | 1,267.5 s (21.1 min) | Strategy finished; task incomplete |
| Putting dirty dishes in sink | 1/4 | 0.25 | 2,991 | 1,967.4 s (32.8 min) | Strategy finished; task incomplete |

The battery task has four goal atoms, including a bin-on-floor atom already true at the start. Its score of
0.25 represents one new satisfied atom; the initial support atom is not credited as a battery transfer.
The ordinary sticky fallback acquired battery 2, then one complete placement plan executed five trajectories
and 510 steps, with maximum joint tracking error 0.00542 rad. It recorded one wrist/desk contact step,
zero carried-object/robot-body contacts, 16 held-battery/desk support steps during pickup and one held-battery/bin
contact step during release. There were no observer errors. Its finalized video contains 1,452 frames.


The toys task completed with no accepted planned trajectory. A sticky grasp acquired toy 7, but placement
and recovery failed before ordinary release. Its strict classifier counts 43 unmatched finger/toy contact
steps during that explicit release (steps 1,090–1,132); the live grasp association had already cleared.
These remain in the raw and strict metrics, separately labeled release-related rather than furniture contact.
There were no robot/furniture contacts observed. Thirteen goal-not-visible rounds and six planning failures
were recorded. This is not evidence of successful collision-free task completion.

The dishes task finished with bowl 1 inside the sink. This followed sticky acquisition, failed inside and
put-down plans, then the ordinary fallback release at the sink. **No planned dish trajectory executed.**
The remaining bowl and both plates were unsatisfied. Its finalized video contains 1,540 frames; the toys video
contains 1,464 frames. All three observers reported zero instrumentation errors.

| Task | Completed planner trajectories | Collision preflight refusals on direct ramps | Direct-ramp tracking aborts |
|---|---:|---:|---:|
| Toys | 0 | 52 | 0 |
| Batteries | 5 (one placement plan) | 10 | 0 |
| Dishes | 0 | 78 | 16 |

All 16 dish-task tracking aborts involved `left_arm_joint1` during folding/refolding, with about 0.48 rad lag.
No corresponding external or carried-object/body contact was observed. Their cause remains unclassified;
tracking error alone is not proof of collision. The battery plan passed every planned-motion preflight.

| Task | Robot/furniture contact steps | Release-related finger/object steps | Held-object/support contact steps | Held-object/non-grasping-robot steps |
|---|---:|---:|---:|---:|
| Toys | 0 | 43 | 34 (floor) | 0 |
| Batteries | 1 (wrist/desk) | 0 | 17 (16 desk + 1 bin at release) | 0 |
| Dishes | 0 | 0 | 47 (booth support) | 0 |

The strict observer retains the 43 toy release rows as unmatched contacts; this table distinguishes their
observed context without deleting or silently relabeling them. Carried support contacts are retained even
while an item is resting on its source before lifting. Observer totals include 180 startup steps per run;
benchmark step counts in the results table exclude those steps. Robot-only self contacts and base/wheel-only
contacts are outside the observer's coverage.

## Follow-up: pre-grasp contact and failed-transfer recovery

Review of the raw contact records after video inspection exposed two further failures. The historical
"intended" contact label checked the target and hand, but did not check the grasp phase; it is **not evidence
of a clean approach**. Before attachment, battery 2 recorded 27 finger/target contact steps, bowl 2 recorded 17,
bowl 1 recorded 16, and plate 1 recorded 10. Open-hand approach contact occurred for the battery and both bowls;
the plate contacted during a later press nudge. These are contact-bearing steps, not impact counts or forces.

Two dish transfers followed the same failed recovery sequence. Bowl 2 was held by task step 623, arrived at the
sink by 713, returned to the booth by 816 after failed inside and floor plans, and was released by 861. Plate 1
arrived at the sink by 1727, returned to the booth by 1830, and was released by 1875. Every requested placement
plan in those sequences failed. `Runner.free_hand` received the next pickup's support, then opened the hand
unconditionally when planned put-downs failed. Bowl 1 instead fell into the sink during the same fallback
release policy; its final predicate does not establish a planned placement.

The subsequent corrections and validation are recorded in [the grasp/recovery audit](GRASP_RECOVERY_AUDIT.md).
These corrections do not alter this batch's saved source, raw metrics, or results.

## Remaining failures identified

- **One observed unexpected battery-task contact:** `left_arm_link6` touched the physical desk during a
  guarded refold after teleportation (task step 100). The desk was present in the map and the path was checked.
  Saved poststep-pose reconstruction is clear, but the event base transform and physics-substep poses were not
  recorded. A separate mesh-coverage check finds up to 3.03 mm undercoverage by that link's buffered sphere;
  this does not establish the cause of the recorded contact. The contact remains counted.
- **Synthetic support geometry falsely blocks a battery pickup:** the first request finds 174 satisfying
  particles, but motion refinement rejects its measured start. GPU replay attributes the collision exclusively
  to the RANSAC table cuboid. Three finger spheres clear the actual physical desk triangles by 85.6–97.8 mm,
  while the synthetic slab reports penetration. This is separate from the earlier real wrist/desk contact.
  No physical collider was removed and no production source was changed during this test batch.

- **Open-finger approximation blocks a held-bowl retreat:** replay of the second dishes request finds the
  planner's fixed-open finger spheres overlapping the physical sink by up to 6.01 mm. The measured fingers
  are closed. Every other static collider, the synthetic table, and all 245 held-bowl spheres are clear.
  This is a planner-model conflict, not evidence of a measured closed-hand collision. The existing bridge
  preflight checks measured finger state but cannot recover a plan that the planner refuses to generate.

## Follow-up indicated by these tests

- Associate a fitted RANSAC support with authoritative physical support geometry before making its proxy
  sampling-only. Uncertain associations and unmapped floors must retain collision coverage.
- Use measured closed-finger geometry during initially-held carrying phases, then model opening at release.
  Particle optimization, IK and MotionGen must agree, and attachment spheres must survive state updates.
- Capture the full base/joint state and physics contact details around future contact events before attributing
  the wrist/desk event or choosing a model margin. Poststep arm joints alone are insufficient for that replay.

These are recommendations from the recorded failures. This batch changes instrumentation and reporting only;
production collision behavior remains frozen throughout all three runs.

## Evidence

Artifacts, commands, source hashes, videos, raw contacts and requests are under
`runs/collision_audit/other_tasks_20260922/`. The shared source is in `source_checkpoint/` and each task has a
separate run directory. Earlier historical task scores use different code and are context, not paired baselines.
A single instance per task cannot establish a success rate or general collision guarantee.

Each run contains `task_outcome.json`, `contact_analysis.json`, `contacts.jsonl`, `actions.jsonl`,
`episode/summary.json`, and a finalized video in `episode/videos/`. The batch also contains `results.json`,
`validation_outcome.json`, and `cleanup.json`. All batch planners and simulators were stopped, ports 8891–8893
were closed, and the preexisting services on 8123 and 8821 were left untouched.

See [the collision audit](COLLISION_AUDIT.md) for the fixes and their known limits. The map is privileged
simulator geometry; teleport endpoint checks do not implement navigation.
