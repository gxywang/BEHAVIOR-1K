# Grasp approach and transfer recovery audit — 2026-09-22

The battery and dish videos exposed behavior that aggregate furniture-collision counts did not describe:
open-hand target contact before grasp closure, and arbitrary release after a failed transfer. Both came from
shared control policy. Sticky attachment succeeding does not make the preceding approach acceptable.

## Historical evidence

The immutable evidence is under `runs/collision_audit/other_tasks_20260922/`. No dish planner trajectory
executed in that batch. See [cross-task validation](CROSS_TASK_VALIDATION.md) for scores and the corrected
interpretation of its contact metrics.

| Object | Contact steps before attachment | Evidence |
|---|---:|---|
| Battery 2 | 27 | `dispose_of_batteries/contacts.jsonl`; task steps 963–990, excluding 966; attachment at 991 |
| Bowl 2 | 17 | `putting_dirty_dishes_in_sink/contacts.jsonl`; steps 579–622; attachment at 623 |
| Bowl 1 | 16 | same file; steps 979–1008; attachment at 1009 |
| Plate 1 | 10 | same file; steps 1584–1593; attachment at 1594 |

The battery and both bowls contacted during the open-hand approach. The plate contacted during a subsequent
press nudge. The observer called active fingers touching the exact grasp target intended regardless of phase;
that label cannot establish acceptable approach timing, force, or object displacement. The old run did not
record forces or continuous object poses, so impact severity cannot be recovered from the counts.

Bowl 2: held at step 623, sink at 713, booth at 816, forced release completed at 861. Plate 1: sink at 1727,
booth at 1830, forced release completed at 1875. Failed floor recovery was followed by a teleport to the support
of the next candidate, then an unconditional open-hand release. This was not a collision-checked placement.
Bowl 1 happened to land inside the sink after the same release policy.

## Recovery correction

`Runner.transfer_one` retains the held item and its intended destination while spending its existing retry
budget. Retrying a placement does not pick the item again. No floor or unrelated support is substituted for a
failed destination. Exhausted attempts raise `TransferBlocked`; the benchmark records a blocked reason and
held-object state, then ends without an opening command. A missing named floor also retains the object.

`put_down` and transfer completion require an empty hand and the exact requested destination predicate. This
uses the benchmark's already privileged simulator evaluator; it is not a perception-only success test.

## Validation

Grasp correction and new simulation validation are in progress. Historical task scores above are not results
of the changed policy. A blocked transfer is a safer failure state, not a solved placement problem.
