# Fix4: six fixes from the sweep3 video review (2026-09-23)

Fix4 changes the bridge, the planner server and cuTAMP to address six systematic failures that a video review of
the sweep3 evaluation found. **Fix4 has no benchmark score.** Its sweep was started with one smoke job and the user
stopped it during that job. Every result below is a unit test or an offline replay of requests and states recorded
in sweep3. Nothing here is validated on the benchmark.

Paths: a bare file name (`r1pro.py`) is in `OmniGibson/omnigibson/tiptop/`. A path starting `tiptop/` is in the
TiPToP submodule, relative to the repository root. "Pre-fix" means sweep3l, the last of the frozen checkpoints the
sweep3 lanes ran; some sweep3 episodes ran on earlier sweep3 checkpoints.
The `runs/` directory is not in git; paths into it are listed only under [Local evidence](#local-evidence-not-in-git).

## Status

| What | Result |
|---|---|
| sweep3 (before fix4) | mean q 0.272 over 114 episodes, against 0.195 the week before (paired +0.085 ± 0.073) |
| fix4 on the benchmark | no scores; one smoke job (laying_tile_floors instance 0) ran and was killed mid-episode |
| Bridge tests | 401 passed (`OmniGibson/tests/test_tiptop_*.py`) |
| Planner tests | 171 passed, `tests/test_tiptop_h5.py` deselected (needs an M2T2 server) |
| Code review | 27 findings: 15 confirmed and fixed, 2 split and left as they are, 5 cleanups applied, 5 confirmed by neither verifier |

## Why: the six issues

The video review watched the sweep3 videos and measured each suspected cause over the 114 latest episodes. Picking is
robust (the sticky fallback takes almost anything). What follows the pick was not. Ranked by cost:

| # | Issue | Measured in sweep3 |
|---|---|---|
| 1 | A sticky pick leaves the robot crouched, and the carried object is modelled as its whole gripper-frame bounding box | The head camera stays at 0.75-1.04 m instead of 1.40 m. Stance searches from that posture end Unreachable 115/164 (70%), against 42/326 (13%) from the ready posture. Sticky carries are put back 150/288 (52%), planned carries 10/123 (8%). 13 episodes froze at a sample-0 refusal with the object in the hand (laying_tile x3, clean_up_your_desk x2, finger vs floor after a floor pick x4) |
| 2 | Furniture goals are planned as "on top of the convex hull" | inside(x, bookcase): 16 of 18 executed places landed on the top board and 0 of 10 untouched items held (open containers: 87% held). The inside region aimed at the case's lowest shelf in 32 of 32 attempts. touching(hallstand): IK 0/256 on the hull top. nextto(x, sink): rejected by cuTAMP in 15 rounds. under(): no translation. 5 of 5 executed table places were planned onto another plane. Upper bound 39 atoms in 18 episodes (about 0.07 mean q), 24 of them bookcase |
| 3 | cuTAMP fills every movable's whole OBB with spheres | An open basket is a solid block around its contents: 0 of 59 planned picks of basket contents in sorting_vegetables. ENCASED: 122 rounds, 25 episodes, 11 tasks, 9,718 s. A wide object blocks its own grasps (pillar candle about 10 cm against the model's 9.7 cm open jaw): WIDE, 123 rounds. The pattern is in 70% of failed planned picks and 14% of held ones |
| 4 | The task layer answers every failure by putting the item back and re-picking | 191 of 375 executed place rounds (51%) are put-backs or put-downs, about 29% of wall time. 83 put-backs repeat a container that was already Unreachable. 3 atoms were undone for certain |
| 5 | The stance proposer and validator disagree, and refusals are not fed back | Rings are bounded by reach around the target's centre (a bed's radius of 1.42 m is past the 1.1 m reach, so every candidate stood on the bed). The clipped pass accepts stances that do not frame the target (GoalNotVisible 30% vs 9%). The 1.1 m retry re-tests the same candidates in 152 of 377 failed searches. Kerbs and thin objects are free to the proposer and solid to the validator |
| 6 | Bridge joint ramps are not checked for self-collision | The capture swing drove the hand into the robot's own head camera 43 times in 30 episodes and 20 tasks, then held 60 steps; about 0.3-0.5 q in total. In the same family: after a wrist capture the right arm stayed off its locked posture, and the planner refused every later left-arm request (7 jobs) |

## The competition data rule

The competition gives the robot a pre-scanned 3D map of the environment (walls, floors, fixed-base furniture) with
localization. Objects are known only as point clouds: depth under masks. Fix4 follows that rule in what it adds,
with one disputed exception (the first bullet below the table). Its first pass read simulator geometry of movable
objects: the carried volume and jaw width in the bridge, the movable collision model in cuTAMP, and a few AABB and
shelf reads. The point-cloud rework moved most of these onto point clouds or limited them to fixed-base objects. The
code review found the rest (GR-2/C5, C1 and C2), and its fix pass moved them too.

| New code reads | Source | Under the rule |
|---|---|---|
| A movable object's shape (carried volume, jaw width, item height) | each view's depth, intrinsics and camera pose under the object's mask, moved into the object's frame (`scene.py: remember_seen`) | point cloud |
| A movable object's collision surface in the planner | `object_pcds`, the per-object cloud perception already builds (`tiptop_run.py: surface_spheres`) | point cloud |
| Shelf boards, a table's top, a fixture's footprint, hard stance obstacles | meshes and AABBs of fixed-base objects only | scanned map |
| Self-collision | robot URDF, cuRobo spheres, the simulator's disabled pairs, the embodiment's `q_home` | robot model |
| Stance refusals | the bridge's own landing check | own checks |

Pre-existing oracle knowledge that fix4 still uses, unchanged: the oracle instance masks, tracked object poses,
fillable volumes, the BDDL goal evaluation, the `--room` collision map and the collision audit.

Reads of a movable object's simulator geometry that remain:

- `r1pro.py: carried_volume` falls back to the pre-fix gripper-frame box of the held object's mesh when no capture
  has seen the object. Only `OracleKnowledge` feeds the seen boxes, so under `--knowledge onboard` the fallback is
  taken every time (review finding C3, left as it is; see [Review](#review)).
- Pre-existing and untouched: the ramp and landing preflight (`_motion_obstacles`, `room_collision_scene`),
  `press_grasp`'s ray casts, the item AABB in `inside_region`'s rest height, and object AABBs in `place_robot_for`.
- Under `--room` the server replaces movable geometry with simulator meshes. After fix4 that only feeds the object's
  OBB, its OBB cover spheres (used after a Place and between movables) and the cuRobo attachment. Robot-vs-movable
  checks before a Pick use the perceived surface in both modes. A fixed object kept as a near() reference keeps the
  mesh from its room entry and gets no perceived surface.

## 1. After a sticky pick

**What changed**

- `bench.py: Episode.pick`: after a successful press grasp, `sim.return_to_ready` carrying the object (allowed
  contacts from `retreat_contacts`) before returning. The destination stance search now starts from the same
  posture a planned pick ends in.
- `scene.py: TiptopSim.remember_seen`, `seen_boxes`: every capture unprojects each view's depth under each
  label's mask with that view's camera pose, moves the points into the object's own frame by the per-view render
  pose, and grows that object's box. The box is the union over captures and keeps bounds only.
- `knowledge.py: OracleKnowledge.describe`: calls `remember_seen` right after the masks are made, before
  `inside_regions`.
- `r1pro.py: own_box`, `carried_volume`: the held object's sphere cover comes from its seen box in its own frame,
  carried into the gripper frame. `ramp_collision` and `base_placement_collision` both use it (their two copies
  of the gripper-frame AABB are gone). An unseen object falls back to the pre-fix box.
- `collision.py: JointPathCollision.check_polyline(excuse_start=...)`: a (link, obstacle) contact present at
  sample 0 is excused while it comes no closer than at the start minus `SETTLED_TOL` (1 mm). Deeper is a hit. A
  lone gripper event (one configuration) gets no excuse. `ramp_collision` and `validate_motion` opt in.
- `r1pro.py: base_placement_collision`: when the destination's floor check refuses the hand or the carried
  volume, and the same hand is already at a floor at the current pose, the hand and carried volume may touch
  fixed-base ground at the destination. A loose flat item (a tile) stays an obstacle.
- `r1pro.py: lift_held`, `press_grasp`, `ramp_to`: a lift refused at path sample 0 by the robot's own body or a
  floor opens the hand where the object rests. `press_grasp` re-reads the hold after the lift, so `bench.pick`
  takes its failure branch. `ramp_to` now returns the preflight's path sample as a fourth element, and the release
  happens only at sample 0.

**Data**: depth, intrinsics and camera pose per view, under the oracle masks, into the object frame by tracked
poses. The robot model. The `fixed_base` flag of ground objects.

**Tests** (`OmniGibson/tests/`):
`test_tiptop_presweep_fixes.py::test_a_sticky_pick_ends_at_the_ready_posture_carrying`,
`::test_a_lift_refused_by_the_robots_own_body_lets_go_where_the_object_rests`;
`test_tiptop_collision_scene.py::test_the_carried_volume_is_the_objects_own_box_not_its_gripper_frame_aabb`,
`::test_a_capture_grows_what_it_saw_of_each_object_in_its_own_frame`,
`::test_a_contact_the_motion_starts_in_is_excused_while_it_comes_no_closer`,
`::test_the_landing_check_excuses_a_hand_already_at_the_floor_here`,
`::test_a_loose_tile_at_the_destination_is_ground_for_the_base_but_not_floor_for_the_hand`;
`test_tiptop_knowledge.py::test_oracle_knowledge_sends_instance_masks_and_every_button_of_the_task` (the capture
is recorded) and `::test_an_inside_goal_ships_the_container_interior_only_when_the_flag_is_on` (before the regions
are computed).

**Offline replay** (CPU; recorded measured states and requests; the robot's spheres and FK):

| Grasp | Pre-fix carried box | Fix4 seen box |
|---|---|---|
| laying_tile 301, step 4752 | lifted to ready refused, carried vs torso_link4 at sample 984 (the same line as the sweep log) | clear at sample 0, through the lift and the ramp to ready |
| laying_tile 302 step 10989, 303 step 2688 | refused, base_link at sample 0 | still refused by base_link at sample 0 (the tile corner lies under the chassis). With the first pass's mesh box the start was within base_link's 2 cm self buffer, physically 1.4 and 1.9 cm clear, and the lift was refused |
| clean_up_your_desk 303, step 5627 | refused, left_arm_link5 at sample 0 | clear at sample 0, through the lift and the ramp to ready |
| clean_up_your_desk 301, step 3533 | refused at sample 0 | refused at sample 0 by left_arm_link5 (a 2-3 mm mask halo tips it); the lift is refused either way |
| clean_up_your_desk 302, steps 4746 and 6267 | lift clear, straight return to ready refused | the same (a planner fallback while carrying is not wired) |
| bringing_in_wood 302, step 893 | fold and return refused, finger vs floor at sample 0 | clear with `excuse_start` (the finger spheres are 4.6 mm above the floor); a motion lowering the fingers to -24.8 cm is still refused at sample 1 |

A full view of a tile gives 0.466 x 0.436 x 0.031 m against its 0.45 x 0.42 x 0.03 m hull: the oracle mask's 8 mm
halo. Four other grasps (folders, pens, a book) are clear under all three boxes. For clean_up_your_desk 303 the
seen laptop is 0.32 x 0.41 x 0.04 m against the room hull's 0.54 x 0.45 x 0.23 m; whether the lid was shut at that
capture is unresolved offline, and the verdict hinges on it.

In the first pass's replay the tile in laying_tile 302 and 303 was released and the round retried from another
stance. Review fix F3 then limited the release to a true sample-0 refusal. With `excuse_start`, a contact the lift
starts in is never refused at sample 0 (robot body, carried volume or room). The sample-0 refusals left are a
sphere that starts inside a closed component of an obstacle, or two carried objects against each other; of these
only a floor lets go, so the release now fires in practice only for a floor. What those two grasps do now was not
replayed.

## 2. Furniture goals

**What changed**

- `r1pro.py: boards()` (module level): the horizontal boards of a mesh (upward faces at one height with at least
  `BOARD_MIN_AREA` 0.02 m², split at `BOARD_GAP` 2.5 cm) and each board's ceiling.
- `r1pro.py: shelf_of`: of a fixture's boards inside a given box, the one with headroom for the item (`HEADROOM`
  3 cm) and within `BOARD_REACH` (1.0 m) of the base, nearest `PLACE_HEIGHT` (0.9 m).
- `r1pro.py: inside_region`: for a fixed-base, joint-less container whose fillable volume is taller than
  `SHELF_CASE_HEIGHT` (0.4 m), the region is that board's compartment (floor at the board, ceiling at the next
  board or the fillable top), not the lowest floor of the whole case. A movable bin keeps the old path.
- `r1pro.py: rest_region`: touching(item, fixture) sends a 2 cm slab at the fixture's reachable board under the
  fixture's label, in place of the hull top. Fixed-base fixtures only.
- `r1pro.py: footprint_region`: a slab over a fixed fixture's AABB footprint. It is the named table's top when the
  goal's supports include exactly one fixed table and no floor or under() fixture, and (through
  `floor_surface(within=f)`) the floor under a fixed fixture for under(x, f).
- `r1pro.py: inside_regions`: restructured around (predicate, item, container). The planner's support label gets
  the floor slab, else the floor under the under() fixture, else the single fixed table's top.
- `tiptop/b1k/bridge/protocol.py: tiptop_goal`: under(x, f) is sent as on(x, table).
- `tiptop/b1k/bridge/strategies.py: PLACE_PREDICATES`: includes "under", so under rounds are scheduled.
- `bench.py: Episode.achieve`: the floor is in the workspace for an under atom. `Episode.satisfied`: a place
  atom holds only when the box test and the task's own evaluator agree; under() by the evaluator alone.
- `tiptop/tiptop/tiptop_websocket_server.py: apply_room_object_roles`: a fixed object that a Near goal atom names
  as its reference stays a cuTAMP movable, carrying the physical mesh from its room entry, so cuTAMP accepts
  nextto(x, sink).
- cuTAMP patch 18 (`tamp_domain.py: get_initial_state(unpickable=...)`, `tamp_world.py: TAMPWorld.initial_state`):
  movables in `env.fixed_labels` get no HasNotPickedUp, so no skeleton picks up the sink.

**Data**: meshes and AABBs of fixed-base furniture only; fillable volumes; the item's height from its seen box
(0 when unseen); the BDDL evaluator.

**Tests**: `test_tiptop_presweep_fixes.py::test_a_case_of_shelves_offers_one_reachable_compartment_not_its_bottom_board`,
`::test_a_touching_goal_gets_a_reachable_board_of_the_fixture_not_its_hull_top`,
`::test_a_named_table_and_a_fixtures_footprint_become_the_planners_support_plane`;
`test_tiptop_protocol.py::test_under_is_a_placement_on_the_planners_support_plane`;
`test_tiptop_strategies.py::test_an_under_goal_is_a_placement_the_runner_schedules`,
`::test_the_episode_judges_rounds_without_the_simulator` (extended);
`tiptop/tests/test_collision_world.py::test_fixed_furniture_a_near_goal_references_stays_a_movable_so_cutamp_accepts_the_goal`,
`::test_a_near_reference_kept_movable_is_never_picked`.

**Offline replay** (CPU; the target's mesh from each recorded request's room entry; first-pass code, with item
heights fixed per category rather than taken from seen boxes):

| Goal | Pre-fix | Fix4 |
|---|---|---|
| inside(x, bookcase), 70 requests | 27 regions at the case's bottom board (0.121 m, 0.047 m); 43 hull tops (the 1.600 m crop, 1.054 m) | 70 of 70 at the board nearest 0.9 m: 0.726 m (ceiling 1.052 m) or 0.722 m (ceiling 0.981 m), near edge within 1.0 m of the base |
| touching(shoe, hallstand), 5 requests | hull top at 1.600 m (IK 0/256 in 4 of 5) | the bench board at 0.549 m in 4, the 0.967 m shelf in 1 |
| nextto(mousetrap, sink), 17 requests | 17 of 17 rejected: "unknown movable literal 'sink_1'" | 17 of 17 plan MoveHolding then PlaceNear |
| ontop(book, table_1), tidying_bedroom i2 requests 009-012 | no surface sent; RANSAC took the floor at -0.003 m | the nightstand's top at 0.342 m as the support |

under() has no replay: sweep3 has no under rounds. No replay executes a placement, so whether items stay inside a
compartment is not measured.

## 3. Movables in the planner

**What changed**

- `tiptop/tiptop/tiptop_run.py: surface_spheres`: each movable's perceived surface as spheres, one per 7.5 mm voxel
  of its cloud (`object_pcds`: depth under its instance mask in every view, merged), radius half a voxel. The
  budget is 3000 spheres per scene; above it one common voxel grows by sqrt(total / 3000).
- `tiptop_run.py: surface_mesh`: the same surface as one cuRobo mesh, a cube per sphere, in the movable's frame.
- `tiptop_run.py: perceived_surfaces`: sets `env.surface_spheres` and `env.surface_meshes` for the non-fixed
  movables. Called by `tiptop_websocket_server.py` after `apply_room_object_roles` (fixed furniture takes no share)
  and by `tiptop_offline.py`.
- cuTAMP patch 17 (`tiptop/install/patches/cutamp-17-movables-by-perceived-surface.patch`):
  - `tamp_world.py: TAMPWorld.get_surface_spheres`: the perceived surface per movable, or the pre-fix OBB cover when
    the environment has no points (cuTAMP's own environments, old pickles).
  - `cost_function.py: CostFunction._validate_rollout`, `collision_costs`: a per-movable schedule. The robot is
    checked against the movable's perceived surface at every timestep up to and including its Pick, not at all
    while it rides in the hand, and against its OBB cover at the placed pose from its Place on (the body) or from
    the next action (the hand, `hand_spheres`: the spheres on links rigid with the end-effector, from cuRobo's
    kinematics). The surface checks use the world's 5 mm activation distance, the same as cuRobo's motion
    planning; the cover checks keep the gripper activation distance (0 by default).
  - `particle_initialization.py: ParticleInitializer`: the grasp filter scores the gripper against the target's
    perceived surface, not its OBB cover, at the world's activation distance.
  - `utils/common.py: get_world_cfg`: cuRobo's world gets the perceived surface mesh in each movable's slot, so the
    motion planner and the cost see the same open basket.
  - `motion_solver.py: _resting_contacts`: statics and movables (the container, packed neighbours) are exempt for
    the carried object during the lift. `_lift_height`: the lift rises in 5 cm steps, up to 30 cm, until the carried
    object clears every support (a basket rim). `plan_retract` uses it.
- `r1pro.py: jaw_spans`, `bench.py: Episode.pick`: an object whose smallest seen extent is not under `JAW_GAP`
  (0.097 m, the model's open-jaw gap) skips the planner's holding round and goes straight to the press. An unseen
  object gets the planner round.

**Data**: object point clouds, with and without `--room`. The robot's kinematics.

**Tests**: `tiptop/tests/test_movable_surface_cover.py` (new, 7):
`test_the_inside_of_an_open_basket_is_free_by_its_surface_and_solid_by_its_cover`,
`test_each_movable_counts_by_surface_until_its_pick_and_by_cover_from_its_place`,
`test_at_the_place_the_body_is_kept_out_of_the_placed_object_and_the_hand_is_not`,
`test_the_hand_is_what_turns_with_the_last_wrist_joint`,
`test_curobo_keeps_the_arm_out_of_a_basket_by_its_perceived_surface_not_its_hull`,
`test_resting_contacts_name_the_basket_and_the_neighbour_but_not_the_item_itself`,
`test_the_lift_rises_until_the_carried_object_clears_the_rim`.
Bridge: `test_tiptop_presweep_fixes.py::test_an_object_wider_than_the_jaw_every_way_round_skips_the_planner_round`,
`::test_the_jaw_test_reads_the_objects_own_level_axes`.

**Offline replay** (448 recorded holding rounds: every flagged round and every executed round; each request's
clouds rebuilt from its views; a round counts as blocked when none of its 256 grasps is clear of every movable at
the Pick):

| Rounds | OBB cover (pre-fix) | Simulator mesh (first pass) | Perceived surface (fix4) |
|---|---|---|---|
| Flagged failed picks, 276 | 274 blocked (99%) | 129 (47%) | 90 (33%) |
| ENCASED (basket contents), of 122 | 121 | 40 | 20 |
| WIDE, of 123 | 122 | 77 | 68 |
| OTHER, of 31 | 31 | 12 | 2 |
| sorting_vegetables, of 57 | 57 | 25 | 10 |
| Executed and held, of 85 | 13 | 0 | 0 |
| Executed, not held, of 87 | 14 | 0 | 1 |

No round that was clear under the OBB cover is blocked by the surface (0 of 448). 45 flagged rounds that the
surface clears are blocked by the true mesh: the cloud of a packed or leafy object has gaps its real shape fills.

End to end through the planning path, with each request's own config and seed: sorting_vegetables i2 round 41
(sweet_corn_3 in the basket; recorded 0/256 satisfying, no plan) now plans in 15.6 s, lifting the corn out of
the basket. sorting_vegetables i0 round 1 (bok_choy_2, 14.6 cm against the 9.7 cm jaw) passes 130/256 at init,
then fails cuRobo IK 22 times and has no plan after 85.6 s: a jaw-width case, which the jaw test sends to the press.

The jaw test over the sweep3 recordings: the 17 pillar candle presses stay skipped. Plates (13 of 13 recorded
holds), bowls, books, toy figures, tiles and sandals seen lying flat (12 of 17) now go to the planner. toilet_tissue
(0.100-0.108 m every way; the planner held it twice in sweep3) and a teddy (0.112 m) are skipped.

Cost of the surface term: at the 3000-sphere budget about 3.3 ms per optimisation step (256 particles, forward and
backward), about 1.6 s per 500-step skeleton; 12,000 unbounded spheres cost 11.6 ms, about 5.8 s. The cube mesh
costs about 50 ms per movable per request that reaches refinement.

The proxy table and the end-to-end runs predate two review fixes: the budget moved from per object to per scene,
and cuRobo's world moved from meshes to the perceived surface. Neither was replayed again.

## 4. The task layer

**What changed** (all in `tiptop/b1k/bridge/strategies.py`):

- `Demand`, `place_demand`: items are keyed per (kind, container), the objects an option actually puts there.
  nextto(sandal_1, bed_1) wants sandal_1 only.
- `Runner.best_options`: before every pass, keeps the options with the most place atoms holding now and reads the
  demand from them. `self.placed` holds the atoms that hold in any option read. When no option agrees with
  everything in place (the 20,000-option sample does this late in gift baskets), it does not narrow.
- `Runner.additive`: an item that already satisfies an atom is picked again only if the move can add an atom. An
  atom whose support is itself still to be moved (a nextto partner) is not protected.
- `Runner.transfer`, `transfer_one`: an Unreachable (item, container) pair is remembered for the instance and not
  tried again.
- `Runner.order_containers`: a container that is itself a pending item sorts last (the bed before sandal_1, the
  floor before tiles).
- `Runner.free_hand`: after both planned floor put-downs fail, `Episode.release()` opens the hand before
  TransferBlocked. TransferBlocked fires only if the hand still holds the object.
- `task_goal_options`: over the 20,000 cap, a seeded random sample (`random.Random(0)`), not the first N (the first
  20,000 gift-basket options all put candle_1 in basket_1).
- `bench.py: Episode.release`: docstring; it is now the runner's last resort.

**Data**: BDDL ground options and the goal evaluator.

**Tests** (`test_tiptop_strategies.py`): `test_a_book_already_on_one_nightstand_is_not_carried_to_the_other`,
`test_the_bed_slot_takes_only_the_sandal_the_goal_names`, `test_a_partner_is_placed_before_anything_is_put_beside_it`,
`test_an_item_is_not_carried_again_to_a_container_no_stance_reached`,
`test_an_item_already_placed_is_only_moved_for_an_atom_it_can_add`,
`test_the_capped_read_samples_the_options_rather_than_taking_the_first`,
`test_tiles_lying_beside_each_other_are_still_carried_to_the_other_floor`,
`test_an_item_in_a_basket_the_goal_accepts_stays_when_the_options_read_lack_the_pairing_that_keeps_it`,
`test_a_full_hand_is_emptied_by_a_planned_floor_put_down_and_opened_only_when_none_plans`,
`test_an_unreachable_container_puts_the_item_back_and_opens_the_hand_only_as_the_last_resort`;
`test_tiptop_knowledge.py::test_assemble_gift_baskets_lets_go_of_an_item_no_plan_can_put_down_and_goes_on`;
`test_tiptop_transfer_recovery.py` (the release is expected as the last call).

**Offline replay** (CPU; a scripted episode that replays the recorded outcomes of each round):

| Episode | Pre-fix | Fix4 |
|---|---|---|
| tidying_bedroom 301 | 8 picks, 5 put-backs (2 repeated), 1 atom undone, 2 wrong-instance carries, q 0.0 (the shape of the recorded log) | 3 picks, 2 put-backs, none repeated, none undone, no wrong instance, q 0.333 |
| tidying_bedroom 303 | 8 picks, 2 undone, 1 wrong instance, q 0.0 | 6 picks, none undone, no wrong instance, q 0.0 (the bed has no stance either way) |
| picking_up_toys 301 | 10 picks, 10 put-backs, 5 repeated | 6 picks, 6 put-backs, none repeated |
| laying_tile_floors, two tiles start beside each other | 2 of 4 tiles reach floor_2 | 4 of 4 |
| setup_a_bar, three cans beside each other | 0 of 3 cans on the countertop | 3 of 3 |
| assembling_gift_baskets, the real 331,776 options, 25% first-pass failures, seeds 0-19 | 12 of 20 take an item out of a correct basket | 20 of 20 fill every basket, no item taken out |

The first three rows ran before review fixes S1 and S2; the last three test those fixes. `best_options` costs 2.6 s
per pass on 20,000 options with 16 atoms, for 2-5 passes an instance.

## 5. Where to stand

**What changed**

- `tiptop/b1k/bridge/geometry.py: best_base_pose`: rings extend to the reach plus the widest target's half-width.
  With boxes, the reach is measured to the target's AABB, not to its centre minus its radius. A target whose box
  top is no higher than `GROUND_ABOVE_BASE` (0.05 m) above the base is ground the robot stands on; it keeps
  centre-aimed rings (review fix F1: the widened rings had put a floor's stance in the next room).
- `geometry.py: frame_objects`: an object with no pixel in the head camera's frame is refused in the clipped
  (non-strict) pass too.
- `r1pro.py: best_base_pose(refused=...)`, `place_robot_for(refused=...)`, `place_robot`: each landing or unfold
  refusal is recorded as (x, y, yaw, obstacle). A candidate within `AVOID_RADIUS` (0.15 m) and `AVOID_YAW` (30°) of
  a refused stance is skipped. A fixed-base obstacle named by a refusal becomes hard: later candidates are
  re-checked against it with `base_placement_collision(only=...)` and `_motion_obstacles(only=...)`. A movable
  obstacle keeps only the avoid disc (review fix C1).
- `bench.py: Episode.stand_for`: one refused list per search, handed to the 0.9 m and 1.1 m searches and across
  the settle retries. It is not carried to the next search.
- `r1pro.py: _footprint_free`: a floor covering (paver, kerb) is ground only while its top is under `GROUND_TOP`
  (5 cm); floors always are. A 9.5 cm kerb now blocks the proposer as it blocks the validator.

**Data**: bridge AABBs, fixed-base room meshes, the landing check's own verdicts.

**Tests**: `test_tiptop_kinematics.py::test_the_rings_reach_past_a_target_wider_than_the_arms_reach`,
`::test_a_floor_is_stood_on_not_reached_from_beside`,
`::test_an_object_beside_the_camera_is_refused_however_big_its_projection_is` (now expects the refusal);
`test_tiptop_collision_scene.py::test_the_landing_checks_verdicts_are_fed_back_into_the_next_search`,
`::test_stance_retry_avoids_rejected_landings_without_repeating_motor_attempts` (updated);
`test_tiptop_presweep_fixes.py::test_the_widened_retry_is_told_what_the_first_search_refused`,
`::test_a_floor_covering_is_ground_only_while_it_is_flat`.

**Offline replay** (CPU; `best_base_pose` on each request's room meshes with the base footprint emulated):

| Search | Pre-fix | Fix4 |
|---|---|---|
| bed, tidying_bedroom 303 request_014 | no stance (3,409 candidates on the bed, as in the live log) | a strictly framed, free stance 0.84 m from the bed's edge (reach 0.9 m) |
| toy box, picking_up_toys 301 request_001, the sweep's three blind stances excluded | no stance | a strictly framed stance 0.99 m from the box |
| toy box, nothing excluded | a stance | the same stance |
| plywood_1, bringing_in_wood 301, the bridge loop with kerbs and planks as the validator | 16 candidates tested, 8 distinct (the 1.1 m retry re-tested the same 8, as in the log), none accepted | 4 candidates; the fourth, along the path, accepted |
| plywood_2, same | none | 3 candidates, accepted |
| a 2.58 x 9.03 m corridor floor and a 4 x 5 m floor, reach 0.9 and 1.1 m | inside the floor | the same stances (the first pass's widened rings had put all four 0.54-0.89 m outside) |

The toy box also lost stances to the arm-rest test in the sweep, which needs IK and is not replayed. The plywood
replay's first ring is 0.9 m where the sweep's was 0.8 m.

## 6. Self-collision in bridge ramps

**What changed**

- `collision.py: JointPathCollision.__init__(disabled_pairs=...)`: the ignored self pairs are cuRobo's
  `self_collision_ignore` plus the simulator's disabled collision pairs. `self_pairs`: sphere pairs across body
  groups (left arm, right arm, the rest). `check_polyline`: a robot-vs-robot distance test on those pairs with the
  robot's own radii and self buffer; a pair already touching at sample 0 is skipped.
- `r1pro.py: _motion_collision_model` passes the robot's disabled pairs. `ramp_collision(scene=False)` checks the
  robot against itself and its carried objects only.
- `r1pro.py: wrist_look`, `swing_collision`: each look offset gets two IK seeds (the current posture and the
  abducted branch). The candidates are ranked first, then the elbow-first swing is self-checked in rank order and
  the first clean one wins (review fix F1: the first pass checked every candidate, about 1 s each).
- `r1pro.py: restore_locked_arm`, `_capture_with_motion`: after a capture the idle arm returns to its locked
  posture by the first of three routes whose ramps pass (straight, elbow first, elbow last), instead of a warning.
  The next request carries the model's locked joints.
- `r1pro.py: open_container`, `tuck_idle_arm`: a handle approach refused by a link of the idle arm (on either side
  of the reported pair; review fix F2) tucks the empty idle arm to its own planner's `q_home` and approaches again.

**Data**: robot URDF, cuRobo spheres, the simulator's disabled pairs, the embodiment's `q_home`.

**Tests**: `test_tiptop_collision_scene.py::test_a_ramp_that_swings_a_hand_into_the_robots_own_head_is_refused`
(the real R1Pro URDF, spheres and disabled pairs);
`test_tiptop_presweep_fixes.py::test_a_look_configuration_that_swings_into_the_robot_is_re_solved_on_the_abducted_branch`,
`::test_a_locked_arm_whose_straight_way_back_is_refused_goes_elbow_first`,
`::test_the_tuck_takes_the_idle_arm_to_its_own_planners_ready_posture_unless_it_holds_something`,
`::test_the_idle_arm_is_tucked_when_it_rides_the_torso_into_the_handle_approach`.

**Offline replay** (CPU; all 7,311 capture, return, fold and unfold legs of the 114 episodes, re-checked on the
commanded path with each tree's `collision.py`):

| Leg | Blocked legs refused, pre-fix / fix4 | Clean legs refused by fix4 |
|---|---|---|
| Capture swing out | 0 / 39 of 43 (the 4 missed are right finger vs left_arm_link6, which the spheres cover about 6 mm short) | 45 of 1,528 (44 overlap on the measured path too, 1 is within the base_link buffer) |
| Return after the capture | 0 / 6 of 6 | 40 of 1,436 (all within 5 mm on the measured path) |
| Fold for travel | 0 / 10 of 39 (the other 29 are torso lag with no self contact) | 2 of 2,671 |
| Unfold after travel | 0 / 4 of 50 | 6 of 1,033 |
| Fold again after the teleport | 0 / 6 of 7 | 4 of 45 |

A 3 rad self-only check costs 0.40 s. An unfold the new check refuses falls back to a partial or elbow-first
unfold. For store_honey 302 the tuck moves the hanging right hand from (0.32, -0.22, 0.35) to (0.0, -0.33, 0.66)
in the base frame and its ramp is self-clear; the handle approach itself is not replayed (the cabinet mesh is not
in the recorded data).

## Also in this round

These come from earlier entries of the sweep3 findings, not the six issues.

- A floor in the planner's world. `tiptop_run.py: FLOOR`, `create_tamp_environment`: a 6 x 6 x 0.5 m static
  cuboid with its top 2 cm under base_link, in every request (installing_a_scanner's presses had been planned
  through the floor and were refused by the bridge's audit). Test: `tiptop/tests/test_collision_world.py::test_every_world_has_a_floor_under_the_robot_that_clears_its_own_wheels`.
  No replay: it needs cuRobo motion generation.
- Press-only requests. `tiptop_run.py: run_perception` sends `FLOOR` as the support when the client sent none, and
  `process_scene_geometry` returns an empty scene instead of raising when a support is given. turning_out_all_lights
  i2 request_001 failed "No object contact points found" and now yields a scene with the floor as its table. Test:
  `tiptop/tests/test_support_surface_fallback.py::test_a_press_rests_on_nothing_so_the_floor_stands_in_even_with_no_object_in_view`.
  The wall press from the switch stance still fails in cuTAMP (ValidPush 0/256).
- `tiptop/b1k/bridge/judgement.py: verdict_caption`: the reason follows the score, so a long reason no longer hides
  q_score in the video. Test: `test_tiptop_knowledge.py::test_verdict_caption_tells_success_from_failure_and_lists_what_is_missing`.

## Review

A review of the combined diff raised 27 findings. Two verifiers checked each bug finding. 15 were confirmed by both
and all 15 are fixed. 2 split (one verifier confirmed, one refuted) and were left, with reasons. 5 cleanups were
applied. The other 5 were bug, risk or compliance findings that neither verifier confirmed; the review workflow
drops those and the result file does not list them.

| ID | Where | Defect | Fix |
|---|---|---|---|
| PG-1 | `r1pro.py: jaw_spans` | judged only the two level axes of a halo-inflated box, so plates, bowls, books and shoes skipped a planner round that held them in sweep3 | the smallest of all three extents against `JAW_GAP` |
| F1 | `r1pro.py: wrist_look` | swing-checked every candidate, about 1 s each, 3-4 s per arm per capture | rank first, check in rank order, stop at the first clean one |
| F2 | `r1pro.py: open_container` | the tuck fired only when the refusal started with the idle arm's name; self pairs report base_link or the working arm first | tuck when either side of the pair names the idle arm |
| F3 | `r1pro.py: lift_held` | released on any preflight refusal by a robot link, whatever the path sample | `ramp_to` returns the path sample; release only at sample 0 |
| GR-2, C5 | `r1pro.py: item_height`, `knowledge.py: OracleKnowledge.describe` | fell back to the item's simulator AABB, and the regions were computed before this capture's points were recorded | `remember_seen` before `inside_regions`; an unseen height is 0.0 (headroom only), never the AABB |
| C1 | `r1pro.py: best_base_pose` | the per-candidate re-check read the simulator mesh of any obstacle a refusal named, movable ones included | only fixed-base obstacles become hard |
| C2 | `r1pro.py: base_placement_collision` | the hand-at-the-floor excuse covered movable flat items such as loose tiles | fixed-base ground only |
| S1 | `strategies.py: Runner.additive` | nextto partners deadlocked: tiles beside each other were never carried to the other floor | an atom whose support is still pending is not protected |
| S2 | `strategies.py: Runner.best_options` | with the 20,000-option sample, correctly placed items dropped out of `self.placed` and were taken out of their baskets | `placed` from atoms that hold in any option read; no narrowing when no option agrees with the state |
| F1 | `geometry.py: best_base_pose` | the widened rings put a floor target's stance off the floor, often in the next room | ground targets keep centre-aimed rings |
| F2 | `tiptop_run.py: surface_spheres` | the 3000 budget was per object, so cluttered scenes sent 10-15k spheres | the budget is per scene |
| F3 | `cost_function.py: collision_costs` | the surface term's cost grew with the total sphere count, with no cap per request, and ran at every timestep before masking | bounded by F2's scene budget; each group runs only up to its last timestep |
| F1 | `motion_solver.py`, cuRobo world | cuRobo still judged a movable by its mesh (a solid hull in competition mode), so picks the cost accepted inside a basket would fail in motion planning | cuRobo gets the same perceived surface as a cube mesh (`surface_mesh`, `get_world_cfg`) |
| F2 | `cost_function.py: _validate_rollout` | nothing checked the robot against a planned-pick object at its Place | the body is checked from the Place, the hand from the next action (`hand_spheres`) |

Split, not fixed:

- GR-3, `protocol.py: tiptop_goal`: under(x, f) is sent as on(x, table) even when no floor region goes out
  (`--knowledge onboard`, `--no-inside-region`, a movable fixture). Not reachable in the default configuration:
  every task spec with an under goal names a fixed-base fixture, and onboard knowledge and `--no-inside-region` send
  no regions for any predicate.
- C3, `r1pro.py: carried_volume`: the mesh fallback for an unseen held object, taken every time under onboard
  knowledge. Left because the mesh read predates fix4 (fix4 put it behind the seen box), and a stand-in at the new
  call sites alone would disagree with the preflight that follows them. The fix needs the planner to report
  perceived extents back to the bridge.

Cleanups applied: PT-1, a stale `ponytail:` comment in `apply_room_object_roles` deleted (patch 18 did the work it
deferred); PT-2, a dead `None` sentinel in `cost_function.py`; PT-3, the robot's own radii computed once in
`check_polyline`; PT-4, `functools.cache` in `best_options`; PT-5, `top_surface` and `floor_surface(within=...)`
merged into `footprint_region`.

## Tests

Both suites were re-run on the final tree on 2026-09-23.

```bash
# Bridge, from the repository root: 401 passed
CUDA_VISIBLE_DEVICES=<gpu> OMP_NUM_THREADS=8 OMNIGIBSON_HEADLESS=1 PYTHONPATH="$PWD/tiptop" \
  ./b1k/bin/python -m pytest OmniGibson/tests/test_tiptop_*.py -q -p no:cacheprovider

# Planner, from tiptop/: 171 passed, 5 deselected
CUDA_VISIBLE_DEVICES=<gpu> OMP_NUM_THREADS=8 .pixi/envs/default/bin/python -m pytest tests -q -p no:cacheprovider \
  --deselect tests/test_tiptop_h5.py
```

`PYTHONPATH="$PWD/tiptop"` matters: without it the sim env may import `b1k` from another checkout. The 5 deselected
tests are `test_tiptop_h5.py`, marked integration (they need an M2T2 server). Run from `tiptop/` they also stop at
`check_cutamp_version`, because the checkout directory `tiptop/cutamp/` shadows the installed package as a namespace
package. That is not fix4's doing.

Fix4 changes 8 OmniGibson test files and adds 36 test functions to 7 of them; 4 of the 36 replace tests whose
contract changed (all four because the hand may now be opened as a last resort). The eighth file,
`test_tiptop_sticky.py`, only has its fakes extended. The planner gains 11:
`test_movable_surface_cover.py` (new, 7), `test_collision_world.py` (+3), `test_support_surface_fallback.py` (+1).
Existing tests updated for the new behaviour: the beside-the-camera kinematics test; five call lists in
`test_tiptop_transfer_recovery.py`; the demand, basket-ordering, full-hand, unlocatable-floor and judging tests in
`test_tiptop_strategies.py`; the caption, oracle-mask and inside-region tests in `test_tiptop_knowledge.py`; the
ramp-preflight return and stance-retry tests in `test_tiptop_collision_scene.py`. Other edits to existing tests only
adapt fakes and stubs to the new signatures.

Lint: ruff 0.9.10 (run from the uv cache, not installed) finds 14 issues in the changed OmniGibson files, the same
14 as on the pre-fix copy; fix4 adds none. `ruff format --check` drift is also pre-existing. In the submodule (no
hook applies) there are 3 new findings: an E731 lambda (ignored by the repo's config) and two F722 from jaxtyping
annotations in `tamp_world.py`, the file's existing style.

Patches: the pre-fix cuTAMP tree plus patches 17 and 18 equals the live tree byte for byte, and reversing 18 then 17
gives back the pre-fix tree. `tiptop/install/install-cutamp.sh` applies `cutamp-*.patch` in order and skips a patch
that already reverse-applies.

## Eval status

- sweep3, before fix4: mean q 0.272 over 114 episodes, against 0.195 the week before (paired +0.085 ± 0.073).
- fix4: the sweep ran from the frozen checkpoint `sweep4a_20260923` with one smoke lane. Its single job,
  laying_tile_floors instance 0, was killed mid-episode when the user stopped the sweep at 23:23 on 2026-09-23.
  There are no fix4 scores. The only fix4 video is that job's partial one; its last recorded step was env step
  2763, during a capture.
- Nothing in this document is validated on the benchmark. The replays say which recorded failures the new code
  would no longer hit; they do not say what the new code does next.
- The checkpoint's source files match the final tree (compared file by file, docs excluded), so a restarted
  sweep from it runs this code.
- When the next sweep is read: robot-vs-movable checks before a Pick now use perceived geometry in cuTAMP and in
  cuRobo, with or without `--room`. Its basket rounds should not be compared with the first-pass replay, which used
  simulator meshes. Where a cloud misses part of an object (45 of the 276 flagged picks that the surface clears are
  blocked by the true mesh), both planners now accept a path through that part.

## Known leftovers and ceilings

### Ceilings marked in the code (`ponytail:` comments)

| Where | Ceiling |
|---|---|
| `collision.py: check_polyline` (self check) | the model's own spheres: the right finger sits about 6 mm inside its hull against left_arm_link6/7, so 4 of the 43 blocked swings still pass |
| `collision.py: check_polyline` (`excuse_start`) | per link, not per sphere: a link keeps what its nearest point touched at the start |
| `r1pro.py: JAW_GAP` | R1Pro's number, hard-coded; read it from the embodiment's sphere file if another gripper arrives |
| `r1pro.py: best_base_pose` | the hard-obstacle re-check re-reads every scene AABB per candidate (about 50 ms) |
| `r1pro.py: base_placement_collision` | no depth comparison across floors; ground higher at the destination (a kerb) is not caught by the floor excuse |
| `scene.py: remember_seen` | bounds, not points, and only of the faces a camera saw (a tile seen only from above is a sheet) |
| `geometry.py: frame_objects` | judged on the box corners ahead of the camera, so a box cut by the near plane whose ahead corners all miss the image is refused even if the cut part would show |
| `geometry.py: best_base_pose` | a flat item lying on the floor counts as ground too and is aimed at from its centre; a step up of more than 5 cm is not ground |
| `strategies.py: transfer_one` | a flat ban per (item, container) once no stance reached it; 7 atoms in sweep3 came from re-attempts that a changed posture made reachable |
| `tiptop_run.py: surface_spheres` | a crowded scene gets a coarser, fatter cover for every object; a small target among large ones may lose its grasps |
| `motion_solver.py: _lift_height` | 5 cm steps up to 30 cm; a deeper container is left to `_validate_lift` |

### Not wired, or open

- `--knowledge onboard` sends no `place_surfaces` and never calls `remember_seen`, so the shelf, touching, table and
  under regions and the seen boxes exist only under oracle knowledge. `--no-inside-region` switches the regions off.
- A held object nobody has seen gets the mesh-derived carried volume (C3 above).
- After review fix F3, `lift_held` releases in practice only for floors. The laying_tile 302 and 303 grasps (the
  carried tile starts against base_link) were not replayed after it; `free_hand`'s release remains the last resort
  there.
- `return_to_ready` while holding is one straight checked ramp. The planner's move request takes no attachment, so
  there is no planner fallback while carrying (clean_up_your_desk 302).
- The carried object is still its OBB cover in cuRobo attachments and in cuTAMP's movable-to-movable checks.
- The perceived surface leaves unseen faces open, for the cost and for cuRobo alike. After a Place, cuRobo moves the
  object's perceived shell rigidly.
- cuRobo's `r1pro_right.yml` lists right_arm_link6 in the attached object's ignore list, so cuRobo never checks the
  carried object against link6; cuTAMP's cost now does.
- nextto(x, fixed sink) reaches cuTAMP, but PlaceNear places onto the "table" plane; a floor slab goes out only when
  the goal also names a floor support. A floor item cannot be nextto a wall-hung sink; the demand should prefer
  under() there.
- `FLOOR`'s top (-0.02 m) is tied to the bridge's floor slab top (-0.005 m); raised ground would need both moved.
- turning_out_all_lights: presses are requested before a stance near the switch, and the wall press still fails.
- The stance for touching(x, hallstand) is chosen for the fixture's centre, not its bench.
- `geometry.footprint_blockers` exempts the pick target that passes under the base by height (laying_tile 303's tile
  corner is 10 cm under the chassis front).
- The server's `validate_locked_joints` still refuses measured locked joints that differ from the model.
- `put_down`'s done-test falls back to the first floor on two-floor tasks.
- The clean_up_your_desk 303 laptop: the seen box and the room hull disagree (lid shut or not); needs the sim or
  the video.
- Offline replays of recorded requests fall back to the OBB cover unless the clouds are rebuilt, and pickled cuTAMP
  goal atoms carry a stale string hash (rebuild them after unpickling).
- The live-robot demo path (`tiptop_run.async_entrypoint`) never calls `perceived_surfaces`, so it keeps the pre-fix
  OBB cover and hull. It is not run in this project.

## How to run an eval of this

Pin the code. The planner packages (`tiptop`, `cutamp`, `curobo`) are editable installs, and the planner's relaunch
loop re-executes from the live submodule after a CUDA fault. A sweep must therefore import only a frozen copy.

1. Freeze a checkpoint: copy `tiptop/tiptop`, `tiptop/b1k`, `tiptop/cutamp/cutamp`,
   `OmniGibson/omnigibson/tiptop` and `tiptop/curobo/src/curobo` into one directory, with a manifest of sha256
   hashes and the git diffs of the root and the submodules. Every job stamps the manifest hash in its report.
2. Run lanes. Each lane pops "task index" lines from a shared queue under `flock`. Per job it starts a fresh planner
   from the checkpoint (`PYTHONPATH=<checkpoint>/source`; `tiptop-server --config
   <checkpoint>/source/tiptop/config/tiptop_sim_r1pro.yml --num-particles 256 --max-planning-time 40 --rerun-mode
   episode --seed 2300`), a second right-arm planner for press-and-hold tasks, then the simulator through the
   checkpoint's `audit_bench.py` (with `AUDIT_SOURCE=<checkpoint>/source` it imports the bridge and `b1k` from the
   copy, then runs the bench with these arguments):

   ```bash
   python <checkpoint>/audit_bench.py --task-name <task> --instances <i> --knowledge oracle \
     --grasping-mode sticky --views head left_wrist right_wrist --port <port> \
     --room --inside-region --no-state-stream --out-dir <job>/episode
   ```

   Start lanes with `setsid nohup`, keep at most about 8 simulators on the host, and stop a lane after its current
   job by touching a stop file.
3. Judge each job with `scripts/read_run.py` (which reads `baselines.json`) and compare with sweep3 per task and
   instance.

For a single run without the harness: `scripts/start_tiptop_server.sh` with
`TIPTOP_CONFIG=tiptop/config/tiptop_sim_r1pro.yml` and `TIPTOP_GPU`, then `scripts/run_bench.sh <bench arguments>`.
`run_bench.sh` exports `PYTHONPATH` with this checkout's `tiptop/` first. Without that, `import b1k` in the sim env
resolves to another checkout that has none of the geometry, strategies or protocol changes; `run_queue.sh` does not
set it.

## Local evidence (not in git)

- `runs/sweep3_20260923/FINDINGS.md` (section "Systematic issues from the video review") and
  `runs/sweep3_20260923/video_review/result.json`: the six issues and their measurements.
- `runs/fix4_eval_prep/coder_reports.json`: what each first-pass change did, its tests and offline replays.
- `runs/fix4_eval_prep/rework_review_result.json`: the point-cloud rework, integration, the review's findings and
  verdicts, the fix pass and the final test runs.
- `runs/collision_audit/sweep3l_20260923/source/`: the pre-fix code. `runs/fix4_baseline/`: the pre-fix tests.
- `runs/collision_audit/sweep4a_20260923/`: the frozen fix4 checkpoint (manifest, `sweep_lane3.sh`, `audit_bench.py`).
  `runs/collision_audit/freeze_checkpoint.py`: the freeze script.
- `runs/sweep4_20260923/`: the stopped fix4 sweep (`WHY_STOPPED.txt`, `report_L1.txt`).
