# Manipulation atoms: build log

The 100 challenge tasks were broken into atomic manipulation primitives and a build order
(`runs/skill_gap_20260924/atomic/atomic_manipulation.md`: section 1 the atom table, section 2 the skill recipes with the
simulator rule each must meet, section 5 the build items A-assist .. N-transport; `runs/manip_atoms_20260924/DESIGN.md`
the per-tier contract). This file records what each tier actually built, on branch `dev/manip-atoms`: per build item,
what changed, the data it reads, the test, the offline replay, and what is still unproven in simulation. Replay
scripts and their outputs are under `runs/manip_atoms_20260924/<package>/`. cuTAMP changes live as numbered patches in
`tiptop/install/patches/`.

Data rule every item follows: environment geometry (walls, floors, fixed-base furniture) may be read from the map;
objects are known only by the points the cameras saw (`scene.seen_boxes`, depth under masks; the planner's perceived
clouds and hulls); privileged simulator state enters only through `OracleKnowledge` hints, named as such.

## Tier 1 (2026-09-24): A-assist, E-tip, E-part, E-near, E-conj, E-pressface, E-region, auto spec

Commits: c207733 / d91c899e5 (first pass) and this section's second-pass commit in both repos (the near aim uncapped,
thickness read from the support, a lawn as the only floor). cuTAMP: `install/patches/cutamp-19-fingers-touch-the-support-near-aim.patch`
and `cutamp-20-near-aim-half-of-allowed.patch`. Suites after integration: bridge `OmniGibson/tests/test_tiptop_*.py`
409 passed (401 before the tier); planner `tiptop/tests/` 180 passed, 5 deselected (171 before).

### A-assist: every pick is a planner round; the sticky press only under sticky

- `bench.py: Episode.pick`. The jaw gate (`R1ProSim.jaw_spans`, `JAW_GAP`) is deleted: every pick runs the planner's
  `holding()` round. The `press_grasp` fallback (closed-hand contact seek) runs only when `args.grasping_mode ==
  "sticky"`: under assisted the hand closes before it touches, so the weld's finger-to-finger ray (`robots/robot.py`
  ~3147-3192) has nothing to hit.
- Reads: the run's grasping mode. Nothing privileged.
- Test: `test_tiptop_presweep_fixes.py::test_under_assisted_grasping_every_pick_is_a_planner_round_and_the_press_never_runs`
  (assisted: two planner rounds, no press; sticky: one round then the press).
- Replay: none (the sticky-only loss of the pressed grasp is counted in the spec, 308/445 in sweep3).
- Unproven in simulation: that the planner's part grasps (E-part) take the objects the press used to take.

### E-tip: the fingertips may touch the support at the grasp (cuTAMP, cutamp-19)

- `cutamp/cost_function.py: finger_spheres` (the planning arm's finger-link sphere mask, now the one definition;
  `motion_solver._non_contact_spheres` is a line over it), `CostFunction.__init__` (`self._fingers`, a zero-activation
  world checker `world.touch_fn` cached per world), `_validate_rollout` (`self._at_pick`, the Pick timesteps),
  `collision_costs` (at Pick timesteps the finger spheres leave `robot_to_world` with radius -1 and their centres are
  checked as 1 mm spheres at zero activation: touch, not 1 mm in). `cutamp/motion_solver.py: _validate_lift` (the
  "robot touches its support" check runs on non-carried, non-finger spheres), the Pick terminal approach (the
  object's `_resting_contacts` disabled for the terminal `plan_pose`, `_validate_contact_motion` per support, re-enabled
  in `finally`, the Push stroke's shape).
- Bridge side, wired at integration: `r1pro.py: validate_motion` exempts the finger links against what the object
  rests against (`rests_against`, the AABB-adjacent bodies, factored out of `lift_held`) for a `Pick(...)` motion; the
  wrist and the rest of the arm stay checked. Without it the bridge refused exactly these plans in sweep4
  (organizing_art_supplies round 18: `('left_gripper_finger_link2', 'tote_93')` on Pick(paintbrush_1)).
- Reads: cuRobo's own sphere model and world; the bridge reads scene AABBs (a tote is a task object, but the check
  only exempts contact, it does not plan on it).
- Tests: `tiptop/tests/test_lift_off_support.py::test_a_fingertip_may_be_in_the_support_at_the_grasp_and_the_lift_the_palm_may_not`,
  `tiptop/tests/test_movable_surface_cover.py::test_at_the_pick_a_fingertip_may_touch_the_desk_and_the_palm_may_not`
  (real cuRobo cuboid checkers); bridge
  `test_tiptop_collision_scene.py::test_planned_validation_uses_live_grasp_state_measured_opposite_arm_and_narrow_contacts`
  (the bookcase the book rests against gets the fingers, a lamp in reach does not, a non-Pick label exempts nothing).
  The bridge test's old assertion that the support is never exempted at a Pick was the old rule and was replaced.
- Replay (`runs/manip_atoms_20260924/T1-cutamp/etip_*.out`, sweep4 organizing_art_supplies_i0, GPU 2, recorded
  env/grasps/q_init): Pick(rubber_eraser_1) 10-42-02 `robot_to_world` 0/256 -> 256/256 satisfying (only the two finger
  links had penetrated, mean 6e-4); Pick(paintbrush_1) 10-47-39 0/256 -> 245/256 (the 11 left are torso_link2 in the
  desk), overall 0 -> 215/256. IK success on the paintbrush grasps is 0/256 before and after: the IK solver's own world
  term rejects them, cuTAMP takes the solutions regardless.
- Unproven in simulation: the whole path (part grasp -> cuTAMP -> bridge validation -> assisted weld) has not run;
  the approach 5 cm above is still IK'd against the full world; the Place-timestep analogue (fingers holding a thin
  object penetrate the table at its placement, 10-44-36) is untouched; after E-tip the next particle blocker for thin
  picks is `robot_to_movables` (fingers vs the object's own perceived surface at 5 mm activation).

### E-part: grasps from points where M2T2 finds none (planner)

- `tiptop/tiptop_run.py: part_grasps` (kinds `tip` under 3 cm tall, `corner` on slabs wider than the jaw both ways and
  7 mm to a finger's length thick, `edge` on bulky objects; box = min-area rectangle of the cloud in xy, not PCA, which
  was 30 deg off on square tiles), `weld_ok` (the assisted weld read on the cloud: the near-tip ray 3.9 mm behind
  the tips runs >= 5 mm inside the hull, the cloud between the fingers is narrower than the jaw by 1 cm, no point inside
  a finger), `fingertip_in_tool` (the embodiment's press point), `process_scene_geometry(fingertip=...)` appends them to
  `filtered_grasps` per label except held labels; the empty-grasps `contacts` shape fixed from (0,0,3) to (0,3).
  Constants `JAW`, `RAY_BEHIND_TIPS`, `FINGER_*` are R1Pro's (ponytail-marked). Support z: the fitted plane's surface
  when the object rests on it (a camera never sees an underside; the paintbrush's lowest seen point was 2 mm above the
  desk), else the cloud's lowest points. Second pass: the thickness is measured from that support, not from the
  cloud's own z range (a puzzle box 1.2 cm over the plane with 6 mm of it seen is 1.8 cm thick, not under the 7 mm
  corner gate), and a sheet thinner than ~6 mm takes the tips down onto the support so the near-tip ray runs under
  its top (S02: tips within height - 4 mm of the support); never so low that the ray leaves the hull.
- Reads: the object's perceived points and hull only; the RANSAC plane or the client's support box.
- Tests: `tiptop/tests/test_support_surface_fallback.py`: a book seen from above gets 8 corner pinches with the tips
  2 mm above its bottom and the ray inside the slab; a 10 cm cube gets 8 edge grasps and nothing top-down; a 1 x 14 cm
  pen gets 8 tip grasps centred within 1 mm, and a finger on it fails `weld_ok`;
  `::test_thickness_is_read_from_the_support_and_a_sheet_takes_the_tips_down_to_it` (the puzzle box: 0 grasps from
  its own 6 mm of cloud, 8 corners with the support given; a 5 mm sheet: 12 tip grasps with the tips on the support).
- Replay (`T1-planner/part_grasps_eval.out`, CPU, 66 object rows over 27 sweep4 pick rounds): tiles 0 -> 4-6 passing,
  eraser 0 -> 4, paintbrush 0 -> 6, magazines 0 -> 4-8, markers 0 -> 6, glue stick 0 -> 2, board games 0 -> 6. With
  the thickness from the support (before copy `part_grasps_eval_c207733.out`): picking_up_toys r01 jigsaw_puzzle_2
  (the goal, 0 M2T2 grasps; the round failed and fell to the sticky press) 0 -> 6, r08 0 -> 6, sorting_bottles r29
  magazine_1 0 -> 6; the desk and countertop rows lose their 8 always-refused corner candidates; every other row is
  unchanged. Still nothing for newspapers (0.4-0.5 cm from the plane, under the 7 mm corner gate); bottles and cans
  stay M2T2's.
- Unproven in simulation: whether cuTAMP's IK and gripper-sphere filter keep these poses, and whether they weld under
  assisted; for the tips on the support (sheets under ~6 mm) E-tip's touch-not-penetrate check must hold at the Pick
  timestep, so a plane estimate a few mm high rejects them (no grasp, as today). Not covered: sheets under 7 mm wider
  than the jaw both ways (newspapers), and slabs 6.5-9.7 cm tall wider than the jaw both ways (a scanner lying flat:
  the corner kind caps at a finger's length; no measured goal object needed it).

### E-near: near() lands next to its reference (cuTAMP + planner)

- `cutamp/cost_function.py: near_placement_costs` (metres the two sphere covers' AABB gap exceeds `near_aim` =
  mean(dims_obj + dims_ref) / 6 / 2, half of what the sim's NextTo allows, uncapped since the second pass (the 1 cm
  cap `NEAR_GAP_AIM` is deleted, cutamp-20), plus the metres a horizontal ray from either centre misses the other's z range;
  `near_thresholds` deleted), `cutamp/particle_initialization.py: near_ring` and the PlaceNear sampler (xy on the ring
  around the reference's AABB at the object's radial extent + aim, clipped to the surface OBB; `place_cache` keyed by
  references too), `tiptop/planning.py: run_planning` (NearPlacement tolerance 2e-3, the stock 5 cm slack is why
  placements landed 8.5-15 cm apart). Deviation from the design: the AABB gap, not the min sphere-surface gap, because
  the sim's NextTo is an AABB gap and a sphere gap rejects two rotated sandals the sim accepts.
- Reads: the cuRobo world's sphere covers of the perceived objects.
- Tests: `test_movable_surface_cover.py::test_a_near_placement_costs_the_gap_past_the_aim_and_a_missed_horizontal_ray`,
  `::test_near_samples_ring_the_reference_inside_the_surface`.
- Replay (`T1-cutamp/enear.out`, the 7 executed nextto rounds of sweep3, aim 1.6 cm for the soda cans and 2.2 cm for
  the sandals; the capped 1 cm run kept as `enear_aim_capped_10mm.out`): the old cost accepted all 7; the new cost
  rejects the 5 the simulator failed (gaps 5.8-15.2 cm) and accepts the 2 it passed. The sphere-surface gap the spec
  names rejects the same 5 and also the sandal PASS (perceived sphere gap 14 cm where the sim's AABB gap is 2.7 cm),
  which is why the AABB gap stays. The conservative cover pads the planner's gap by ~1 cm, so the aim lands ~1 cm
  wider in the sim.
- Unproven in simulation: no near round has been planned live with the new cost; the `/ 2` in `near_aim` is the one
  knob if placements land too far or too close.

### E-conj: a placement round asks for every atom the placement decides (policy + planner + cuTAMP)

- `b1k/bridge/strategies.py: Runner.transfer` sends the primary atom plus `Runner.co_atoms` (every nextto partner of
  the item that is not itself pending, and, for a nextto primary, the item's one on-type atom; never a second on-type
  atom); the verdict is the primary's alone. `tiptop/tiptop_run.py: create_tamp_environment` groups near atoms by
  their first argument: the first is the Near goal fluent, the rest ride on `env.near_also`, which cuTAMP's near cost
  and sampler read.
- Reads: the task's goal atoms and the evaluator's current truth (task-class inputs only).
- Tests: `test_tiptop_strategies.py::test_a_placement_asks_for_every_atom_of_the_item_it_decides`,
  `::test_a_partner_still_to_move_is_not_asked_for_beside_the_item`, `::test_a_stacking_round_does_not_also_ask_for_the_bookcase`;
  `tiptop/tests/test_inside_region.py::test_further_near_references_ride_on_near_also_not_as_second_fluents`.
- Replay (`T1-policy/replay.out`, CPU, FakeEpisode with a truth-set placement model): sorting_household_items 301
  all-succeed world 7/8 -> 8/8 atoms; getting_organized_for_work 301 7/10 -> 10/10 with 11 -> 7 achieves (the
  ontop/nextto ping-pong is gone). As recorded (most items unreachable) before == after.
- Unproven in simulation: a round carrying two atoms has not been planned live; a start-true nextto whose partner is
  itself pending is still lost (the contract's own pending rule).

### E-pressface: a button on the body's face, at the body's surface (bridge)

- `r1pro.py: button_world` (vertices from `collision_mesh_world`, so the toggle marker's own sphere no longer enters the
  box; of the faces within half the marker width the most upward wins, else the nearest; the position is where a ray
  along the normal from 5 cm out meets the physical surface). `b1k/bridge/protocol.py: face_normal_local(vertices,
  point, within=None, up=None)`.
- Reads: privileged (the ToggledOn marker pose), only through `OracleKnowledge.button_hints`.
- Tests: `test_tiptop_presweep_fixes.py::test_a_button_sits_on_the_body_face_the_press_is_braced_on_and_its_position_is_on_that_surface`
  (wall switch on a wall, lighter flat and stood on end, washer); `test_tiptop_protocol.py::test_face_normal_local_prefers_the_braced_face_among_those_within_the_markers_radius`.
- Replay (`T1-bridge/button_faces.out`, 18 assets): 16/18 match the body-only face (the lighter goes to its top face
  by design; the vacuum to +z, 24.6 mm from its marker); the 5 mm stroke now ends past the surface on 18/18 (the washer's
  marker floats 11.2 mm proud, and its stroke used to stop 6.2 mm short). `button_faces_within_radius.out`: the
  design's `within=radius` (the full marker width) gives 15/18 (the strbnw wall switch flips to its +z top), so half
  the width stays.
- Unproven in simulation: no press has run with the projected position.

### E-region: floor-type supports, movable tables, reachable boards (bridge)

- `b1k/bridge/protocol.py: FLOOR_CATEGORIES = ("floor", "lawn")`; `bench.py: Episode.is_floor`, `r1pro.py: task_scope,
  floor_name, scope_floor, inside_regions, floor_surface` and the `track_task_objects` skip list treat a lawn as ground
  (hiding_Easter_eggs has no floor; `floor_name` raised in `Episode.__init__` on that scope).
  `r1pro.py: footprint_region` gives a movable table's slab from its own seen box (`own_box`, points), None until a
  capture has seen it (putting_up_Christmas_decorations_inside). `r1pro.py: shelf_of` drops boards above
  `PLACE_HEIGHT_MAX = 1.5` m and, when no board is within `BOARD_REACH`, takes the one nearest `PLACE_HEIGHT` anyway
  (the stance search moves the robot; putting_shoes_on_rack IK-failed 6 rounds on a hallstand's 2.37 m top).
- Reads: the map's AABBs for fixed furniture; `seen_boxes` for a movable table.
- Tests: `test_tiptop_presweep_fixes.py::test_a_lawn_is_the_floor_under_the_robot` (`floor_surface`, `is_floor`, and
  `floor_name` on a scope with eggs and a lawn only), `::test_a_movable_tables_slab_is_the_world_box_of_the_points_that_saw_it`,
  `::test_a_board_above_the_arms_reach_is_never_chosen_even_when_it_is_the_only_one_in_reach`.
- Replay (`T1-bridge/shelf_of_replay.out`, putting_shoes_on_rack i0 sweep4, decrypted hallstand mesh at each logged
  stance): the 6 rounds that chose 2.373 m now choose 0.973 m (the shelf; the 0.554 m bench when it is in reach).
- Unproven in simulation: a touching round onto the 0.973 m shelf from a stance the search moves to.

### Auto spec: a task without a yaml runs on its name (policy + bridge)

- `b1k/bridge/strategies.py: strategy_for` falls back to `TaskSpec(task, task.replace("_", " "), press="in_place")`;
  `bench.py: parse_args` loses the `require_strategy` refusal (no caller passed it).
- Test: `test_tiptop_strategies.py::test_a_task_without_a_description_runs_on_its_name`.
- Unproven in simulation: a yaml-less tier-1 task run end to end.

### Open after tier 1

- The bridge's `validate_motion` exemption is broader than cuTAMP's: it covers the whole Pick polyline and every
  AABB-adjacent body, cuTAMP only the terminal segment against the resting contacts. The planner already kept those
  clear, so the bridge check there was redundant; a finger through a thin wall (cuTAMP's own ceiling) now executes and
  is stopped by the ramp's lag detector instead of refused.
- `b1k/bridge/geometry.py: FLOOR_COVERINGS` has no "lawn", so the stance search's `footprint_blockers` treats the four
  garden lawn bodies of hiding_Easter_eggs as candidate obstacles; whether a base standing on one is refused depends
  on the lawn top vs the base underside. Add "lawn" there only if a run refuses a garden stance for it.
- Sweep-level evaluation of every item above waits for the simulator: the A-assist run (the 13 step-1 tasks and the
  thin-sheet probe with `--grasping-mode assisted`, judged by `read_run.py` against sweep4), planned part grasps on
  tile / eraser / paintbrush / magazine / jigsaw_puzzle rounds, and one episode each of hiding_Easter_eggs,
  putting_shoes_on_rack and putting_up_Christmas_decorations_inside for E-region.

## Tier 2 (2026-09-24): F-reader, F-revopen, F-close, E-side, E-region petcxr, W-seq

The kitchen group (DESIGN.md section 4). cuTAMP: `install/patches/cutamp-21-side-grasp-lift-first-clear-centimetre.patch`.
Suites after integration: bridge 422 passed (409 after tier 1); planner 182 passed, 5 deselected (180 after tier 1).
Integration wired two test fakes the packages could not reach: `test_tiptop_transfer_recovery.py` `TransferEpisode.pick`
takes `into=None` (7 tests), and `test_movable_surface_cover.py::test_the_lift_rises_until_the_carried_object_clears_the_rim`
mocks `ee_pose` / `tool_from_ee` for the new `_lift_height` (its expectations are unchanged: a top-down approach).
Nothing in this tier ran in the simulator; every "unproven" line below is the same fact.

### F-reader: the joint frame is the joint's own, and "closed" is directional (bridge + policy)

- `b1k/bridge/articulation.py: joint_frame(parent_pos, parent_rot, local_pos0, local_rot0_wxyz, letter)` -> (axis,
  origin) = R_parent @ (R_local0 @ letter), parent_pos + R_parent @ localPos0. `OmniGibson/omnigibson/tiptop/articulation.py:
  openable_joints` uses it with `joint.local_position_0` (times the instance scale) and `local_orientation_0` (xyzw
  converted), and adds `"closed"`: the lower limit, or the upper one where `open_state._get_relevant_joints` lists the
  joint with direction -1. `is_open(..., closed=)` and `opening_travel(..., closed=)` measure from that end;
  `bench.is_shut`, `r1pro.inside_rect`, `container_grasps`, `open_container`, `push_joint` pass it.
- Reads: the joint frames of fixed objects (map) and the asset's openable_joint_ids direction metadata. A NON-fixed
  object's joints (laptop, jar, toolbox) are oracle data read here unnamed: a `# ponytail:` note in `openable_joints`
  marks it until an OracleKnowledge hint replaces the reader for movables.
- Tests: `test_tiptop_articulation.py::test_joint_frame_turns_the_axis_letter_by_the_joints_own_rotation_and_puts_the_hinge_at_its_own_origin`,
  `::test_a_lid_resting_at_its_open_limit_reads_open_and_closes_toward_its_closed_end`.
- Replay (`T2-articulation/joint_reader_check_after.out`, offline USD, 39 revolute joints of 27 challenge assets): lead
  error 0.0 deg on 39 of 39 (skill_gap's reader: 38 of 39 over 30 deg), handle miss at 0.5 rad 0.0 cm on 39 of 39 (was
  9-89 cm), hinge offset 0.000 m. The two assets whose +travel closes (laptop_nvulcs, trash_can_ifzxzj) are what
  `"closed"` now reads. store_honey's slgzfc prismatic axes are unchanged.
- Unproven in simulation: any hinge pull with the corrected frame (only drawer pulls were ever recorded, and their
  prismatic axes did not change).

### F-revopen: doors to 90 deg, lids past balance, grips that weld under assisted (bridge)

- `r1pro.py: container_grasps`: a vertical hinge's travel is clipped to `DOOR_TRAVEL_MAX = pi/2`; a horizontal hinge
  (lid) asks for at least `LID_FRACTION = 0.85` of its range; under assisted grasping the pressed-face column is dropped
  (no second finger, no ray between the pads) and a new `edge` grip pinches the panel's top edge from above
  (`EDGE_INSET` 1.5 cm below the edge, `EDGE_THICKNESS`: the panel is assumed 2 cm thick, ponytail-marked). The stance /
  reach / approach / hold / pull / retreat of `open_container` is factored into `_drive_joint`, which `push_joint`
  shares; it solves `with_torso=False` while the other hand holds. A revolute pull that stops more than `JOINT_TOL`
  (5 % of range) short of its target continues with `push_joint` on the door's inner face.
- Reads: the robot's grasping mode and `hands()`; the container's mesh as before (fixed: the map).
- Tests: none of its own. The caps and the edge grip run only through the updated `open_container` fakes in
  `test_tiptop_presweep_fixes.py` (`test_the_idle_arm_is_tucked_...`, `test_an_opening_stance_the_destination_check_refuses_is_reported_not_raised`,
  which now inspects `_drive_joint`, where the except moved).
- Replay: none; no recorded round opens a hinge.
- Unproven in simulation: whether the edge grip welds under assisted (the finger-to-finger ray must cross a 2 cm panel
  pinched 1.5 cm below its edge; `EDGE_THICKNESS` is assumed), the 90 deg door pull, the lid at 0.85, the push
  continuation.

### F-close: a close is a push on the moving link (bridge)

- `r1pro.py: push_grasps` (a column of points on the face that TRAILS the link's motion: `handle_on` read along the
  motion reversed, the hand coming in along the motion, closed, no nudges) and `push_joint(arm, name, joint, target)` ->
  {reached, why, position, ...} via `_drive_joint(take_hold=False)`; the arm must clear the container's body
  (`solve_pull`'s body check). `bench.py: Episode.open_up(name, 0.0)` routes to it: every joint of the container that
  `is_open` from its closed end is pushed to its `"closed"` value, records get `{"close": name, ...}`, and it returns
  `is_shut(name)`.
- Reads: the moving link's mesh (fixed: the map); the joint read-back after the push (privileged, as it always was).
- Tests: `test_tiptop_articulation.py::test_a_door_open_80_deg_is_pushed_on_the_face_that_trails_its_motion_whichever_way_it_goes`,
  `test_tiptop_presweep_fixes.py::test_closing_pushes_every_open_joint_to_its_closed_end_and_a_roofed_pick_is_marked_for_the_side_grasp`
  (a lid 2 % from its upper closed end is left alone).
- Replay: none; no recorded round closes anything.
- Unproven in simulation: a live push close. `push_joint` plans against the container's body but executes with the
  whole container as allowed contact (`grasp_contacts`), not the moving link alone.

### E-side: a grasp from the side where a roof stops a top-down hand (planner + cuTAMP + bridge)

- Planner: `tiptop/tiptop_run.py: part_grasps` gains the `side` kind: 8 yaws about the vertical, approach horizontal
  at mid-height, the jaw across the width seen from that yaw, tips half the depth in (at most 4 cm), tool columns
  [up, cross(n, up), n]; each candidate and its 180 deg twin pass `weld_ok`, confidence PART_GRASP_CONFIDENCE. Two gates
  the design did not name: a yaw whose seen width is >= jaw - 1 cm is skipped (the far face is behind the tips on a
  partial cloud, so `weld_ok` cannot refuse it), and only objects at least 2 * FINGER_HALF_WIDTH (2 cm) tall get the
  kind (the finger's width is centred on the tips). `process_scene_geometry(side_grasp=set)`: those labels' side grasps
  (the one kind with a horizontal approach) get confidence 1.0; `run_perception` passes the request's `side_grasp`;
  `perception_wrapper.extract_gt_detections` reads the key.
- cuTAMP (cutamp-21): `motion_solver._lift_height` reads the approach axis from the kinematic state it already
  computes (`ee_pose` @ inverse(`world.tool_from_ee`), column z); |z . up| < 0.7 (within ~45 deg of horizontal; the
  45 deg edge kind at 0.7071 keeps LIFT_HEIGHT) steps 1 cm up to 5 cm, else the old 5 cm steps to 30 cm. The Place is
  untouched (approach and retreat along the gripper axis).
- Bridge: `r1pro.py: HAND_STACK = 0.21`, `roof_over(mesh, centre, half, z_lo, z_hi)` (3 x 3 rays up over the
  rectangle's inner half) and `R1ProSim.side_entry(item, container)`: the container must be `fixed_base` (a movable's
  mesh is never read), the rectangle is `inside_rect`'s, the band is [floor + `item_height`, + HAND_STACK + HEADROOM].
  `bench.py: Episode.pick(bddl, into=None)` sets `sim.side_grasp = {bddl}` around each pick round when
  `side_entry(bddl, into)` or `side_entry(bddl, support_of(bddl))` (floors skipped), cleared after;
  `knowledge.py: OracleKnowledge.describe` writes `request["side_grasp"]` = the sorted labels; `protocol.py:
  KNOWLEDGE_JSON_KEYS` and `request_from_observation`'s copy list carry it. Stance: `bench.py` records
  `opened_at[name]` = the (x, y, yaw) `open_container` stood at; `stand_for(name)` teleports back there first (when the
  pose is not in the names' avoid list) and searches only otherwise or when `place_robot` refuses it.
- Reads: planner, the label's depth cloud and hull, cuRobo kinematics and obstacle costs; bridge, a fixed container's
  collision mesh (map), the item's height from the captures' own box (points). The privileged decision (a roof within
  the hand stack) never enters the planner: it arrives as the request key `side_grasp`, filled only through
  `OracleKnowledge.describe` from `Episode.pick`'s per-round mark.
- Tests: `tiptop/tests/test_support_surface_fallback.py::test_a_standing_book_gets_side_grasps_and_side_grasp_makes_them_certain`
  (a 22 x 3 x 30 cm book on edge: 4 side grasps from its two ends, then 1.0 with the label marked, M2T2's 0.6 and an
  unmarked book's 0.2 untouched), `tiptop/tests/test_lift_off_support.py::test_a_side_grasped_lift_stops_at_the_first_clear_centimetre`
  (5 mm in the desk: LIFT_HEIGHT top-down, 0.01 sideways; 3.5 cm in: 0.04; 15 cm in: the 0.05 cap);
  `test_tiptop_presweep_fixes.py::test_side_entry_is_a_roof_within_the_hand_stack_over_the_item_and_an_open_top_or_a_movable_has_none`,
  `::test_the_stance_a_container_was_opened_from_is_stood_at_again_before_any_search`, the side-mark half of
  `::test_closing_pushes_every_open_joint_...`; `test_tiptop_knowledge.py::test_a_pick_round_marked_for_a_side_entry_names_the_label_the_planner_takes_from_the_side`;
  `test_tiptop_protocol.py::test_the_side_grasp_labels_survive_the_h5_so_a_side_entry_round_replays_exactly`.
- Replay, planner (`T2-planner/side_grasps_eval_{before,after}.out`, CPU, every book of the 9 recorded pick rounds of
  sweep4 re_shelving_library_books i0, the pool = recorded grasps.pt + part grasps through ParticleInitializer's
  top-N-of-2N rule): before, 0 side grasps anywhere. After: the pick targets lie FLAT (15-26 cm each way, 2.6-4 cm
  tall) in 9 of 11 goal rows and get 0 side grasps, since every yaw's width is over the jaw: the jaw is the ceiling,
  not the code, and the S13 bookcase failure (hand straight down on a flat book) needs something else. Small or partial
  clouds get them: r12 book_2 (7.4 x 3.8 x 2.4 cm, no M2T2) 16 side grasps -> 1024/1024 particles; r26 book_3 12 ->
  1024/1024 (0 part grasps before). The only rows with both M2T2 and side grasps (r01 book_2, 180 / 336 M2T2, 2 side):
  at 0.2 they keep 0 of 1024 slots, at 1.0 they keep 21 and 15 (2 distinct poses), i.e. 1.0 makes every draw of them
  survive, about 2N x side/pool particles, not "the kept half" as DESIGN.md put it.
- Replay, bridge (`T2-articulation/side_entry_check.out`, decrypted USDs, joint at 80 % of range, a 10 cm item): fridge
  petcxr True (shelf underside 21.1 cm above the item), bookcase otwukr True, bottom_cabinet slgzfc at store_honey's
  scale (the pulled-out drawer) False, microwave hjjxmi (17.8 cm cavity) True. Before HEADROOM the right column's 4 mm
  surplus over the bare stack read False: petcxr's right bays are 31.3 cm tall and leave a 10 cm item 21.3 cm, which no
  planner margin lets through. DESIGN.md correction: the band is HAND_STACK + HEADROOM, not HAND_STACK.
- Unproven in simulation: whether a side grasp welds under assisted, whether the palm clears the shelf at mid-height,
  the 1 cm lift under a shelf, whether ~20 particles on 2 poses are enough beside M2T2's, the stance reuse, a
  side-marked pick round end to end. M2T2's own horizontal grasps are not raised to 1.0 for the marked labels (one line
  in `process_scene_geometry` if a live run wants them).

### E-region petcxr: the bay behind the opened door, on its own floor (bridge)

- `r1pro.py: inside_rect` (centre, half, floor, ceiling) is split out of `inside_region` (the box);
  `R1ProSim.bay(link, lo, hi, near=None, n=16, levels=8)` -> (centre, half, floor): an n x n x levels grid over the
  fillable AABB, the accepted point nearest the opened link's AABB centre (the lowest of equals), the run of accepted
  points about it along x and y, and the bay's OWN floor probed 1 cm at a time straight down from it. The floor was not
  in the contract: petcxr's two columns start 26 cm apart (left from z = -0.243, right-door column from z = 0.019), so
  the volume's bottom at the item's rest height is the OTHER column, behind the shut door; the single-height bay landed
  there. `"opened"` is read on the directional joint (F-reader).
- Reads: the fillable meta-link volumes and the moving link's AABB of fixed objects (map). `inside_rect`'s pre-existing
  read of the item's AABB is unchanged.
- Test: `test_tiptop_presweep_fixes.py::test_a_two_column_fillable_gives_a_rectangle_in_the_bay_nearest_the_opened_door`
  (columns with floors 26 cm apart: the right column's rectangle and its own floor).
- Replay (`side_entry_check.out`): before the bay-floor fix the region sat in the left column at z = -0.243 (behind the
  shut door); after, the right column at z = 0.363. DESIGN.md correction: `bay()` returns the floor too.
- Unproven in simulation: a place into petcxr.

### W-seq: open the source, close what the goal wants shut, after the transfers (policy)

- `b1k/bridge/strategies.py: Runner.transfer`: after the target open (now added to `self.opened`), and only with the
  hand empty, `support` (the `ep.support_of(item)` `transfer_one` already passes) is opened with OPEN_FRACTION_REACH when
  it is not a floor, `ep.is_shut(support)` and `Runner.holds(ep, "inside", item, support)`; a failed open returns False
  like the target case; then `ep.pick(item, into=container)`. `Runner.run`: the opens loop opens only what the goal wants
  open, before the transfers; a loop after the transfers (before the presses, so a microwave is shut before its press)
  shuts every wanted-shut name and every `self.opened` container the goal does not want open, when `not ep.is_shut(name)`,
  via `open_up(name, 0.0)`. `self.opened` is cleared in `run` (the Runner is reused across instances).
- Reads: `ep.support_of` (localized boxes), `ep.is_shut` (joints, the existing privileged read), `ep.goal_already_holds`
  (the existing oracle verdict), `ep.is_floor`. Nothing new on the wire.
- Tests (`test_tiptop_strategies.py`, on a `Kitchen(FakeEpisode)` that answers from (predicate, item, container) facts
  the way a :init states them, since boxes cannot tell an item inside a cabinet from one on its top):
  `::test_a_shut_source_container_holding_the_item_is_opened_before_the_pick_and_closed_at_the_end` (freeze_fruit),
  `::test_an_item_on_a_shut_articulated_counter_opens_nothing` (setup_a_bar),
  `::test_a_container_the_goal_wants_shut_is_shut_after_the_transfers` (store_produce; fails on the committed runner,
  which closed first and left it open), `::test_a_container_the_goal_wants_open_is_not_shut_at_the_end`.
- Replay (`T2-policy/dryrun.py` -> `dryrun.txt`, every task's problem0.bddl grounded offline, `Runner.run` on a Kitchen
  from its :init; `before_after.out` against the committed runner): 0 of 100 raise; 23 tasks run no rounds (tier-3
  predicates: not-covered/stained 12, cooked 5, attached 4, real 3, contains+real 3, covered 2, on_fire, frozen,
  filled); 28 end with a close. freeze_fruit before: open(fridge) -> picks/places, the tupperware's cabinet never
  opened, nothing shut; after: open(fridge) -> open(cabinet) -> pick -> inside x6 -> close(fridge) -> close(cabinet).
  storing_food, store_produce and clearing_food_from_table_into_fridge likewise end with a close. Tiers 3 and 4 rerun
  `dryrun.py` and diff `dryrun.txt`.
- Unproven in simulation: the whole chain, and one known gap on the live path: `Episode.support_of` (bench.py) uses
  `judgement.highest_support` -> `placed_over(from_bottom=False)`, which accepts only an item whose bottom is within
  -2..+15 cm of the candidate's TOP. An item on a shelf inside a shut cabinet or fridge (freeze_fruit's tupperware in
  cabinet gjeoer) has its bottom far below the cabinet top, so `support_of` answers the floor and the source open never
  fires live; the dry run passes because the Kitchen answers from :init. Not wired at integration: a behaviour change
  with no owner in this tier (see below).

### Open after tier 2

- `Episode.support_of` for an item inside a container (above). Two candidate fixes: fall back to the highest candidate
  whose box contains the item (`placed_over(from_bottom=True)`) when no "on" support is found, which mis-answers an item
  on the floor under a table's footprint; or tier 3's `Runner.scope` letting `transfer` test `holds("inside", item, c)`
  over the scope's shut containers, which needs no geometry. Until one lands, W-seq's source open is dead live.
- `EDGE_THICKNESS` is an assumption (2 cm panel); `push_joint` allows contact with the whole container; no
  OracleKnowledge hint yet for a non-fixed object's joints (the ponytail notes name all three).
- curobo imports from the MAIN tree's editable install (`BEHAVIOR-1K/tiptop/curobo/src`) even under the worktree
  PYTHONPATH; nothing in this tier touches curobo, so the suites are unaffected, but a future curobo change in the
  worktree would not be what the tests run.
- `runs/manip_atoms_20260924/T2-articulation-partial/` is superseded by `T2-articulation/{root,tiptop}.diff`.
- Live checks when a sim slot is free: one fridge task (freeze_fruit or storing_food, petcxr / dszchb) for the open ->
  side pick -> stance reuse -> push close chain under `--grasping-mode assisted`; store_honey for the unchanged drawer
  path; a standing-book or cup pick with `side_grasp` set; one place into a bookcase bay with a side-grasped item.

## Tier 3 (2026-09-25): W-goals, W-ipress, W-dwell, E-yaw, E-stamp, E-knife + N-touch, E-heat, E-aim, E-attach

The goal sub-plans (DESIGN.md section 5). cuTAMP: `install/patches/cutamp-22-place-rotation.patch` (tier 2 took 21, so
this tier's patch is 22, not the 21 DESIGN.md names). Suites after integration: bridge 438 passed (422 after tier 2);
planner 184 passed, 5 deselected (182 after tier 2). Integration wired two things the packages asked for:
`tests/test_presweep_planner.py`'s fake world gets an `env` (the new yaw loop in `stable_placement_costs` reads
`world.env` the way `near_placement_costs` already did; the real TAMPWorld always has one), and
`tiptop/planning.py: run_planning` sets `constraint_to_tol[StablePlacement][f"{obj}_yaw"] = 0.0` for every object in
`env.place_rotation`, which is the checker's default value, so plans are unchanged and its per-plan "No tolerance found"
line is gone. Nothing in this tier ran in the simulator; no stamp, cut, heat, aim, attach or dwell round has ever
executed, and every "unproven" line below is that fact.

### E-yaw: a placement carries a rotation and a yaw band (cuTAMP + planner)

- cuTAMP: `rollout.py: place_pose(env, obj, p)` = `action_4dof_to_mat4x4(p) @ R4(R)` for `obj in env.place_rotation`
  (obj -> (R 3x3 on the device, yaw tolerance in radians or None)); `RolloutFunction`'s Place, `motion_solver.py`'s
  Place (`world_from_obj`) and `particle_initialization.py`'s Place use it. The Place sampler stands the spheres rotated
  by R on the surface (its bottom offset and radial extent, and the PlaceNear ring's `radial`, are the rotated
  shape's); the yaw is redrawn uniformly in [-tol, tol] when tol is not None (free when None); the collision filter and
  IK run on the rotated pose. `cost_function.py: stable_placement_costs` adds `f"{obj}_yaw" = relu(|yaw| - tol)` per
  StablePlacement constraint whose object has a tol, yaw = atan2 of the placed pose's rotation block times R^T (summed
  when the same object is placed twice in one skeleton). Without the key every path is the old code.
- Planner: `tiptop_run.py: create_tamp_environment` copies each `place_surfaces` box and pops `rotation` /
  `yaw_tolerance` before `Cuboid(**box)` (support-label and hull-replacement branches alike); `rotations[label] =
  (tensor_args.to_device(rotation), tol)`; every `on(x, label)` atom with x a movable sets `place_rotation[x]`;
  `env.place_rotation` is set next to `env.near_also` ({} when no region carries one). `planning.py: run_planning`:
  the `{obj}_yaw` tolerance (integration, above).
- Reads: only the two keys on the region box the bridge sends (`stamp_region`, `aim_target`, `attach_target` below);
  nothing from the simulator or a mesh on the planner side. The keys ride inside `place_surfaces`, already in
  `KNOWLEDGE_JSON_KEYS`, so a recorded request replays them.
- Tests: `tiptop/tests/test_movable_surface_cover.py::test_a_place_rotation_lays_the_object_down_on_the_surface_and_bounds_its_yaw`
  (R = 90 deg about x, tol 0.2, on a fake world: the neck level with the base, the lowest rotated sphere the sampler's
  1-10 mm plus activation above the top, every yaw within +-0.2; `bottle_yaw` = [0, 0, 0.3] for yaws 0.1, -0.2, 0.5);
  `tiptop/tests/test_inside_region.py::test_a_region_rotation_is_the_placed_objects_not_the_cuboids` (the Cuboid's
  dims/pose equal the slab's, `place_rotation["jar_1"]` a (3, 3) device tensor with tol 0.2, None without
  `yaw_tolerance`); `::test_without_the_key_nothing_changes` asserts `place_rotation == {}`.
- Replay (`T3-cutamp/eyaw_*.out`, GPU 2, a few tensors: sweep4 assembling_gift_baskets_i0_0924_003546 round 3
  inside(candle_1, wicker_basket_2), seed 2302, 256 particles, 200 Adam steps): no key reproduces the pre-edit snapshot
  and the server's live log exactly (1 sampled -> 132 satisfying, yaws spread -127..165 deg). 20 deg +-10 deg (the
  design's numbers): 0/256 sampled and 0/256 optimised, all lost to `wicker_basket_2_in_xy`: the candle's 252-sphere
  footprint spans 11.5-11.8 cm at world yaw 10-30 deg against the 13.2 x 9.9 cm compartment and fits only at 50-75,
  140-165, 230-255, 320-345 deg (`yaw_fit.out`), so the constraint refuses an orientation that does not fit, which is
  right. 60 deg +-10 deg: 7/256 sampled (7/7 within), 215/256 after 200 steps, 215/215 within [50, 70] deg (before:
  31/132 within, by chance). `T3-planner/place_rotation_eval.out`: 11/11 recorded inside rounds with a region build
  the same env apart from `place_rotation` once a rotation is added to their box; HEAD raised `TypeError` on the Cuboid.
- Unproven in simulation: any rotated placement. The bridge has to choose a band the footprint fits (an
  `attach_target` sizes its box for the rotated body; a stamp's identity rotation is the fit itself), or the round has
  no satisfying particle by design.

### E-stamp: a stamp is a placement over the densest particle cluster the tool covers (protocol + bridge + policy)

- `b1k/bridge/protocol.py`: `INTENT_PREDICATES = ("stamp", "cut", "heat", "aim")`, `KEEP_HOLD_PREDICATES = ("stamp",)`;
  `tiptop_goal` maps attached/stamp/cut/heat -> on(x, y) and aim -> on(tool, PLANNER_SUPPORT) (the `under` branch);
  `keep_holding(plan)` on a parsed plan drops the last gripper `open` and everything after it and appends the two
  trajectories before it reversed (positions flipped, velocities negated and flipped); no `open` -> the same plan,
  warned. `run.py: do_execute` applies it when a round atom's predicate is in KEEP_HOLD_PREDICATES. `bench.py:
  Episode.satisfied`: an intent predicate is satisfied by its round having run without error (open loop, like a press;
  what it changed is the evaluator's); `attached` joins `under` as evaluator-only.
- `r1pro.py: inside_regions(atoms, oracle=None)` splits the intent atoms out of `pairs` (a floor stamp target no longer
  triggers `floor_surface`) and dispatches stamp/cut/heat/aim/attached to the region functions below over hints fetched
  through `oracle` (the describing OracleKnowledge), keyed by the target's label, or PLANNER_SUPPORT for aim and for a
  floor target; with no oracle they get none. `seen_aabb(obj)` is the world AABB of `own_box` (`footprint_region` uses
  it). `stamp_region(tool, target, particles, projection=None)`: greedy densest cluster (every particle as one of the
  4 footprint corners, ponytail-marked) that fits the tool's world box as held + `STAMP_MARGIN` 2 cm (an adjacency
  remover's rule, particle_modifier.py:524-531) or the projection slab with no margin; box = the covering tool centres
  grown by the tool's half extents (cuTAMP keeps every sphere inside the surface); top = the cluster's lowest particle,
  or for a projection remover the middle of the slab's straddle band (`VACUUM_HOVER`); `rotation` identity,
  `yaw_tolerance` `STAMP_YAW_TOL` 5 deg, so the tool lands at the yaw the fit was made at.
- Policy (`b1k/bridge/strategies.py`): `strategy_for(..., scope=())`, `Runner.scope` (BDDL names at reset, from
  `bench.main`'s `sorted(sim.task_scope())`), `Runner.real` = scope plus what `ep.after_transition()` names;
  `transfer_one` skips a demand item not in `real`. `Runner.run`: opens -> `run_goals` -> transfers -> closes ->
  presses -> a final `dwell(state_atoms())`. `run_goals` groups the goal: `not covered` -> `clean()`: remover = a scope
  particleRemover that is not itself a target (gloves, caps and teddies are removers by taxonomy),
  `press_instrumental` when toggleable, one pick, `stamp()` per target (`stand_for` / `walk_to_floor` for a floor,
  `achieve([stamp(tool, t)])`, stop when the evaluator reads clean, at `MAX_STAMPS` 20, or after `attempts` consecutive
  failed rounds), `free_hand`; no remover -> the washer (category "washer", transition_rules.WasherRule): every target
  inside, one `press_instrumental` (which shuts it first). `has_ability(name, ability)` reads bddl's ObjectTaxonomy.
- Reads: `OracleKnowledge.particles(target, system=None)` (every visual system's group on the target, or one system's:
  privileged, named) and `projection_box(tool)` (the remover's projection cube hung below its meta link, in the tool
  frame, from `_projection_mesh_params`: privileged, named; None for an adjacency remover); the tool's seen box
  (points). "Still covered" is `Episode.goal_already_holds`, the existing evaluator read.
- Tests: `test_tiptop_presweep_fixes.py::test_a_stamp_box_covers_the_densest_cluster_the_tool_and_its_margin_can_take_and_no_more`
  (a brush over a 5-particle patch beats a 3-particle one: a 0.252 x 0.086 box at the patch's height, a millimetre
  more would miss a particle; the vacuum's slab hovering 11 mm up, the body set back from the slab);
  `test_tiptop_protocol.py::test_the_effect_predicates_translate_to_on_and_aim_names_the_support_plane`,
  `::test_keep_holding_drops_the_release_and_comes_back_up_the_way_it_went_down`;
  `test_tiptop_strategies.py::test_a_covered_target_is_stamped_with_the_scopes_remover_until_the_episode_reads_it_clean`,
  `::test_a_loaded_washer_is_shut_before_its_one_press`.
- Replay (`T3-bridge/stamp_counts.out`, CPU, the study's particle dumps, 300 instances per task; stamps until no
  particle is left, by `stamp_region`'s clustering at the tool's yaw 0 or 90, vs the study's greedy set cover which
  also turned the tool 45 deg): within +-1 of the study on clean_a_keyboard 299/300, scrubbing_bathroom_floor 298/300,
  vacuuming_floors 300/300, garden tools 291-298/300 (one fewer on most: the footprint is the tool's own box);
  sweeping_garage 180/300 and clean_a_patio 100/300, 1-2 stamps more than the study (300/300 when the 45 deg yaw is
  allowed too; the broom is long and the region holds one yaw). `keep_holding.out`: on the 9 executed place plans of
  sweep4 assembling_gift_baskets the cut plan ends 0 rad from the approach's first waypoint and the seams add no joint
  step beyond the plan's own.
- Unproven in simulation: a pick of a brush, broom or vacuum under assisted grasping (two fingers and the ray); a stamp
  planned as a Place landing within the margin at the cluster's height; `keep_holding` played by the executor; the
  vacuum's hover and its switch; a floor stamp through the PLANNER_SUPPORT region.

### E-knife + N-touch: a cut is the knife set down on the food (bridge + policy)

- `r1pro.py: knife_region(knife, food)`: the food's seen top, a square of the food's extent plus the knife's longest
  seen extent about the food's centre (the slicer fires on any knife-link contact while armed, slicer_active.py:72-135:
  the place is the cut). Policy `cut()`: real(x) / contains(c, x) with x a half__/sliced__/diced__ category (cooked__
  stripped) -> the scope wholes of the base category; the whole into c first for contains; `carry("cut", knife,
  whole)` then `self.real |= ep.after_transition()`; diced: cut each appeared half__<base> (the pick between cuts
  re-arms the knife); a cooked__ product then `warm()`s the container waiting on real(product). `transfer()`: the
  verdict for a non-place predicate is the round having run; `reach_into(ep, name, support)` (factored out of the
  source open) walks up to 3 supports and is applied to the target's enclosure too (a knife set down on an egg still
  in the fridge). `holds(ep, predicate, *args)` takes unary atoms and reads a TypeError as not holding.
- Reads: the knife's and food's seen boxes (points); `OracleKnowledge.appeared()` (privileged, named: task-scope
  objects that exist and were not tracked, tracked one by one under their labels; a scope entry that became None is
  dropped); `OracleKnowledge.localize` leaves out a name `scene_object` cannot resolve instead of raising.
  `Episode.after_transition()` -> `knowledge.appeared()` (`KnowledgeSource.appeared()` -> []).
- Tests: `test_tiptop_presweep_fixes.py::test_a_knife_is_set_down_on_the_foods_top_over_its_centre`;
  `test_tiptop_knowledge.py::test_the_oracle_leaves_out_what_it_cannot_resolve_and_tracks_what_a_transition_created`;
  `test_tiptop_strategies.py::test_a_diced_goal_cuts_the_whole_in_its_bowl_then_each_half_the_cut_made`,
  `::test_a_half_is_cut_before_it_is_transferred_and_one_that_never_appears_is_not_picked`.
- Replay: none for the region (no recorded round cuts). `T3-policy/dryrun.txt` (every task's problem0.bddl grounded
  offline on a Kitchen from its :init, 0 of 100 raise): the onion into the bowl first, four logs, the egg in the opened
  fridge then the halves onto the plate, five vegetables, steak and pineapple, four sprouts then the tupperware into
  the oven.
- Unproven in simulation: whether a knife placed flat on a fruit makes knife-link contact while armed; the re-arm (60
  steps without contact) between cuts; `appeared()` live after a slice. Known gap (ponytail note in `cut()`): a half
  gets a BDDL name only when the :init lists it as a future object; chop_an_onion, slicing_vegetables, cook_cabbage and
  canning_food list only the diced__ systems, so their dice stops after the slice.

### E-heat, W-ipress, W-dwell, frozen, on_fire (bridge + policy)

- `r1pro.py: heat_region(item, source, heat_link)`: the square inscribed in the heat sphere at the cooking surface's
  height, grown by the item's seen half extents; the top from 9 rays down at and around the link on a fixed source's
  map mesh (a grate, not the panel behind or the drip tray under), capped at link z + `HEAT_TOP_ABOVE` 5 cm, or a
  movable source's seen top; None with no link (an oven heats inside: the policy routes `inside` there).
  `bench.py: Episode.fixture_for(ability, near=None)`: the nearest fixed-base scene object whose category's synset has
  the taxonomy ability, one without an openable joint first, tracked under its scene name (`sim.track(name)`,
  `bddl_names[name] = name`) so `toggled_on(stove_ykretu_0)` translates; `Episode.dwell(steps)`: `sim.hold` with the
  last gripper command, capped by the steps left; `Episode.goal_already_holds(predicate, *args)` any arity.
  `knowledge.py: OracleKnowledge.describe` calls `button_hints(self.goal + atoms)`, so an instrumental press of a
  fixture the goal never names has its button described.
- Policy `warm(predicate, items)`: source = a scope heatSource, doorless first, else `ep.fixture_for`; route `inside`
  for an openable source else `heat`; carrier = the movable support of the item (not a taxonomy sceneObject) else the
  item, falling back to the item when the carrier cannot be taken; skips an item whose state holds or that already sits
  where a goal placement wants it; `press_instrumental(source)` once after the carries; inline `dwell` until the atoms
  hold. frozen: the same into the coldSource, no press, no carry for an item the goal itself places (freeze_pies), the
  dwell at the end behind the shut door. `press_instrumental(ep, obj)`: skipped when `ep.switched_on(obj)`; an openable
  that is not shut is closed first (a microwave or washer refuses ON while open); `run_press(style="in_place")`.
  `dwell(ep, atoms)`: `ep.dwell(30)` while an atom (or its negation) does not hold, bounded by `MAX_DWELL_STEPS` 600
  and `spend_what_is_left`'s 0.6 budget share; stops when the episode returns fewer steps than asked.
- Reads: `OracleKnowledge.heat_link(source)` (HeatSourceOrSink.link world xyz and distance_threshold; None when
  requires_inside: privileged, named); a fixed source's collision mesh (map); the item's seen box; `ep.switched_on`
  (the existing privileged read); bddl's taxonomy (task class).
- Tests: `test_tiptop_presweep_fixes.py::test_a_heat_box_keeps_the_item_inside_the_heat_links_sphere_on_the_cooking_surface`
  (the grate, not the tray under it or the panel behind; every corner within the 0.2 m disk, most of it used),
  `::test_fixture_for_tracks_the_nearest_doorless_heat_source_so_a_goal_atom_can_name_it` (a stove 3 m off beats a
  microwave 1 m off; a movable table is never a fixture; the pressed atom translates);
  `test_tiptop_knowledge.py::test_the_oracle_describes_the_button_of_a_round_atom_the_goal_never_names`;
  `test_tiptop_strategies.py::test_cooked_carries_the_tray_to_the_stove_presses_it_once_and_waits`,
  `::test_frozen_puts_the_item_in_the_cold_source_shuts_it_and_waits`,
  `::test_on_fire_lights_the_lighter_and_never_carries_off_an_item_already_where_the_goal_wants_it`.
- Replay (`T3-bridge/heat_disk.out`, CPU, the study's burner link positions with a cooking surface 5 mm under the
  link): cook_bacon, cook_broccolini, cook_cabbage: every pan centre the region allows is within 0.200 m of the link
  and the square uses 64 % of the disk; 89/300 cook_bacon instances start with the pan already in the disk (45/300 in
  the region), 0/300 for the other two. `T3-policy/dryrun.txt`: bacon's tray to the stove, knob press, dwell; brisket
  and pie into the oven, close, press, dwell, then out to the board or tray; hot dogs and popcorn into the microwave;
  broccolini's plate to the stove; cabbage and chili into the pan, cut, pan to the stove; freeze_pies transfers only,
  close, dwell; setting_the_fire: three items to the lighter, press on, dwell, transfers, press off.
- Unproven in simulation: all of it: the heat placement staying in the sphere after the release, the in_place knob
  press on a stove, `fixture_for` live, the dwell counts (hot dogs 298-322 steps in the study). Known: cook_hot_dogs and
  make_microwave_popcorn take the in-scope microwave, not the study's burner route (the burner route needs
  `fixture_for`, which fires only when the scope has no heatSource); setting_the_fire lights the newspaper at the
  lighter and never moves it into the fireplace.

### E-aim and E-attach: the tool turned at its target, the child at the parent's frame (bridge + policy)

- `r1pro.py: aim_target(tool, target, nozzle)`: a box on the robot's side of the target within the nozzle's reach, at
  the target's bottom height under PLANNER_SUPPORT, `rotation` = the yaw that turns the nozzle's -z direction now onto
  the target, `yaw_tolerance` `AIM_YAW_TOL` 15 deg. `attach_target(child, parent, frames)`: R = R_F R_M^T; the child's
  body centre with the male frame on the female one; half extents of the rotated body plus `ATTACH_TOL` 5 cm / sqrt 2;
  top = the aligned bottom + `ATTACH_LIFT` 1 cm (at or above, never below: the snap allows 5 cm); `rotation` in the base
  frame (base^T R base), `yaw_tolerance` `ATTACH_YAW_TOL` 10 deg (the snap allows 15). Policy `coat()`: the scope's
  particleApplier, `press_instrumental` at rest, `carry("aim", tool, t)` per still-uncovered target; `attached` is a
  transfer (`PLACE_PREDICATES += "attached"`), judged by the evaluator alone.
- Reads: `OracleKnowledge.nozzle(tool)` (the ParticleApplier meta link's frame and the projection extent along -z) and
  `attach_frames(child, parent)` (the first free (male, female) pair from AttachedTo's own candidates): privileged,
  named; seen boxes otherwise; the robot's base pose.
- Tests: `test_tiptop_strategies.py::test_an_attached_goal_is_carried_like_a_placement`; the protocol test above (aim
  -> on(tool, table), attached -> on). `aim_target` and `attach_target` themselves have no unit test.
- Replay: none.
- Unproven in simulation: all of it. The frame convention (the planner's object frame is base-aligned at the capture,
  R applied about the object's origin under the particle's yaw) has been checked on paper against `place_pose` only.

### Open after tier 3

- `appeared()` cannot name a half with no future entry in the :init; four dicing tasks stop after the slice.
- `Episode.support_of` for an item inside a container (tier 2's open item) is still open: `reach_into` walks
  `support_of`, which answers the floor for an item on a shelf inside a shut cabinet, so that source open is dead live.
- `aim_target` / `attach_target`: no unit test, no replay; whether the attach rotation leaves the child's footprint
  fitting its region is by construction only.
- The `{obj}_yaw` tolerance is 0.0 (exact). If live plans lose particles at the checker to optimiser noise, a 1e-2 rad
  slack in `planning.run_planning` is the one-line change.
- The carrier rule reads fixed vs movable off the taxonomy's sceneObject ability, not `obj.fixed_base` (the policy has
  no fixed_base channel); a fixed fillable without sceneObject is tried and falls back to the item after a failed pick.
- `filled` and `not contains` have no sub-plan (canning_food); make_pizza's real(pizza) is a recipe transition (tier 4).
- Live checks when a sim slot is free, all under `--grasping-mode assisted`: clean_a_keyboard (stamp, keep_holding),
  wash_a_baseball_cap (load, push-close, in_place press), halve_an_egg (cut in the open fridge, halves appear,
  transfers), cook_bacon (tray to the heat region, knob press, dwell), freeze_pies (the dwell within the 0.6 budget
  share), one attach round (a camera on its tripod) and one aim round (the atomizer) for the rotation on the wire.
- curobo still imports from the main tree's editable install (tier 2 note); nothing in this tier touches it.

## Tier 4 (2026-09-25): E-level, E-stack, E-6dof, N-push, N-rotate (pour and the recipe)

cuTAMP: `install/patches/cutamp-23-level-carry-stack-centre-push-depth.patch` (E-level, E-stack, N-push depth; every
change reads an env attribute with a getattr default, so it is inert until the planner sets it). Suites after
integration: bridge 445 passed (438 after tier 3); planner 190 passed, 5 deselected (184 after tier 3). Lint: no new
findings on the changed OmniGibson files. Integration wired nothing new: what each package asked of the others had
landed in the same step (`level` and a button's `depth` on the wire, `env.level` / `env.push_depths` /
`env.stack_surfaces` from the planner, `Episode.pour` in the bench). Nothing in this tier ran in the simulator; no
push, pour, level carry or book-on-book placement has ever executed.

### E-level: a loaded carrier is carried level (oracle + bridge + planner + cuTAMP)

- Oracle (`knowledge.py: OracleKnowledge.passengers(label)`): the tracked labels whose localized box rests on the
  carrier's (`judgement.highest_support`: bottom within -2..+15 cm of its top, centre over it). `describe` writes
  `request["level"]` for every in-hand carrier and the `holding(x)` target that has passengers, sets `sim.level`, and
  once the carrier is in the hand merges each passenger's mask rows into the carrier's in every view (the pizza's
  pixels become the plate's: what the hand carries, not a body standing at the hand for the planner to avoid). The
  workspace top is raised to 0.3 m over the highest region top when a region lies above it (the 1.71 m nail of
  installing_smoke_detectors was cropped out of every view).
- Wire (`protocol.py`): `level` in `KNOWLEDGE_JSON_KEYS` and `request_from_observation`'s copy list.
  Planner (`perception_wrapper.extract_gt_detections`): `level` is a trigger key, returned as `gt["level"]`;
  `tiptop_run.create_tamp_environment(level=)` sets `env.level` (set of labels).
- cuTAMP (`motion_solver.py: _level_query(plan_config, tensor_args)`): the plan config cloned with
  `PoseCostMetric(hold_partial_pose=True, hold_vec_weight=[1,1,0,0,0,0], project_to_goal_frame=False)`; `solve_curobo`
  tracks `held` (initial_holding, set at a Pick's attach, cleared at a Place's detach) and `plan_pose` swaps in the
  level query for every leg while `held in env.level`. Deliberately NOT projected to the goal frame: unprojected,
  cuRobo's rotation error is in the world frame (pose_distance_kernel.cu:196-221), so [1,1,0] pins world roll and
  pitch and frees yaw; projected, `MotionGen.update_pose_cost_metric` refuses any goal whose orientation differs
  from the start's by more than 0.05 rad, which every yawing approach does.
- Bridge (`r1pro.py`): `LEVEL_TILT` 20 deg (rigid passengers slide at ~27 deg, inferred); `present_held` solves with
  `tolerance_rad` 20 deg instead of 1.2 when the arm holds a level label; `return_to_ready` refuses (returns False,
  the hand stays) when the ready posture would tip the load past 20 deg (`held_tilt(quat_from, quat_to)`: the angle
  the load's up axis leaves the vertical by; a turn about the vertical tips nothing).
- Reads: localization boxes (the existing oracle read) for the passengers; nothing from a mesh.
- Tests: `test_tiptop_knowledge.py::test_a_carrier_with_passengers_is_sent_level_and_its_passengers_ride_under_its_label_once_in_hand`,
  `::test_a_region_above_the_planners_box_raises_the_box_over_it`;
  `test_tiptop_presweep_fixes.py::test_return_to_ready_ramps_then_asks_the_planner_for_the_ready_configuration`
  (the rolled ready posture refused, the yawed one taken); `tiptop/tests/test_inside_region.py::test_level_push_depths_and_stack_surfaces_reach_the_env`,
  `::test_the_wire_keeps_a_button_depth_the_level_labels_and_the_rooms_fixed_labels`;
  `tiptop/tests/test_lift_off_support.py::test_a_level_held_lift_keeps_the_hand_level_while_it_yaws` (a real
  r1pro_left MotionGen in a far-away placeholder world: a 10 cm lift that also yaws 60 deg succeeds under
  `_level_query` with the hand's z axis tilting < 5 deg), `::test_a_held_plate_in_level_makes_every_query_the_level_one`
  (wiring: a held plate in env.level gives the closing retreat hold_vec_weight [1,1,0,0,0,0]; unlisted, the old
  [0.1,0.1,0.1,0.1,0.1,0]).
- Replay (`T4-cutamp/level_probe.out`, GPU 2): the one real plan above, max tilt 0.18 deg under the level query
  (0.37 deg for the plain query on the same motion; 62 waypoints, 2.0 s vs 0.2 s).
- Unproven in simulation: a loaded plate carried under `level` (whether the passenger stays on through the pick's
  lift, the carry and the place; whether the 20 deg bound is right); the mask merge in a live capture; the raised
  workspace on the nail.

### E-stack: a placement on a perceived movable is judged by its centre (cuTAMP + planner)

- cuTAMP (`cost_function.py`, `particle_initialization.py`): for a surface in `env.stack_surfaces`, `get_object_obb`
  is built with shrink 0 (sampler and cost) and `stable_placement_costs` scores `{surface}_in_xy` as the distance of
  the placed object's AABB centre (`get_aabb_from_spheres`: the column the simulator's VerticalAdjacency ray runs
  down, adjacency.py:93-95, on_top.py:43-52) from the OBB shrunk by `placement_shrink_dist` and clamped at 0,
  instead of the per-sphere sum. Two deviations from the spec text, deliberate: the AABB centre, not the sphere
  centroid (the sim's ray origin); and the 1 cm shrink kept as the in_xy margin, because the live tolerance is 1e-2
  (planning.py:114), so with no margin a centre 1 cm past the perceived footprint would satisfy and on a true edge the
  book tips and the ray misses. Clamping instead of shrinking the OBB means a sliver narrower than 2 cm no longer
  raises "Shrunk OBB ... half extents <= 0" (28 such plan losses in sweep3's planner logs). The PlaceNear ring uses
  the same unshrunk OBB.
- Planner (`tiptop_run.create_tamp_environment(fixed=)`): `env.stack_surfaces` = {perceived on() targets} minus the
  `place_surfaces` labels (container regions, the support box) minus `fixed` (the room map's fixed-base task labels,
  `gt["fixed"]` from `extract_gt_detections`, read off `room[..].fixed_base`, a map read). The support plane is never
  in `surfaces`. `apply_room_object_roles` (websocket server) computes the same fixed set afterwards for
  `env.fixed_labels`; the duplicate exists because `create_tamp_environment` runs first.
- Reads: labels only; the target's hull is the perception pipeline's (depth under masks).
- Tests: `tiptop/tests/test_movable_surface_cover.py::test_a_same_size_book_stack_has_satisfying_placements` (real
  ParticleInitializer + `stable_placement_costs`: a same-size book on a book, per-sphere rule every sample > 1e-2,
  stack rule > 90 % <= 1e-2 with support <= 1e-2; a book on an 8 x 2.4 cm sliver samples without raising and > 50 %
  place; a centre 3 cm past the sliver's edge costs the overhang past the margin); `test_inside_region.py` (above:
  on(cup, book) is a stack, on(jar, cabinet) with a region box, on(vase, shelf) fixed and on(book, table) are not).
- Replay (`T4-cutamp/estack_*.out`, GPU 2, 256 particles, 200 steps, seed 2302, the six recorded book-on-book
  requests of sweep4 sorting_books_on_shelf_i1_0924_120416): `{surface}_in_xy` satisfying, sampled -> optimised,
  before -> after: comic_book_2 on comic_book_3 seen as 8 x 2.4 cm, 0 -> 0 before, 174 -> 233 after (other stance
  99 -> 203); notebook_2 on notebook_1 seen as 2.6 x 7.2 cm, 0 -> 0 before, 226 -> 256 after; notebook_1 seen as
  13 x 23 cm, 15 -> 251 before, 244 -> 256 after; the remaining two 246 -> 255 and 240 -> 255. Overall satisfying
  stays 0 on all six: the binding constraint after the change is `robot_to_world` (4-8 of 256), the placing hand
  inside `sim_bookcase_otwukr_3` at the place pose for 22-31 of the 32 best particles, live as well (planner.log
  robot_to_world 0/256 on all six rounds). E-stack removes the in_xy blocker; the compartment reach (tier 2 open
  item 3) still blocks these rounds. The control (worktree code, key unset) reproduces the before numbers exactly.
- Unproven in simulation: whether a stack placed with its centre over a sliver footprint survives the drop; a same-size
  top book may be placed with its centre anywhere over the seen footprint (no centring term), overhanging up to half
  its length.

### E-6dof: an item longer than its vessel is wide goes in on end (bridge)

- `r1pro.py: upright_rotation(item)`: the base-frame 3x3 rotation turning the longest seen-box axis vertical the
  shorter way round; `inside_region` adds `rotation=upright_rotation(item), yaw_tolerance=None` when the item's
  longest extent exceeds the vessel's narrower side (roses by flat drop 2/5, spec S21). The attach rotation
  (tier 3's `attach_target`, R = R_F R_M^T) already carries roll and pitch; the smoke detector's parent is
  wall_nail.n.01_1, so the workspace raise above is what it needed.
- Reads: the item's seen box (points); the vessel's interior from tier 1's `inside_rect`.
- Tests: `test_tiptop_presweep_fixes.py::test_a_rose_longer_than_the_vase_is_wide_is_stood_on_end_yaw_free_and_a_nail_target_hangs_the_alarm_vertical`
  (R turns the rose's long axis onto +z and is a proper rotation; a short item lies; the alarm's region rotation is
  the nail's frame, its top 5 cm below the nail plus `ATTACH_LIFT`).
- Replay: none.
- Unproven in simulation: a rose planned on end into a vase (the sampler's rotated footprint against a 12 cm mouth),
  the alarm hung vertical on the nail.

### N-push: a flat item under a shelf board is slid out to the board's edge (bridge + protocol + planner + cuTAMP)

- Bridge (`r1pro.py: push_face(item)`): from the item's `own_box` (points), only for a flat item (its thinnest
  seen-box axis vertical) lying on a board of FIXED furniture (the map's `boards()`) under another board; the face
  is the side turned away from the robot, at mid-height, so the stroke runs toward the open front; the depth leaves
  the near edge `PUSH_OVERHANG` 3 cm past the board's edge on the stroke's own line (a ray on the map mesh from just
  inside the board's top out along the stroke; the board's axis-aligned box as fallback); None when it already hangs
  over (self-limiting), when standing on edge, on an open top board or on a movable case. `button_hints` describes a
  `push` atom through it (`<item>_button`, radius `PUSH_RADIUS` 1 cm, depth). `bench.py: Episode.pick` runs
  `achieve([push(item)])` after `stand_for` whenever `push_face(item)` is not None (each attempt). `run.py:
  do_execute` blocks the grasp assist during a push as during a press.
- Wire (`protocol.py`): `tiptop_goal` maps push(x) -> pressed(<x>_button); `push` in `INTENT_PREDICATES` (satisfied
  by having run); `attach_knowledge` validates an optional `depth` > 0 and keeps it in the rebuilt button.
  Planner: `_parse_buttons` keeps `depth`; `run_perception` passes `push_depths={label: depth}` from the given
  buttons; `create_tamp_environment(push_depths=)` sets `env.push_depths`.
- cuTAMP (`motion_solver.py` Push): `travel = standoff + env.push_depths.get(button, config.push_depth)`; the stroke's
  exempt list is `[button, *hosts, *resting supports of every movable host]` via the existing `_resting_contacts`
  (the fingers at a flat book's mid-height reach its shelf board), each still checked by `_validate_contact_motion`.
- Reads: the item's points; the map (fixed fixture meshes and boxes); the robot's base pose.
- Tests: `test_tiptop_presweep_fixes.py::test_the_push_face_of_a_book_lying_under_a_shelf_board_points_into_the_compartment_with_the_travel_to_its_edge`
  (face, normal, depth against a three-board case; the other face from behind the case; None once hanging over, on
  edge, on the top board, on a movable case), `::test_a_flat_book_under_a_shelf_board_is_pushed_out_before_its_pick_round_and_a_pour_tilts_after_its_round`;
  `test_tiptop_protocol.py::test_a_push_is_a_press_on_the_items_own_face_and_a_pour_a_placement_and_their_keys_survive_the_h5`;
  `tiptop/tests/test_lift_off_support.py::test_a_push_stroke_travels_the_asked_depth_with_the_hosts_supports_exempt`
  (solve_curobo with cuRobo stubbed: the stroke is standoff + 0.05 along the hover's z; between hover and stroke
  exactly {book_button, book, board} are off, the roof is not, all back on before the retreat).
- Replay (`T4-bridge/push_faces.out`, CPU: the 11 recorded holding rounds of sweep4 sorting_books_on_shelf
  i0_0924_003716, the books' clouds from depth under the oracle masks against the room map's bookcases): every book
  on a roofed board gets a face pointing into the case, depth 0.13-0.44 m, its near edge ending 3 cm past the
  board's edge; 5 of 70 book views get no push (a box seen too thin to be flat, or not over a roofed board).
  `T4-planner/replay_press_depth.out`: the two recorded radio requests parsed before and after adding `depth`; the
  one changed key is `buttons.radio_receiver_1_button.depth`. No recorded push of a movable exists to replay.
- Unproven in simulation: all of it: whether the assisted hand at a book's mid-height clears the board above, whether
  `_validate_contact_motion` passes with the board exempt, whether the book slides (friction) rather than tips, and
  whether the pinch on the 3 cm overhang welds under assisted grasping.

### N-rotate: a pour, and the recipe that needs it (protocol + bridge + policy)

- Wire (`protocol.py`): pour(x, y) -> on(x, y); `pour` in `INTENT_PREDICATES` and `KEEP_HOLD_PREDICATES` (the plan is
  cut with `keep_holding`: the hand stops above the target, still holding). Bridge (`r1pro.py`): `inside_regions`
  gives a pour no region (the target's own hull top); `tilt_wrist(arm)` ramps `<arm>_arm_joint7` by `POUR_TILT` 90 deg
  the way its limit allows, holds `POUR_HOLD_STEPS` 60, ramps back, both through `ramp_to` (checked). `bench.py:
  Episode.pour(item, target)` = `achieve([pour(item, target)])` then `sim.tilt_wrist(sim.arm)`.
- Policy (`strategies.py`): `Runner.cut` hands a real(x) that no cut or cook makes to `Runner.recipe(ep, product)`:
  bddl's own cooking recipes (`bddl.transition_rules.load_cooking_recipes`, cached) for x's category that the scope
  can supply (make_pizza gets `simple_pizza`, not `pizza`, which needs tomato sauce and marjoram); the recipe's
  input_states name the root (the dough) and how each input goes on it: `ontop` inputs first (a half__ one cut first,
  `cut(limit=ceil(n/2))`, then n items of that category carried onto the root), then `covered` inputs (a diced__ one
  diced in a scope bowl that is not the recipe's vessel and poured; a substance poured from the container the
  episode reads as `filled` with it, skipped when the root already reads covered); the knife goes back to its support
  after every cut (left on the half it blocks the half's pick, left in the bowl it is poured onto the dough); then
  `warm(ep, "cooked", [root], wait=[real(product)])` bakes the vessel and `self.real` learns the product from
  `ep.after_transition()`. `Runner.pour(ep, x, target)`: pick x (`reach_into`), `stand_for(target)`, `ep.pour`,
  `free_hand`. The three push tasks have no runner-side push step (`Episode.pick` pushes).
- Reads: BDDL scope names, the taxonomy, bddl's heat_cook.json (task class); which container holds the cheese is the
  existing `Episode.goal_already_holds("filled", c, x)` verdict; no particle read in the policy.
- Tests: `test_tiptop_strategies.py::test_a_recipe_tops_the_dough_then_pours_over_it_and_bakes_the_sheet` (the exact
  event list on the Kitchen fake: pepperoni on the dough; mushroom cut, knife back, half on the dough; cheese poured;
  onion into the bowl, cut x3, knife back, bowl poured; sheet into the oven, press, dwell; the goal's ontop(pizza,
  sheet); both emptied containers set down; the oven shut before its press); the two bridge tests named under
  N-push (`Episode.pour`, the wire).
- Replay (`T4-policy/dryrun.txt`, CPU, every task's problem0.bddl grounded on the Kitchen fake): 0 of 100 raise;
  make_pizza went from "real(pizza): no cut or cook makes it; (no rounds)" to the 34-round plan in
  `dryrun_make_pizza.txt`. The harness synthesizes halves for a sliceable whole the :init lists no futures for, which
  adds cut(half) rounds to can_meat, chop_an_onion and cook_cabbage relative to tier 3's dry run (a harness change,
  not a policy one).
- Unproven in simulation: a keep-hold placement over the dough's hull (which the planner now judges by the E-stack
  centre rule, the dough being a perceived movable on() target), the wrist tilt spilling the tupperware's contents
  onto the dough, the diced onion staying in the bowl through its carry, `after_transition` naming the pizza.

### Open after tier 4

- `OracleKnowledge.appeared()` names only scope futures: make_pizza's :init lists only the pizza, so neither the
  mushroom halves nor the onion halves get a name live, and without a named half__mushroom the recipe's ontop(half,
  dough) is never carried and simple_pizza cannot fire (the dry run synthesizes the halves). Tier 3's open item 1,
  now load-bearing: naming a transition's products by category is the next bridge item.
- Every recorded book-on-book round stays unplannable after E-stack: the placing hand is inside the bookcase mesh at
  the place pose (robot_to_world), tier 2's open item 3 (stance / side entry into a compartment).
- `gt["fixed"]` (perception_wrapper) and `apply_room_object_roles`'s `fixed_labels` (websocket server) are the same
  room read twice; fold them if the server is ever refactored so the roles are applied before `create_tamp_environment`.
- The recipe's heat source is `Runner.source` (the scope's doorless heatSource first), not the recipe's heatsource
  synset; the dicing bowl is the first fillable, non-openable scope object that is not the recipe's vessel
  (ponytail-marked); emptied containers go to the floor; the recipe's unary input_states (cooked) are not checked.
- No sampler bias toward the middle of a stack footprint; add a centring term only if a live stack topples.
- No push of a movable, no pour and no level carry is recorded anywhere; the live checks when a sim slot is free,
  under `--grasping-mode assisted`: sorting_books_on_shelf (a push of a flat book, the board above, the overhang
  pinch), make_pizza (pour, recipe order), a loaded plate carried under `level`, and roses into a vase on end.
- curobo still imports from the main tree's editable install (tier 2 note); nothing in this tier touches it.
