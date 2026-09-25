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
