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

Commit: this section's commit in both repos. cuTAMP: `install/patches/cutamp-19-fingers-touch-the-support-near-aim.patch`.
Suites after integration: bridge `OmniGibson/tests/test_tiptop_*.py` 409 passed (401 before the tier); planner
`tiptop/tests/` 179 passed, 5 deselected (171 before).

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
  desk), else the cloud's lowest points.
- Reads: the object's perceived points and hull only; the RANSAC plane or the client's support box.
- Tests: `tiptop/tests/test_support_surface_fallback.py`: a book seen from above gets 8 corner pinches with the tips
  2 mm above its bottom and the ray inside the slab; a 10 cm cube gets 8 edge grasps and nothing top-down; a 1 x 14 cm
  pen gets 8 tip grasps centred within 1 mm, and a finger on it fails `weld_ok`.
- Replay (`T1-planner/part_grasps_eval.out`, CPU, 66 object rows over 27 sweep4 pick rounds): tiles 0 -> 4-6 passing,
  eraser 0 -> 4, paintbrush 0 -> 6, magazines 0 -> 4-8, markers 0 -> 6, glue stick 0 -> 2, board games 0 -> 6; nothing
  for sheets under ~6 mm as perceived (newspapers, jigsaw_puzzle_2), bottles and cans stay M2T2's.
- Unproven in simulation: whether cuTAMP's IK and gripper-sphere filter keep these poses, and whether they weld under
  assisted. Not covered: the 6 mm clove and thinner sheets (the ray at tips + 3.9 mm clears them).

### E-near: near() lands next to its reference (cuTAMP + planner)

- `cutamp/cost_function.py: near_placement_costs` (metres the two sphere covers' AABB gap exceeds `near_aim` =
  min(1 cm, mean(dims)/6/2), plus the metres a horizontal ray from either centre misses the other's z range;
  `near_thresholds` deleted), `cutamp/particle_initialization.py: near_ring` and the PlaceNear sampler (xy on the ring
  around the reference's AABB at the object's radial extent + aim, clipped to the surface OBB; `place_cache` keyed by
  references too), `tiptop/planning.py: run_planning` (NearPlacement tolerance 2e-3, the stock 5 cm slack is why
  placements landed 8.5-15 cm apart). Deviation from the design: the AABB gap, not the min sphere-surface gap, because
  the sim's NextTo is an AABB gap and a sphere gap rejects two rotated sandals the sim accepts.
- Reads: the cuRobo world's sphere covers of the perceived objects.
- Tests: `test_movable_surface_cover.py::test_a_near_placement_costs_the_gap_past_the_aim_and_a_missed_horizontal_ray`,
  `::test_near_samples_ring_the_reference_inside_the_surface`.
- Replay (`T1-cutamp/enear.out`, the 7 executed nextto rounds of sweep3): the old cost accepted all 7; the new cost
  rejects the 5 the simulator failed (gaps 5.8-15.2 cm) and accepts the 2 it passed. The conservative cover pads the
  planner's gap by ~1 cm, so a 1 cm aim lands ~2 cm in the sim.
- Unproven in simulation: no near round has been planned live with the new cost; `NEAR_GAP_AIM` is the one knob if
  placements land too far.

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
  marker floats 11.2 mm proud, and its stroke used to stop 6.2 mm short).
- Unproven in simulation: no press has run with the projected position.

### E-region: floor-type supports, movable tables, reachable boards (bridge)

- `b1k/bridge/protocol.py: FLOOR_CATEGORIES = ("floor", "lawn")`; `bench.py: Episode.is_floor`, `r1pro.py: task_scope,
  floor_name, scope_floor, inside_regions, floor_surface` treat a lawn as ground (hiding_Easter_eggs has no floor).
  `r1pro.py: footprint_region` gives a movable table's slab from its own seen box (`own_box`, points), None until a
  capture has seen it (putting_up_Christmas_decorations_inside). `r1pro.py: shelf_of` drops boards above
  `PLACE_HEIGHT_MAX = 1.5` m and, when no board is within `BOARD_REACH`, takes the one nearest `PLACE_HEIGHT` anyway
  (the stance search moves the robot; putting_shoes_on_rack IK-failed 6 rounds on a hallstand's 2.37 m top).
- Reads: the map's AABBs for fixed furniture; `seen_boxes` for a movable table.
- Tests: `test_tiptop_presweep_fixes.py::test_a_lawn_is_the_floor_under_the_robot`,
  `::test_a_movable_tables_slab_is_the_world_box_of_the_points_that_saw_it`,
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
- Sweep-level evaluation of every item above waits for the simulator.
