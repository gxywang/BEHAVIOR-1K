# Point Goal
## Benchmark generation
### (scene)
```
python OmniGibson/scripts/navigation/generate_nav_benchmark.py \
  --scene Rs_int\
  --num-episodes 20 \
  --output outputs/navigation/task_benchmarks_extra_clearance/nav_benchmark_test.json \
  --extra-clearance 0.2
```

### (scene, task)
```
python OmniGibson/scripts/navigation/generate_nav_benchmark.py \
  --scene hotel_suite_large \
  --task polishing_shoes \
  --num-episodes 20 \
  --output outputs/navigation/task_benchmarks_more_clearance_v3/nav_benchmark.json \
  --extra-clearance 0.2 \
  --safety-clearance 0.1 \
  --seed 0
```


# Running nav2py
###  batch
```
 python OmniGibson/scripts/navigation/batch_nav_benchmarks.py run \
  --benchmark-dir outputs/navigation/task_benchmarks_extra_clearance_hotel_suite_large/ \
  --output-dir outputs/navigation/nav2py_results_extra_clearance_house_single_floor \
  -- \
  --nav2py-root ../nav2py \
  --costmap-source og-eroded-soft \
  --soft-cost-radius 0.5 \
  --soft-cost-scaling-factor 5.0 \
  --disable-path-smoothing \
  --trace-failures \
  --desired-linear-velocity 0.3 \
  --profile-max-linear-velocity 0.75 \
  --profile-max-angular-velocity 1.75 \
  --command-max-linear-velocity 0.75 \
  --command-max-angular-velocity 1.0 \
  --success-distance 0.5 \
  --seed 0
```

### run one task (can also specify episode-ids to run specific episode)
python OmniGibson/scripts/navigation/run_nav2py_benchmark.py \
  --benchmark  outputs/navigation/task_benchmarks_more_clearance_v4/nav_benchmark_house_double_floor_upper_clean_a_trumpet.json \
  --output outputs/navigation/nav2py_res_benchmark_more_clearance_v4_house_double_floor_upper_clean_a_trumpet.json \
  --nav2py-root ../nav2py \
  --costmap-source og-eroded-soft \
  --soft-cost-radius 0.5 \
  --soft-cost-scaling-factor 5.0 \
  --disable-path-smoothing \
  --trace-failures \
  --desired-linear-velocity 0.3 \
  --profile-max-linear-velocity 0.75 \
  --profile-max-angular-velocity 1.75 \
  --command-max-linear-velocity 0.75 \
  --command-max-angular-velocity 1.0 \
  --success-distance 0.5 \
  --runtime-extra-clearance 0.2
  --seed 0


# Costmap sources

`run_nav2py_benchmark.py --costmap-source` selects the map the planner runs on.

| source | map | clearance |
| --- | --- | --- |
| `nav2py-inflated` | OmniGibson `floor_trav_*.png` | nav2py inflates for the robot |
| `og-eroded` | OmniGibson `floor_trav_*.png` | OmniGibson's robot erosion, baked in |
| `og-eroded-soft` | as `og-eroded` | plus non-lethal cost near obstacles |
| `b1k-gt` | b1k ground-truth `navigation_2d` | robot erosion applied here, baked in |
| `b1k-gt-soft` | as `b1k-gt` | plus non-lethal cost near obstacles |

The `b1k-gt*` sources read `<--b1k-map-root>/<scene_model>/navigation_2d` (default
`/scratch/gxwang2/b1k/b1k_gt_out/final_v4`) through `benchmarks.costmaps.load_navigation_2d`
in nav2py, so the resolution, world origin and axis convention all come from the artifact's
own JSON sidecar. Traversable cells are free, obstacle cells lethal, and **unknown cells take
nav2py's no-information cost -- they are never free**, so unknown space is neither planned
through nor eroded across.

The artifact is ground-truth geometry at 2 cm, not a robot-eroded map, so these sources erode
for the robot themselves and report `clearance_is_in_costmap=True` to `make_robot_profile`,
exactly as `og-eroded` does. The erosion removes the same number of *metres* OmniGibson removes
at its own map resolution, not a re-run of its `ceil(radius / resolution)` square at 2 cm --
that formula is resolution-dependent and would remove 0.34 m where OmniGibson removes 0.30 m,
rejecting episode endpoints for a reason unrelated to map content.

Same invocation as above with `--costmap-source b1k-gt` substituted:

```
python OmniGibson/scripts/navigation/run_nav2py_benchmark.py \
  --benchmark <in.json> --output <out.json> \
  --nav2py-root ../nav2py \
  --costmap-source b1k-gt \
  --b1k-map-root /scratch/gxwang2/b1k/b1k_gt_out/final_v4 \
  --soft-cost-radius 0.5 --soft-cost-scaling-factor 5.0 \
  --disable-path-smoothing --trace-failures --desired-linear-velocity 0.3 \
  --profile-max-linear-velocity 0.75 --profile-max-angular-velocity 1.75 \
  --command-max-linear-velocity 0.75 --command-max-angular-velocity 1.0 \
  --success-distance 0.5 --runtime-extra-clearance 0.2 --seed 0
```


# Object Goal

## Deriving recipes from the teleoperation demonstrations

`derive_object_nav_recipes.py` turns the BEHAVIOR-1K 20k demo corpus into object-navigation
recipes. The demonstrated `navigation: move to` skills are the source of truth: their
`object_id` values are resolved against the task's `0_0` template by
`list_object_nav_references.resolve_reference`, kept in demonstrated order, and chained. A
reference that is ambiguous (a category with several instances) or unresolved is skipped, never
guessed, and every drop is counted by reason.

Each leg is then verified for **reachability**, not merely traversability, on the ground-truth
`navigation_2d` map: the map is eroded exactly as `run_nav2py_benchmark.make_b1k_costmap` erodes
it for `--costmap-source b1k-gt` (OmniGibson's own 0.30 m robot erosion, then a 0.2 m clearance
disk), and a leg survives only if the chosen approach pose and the leg's start are in the same
connected component of that eroded map, with a geodesic distance inside the distance bounds.
The approach pose is chosen the way `generate_object_nav_benchmark.project_goal_near_object`
chooses it -- rings around the goal object, `min` by projection radius then geodesic distance --
but only over reachable candidates, so when the closest ring is cut off by inflation the leg
falls through to a farther reachable ring instead of being discarded.

`--min-distance` defaults to 2.0 m rather than the generator's 1.0 m: with a 0.5 m success
radius a 1.0 m leg is a 0.5 m drive, which is what made the point-goal suite undiscriminating.
Every demonstrated chain is tried at that floor first; a task still short of
`--recipes-per-task` is then topped up at `--fallback-min-distance` (1.0 m) from the chains the
primary floor rejected, and each recipe is labelled with the floor it was derived at. The
summary reports each floor's travel distribution separately, so short episodes are visible
rather than silent. A task is never padded with duplicates: recipes are deduplicated by their
goal sequence and each comes from a different demonstrated chain, so a task short of five
recipes reports why (usually that its demonstrations hold fewer distinct move-to chains).

`--task-scope-disambiguation` allows the one narrowing that comes from the task definition
rather than from us: when a demo reference resolves to several instances of a category and
exactly one of them is in the task's own BDDL object scope, that instance is used and the leg
records `resolution: bddl_task_scope`. Without the flag every such reference is dropped as
ambiguous; over the 100 demo tasks the flag resolved 52 of the 1297 emitted legs.

Nothing in the script imports omnigibson; it runs on a login node in about half an hour for all
100 tasks.

```
python OmniGibson/scripts/navigation/derive_object_nav_recipes.py \
  --task-scope-disambiguation \
  --output-root /scratch/gxwang2/b1k/objnav/recipes \
  --report /scratch/gxwang2/b1k/objnav/derivation_report.json
```

Each recipe carries, besides the `scene` / `task` / `navigation_chain` the generator requires,
a `derivation` block (which demo episodes back the chain, what was dropped and why), a
`reachability_check` block (map, erosion, per-leg approach pose and geodesic distance) and a
`generator_invocation` block naming the arguments the recipe was verified under.

## Generating episodes from a recipe

The `0_0` template is the only instance `find_templates` matches, so `--num-instances` must be 1.
The template copy it resolves under the default root exposes only the generic `robot` start pose,
so `--robot-pose-key robot` is required (each recipe records the key it was verified with).

```
python OmniGibson/scripts/navigation/generate_object_nav_benchmark_from_recipe.py \
  --recipe /scratch/gxwang2/b1k/objnav/recipes/house_double_floor_lower/picking_up_trash_1.json \
  --task-instances-root datasets/2026-challenge-task-instances \
  --scene house_double_floor_lower \
  --robot-pose-key robot \
  --num-instances 1 \
  --min-distance 2.0 \
  --extra-clearance 0.2 \
  --output outputs/navigation/objnav/house_double_floor_lower_picking_up_trash_1.json
```
