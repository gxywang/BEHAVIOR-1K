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
