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
