# BEHAVIOR-1K 2026 Navigation Workstream — Codex Handoff

_Last updated: 2026-09-02_

## 1. Project Goal

This work is for the **2026 BEHAVIOR Challenge** using the team repository:

- Repository: `gxywang/BEHAVIOR-1K`
- Important team branch/tag: `blackwell_v391`
- **Do not branch from `main`** for challenge work.
- User GitHub username: `NaomiEX`

The user is responsible for the **navigation workstream**, specifically:

### Issue #3 — Create nav benchmark

GitHub issue title: `Create nav benchmark`

Original checklist:

- Use the existing 7 challenge scenes.
- Sample `(start, goal)` navigation points.
- Eventually make sampling balanced across tasks when using the dataset.
- Log evaluation results in the team Google Sheet.
- Iterate until success rate is ~100% or as close as possible.

Important teammate clarification on 2026-09-02:

> The 20k dataset is on the server.  
> For this week, just sample along the scenes.  
> Do not need to use the dataset.  
> That one is mostly used for training.

**Therefore, the current benchmark should sample navigation start/goal pairs directly from the challenge scenes. Do not block on the 20k demonstrations.**

### Issue #4 — Tune Python navigation

After the benchmark is working:

- Build on `https://github.com/gxywang/nav2py`
- First run existing `nav2py` unchanged.
- Log baseline results.
- Then tune/improve it using the benchmark.

Do **not** start by tuning navigation before the benchmark infrastructure is usable.

---

## 2. Current Working Environment

The user is currently developing on the shared lab machine **alpha**.

### Alpha machine

- Host/IP: `172.16.97.180`
- Shared account username: `shenlong`
- Machine GPU: NVIDIA GeForce RTX 4070 Ti
- GPU VRAM: ~12 GB
- NVIDIA driver: `560.35.03`
- `nvidia-smi` reports CUDA support level 12.6
- Ubuntu 22.04
- Local disk: ~916 GB total, ~53 GB free at last check
- No `/shared/perception/...` cluster filesystem is mounted on alpha.

Because alpha only has ~53 GB free, **do not copy/download the full 20k BEHAVIOR demonstration dataset onto alpha**.

### Repository checkout

Current local checkout:

```bash
~/Projects/michelle/BEHAVIOR-1K
```

Environment:

```bash
~/Projects/michelle/BEHAVIOR-1K/b1k_m
```

This is a **uv virtual environment**, not a conda environment.

Activate with:

```bash
cd ~/Projects/michelle/BEHAVIOR-1K
source b1k_m/bin/activate
```

The shell prompt may show both:

```text
(b1k_m) (base)
```

That is okay. The important check is that `which python` points inside `b1k_m/bin/python`.

### Python / major packages

The custom environment was created with Python 3.11.

Installed / working:

- Python 3.11
- PyTorch 2.7.0 + CUDA 12.6 wheel
- OmniGibson
- Isaac Sim 5.1
- JoyLo likely installed
- BEHAVIOR / OmniGibson assets sufficiently installed to load scenes and run examples

One optional eval dependency failed during setup:

- `torch-cluster` install failed because `data.pyg.org` had DNS/network resolution issues.

This is **not currently blocking navigation benchmark work**.

---

## 3. Setup Command Used

The environment was created using the repo's setup script:

```bash
bash setup_uv.sh \
  --new-env b1k_m \
  --omnigibson \
  --bddl \
  --dataset \
  --joylo \
  --eval \
  --accept-nvidia-eula \
  --accept-dataset-tos \
  --cuda-version 12.6
```

Notes:

- Cluster README uses CUDA 13.0 modules, but alpha is not the cluster and does not use that module setup.
- `--cuda-version 12.6` was chosen for alpha.
- `--primitives` was intentionally not used because it requires matching system `nvcc`.
- Setup initially hit network failures but core simulator packages now work.

---

## 4. Useful Environment Variables

For normal headless development on alpha:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMNIGIBSON_HEADLESS=1
unset OMNIGIBSON_REMOTE_STREAMING
```

The team README suggests WebRTC:

```bash
export OMNIGIBSON_REMOTE_STREAMING=webrtc
```

but streaming is currently being skipped because of network limitations. Do not make WebRTC a dependency for benchmark work.

---

## 5. WebRTC Status — Ignore for Now

OmniGibson successfully started its streaming extension and printed:

```text
Streaming server started.
Now streaming on: http://172.16.97.180:8211/?server=172.16.97.180
```

However:

- TCP 49100 is reachable from the user's Windows machine.
- Port 8211 is not actually listening on alpha.
- UDP 47998 was not observed listening.
- Native Isaac Sim WebRTC client remained blank / disconnected.
- User suspects current network configuration prevents streaming.

**Decision: stop spending time on WebRTC. Continue headless.**

---

## 6. Simulator Smoke Tests Already Completed

### Robot control example

Command used:

```bash
python -m omnigibson.examples.robots.robot_control_example --quickstart
```

Result:

- Isaac Sim starts successfully.
- GPU/Vulkan works.
- OmniGibson initializes.
- Scene imports successfully.
- Robot demo reaches:

```text
Running demo.
Press ESC to quit
```

So the simulator installation is functional.

### Navigation example

Command:

```bash
unset OMNIGIBSON_REMOTE_STREAMING
python -m omnigibson.examples.environments.navigation_env_demo
```

The demo launched but eventually logged:

```text
Failed to sample initial and target positions within requested path range
```

and shut down.

This is **not considered an environment/setup failure**.

The example config is:

```text
OmniGibson/omnigibson/configs/turtlebot_nav.yaml
```

Relevant defaults:

```yaml
scene:
  type: InteractiveTraversableScene
  scene_model: Rs_int
  trav_map_resolution: 0.1
  default_erosion_radius: 0.0
  trav_map_with_objects: true

robots:
  - type: Turtlebot

task:
  type: PointNavigationTask
  floor: 0
  initial_pos: null
  initial_quat: null
  goal_pos: null
  goal_tolerance: 0.36
  path_range: [1.0, 10.0]
  reward_type: geodesic
```

The warning happens because the `PointNavigationTask` sampler fails to find a start/goal pair whose geodesic path satisfies the configured range within the retry budget.

The repeated Gymnasium message:

```text
UserWarning: WARN: Casting input x to numpy array.
```

is only a warning printed through stderr, not the root failure.

Do not spend time fixing this generic Turtlebot demo unless it becomes useful as a reference.

---

## 7. Existing OmniGibson Navigation Utilities

`PointNavigationTask` already contains useful logic for benchmark generation.

Important code location:

```text
OmniGibson/omnigibson/tasks/point_navigation_task.py
```

The task can sample a start:

```python
_, initial_pos = env.scene.get_random_point(
    floor=self._floor,
    robot=env.robots[self._robot_idn],
)
```

It samples candidate goals:

```python
_, goal_pos = env.scene.get_random_point(
    floor=self._floor,
    reference_point=initial_pos,
    robot=env.robots[self._robot_idn],
)
```

It checks reachability / geodesic distance:

```python
_, dist = env.scene.get_shortest_path(
    self._floor,
    initial_pos[:2],
    goal_pos[:2],
    entire_path=False,
    robot=env.robots[self._robot_idn],
)
```

It accepts the candidate when:

```python
dist is not None
```

and, when a path range is specified:

```python
path_range[0] < dist < path_range[1]
```

This means **the benchmark generator should reuse these scene traversal APIs instead of inventing its own collision/grid logic initially.**

---

## 8. Seven Challenge Scenes

The 2026 challenge task metadata is locally available at:

```text
datasets/2026-challenge-task-instances/metadata/available_tasks.yaml
```

The seven scene models found there are:

```text
house_double_floor_lower
house_double_floor_upper
house_single_floor
office_cubicles_right
restaurant_diner
hotel_suite_large
Rs_int
```

Treat `house_double_floor_lower` and `house_double_floor_upper` as separate benchmark scene entries for now.

To confirm locally:

```bash
grep "scene_model:" \
  datasets/2026-challenge-task-instances/metadata/available_tasks.yaml \
  | awk '{print $2}' \
  | sort -u
```

---

## 9. Important Dataset Clarification

There are two different categories of data involved.

### A. Simulation / challenge assets

These are installed locally enough for OmniGibson to run scenes.

Examples:

- OmniGibson robot assets
- BEHAVIOR-1K assets
- 2026 challenge task instances

These are what current simulator development relies on.

### B. BEHAVIOR 20k demonstration dataset

A path previously expected on the cluster was:

```text
/shared/perception/datasets/behavior1k-20k
```

Alpha cannot see this path.

`df -hT` on alpha showed no separate shared filesystem mount.

But this dataset is **not required this week**, per teammate instruction.

Do not block benchmark development on it.

Later, when task-balanced sampling from real demonstrations is needed, likely perform that stage on the cluster/server where the 20k dataset exists.

---

## 10. Current Benchmark Plan

### Current scope

Build a **scene-sampled navigation benchmark** across the seven challenge scenes.

Do not involve:

- VLMs
- localization noise
- learned goal prediction
- demonstration extraction
- task balancing
- nav2py tuning

until the minimal benchmark pipeline works.

### First milestone

Do this for **one scene only**:

```text
load challenge scene
    ↓
load / spawn R1Pro
    ↓
sample valid navigable start pose
    ↓
sample valid navigable goal
    ↓
verify shortest path exists
    ↓
record geodesic distance
    ↓
repeat 5 times
    ↓
write benchmark file
```

Recommended first scene:

```text
house_single_floor
```

because it avoids multi-floor ambiguity.

### Suggested first benchmark size

Start with:

```text
1 scene × 5 pairs
```

Then expand to:

```text
7 scenes × 10–20 pairs
```

Only after the full pipeline works.

---

## 11. Sampling Strategy

Randomly sampling arbitrary reachable points may overrepresent easy short-range episodes. Prefer stratified distance buckets eventually.

Suggested buckets:

```text
short:   ~1–3 m
medium:  ~3–6 m
long:    ~6–10+ m
```

Do not hardcode these as permanent challenge definitions without team agreement; they are a useful development stratification.

For each sampled pair:

1. `get_random_point(...)` for the start.
2. `get_random_point(..., reference_point=start)` for the goal.
3. `get_shortest_path(...)` to verify reachability.
4. Reject if `dist is None`.
5. Reject if outside desired distance bucket.
6. Store deterministic benchmark data.

Sampling should have a clear maximum retry budget and should fail gracefully rather than silently storing a bad goal.

---

## 12. Recommended Benchmark Record Format

A simple JSON / JSONL format is sufficient initially.

Example:

```json
{
  "episode_id": "house_single_floor_000",
  "scene_model": "house_single_floor",
  "floor": 0,
  "start_position": [1.2, 3.4, 0.0],
  "start_yaw": 1.57,
  "goal_position": [5.6, 2.1, 0.0],
  "geodesic_distance": 6.3,
  "difficulty": "long"
}
```

Eventually evaluation output can add:

```json
{
  "success": true,
  "steps": 123,
  "path_length": 7.1,
  "spl": 0.88,
  "final_xy_error": 0.12,
  "collision_count": 2,
  "failure_reason": null
}
```

### Minimum benchmark fields

Keep at least:

- `episode_id`
- `scene_model`
- `floor`
- exact start pose
- exact goal position
- geodesic shortest-path distance

Do **not** generate a fresh random benchmark on every evaluation run. The benchmark should be saved once and replayed deterministically so navigation methods can be compared fairly.

---

## 13. Success Criterion

`PointNavigationTask`'s existing logic can be used as a reference.

The generic Turtlebot config uses:

```yaml
goal_tolerance: 0.36
```

but R1Pro may need a different tolerance based on its footprint and the team's desired benchmark definition.

For the first infrastructure test, a simple XY goal tolerance is fine.

Before finalizing benchmark numbers, align the exact R1Pro success tolerance with team expectations.

Metrics worth logging:

- binary success
- final XY error
- number of steps
- path length
- shortest-path / geodesic distance
- SPL
- collisions
- timeout / stuck / unreachable failure reason

---

## 14. R1Pro Is the Target Robot

The generic nav demo uses Turtlebot only as an example.

The **challenge default / target embodiment is R1Pro**.

Relevant config files visible in the repo include:

```text
OmniGibson/omnigibson/configs/r1pro_behavior.yaml
OmniGibson/omnigibson/configs/r1pro_primitives.yaml
```

A very important next step is to inspect how the challenge code configures R1Pro and instantiate the benchmark using the same robot configuration, rather than adapting the Turtlebot example blindly.

Useful commands:

```bash
grep -R "R1Pro" \
  --include="*.py" \
  --include="*.yaml" \
  --include="*.yml" \
  . | head -100
```

and:

```bash
sed -n '1,180p' \
  OmniGibson/omnigibson/configs/r1pro_behavior.yaml
```

Codex should inspect existing challenge/eval code before introducing a new R1Pro config.

---

## 15. Suggested Implementation Structure

Do not modify core OmniGibson internals just to generate a benchmark.

Prefer a standalone team-level utility, for example:

```text
scripts/navigation/generate_nav_benchmark.py
scripts/navigation/run_nav_benchmark.py
configs/navigation/nav_benchmark.json
```

Exact location should follow existing repo conventions after inspecting the repo.

### `generate_nav_benchmark.py`

Responsibilities:

- define the seven scene names
- load each desired scene
- load R1Pro
- sample reachable start/goal pairs
- enforce distance stratification
- save deterministic episodes
- seed Python / NumPy / Torch as appropriate
- log rejected / failed samples

Possible CLI:

```bash
python scripts/navigation/generate_nav_benchmark.py \
  --scene house_single_floor \
  --num-episodes 5 \
  --output nav_benchmark_test.json \
  --seed 0
```

Eventually:

```bash
python scripts/navigation/generate_nav_benchmark.py \
  --all-scenes \
  --num-episodes-per-scene 20 \
  --output nav_benchmark_v1.json \
  --seed 0
```

### `run_nav_benchmark.py`

Responsibilities:

- load saved episodes
- instantiate the specified scene and robot
- reset R1Pro to exact stored start pose
- execute a navigation policy
- stop on success / timeout
- compute and save metrics
- make failures debuggable

Keep **benchmark generation** and **navigation evaluation** separate.

---

## 16. Development Order

Recommended sequence:

```text
[done] Install local b1k_m environment
[done] Verify Isaac Sim / OmniGibson launch
[done] Verify scene + robot example runs
[done] Identify seven challenge scenes
[done] Confirm 20k dataset is not required this week

[next] Inspect R1Pro challenge configuration
    ↓
Load one challenge scene + R1Pro
    ↓
Sample one valid start/goal pair
    ↓
Sample and save 5 deterministic pairs
    ↓
Replay those exact 5 starts/goals
    ↓
Expand generator to all seven scenes
    ↓
Create baseline benchmark file
    ↓
Integrate nav2py unchanged
    ↓
Run baseline + log metrics
    ↓
Tune navigation
```

---

## 17. Things NOT to Do Yet

Avoid these until the basic benchmark works:

- Do not download the full 20k dataset to alpha.
- Do not spend more time on WebRTC.
- Do not tune `nav2py` before benchmark replay works.
- Do not build localization / SLAM.
- Do not infer start/goals from demonstrations.
- Do not modify system NVIDIA drivers on the shared alpha machine.
- Do not modify the shared old conda `omnigibson` environment.
- Do not branch from repo `main`.
- Do not write benchmark-specific hacks into core OmniGibson if standalone code can use the existing APIs.

---

## 18. Git / Branch Safety

The team explicitly uses:

```text
blackwell_v391
```

for current challenge work.

Before coding:

```bash
git status
git branch --show-current
git log -1 --oneline
```

If creating a personal branch, branch from `blackwell_v391`, e.g.:

```bash
git switch blackwell_v391
git pull
git switch -c michelle-nav-benchmark
```

Do not do this blindly if the checkout already has uncommitted changes; inspect `git status` first.

---

## 19. Known Warnings That Are Not Currently Blocking

Examples seen during simulator runs:

```text
GLFW initialization failed.
failed to open the default display.
```

Expected / acceptable in headless mode.

```text
gymnasium/spaces/box.py:
UserWarning: WARN: Casting input x to numpy array.
```

Noisy warning, not the reason the nav demo shut down.

Various deprecated Isaac Sim / PhysX APIs were logged.

There was also:

```text
Warp 1.8.2 initialized:
CUDA Toolkit 12.8, Driver 12.6
```

and a CUDA interop version compatibility warning.

Since scene loading and simulation proceed, do not treat this as the current blocker. Revisit only if a concrete simulator/runtime failure appears.

---

## 20. Current Immediate Task for Codex

The immediate task should be:

> Inspect the existing repo's R1Pro / 2026 challenge environment configuration and create the smallest possible headless script that loads `house_single_floor` with R1Pro and samples **5 valid reachable `(start, goal)` navigation episodes**, using existing OmniGibson traversability APIs. Save them deterministically to JSON. Do not modify core OmniGibson unless necessary.

Before writing code, Codex should inspect:

```text
datasets/2026-challenge-task-instances/metadata/available_tasks.yaml
OmniGibson/omnigibson/configs/r1pro_behavior.yaml
OmniGibson/omnigibson/configs/r1pro_primitives.yaml
OmniGibson/omnigibson/tasks/point_navigation_task.py
OmniGibson/omnigibson/examples/environments/navigation_env_demo.py
```

Also search the repo for:

```text
R1Pro
house_single_floor
2026-challenge-task-instances
get_random_point
get_shortest_path
```

Prefer to reuse existing challenge helpers/configs if they already load scenes and R1Pro correctly.

---

## 21. Expected First Deliverable

A successful first implementation should produce something like:

```text
Loaded scene: house_single_floor
Robot: R1Pro

Sampled episode 0:
  start = (...)
  goal = (...)
  shortest path = 2.7 m

Sampled episode 1:
  ...
...
Saved 5 episodes to:
  nav_benchmark_test.json
```

Then a replay check should verify all five stored starts/goals can be loaded again and shortest paths still exist.

Only after this should the generator scale to all seven scenes.

---

## 22. Longer-Term Benchmark Refinement

Once the scene-sampled benchmark is stable:

1. Expand sampling across all seven scenes.
2. Decide number of episodes per scene.
3. Stratify by geodesic distance.
4. Add difficult geometry / narrow passages if underrepresented.
5. Agree on R1Pro success tolerance.
6. Add collision, stuck, timeout, path-length, SPL logging.
7. Run `nav2py` unchanged for baseline.
8. Log results in the team evaluation sheet.
9. Tune navigation.
10. Later, when server/cluster access permits, replace or supplement scene-random samples with task/demo-derived starts/goals and balance across tasks.

---

## 23. Mental Model

The navigation workstream should initially isolate navigation itself.

The first benchmark assumes:

```text
known static scene
+ exact start pose
+ exact goal pose
+ known traversability / map available
----------------------------------------
test navigation planner/controller
```

Do **not** combine localization uncertainty, semantic goal finding, or task planning into the first navigation benchmark.

This makes failures attributable to navigation and gives a clean environment for tuning `nav2py`.

---

## 24. Key Context Summary

If only a few facts are remembered, remember these:

1. **Repo:** `gxywang/BEHAVIOR-1K`
2. **Base branch:** `blackwell_v391`
3. **Alpha checkout:** `~/Projects/michelle/BEHAVIOR-1K`
4. **Env:** `source b1k_m/bin/activate`
5. **Headless works. WebRTC can be ignored.**
6. **Target robot:** R1Pro, not Turtlebot.
7. **Current assignment:** create navigation benchmark first.
8. **This week:** sample directly from scenes; do not use 20k dataset.
9. **Seven scenes:**
   - `house_double_floor_lower`
   - `house_double_floor_upper`
   - `house_single_floor`
   - `office_cubicles_right`
   - `restaurant_diner`
   - `hotel_suite_large`
   - `Rs_int`
10. **Existing APIs:** `scene.get_random_point()` + `scene.get_shortest_path()`.
11. **First milestone:** `house_single_floor`, R1Pro, 5 deterministic valid start/goal pairs.
12. After benchmark works, integrate `nav2py` and record baseline before tuning.
