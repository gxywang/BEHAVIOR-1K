# DEBUG.md

Debugging notes for running OmniGibson / Isaac Sim 5.1 on the Illinois Campus
Cluster (`shenlong`, `shenlong2` partitions). Nodes are RHEL 9.8, glibc 2.34.

Quick start: `source cluster_env.sh` before launching anything.

## Verified working configuration

`python -m omnigibson.examples.environments.behavior_env_demo` (option 1,
pre-sampled scene) on `ccc0325` / RTX A6000 / driver 580.173.02, partition
`shenlong2`:

| | user's original env | after `cluster_env.sh` |
| --- | --- | --- |
| exit code | 0 | **0** |
| episodes completed | 11 | **11** |
| Kit ready | 270.6 s | **199.4 s** |
| total wall clock | 867 s | **855 s** |
| `NVML_ERROR_DRIVER_NOT_LOADED` | 6 | **0** |
| `No CUDA devices found` | 5 | **0** |
| `Unable to create PxCudaContextManager` | 5 | **0** |
| `livestream-rtc` failures | 3 | **0** |
| GPU PhysX available | no | **yes** |
| WebRTC streaming | broken | **works** |
| `GLIBC_2.35` extension failures | 15 | 15 (see issue 2) |

The headline: the demo *always* exited 0 — the failures were silent. Anything
needing GPU physics or video encode was broken without any fatal error.

---

## 1. `No CUDA devices found` / `NVML_ERROR_DRIVER_NOT_LOADED` (root cause of most failures)

**Symptoms** in the Kit log (`$OMNIGIBSON_APPDATA_PATH/local/logs/Kit/OmniGibson/3.9/kit_*.log`):

```
[Error] [carb.cudainterop.plugin] Could not initialize NVML: return code 9
        (NVML_ERROR_DRIVER_NOT_LOADED: NVIDIA driver is not loaded.)
[Error] [omni.physx.tensors.plugin] CUDA context validation failed
[Error] [omni.physx.plugin] No CUDA devices found
[Error] [omni.physx.plugin] Unable to create PxCudaContextManager!
[Error] [carb.livestream-rtc.plugin] Couldn't initialize the capture device.
```

`nvidia-smi` works fine from the shell, which makes this confusing.

**Cause.** `module load cuda/12.8` (your `~/.bashrc` does this) puts
`/sw/apps/cuda/12.8/lib64/stubs` on `LD_LIBRARY_PATH`. That directory holds
*link-time stub* libraries — they export the symbols but contain no driver.

The stub directory contains **only the unversioned names**
(`libcuda.so`, `libnvidia-ml.so`) — there is no `libcuda.so.1` or
`libnvidia-ml.so.1` in it. The loader searches `LD_LIBRARY_PATH` by
**filename**, so only code that dlopens the *unversioned* soname is affected;
anything asking for `.so.1` falls through to `/usr/lib64` and works.

Measured on `ccc0325`, one fresh process per name:

| dlopen name | stubs on path | stubs stripped |
| --- | --- | --- |
| `libnvidia-ml.so` | stub → `nvmlInit` **rc=9** | real driver → rc=0 |
| `libnvidia-ml.so.1` | real driver → rc=0 | real driver → rc=0 |
| `libcuda.so` | stub → `cuInit` **rc=34** | real driver → rc=0 |
| `libcuda.so.1` | real driver → rc=0 | real driver → rc=0 |

Isaac Sim's `carb.cudainterop` plugin evidently dlopens the unversioned names,
which is why it fails while `nvidia-smi` (linked against `.so.1`) works fine.

Caveat when testing this yourself: probe **one name per process**. If you load
`libnvidia-ml.so` first, its SONAME is `libnvidia-ml.so.1`, so a later request
for `libnvidia-ml.so.1` in the *same* process is satisfied by the already-loaded
stub and you will wrongly conclude the versioned name is shadowed too.

**Verified on `ccc0325` (RTX A6000, driver 580.173.02):**

Same demo, same node, same GPU, compared at the identical startup phase:
**6** `NVML_ERROR_DRIVER_NOT_LOADED` errors with stubs vs **0** without.

**Fix** — strip any `*/stubs` entry from `LD_LIBRARY_PATH`. Stubs are only ever
needed at link time, so removing them at runtime is safe:

```bash
export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/stubs$' | paste -sd:)"
```

**Check it in one line:**

```bash
python3 -c "import ctypes;print(ctypes.CDLL('libnvidia-ml.so.1').nvmlInit_v2())"   # must print 0
```

**What it actually breaks.** Verified end-to-end: `behavior_env_demo` still
finishes with **exit code 0** and completes all 10 reset/100-step episodes even
with the stubs on the path. So this is a *silent degradation*, not a hard
block. What you lose:

| | stubs on path | stubs stripped |
| --- | --- | --- |
| exit code | 0 | 0 |
| episodes completed | 11 | 11 |
| `NVML_ERROR_DRIVER_NOT_LOADED` | 6 | **0** |
| `No CUDA devices found` | 5 | **0** |
| `Unable to create PxCudaContextManager` | 5 | **0** |

The consequence is no GPU PhysX context and no NVENC. Anything that needs
`gm.USE_GPU_DYNAMICS = True`, GPU tensor APIs, or hardware video encode
(i.e. WebRTC streaming) fails. This demo sets `USE_GPU_DYNAMICS = False`, which
is why it survives.

Note Vulkan is *not* affected — the renderer initializes and reports the GPU
correctly even while CUDA is broken, so "the GPU is detected" is not evidence
that CUDA works.

---

## 2. `GLIBC_2.35' not found` — extensions that cannot load on RHEL 9

Isaac Sim 5.1 wheels are built for Ubuntu 22.04 (glibc 2.35). Campus cluster
nodes run RHEL 9.8 (glibc **2.34**), so these fail to load:

| Library | Extension | Impact |
| --- | --- | --- |
| `libusd_ts.so` | `omni.usd.libs` | USD tsplines; non-fatal |
| `lula.cpython-311-*.so` | `isaacsim.robot_motion.lula` | **blocks motion planning / curobo** |
| (same) | `isaacsim.robot_motion.motion_generation` | **blocks action primitives** |
| (same) | deprecated `omni.isaac.franka`, `omni.isaac.universal_robots` | unused |

The gap is narrow — the only missing versioned symbol is `hypot@GLIBC_2.35`
(`hypot` itself exists in 2.34 as `hypot@GLIBC_2.2.5`; glibc 2.35 added a new
version node for it).

`behavior_env_demo` and other basic env/scene work run fine despite these
errors. Motion-planning work needs a real fix: run inside a container.
`apptainer` and `singularity` are both available on the cluster, and the repo
ships [docker/Dockerfile](docker/Dockerfile) plus
[docker/sbatch_example.sh](docker/sbatch_example.sh).

---

## 3. Kit startup is slow — keep the shader cache on node-local disk

`gm.APPDATA_PATH` defaults to `OmniGibson/appdata` *inside the repo*
(see [macros.py:134](OmniGibson/omnigibson/macros.py#L134)), i.e. on **Lustre**
(`/projects`, currently 92% full). It grows to ~7 GB of Omniverse shader,
texture, and extension-registry cache.

Measured on `ccc0325` (RTX A6000), time to Kit "app ready", same demo:

| appdata location | cache state | Kit ready |
| --- | --- | --- |
| `OmniGibson/appdata` (Lustre) | warm | 270.6 s |
| `/tmp/og_appdata_*` (node-local) | **cold** | 274.8 s |
| `/tmp/og_appdata_*` (node-local) | **warm** | **199.4 s** |

So node-local caching is worth ~70 s (26%) — but only once the cache is warm.
The first job on a new node pays full price, which is why a cold local cache
looks no better than warm Lustre. The code's own "This will take 10-30 seconds"
message is simply wrong for this environment.

**Fix:**

```bash
export OMNIGIBSON_APPDATA_PATH=/tmp/og_appdata_$USER
```

Use a `$USER`-stable (not `$SLURM_JOB_ID`-stable) path so successive jobs
landing on the same node reuse the warm cache. Add `--nodelist=<node>` to
`srun` when you want to guarantee that.

Trade-offs: `/tmp` is node-local, so the Kit log is only readable from that
node — copy it back if you need it. Same reasoning applies to
`TORCHINDUCTOR_CACHE_DIR`; `BehaviorTask` sampling calls `torch.compile` on
`get_base_aligned_bbox`.

Some of the remaining ~200 s is the failed-extension retries from issue 2.

---

## 4. WebRTC streaming does not connect

`gm.PUBLIC_IP` is **hardcoded** to `172.22.224.37` in
[macros.py:149](OmniGibson/omnigibson/macros.py#L149) — that is
`shenlong-gpu-01`, not a campus cluster node. The livestream then advertises an
endpoint on a host that isn't running the sim.

Also note the address in `USAGE_DOCS.md` for the `shenlong` partition
(`172.29.128.5`) is a **login node**, not the compute node your job lands on.

**Fix** — advertise the actual compute node, and tunnel to it:

```bash
# inside the job. NB: plain `hostname -I` puts a useless link-local
# 169.254.x address FIRST on these nodes -- select the 172.29.x one.
export OMNIGIBSON_PUBLIC_IP="$(hostname -I | tr ' ' '\n' | grep -E '^172\.29\.' | head -1)"
export OMNIGIBSON_REMOTE_STREAMING=webrtc
```

Compute node addresses, for reference (`ccc0325`):

| Address | Network | Reachable from |
| --- | --- | --- |
| `169.254.3.1` | link-local | nothing — never use |
| `172.29.130.70` | cluster-internal | login nodes |
| `141.142.114.70` | campus public | likely firewalled on 8211/49100 |

Compute nodes are on the private `172.29.x.x` network, so from a laptop you
need an SSH tunnel through the login node (ports 8211 HTTP and 49100 WebRTC):

```bash
ssh -L 8211:<node>:8211 -L 49100:<node>:49100 <login-host>
```

**Streaming depends on NVENC, which needs a working `libcuda` — so issue 1 is
the actual blocker.** Verified: with stubs on the path the user's log showed

```
[Error] [carb.livestream-rtc.plugin] Stream Server: Net Stream Creation failed, 0x800E850A
[Error] [carb.livestream-rtc.plugin] Could not initialize streaming components
[Error] [carb.livestream-rtc.plugin] Couldn't initialize the capture device.
```

With stubs stripped, the same demo on the same node starts
`omni.kit.livestream.webrtc` cleanly with **zero** `livestream-rtc` errors and
prints its `Now streaming on: ...` banner. Fix issue 1 and streaming works.

**Gotcha — the printed URL is not the advertised endpoint.** In
[simulator.py](OmniGibson/omnigibson/simulator.py#L275-L300) the endpoint Kit
actually advertises comes from `gm.PUBLIC_IP`:

```python
app.set_setting("/app/livestream/publicEndpointAddress", gm.PUBLIC_IP)
```

but the URL printed to your terminal comes from a *separate* default-route
lookup (`socket.connect(("8.8.8.8", 80))` then `getsockname()`):

```python
print(f"Now streaming on: http://{ip}:{gm.HTTP_PORT}/?server={ip}")
```

On `ccc0325` those disagree — the banner said `141.142.114.70` (campus public)
while `OMNIGIBSON_PUBLIC_IP` was `172.29.130.70` (cluster-internal). Don't
trust the banner; set `OMNIGIBSON_PUBLIC_IP` to whatever your client can
actually reach and connect there.

---

## 5. `behavior_env_demo` appears to hang

Two separate things to know:

- The demo calls `choose_from_options`, which blocks on `input()`. Under
  `srun` without a TTY it will sit there. Pipe a choice: `echo 1 | python -m ...`
- Option **2** ("sample the BEHAVIOR activity in an online fashion") runs full
  BDDL object sampling and is very slow. Option **1** (pre-sampled cached
  scene) is the right smoke test. A previous interrupted run shows the signal
  arriving during `sample_kinematics` under online sampling.

The `FileNotFoundError: '/tmp/tmpXXXX'` + `torch._inductor.exc.InductorError`
traceback on Ctrl-C is *fallout* of shutdown racing torch-inductor's temp
directory, not the original failure. Ignore it and look further up the log.

---

## 6. Harmless warnings you can ignore

These appear in every run on this cluster and are **not** the problem:

```
[Warning] [carb.cudainterop.plugin] CUDA_VISIBLE_DEVICES environment variable is set.
[Warning] [carb.cudainterop.plugin] Note CUDA device enumeration and Omniverse device enumeration are different.
[Warning] [carb.cudainterop.plugin] Setting CUDA_VISIBLE_DEVICES can lead to undesired behavior or crashes.
```

Slurm sets `CUDA_VISIBLE_DEVICES` itself when you request GPUs, so this is
unavoidable. It is benign for a single-GPU allocation. If you request more than
one GPU and want to pin a specific one, use `OMNIGIBSON_GPU_ID` (which maps to
Isaac's `active_gpu`/`physics_gpu`) rather than fighting the variable — and
keep `CUDA_DEVICE_ORDER=PCI_BUS_ID` so the indices match `nvidia-smi`.

Also benign:

- `OmniHub: Hub failed to launch: Io("Resource temporarily unavailable")` — the
  Omniverse Hub service isn't available; nothing here needs it.
- `pxr.Semantics is deprecated`
- `Could not find category 'Replicator...' for removal` at shutdown
- `Recursive unloadAllPlugins() detected!` at shutdown

---

## Reading the real error

OmniGibson suppresses Omniverse logging until an error, so stdout is often
uninformative. The full log is at:

```
$OMNIGIBSON_APPDATA_PATH/local/logs/Kit/OmniGibson/3.9/kit_<timestamp>.log
grep -E "\[Error\]|\[Fatal\]" <that file> | sed 's/.*\[Error\] //' | sort | uniq -c | sort -rn
```

Set `OMNIGIBSON_DEBUG=1` (`gm.DEBUG`) to stop the suppression entirely.
