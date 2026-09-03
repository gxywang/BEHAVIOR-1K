# Usage docs for lab
## Basics
- `module load cuda-toolkit/13.0`
- on server: `export OMNIGIBSON_HEADLESS=1` and `export OMNIGIBSON_REMOTE_STREAMING=webrtc`
  - if using `CUDA_VISIBLE_DEVICES=`, make sure to add this to `.bashrc` to make GPU ordering consistent with `nvidia_smi`: `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- to run `uv` install script: `bash setup_uv.sh   --new-env b1k   --omnigibson   --bddl   --dataset  --joylo  --eval --accept-nvidia-eula   --accept-dataset-tos`
- download [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
- on streaming client, connect to server `172.22.224.37` for `shenlong-gpu-01`
  - for `campus-cluster`'s `shenlong` partition: `172.29.128.5`
  - for `shenlong-gpu-02`: `172.22.224.85`

## TiPToP submodule (branch `dev/tiptop`)
- `tiptop/` is a git submodule: BEHAVIOR-1K only pins a tiptop commit (URL/branch in `.gitmodules`; `git submodule status` prefix `-` = not checked out, `+` = checkout differs from the pin)
  - `.gitmodules` points at the private repo `WenzhouDing/tiptop` (read access needed); upstream `tiptop-robot/tiptop` is only reachable through the `upstream` remote inside `tiptop/`
- fresh clone: `git clone --recurse-submodules -b dev/tiptop https://github.com/gxywang/BEHAVIOR-1K.git`
  - existing clone / empty `tiptop/`: `git submodule update --init`
  - once the submodule points at the private repo you need read access to it for these to succeed; the rest of the clone works without it
- after every `git pull`: `git submodule update --init` (`git pull` alone leaves `tiptop/` at the old commit; `git status` then shows `modified: tiptop (new commits)`)
  - consumers only, once: `git config submodule.recurse true` makes `git pull`/`git checkout` update `tiptop/` automatically (skip if you develop in `tiptop/`: every checkout or pull that moves the pin detaches your `tiptop/` branch)
- install: `pip install -e tiptop` after `git submodule update --init` (version comes from tiptop's git tags; `setup.sh` does not install it)
- develop in `tiptop/` (fresh clones and `git submodule update` leave it on a detached HEAD; commits made there are hidden by the next `git submodule update`):
  - `git -C tiptop checkout main && git -C tiptop pull --ff-only`
  - edit, `git -C tiptop commit`, `git -C tiptop push` (push tiptop FIRST)
  - `git add tiptop && git commit -m "tiptop: bump to $(git -C tiptop rev-parse --short HEAD)" && git push`
  - once per clone: `git config push.recurseSubmodules check` (refuses to push a pin nobody can fetch), `git config diff.submodule log`, `git config status.submoduleSummary true`
  - hooks: `cd tiptop && pre-commit install` (BEHAVIOR-1K hooks never run on submodule commits)
  - never `git commit -a` / `git add -A` in BEHAVIOR-1K unless you mean to move the pin; undo a stray move: unstaged: `git submodule update`; after `git add -A`: `git restore --staged tiptop && git submodule update`; after `git commit -a`: `git reset --soft HEAD~1 && git restore --staged tiptop && git submodule update`
  - on branches without `.gitmodules` (e.g. `main`) `tiptop/` shows as untracked (with `submodule.recurse true` its working tree is removed and restored on checkout); leave it alone and never `git clean -ffd` / `-ffdx` in BEHAVIOR-1K (`-fd` skips nested repos, the second `-f` deletes them)
- move the pin to latest `main` without developing: `git -C tiptop checkout main && git -C tiptop pull --ff-only && git add tiptop && git commit -m "tiptop: bump"` (`git submodule update --remote tiptop` does the same fetch but leaves a detached HEAD)
- `upload-pack: not our ref <sha>` / `Fetched in submodule path 'tiptop', but it did not contain <sha>`: the pinned commit is not on the URL your `tiptop/` uses; either the author forgot `git -C tiptop push`, or your URL is stale (next bullet)
- re-point the submodule from upstream to the private repo (done 2026-09-02 for `WenzhouDing/tiptop`; kept for reference):
  - `git submodule set-url tiptop https://github.com/<user>/tiptop.git && git -C tiptop push -u origin main --tags` (`set-url` also re-points `origin` inside `tiptop/`; `--tags` keeps the setuptools_scm versions)
  - `git add .gitmodules && git commit -m "tiptop: track private repo"`, and push this BEFORE the first pin that only exists in the private repo
  - everyone else, once that commit is on the remote: `git pull --no-recurse-submodules && git submodule sync tiptop && git submodule update --init` (a plain `git pull` aborts with `not our ref` because it tries to fetch the new pin from the old URL, and `sync` only sees the new URL after the `.gitmodules` change is merged, hence `--no-recurse-submodules` first, then `sync`, then `update`)
  - use ssh while `.gitmodules` stays https (fetch and push, survives `git submodule sync`): `git -C tiptop config url."git@github.com:<user>/".insteadOf "https://github.com/<user>/"`
  - pull upstream tiptop-robot changes: `git -C tiptop remote add upstream https://github.com/tiptop-robot/tiptop.git` (once per clone; skip if `git -C tiptop remote -v` already lists `upstream`), then `git -C tiptop fetch upstream && git -C tiptop merge upstream/main`
  - CI: `.github/workflows/tests.yml` and `profiling.yml` check out with `submodules: true`; with a private submodule those checkouts fail unless the checkout step is given a token or ssh key with access
