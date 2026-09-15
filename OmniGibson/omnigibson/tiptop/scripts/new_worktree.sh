#!/bin/bash
# Make a git worktree that can actually RUN the pipeline.
#
#   scripts/new_worktree.sh BRANCH DIR [BASE]
#
# A fresh worktree cannot run anything until three things are true, and all three have cost whole sessions:
#
#   1. datasets/ is a git-tracked directory holding only .gitignore and README, so `ln -s .../datasets` lands
#      INSIDE it and the run dies on a missing B100_task_misc.csv. Each entry is linked individually instead.
#   2. b1k/ (the uv env) and tiptop/ (the submodule, empty in a worktree) must be linked from the main repo.
#   3. PYTHONPATH MUST point at the worktree's OmniGibson on every python invocation. `omnigibson` is an editable
#      install pinned to the MAIN tree, so without it pytest and bench import code the branch did not write. On
#      2026-09-15 three branches finished with failing tests for exactly this reason, and one shipped a call to a
#      method that does not exist without ever finding out.
#
# The script prints the PYTHONPATH line to use. Read it; do not skip it.
set -eu
BRANCH="${1:?usage: new_worktree.sh BRANCH DIR [BASE]}"
DIR="${2:?}"
BASE="${3:-dev/tiptop}"
MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$MAIN"
git worktree add -b "$BRANCH" "$DIR" "$BASE"
ln -sfn "$MAIN/b1k" "$DIR/b1k"
rm -rf "$DIR/tiptop"
ln -sfn "$MAIN/tiptop" "$DIR/tiptop"
for entry in "$MAIN"/datasets/*; do
  [ -e "$entry" ] || continue
  ln -sfn "$entry" "$DIR/datasets/$(basename "$entry")"
done
test -f "$DIR/datasets/2026-challenge-task-instances/metadata/B100_task_misc.csv" \
  || { echo "datasets link is wrong: B100_task_misc.csv not reachable" >&2; exit 1; }
cat <<MSG

worktree ready: $DIR  (branch $BRANCH off $BASE)

Run EVERYTHING from the main repo with PYTHONPATH pointing here:

  cd $MAIN
  export PYTHONPATH=$DIR/OmniGibson
  ./b1k/bin/python -m pytest OmniGibson/tests/test_tiptop_*.py -q

Confirm it took effect before trusting any result:

  PYTHONPATH=$DIR/OmniGibson ./b1k/bin/python -c "import omnigibson; print(omnigibson.__file__)"
MSG
