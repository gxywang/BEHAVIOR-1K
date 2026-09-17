"""Moved to ``b1k.bridge.replay``: replaying a saved round against a running planner is a CLI over the websocket
client, and deliberately runs without the simulator, so it belongs with the policy.

Kept here as a re-export, main guard included, so ``python -m omnigibson.tiptop.replay`` (what README.md and the
run notes give) keeps working; new code should use ``python -m b1k.bridge.replay``.
"""

from b1k.bridge.replay import *  # noqa: F401,F403
from b1k.bridge.replay import main

if __name__ == "__main__":
    main()
