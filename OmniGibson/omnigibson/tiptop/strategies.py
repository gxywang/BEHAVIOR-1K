"""Moved to ``b1k.bridge.strategies``: how a task is broken into rounds is the policy's own decision, worked out
from the task's BDDL goal and ``tasks/<task>.yaml``, with no simulator in it. The 38 task yamls moved with it
(``b1k/bridge/tasks/``), because ``TASKS_DIR`` is resolved from the module's own directory.

Kept here as a re-export; new code should import ``b1k.bridge.strategies`` directly.
"""

from b1k.bridge.strategies import *  # noqa: F401,F403
