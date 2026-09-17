"""Moved to ``b1k.bridge.protocol``: the wire protocol has no simulator in it, so it lives with the policy.

Kept here as a re-export because seven unmerged branches, the bench scripts and the tiptop tests all import it
from this path. New code should import ``b1k.bridge.protocol`` directly.
"""

from b1k.bridge.protocol import *  # noqa: F401,F403
