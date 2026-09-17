"""Moved to ``b1k.bridge.client``: the websocket client speaks to the planning server, not to the simulator.

Kept here as a re-export; new code should import ``b1k.bridge.client`` directly.
"""

from b1k.bridge.client import *  # noqa: F401,F403
