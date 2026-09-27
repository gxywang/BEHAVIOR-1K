"""The connector host's routing profiles (WEEK4_PLAN §3.1, §3.3): PARITY sends every EPISODE_SPECS skill to the
legacy backend with the scorer as the one GoalChecker and no shadow; NATIVE is routing.yaml's skill lines with every
intent on legacy and the shadows off unless asked; overlay() applies the ladder's ``--route SKILL[.QUAL]=BACKEND``
lines on a copy. A profile is the mapping SkillRegistry and goal_panel() read (load_routing's shape)."""

import copy

from b1k.connector.skills import Rel
from b1k.skills.registry import load_routing
from omnigibson.tiptop.host.legacy_episode import EPISODE_SPECS

RELATIONS = frozenset(r.value for r in Rel)  # a qualifier naming one: by_relation
JOINTS = frozenset({"prismatic", "revolute"})  # by_joint (b1k.connector.world.JointFrame.kind)
GOAL_LINES = ("goal_checker", "goal_checkers_shadow")


def parity() -> dict:
    return {"goal_checker": "scorer", "goal_checkers_shadow": [], **{k: {"default": "legacy"} for k in EPISODE_SPECS}}


def native(shadow: bool = False) -> dict:
    """routing.yaml's skill lines as they stand, every intent.* on legacy, the shadow checkers only with ``shadow``."""
    cfg = load_routing()
    out = {"goal_checker": cfg.get("goal_checker", "scorer"),
           "goal_checkers_shadow": list(cfg.get("goal_checkers_shadow", ())) if shadow else []}
    for k in EPISODE_SPECS:
        out[k] = copy.deepcopy(cfg[k]) if k in cfg else {"default": "legacy"}
    return out


PARITY = parity()
NATIVE = native()


def overlay(profile: dict, routes) -> dict:
    """``routes``: "place.on=tiptop" (by_relation), "close.prismatic=tiptop" (by_joint), "press=tiptop" (default),
    "intent.pour=legacy" (a dotted skill name, no qualifier). An unknown skill or qualifier is refused."""
    out = copy.deepcopy(profile)
    skills = set(EPISODE_SPECS) | {k for k in profile if k not in GOAL_LINES}
    for route in routes:
        key, sep, backend = route.partition("=")
        if not sep or not backend:
            raise ValueError(f"route {route!r}: expected SKILL[.QUAL]=BACKEND")
        if key in skills:
            skill, qual = key, ""
        else:
            skill, _, qual = key.rpartition(".")
        if skill not in skills:
            raise ValueError(f"route {route!r}: unknown skill {skill or key!r}")
        rule = out.setdefault(skill, {})
        if not qual:
            rule["default"] = backend
        elif qual in RELATIONS:
            rule.setdefault("by_relation", {})[qual] = backend
        elif qual in JOINTS:
            rule.setdefault("by_joint", {})[qual] = backend
        else:
            raise ValueError(f"route {route!r}: qualifier {qual!r} is neither a relation nor a joint kind")
    return out
