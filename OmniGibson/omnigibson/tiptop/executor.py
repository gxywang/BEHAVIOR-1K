"""Grading a round's goal atoms against the simulator, and the old address of the executor.

``check_success`` asks OmniGibson's ``Inside`` / ``OnTop`` / ``ToggledOn`` object states whether an atom holds and
falls back to the objects' true AABBs when a state cannot answer. That is the simulator marking its own homework:
privileged, and the one thing in this module a policy must not be able to do, so it stays here. The policy judges
a round from localization and its own readings instead (``b1k.bridge.strategies``, ``Episode.satisfied``).

Everything else moved to ``b1k.bridge.executor`` -- ``PlanExecutor``, ``leash``, ``plan_dt``, ``compose_views``,
``VideoRecorder`` and the execution constants -- and is re-exported here, so
``from omnigibson.tiptop.executor import PlanExecutor, check_success`` is unchanged.
"""

import numpy as np

from b1k.bridge.executor import *  # noqa: F401,F403


def check_success(sim, atoms: list[dict]) -> dict:
    """Evaluate goal atoms with OmniGibson object states plus a geometric fallback."""
    from omnigibson.object_states import Inside, OnTop

    results = {}
    for atom in atoms:
        pred, args = atom["predicate"], atom["args"]
        key = f"{pred}({', '.join(args)})"
        if pred in ("on", "in") and len(args) == 2:
            a, b = sim.objects.get(args[0]), sim.objects.get(args[1])
            if a is None or b is None:
                results[key] = {"success": None, "reason": "object not in scene"}
                continue
            a_lo, a_hi = [v.cpu().numpy() for v in a.aabb]
            b_lo, b_hi = [v.cpu().numpy() for v in b.aabb]
            a_c = (a_lo + a_hi) / 2
            xy_inside = bool(np.all(a_c[:2] > b_lo[:2] - 0.02) and np.all(a_c[:2] < b_hi[:2] + 0.02))
            z_ok = bool(a_lo[2] > b_lo[2] - 0.03 and a_c[2] < b_hi[2] + 0.12)
            geometric = xy_inside and z_ok
            states = {}
            for name, state in (("Inside", Inside), ("OnTop", OnTop)):
                try:
                    states[name] = bool(a.states[state].get_value(b))
                except Exception as e:
                    states[name] = f"n/a ({type(e).__name__})"
            results[key] = {
                "success": bool(geometric or any(v is True for v in states.values())),
                "geometric": geometric,
                "states": states,
                "a_center": a_c.tolist(),
                "b_aabb": [b_lo.tolist(), b_hi.tolist()],
            }
        elif pred == "toggled_on" and len(args) == 1:
            from omnigibson.object_states import ToggledOn

            a = sim.objects.get(args[0])
            if a is None or ToggledOn not in a.states:
                results[key] = {"success": None, "reason": "object not in scene or has no toggle button"}
                continue
            state = a.states[ToggledOn]
            results[key] = {
                "success": bool(state.get_value()),
                "finger_on_button_steps": int(state.robot_can_toggle_steps),
            }
        elif pred == "holding" and len(args) == 1:
            a = sim.objects.get(args[0])
            grasping = str(sim.robot.is_grasping())
            # lifted relative to where the object rested at capture time (the base may be on the floor, not the table)
            z0 = sim.capture_object_aabb_min_z.get(args[0])
            if z0 is None and a is not None:
                z0 = sim.base_pose()[0][2].item()
            lifted = bool(a is not None and a.aabb[0][2].item() > z0 + 0.05)
            results[key] = {"success": lifted, "is_grasping": grasping, "aabb_min_z_at_capture": z0}
        else:
            results[key] = {"success": None, "reason": "unsupported predicate"}
    results["all"] = bool(atoms) and all(v.get("success") for k, v in results.items() if k != "all")
    return results
