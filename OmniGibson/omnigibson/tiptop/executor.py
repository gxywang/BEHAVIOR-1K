"""Open-loop execution of a TiPToP plan in OmniGibson, success checks, and video recording."""

import logging
import time

import numpy as np

from omnigibson.tiptop.protocol import resample_trajectory

log = logging.getLogger(__name__)


class VideoRecorder:
    def __init__(self, path, fps: int = 15, every: int = 2):
        import imageio

        self.path, self.every, self.count = str(path), every, 0
        self.writer = imageio.get_writer(self.path, fps=fps, codec="libx264", quality=7, macro_block_size=None)

    def add(self, frame: np.ndarray | None) -> None:
        self.count += 1
        if frame is not None and self.count % self.every == 0:
            self.writer.append_data(frame)

    def close(self) -> None:
        self.writer.close()
        log.info(f"wrote video {self.path}")


class PlanExecutor:
    """Streams absolute joint targets to the sim at the env rate; gripper events hold the arm and toggle the fingers."""

    def __init__(
        self,
        sim,
        gripper_hold_steps: int = 25,
        converge_tol: float = 0.01,
        converge_max_steps: int = 90,
        video: VideoRecorder | None = None,
    ):
        self.sim = sim
        self.gripper_hold_steps = gripper_hold_steps
        self.converge_tol = converge_tol
        self.converge_max_steps = converge_max_steps
        self.video = video
        self.gripper = sim.OPEN
        self.n_steps = 0

    def _step(self, q_arm) -> np.ndarray:
        self.sim.step(q_arm, self.gripper)
        self.n_steps += 1
        if self.video is not None:
            self.video.add(self.sim.camera_rgb())
        return self.sim.q_arm()

    def converge(self, q_target, tol=None, max_steps=None) -> float:
        tol = self.converge_tol if tol is None else tol
        max_steps = self.converge_max_steps if max_steps is None else max_steps
        err = np.inf
        for _ in range(max_steps):
            err = float(np.abs(self._step(q_target) - q_target).max())
            if err < tol:
                break
        return err

    def home_to(self, q_target, tol: float = 0.02, max_steps: int = 300) -> float:
        err = self.converge(np.asarray(q_target, np.float32), tol=tol, max_steps=max_steps)
        log.info(f"homed to q_init with max joint error {err:.4f} rad")
        return err

    def set_gripper(self, action: str, q_hold=None) -> None:
        self.gripper = self.sim.CLOSE if action == "close" else self.sim.OPEN
        q_hold = self.sim.q_arm() if q_hold is None else q_hold
        for _ in range(self.gripper_hold_steps):
            self._step(q_hold)

    def execute(self, plan: dict) -> dict:
        """Execute a parsed plan; returns tracking statistics."""
        t0 = time.time()
        stats = {"trajectories": [], "gripper_events": [], "start_error_rad": None}
        if plan.get("q_init") is not None:
            stats["start_error_rad"] = self.home_to(plan["q_init"])
        q_last = self.sim.q_arm()
        for i, step in enumerate(plan["steps"]):
            if step["type"] == "trajectory":
                traj = resample_trajectory(step["positions"], step["dt"], self.sim.dt)
                start_gap = float(np.abs(traj[0] - self.sim.q_arm()).max())
                errs = []
                for q in traj:
                    errs.append(float(np.abs(self._step(q) - q).max()))
                final_err = self.converge(traj[-1])
                q_last = traj[-1]
                stats["trajectories"].append(
                    {
                        "step": i,
                        "label": step["label"],
                        "waypoints": int(len(step["positions"])),
                        "resampled": int(len(traj)),
                        "start_gap_rad": start_gap,
                        "max_tracking_error_rad": float(max(errs)),
                        "final_error_rad": final_err,
                    }
                )
                log.info(
                    f"[{i}] {step['label']}: {len(step['positions'])} wp -> {len(traj)} steps, "
                    f"max lag {max(errs):.3f} rad, final err {final_err:.4f} rad"
                )
            else:
                fingers_before = self.sim.q_fingers().tolist()
                self.set_gripper(step["action"], q_hold=q_last)
                fingers_after = self.sim.q_fingers().tolist()
                grasping = None
                try:
                    grasping = str(self.sim.robot.is_grasping())
                except Exception:
                    pass
                stats["gripper_events"].append(
                    {
                        "step": i,
                        "label": step["label"],
                        "action": step["action"],
                        "fingers_before": fingers_before,
                        "fingers_after": fingers_after,
                        "is_grasping": grasping,
                    }
                )
                log.info(
                    f"[{i}] gripper {step['action']}: fingers {fingers_before} -> {fingers_after}, is_grasping={grasping}"
                )
        self.sim.hold(15, self.gripper)
        stats["env_steps"] = self.n_steps
        stats["sim_time_s"] = self.n_steps * self.sim.dt
        stats["wall_time_s"] = time.time() - t0
        return stats


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
