"""Open-loop execution of a TiPToP plan in OmniGibson, success checks, and video recording."""

import logging
import time

import numpy as np

from omnigibson.tiptop.protocol import resample_trajectory

log = logging.getLogger(__name__)


def compose_views(views: dict, column_width: int = 560, caption: str | None = None) -> np.ndarray:
    """One video frame from the simulator's views ({name: (H, W, 3) uint8}, the capture camera first): the first
    view full size on the left, the others scaled to ``column_width`` and stacked down the right, each labelled;
    ``caption`` goes in the top-left corner."""
    from PIL import Image, ImageDraw, ImageFont

    names = list(views)
    main = Image.fromarray(np.ascontiguousarray(views[names[0]][..., :3]))
    tiles, left = [], main.height
    for name in names[1:]:
        img = Image.fromarray(np.ascontiguousarray(views[name][..., :3]))
        scale = min(column_width / img.width, left / img.height)
        if scale <= 0:
            break
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
        tiles.append((name, img))
        left -= img.height
    canvas = Image.new("RGB", (main.width + (column_width if tiles else 0), main.height))
    canvas.paste(main, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    y = 0
    for name, img in tiles:
        canvas.paste(img, (main.width + (column_width - img.width) // 2, y))
        draw.text((main.width + 6, y + 4), name, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        y += img.height
    if caption:
        draw.text((8, 6), caption, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(canvas)


# A fragmented MP4 (a fragment every 2 s) keeps its index in every fragment instead of at the end of the file, so the
# video plays up to the last flushed fragment while a run is still going, and after a run that was killed or crashed
# before close(). The encoder's look-ahead still holds the last few seconds until close().
FRAGMENTED_MP4 = ["-movflags", "empty_moov+default_base_moof+frag_keyframe", "-frag_duration", "2000000"]


class VideoRecorder:
    """Writes every ``every``-th simulator step as one composed frame (``compose_views``). Register it in
    ``sim.recorders``: the simulator feeds it from ``step()``, so holds, captures and arm switches are in the video
    too, not only the executed plan."""

    def __init__(self, path, fps: int = 15, every: int = 2, column_width: int = 560):
        import imageio

        self.path, self.every, self.count, self.column_width = str(path), every, 0, column_width
        self.writer = imageio.get_writer(
            self.path, fps=fps, codec="libx264", quality=7, macro_block_size=None, output_params=FRAGMENTED_MP4
        )

    def due(self) -> bool:
        self.count += 1
        return self.count % self.every == 0

    def write(self, views: dict, caption: str | None = None) -> None:
        if views:
            self.writer.append_data(compose_views(views, self.column_width, caption))

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
        press_done=None,
    ):
        """``press_done``: no-argument callable; a ``Push(...)`` trajectory stops as soon as it returns True (the
        button flipped), so the fingers do not keep pushing against the surface for the rest of the segment."""
        self.sim = sim
        self.gripper_hold_steps = gripper_hold_steps
        self.converge_tol = converge_tol
        self.converge_max_steps = converge_max_steps
        self.press_done = press_done
        # start from the gripper's current command: a plan for an object already in the hand must keep it closed
        self.gripper = sim.last_gripper  # a hand that holds something stays closed through the plan's start
        self.n_steps = 0
        self.close_eef = None  # base-frame eef pose at the last gripper close (where a held object was taken)

    def _step(self, q_arm) -> np.ndarray:
        self.sim.step(q_arm, self.gripper)
        self.n_steps += 1
        return self.sim.q_arm()

    def converge(self, q_target, tol=None, max_steps=None, stop=None) -> float:
        tol = self.converge_tol if tol is None else tol
        max_steps = self.converge_max_steps if max_steps is None else max_steps
        err = np.inf
        for _ in range(max_steps):
            err = float(np.abs(self._step(q_target) - q_target).max())
            if err < tol or (stop is not None and stop()):
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
        if action == "close":
            self.close_eef = self.sim.eef_pose_base(self.sim.arm)

    def execute(self, plan: dict) -> dict:
        """Execute a parsed plan; returns tracking statistics."""
        t0 = time.time()
        stats = {"trajectories": [], "gripper_events": [], "start_error_rad": None}
        if plan.get("q_init") is not None:
            stats["start_error_rad"] = self.home_to(plan["q_init"])
        q_last = self.sim.q_arm()
        pressed = set()  # Push ops whose button has flipped: their remaining segments (the back-off) run unstopped
        for i, step in enumerate(plan["steps"]):
            if step["type"] == "trajectory":
                traj = resample_trajectory(step["positions"], step["dt"], self.sim.dt)
                start_gap = float(np.abs(traj[0] - self.sim.q_arm()).max())
                pressing = step["label"].startswith("Push(") and step["label"] not in pressed
                stop = self.press_done if (self.press_done is not None and pressing) else None
                errs = []
                stopped_early = False
                for q in traj:
                    errs.append(float(np.abs(self._step(q) - q).max()))
                    if stop is not None and stop():
                        stopped_early = True
                        break
                final_err = self.converge(traj[-1], stop=stop) if not stopped_early else float(errs[-1])
                if stopped_early:
                    pressed.add(step["label"])
                q_last = traj[-1]
                stats["trajectories"].append(
                    {
                        "step": i,
                        "label": step["label"],
                        "waypoints": int(len(step["positions"])),
                        "resampled": int(len(traj)),
                        "executed": int(len(errs)),
                        "start_gap_rad": start_gap,
                        "max_tracking_error_rad": float(max(errs)),
                        "final_error_rad": final_err,
                        "stopped_early": stopped_early,
                    }
                )
                log.info(
                    f"[{i}] {step['label']}: {len(step['positions'])} wp -> {len(traj)} steps"
                    f"{f' (button flipped after {len(errs)})' if stopped_early else ''}, "
                    f"max lag {max(errs):.3f} rad, final err {final_err:.4f} rad"
                )
            else:
                fingers_before = self.sim.q_fingers().tolist()
                self.set_gripper(step["action"], q_hold=q_last)
                fingers_after = self.sim.q_fingers().tolist()
                grasping = str(self.sim.robot.is_grasping())
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
