"""Drive the base to a stance with nav2py: the simulator side of the navigation seam (``R1ProSim.move_base``).

The challenge's robot has a velocity-controlled holonomic base (eval/r1pro.yaml: +-0.75 m/s, +-1.0 rad/s), so a
stance can only ever be driven to. nav2py (``third-party/nav2py``, NumPy only, never imports omnigibson) plans
and tracks; this module reads the robot's pose and velocity out of the simulator, hands nav2py a state estimate
each env step and writes its command into the base action. Everything else about ARRIVING -- the travel fold,
the unfold, the contact probe -- stays in ``place_robot``.

The costmap is the scene's ``navigation_2d`` artifact (3dmap; ``--nav-map-root/<scene>/navigation_2d``) built the
way nav2py's point-goal benchmark that tuned it did (``b1k-gt-soft``: the 0.30 m robot erosion OmniGibson applies at
its own grid, 0.2 m of extra clearance, then a soft cost out to 0.5 m). That map is coarser than the stance search:
``best_base_pose`` tests the base's own rectangle against object boxes and accepts a stance 10 cm from a table,
which the eroded map calls lethal. A stance the map refuses is driven to the nearest free cell within
``SNAP_RADIUS_M`` and the last few centimetres are closed by ``creep``, a measured-pose P controller with no
collision check (the stance search already vetted the footprint at the goal). The creep is undone before the next
drive when it left the robot in a cell the planner would refuse to start from.

nav2py is found through PYTHONPATH (its repo root supplies both ``nav2py`` and ``benchmarks``); it is not installed
in the shared simulator environment on purpose.
"""

import logging
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

log = logging.getLogger("omnigibson.tiptop")

ROBOT_EROSION_M = 0.30  # m OmniGibson's own traversability map removes for the R1Pro (nav2py benchmarks/costmaps.py)
EXTRA_CLEARANCE_M = 0.20  # the point-goal benchmark's --runtime-extra-clearance
SOFT_RADIUS_M, SOFT_FACTOR = 0.5, 5.0  # --soft-cost-radius / --soft-cost-scaling-factor of the b1k-gt-soft source
MAX_LINEAR, MAX_ANGULAR = 0.75, 1.0  # m/s, rad/s: the challenge controller's output limits (eval/r1pro.yaml)
PROFILE_MAX_ANGULAR = 1.75  # what the benchmark let the profile ask for before the command cap
DESIRED_LINEAR = 0.3  # m/s cruising speed the benchmark settled on
PLANNING_TIME_S = 10.0  # nav2py's A* budget; its 1 s default reports a 7 m goal on this 2 cm map as unreachable
SNAP_RADIUS_M = 0.45  # a stance the map calls lethal is driven to the nearest free cell this close, then crept to
CREEP_SPEED, CREEP_YAW_SPEED = 0.15, 0.5  # m/s, rad/s of the last, unplanned centimetres
CREEP_TOL, CREEP_YAW_TOL = 0.02, 0.05  # m, rad: when the creep is done
STOP_STEPS = 10  # env steps of zero command after a drive, so the base is still when it is read back
STEP_BUDGET_MIN, STEP_BUDGET_PER_M = 600, 400  # env steps a drive may take: this plus this per metre of crow-flight
TERMINAL = frozenset({"succeeded", "failed", "blocked", "canceled"})


class BaseDriver:
    """nav2py behind ``R1ProSim.move_base``: one costmap and profile per scene, one navigator per drive."""

    def __init__(self, sim, map_root):
        self.sim = sim
        self.map_root = Path(map_root)
        self._parts = None
        self._creep = np.zeros(2)  # world xy the last creep moved off the map's free space; undone before the next drive

    # ---------------------------------------------------------------- nav2py parts
    def _build(self) -> None:
        from benchmarks.costmaps import make_soft_costmap, planning_costmap_from_navigation_2d
        from benchmarks.nav2py_driver import make_command_limits, make_navigation_config, make_robot_profile

        scene = self.sim.env.scene.scene_model
        map_dir = self.map_root / scene / "navigation_2d"
        hard = planning_costmap_from_navigation_2d(map_dir, ROBOT_EROSION_M, EXTRA_CLEARANCE_M)
        costmap = make_soft_costmap(hard, SOFT_RADIUS_M, SOFT_FACTOR)
        profile = make_robot_profile(
            "r1pro",
            0.5,
            clearance_is_in_costmap=True,
            max_linear_velocity=MAX_LINEAR,
            max_angular_velocity=PROFILE_MAX_ANGULAR,
        )
        config = make_navigation_config(desired_linear_velocity=DESIRED_LINEAR, disable_path_smoothing=True)
        config = replace(config, planner=replace(config.planner, max_planning_time=PLANNING_TIME_S))
        limits = make_command_limits(MAX_LINEAR, MAX_ANGULAR)
        self._parts = (costmap, profile, config, limits)
        free = int((costmap.data < int(config.planner.lethal_cost)).sum())
        log.info(
            f"nav2py costmap for {scene} from {map_dir}: {costmap.data.shape[1]}x{costmap.data.shape[0]} cells at "
            f"{costmap.resolution} m, {free} free after {ROBOT_EROSION_M} + {EXTRA_CLEARANCE_M} m of erosion"
        )

    # ---------------------------------------------------------------- the robot, as nav2py sees it
    def _pose_rot(self) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = self.sim.base_pose()  # torch tensors, quaternion (x, y, z, w)
        return np.asarray(pos.cpu().numpy(), dtype=np.float64), Rotation.from_quat(np.asarray(quat.cpu().numpy(), dtype=np.float64)).as_matrix()

    def _pose(self) -> tuple[float, float, float]:
        pos, rot = self._pose_rot()
        return float(pos[0]), float(pos[1]), float(math.atan2(rot[1, 0], rot[0, 0]))

    def _state(self, now: float):
        from benchmarks.nav2py_driver import state_estimate

        robot = self.sim.robot
        pos, rot = self._pose_rot()
        v = rot.T @ np.asarray(robot.get_linear_velocity().cpu().numpy(), dtype=np.float64)  # body frame
        w = rot.T @ np.asarray(robot.get_angular_velocity().cpu().numpy(), dtype=np.float64)
        yaw = float(math.atan2(rot[1, 0], rot[0, 0]))
        return state_estimate(now, float(pos[0]), float(pos[1]), yaw, float(v[0]), float(v[1]), float(w[2]))

    def _step(self, vx: float, vy: float, wz: float, hold_q, gripper) -> None:
        self.sim.step_base((float(vx), float(vy), float(wz)), hold_q, gripper)

    def _stop(self, hold_q, gripper) -> None:
        for _ in range(STOP_STEPS):
            self._step(0.0, 0.0, 0.0, hold_q, gripper)

    # ---------------------------------------------------------------- the map's word on a point
    @staticmethod
    def _nearest_free(costmap, x: float, y: float, lethal: int):
        """World xy of the nearest cell under ``lethal`` within SNAP_RADIUS_M of (x, y), or None."""
        cell = costmap.world_to_map(x, y)
        if cell is None:
            return None
        r = int(math.ceil(SNAP_RADIUS_M / costmap.resolution))
        rows, cols = costmap.data.shape
        r0, r1 = max(0, cell[0] - r), min(rows, cell[0] + r + 1)
        c0, c1 = max(0, cell[1] - r), min(cols, cell[1] + r + 1)
        window = costmap.data[r0:r1, c0:c1] < lethal
        if not window.any():
            return None
        rr, cc = np.nonzero(window)
        xs = costmap.origin_x + (cc + c0 + 0.5) * costmap.resolution
        ys = costmap.origin_y + (rr + r0 + 0.5) * costmap.resolution
        d = np.hypot(xs - x, ys - y)
        k = int(d.argmin())
        return (float(xs[k]), float(ys[k])) if d[k] <= SNAP_RADIUS_M else None

    # ---------------------------------------------------------------- moving
    def creep(self, x: float, y: float, yaw: float, hold_q, gripper) -> tuple[float, float]:
        """Close the last centimetres to (x, y, yaw) on the measured pose, straight, with no map: (off_xy, off_yaw)."""
        dt = self.sim.dt
        px, py, pyaw = self._pose()
        budget = int(2 * math.hypot(x - px, y - py) / (CREEP_SPEED * dt)) + int(2 * math.pi / (CREEP_YAW_SPEED * dt))
        for _ in range(budget):
            px, py, pyaw = self._pose()
            dx, dy = x - px, y - py
            dyaw = math.atan2(math.sin(yaw - pyaw), math.cos(yaw - pyaw))
            dist = math.hypot(dx, dy)
            if dist < CREEP_TOL and abs(dyaw) < CREEP_YAW_TOL:
                break
            speed = min(CREEP_SPEED, 2.0 * dist)  # slow into the last few centimetres
            bx = (math.cos(pyaw) * dx + math.sin(pyaw) * dy) / max(dist, 1e-6) * speed
            by = (-math.sin(pyaw) * dx + math.cos(pyaw) * dy) / max(dist, 1e-6) * speed
            wz = max(-CREEP_YAW_SPEED, min(CREEP_YAW_SPEED, 2.0 * dyaw))
            self._step(bx if dist >= CREEP_TOL else 0.0, by if dist >= CREEP_TOL else 0.0, wz, hold_q, gripper)
        self._stop(hold_q, gripper)
        px, py, pyaw = self._pose()
        return math.hypot(x - px, y - py), abs(math.atan2(math.sin(yaw - pyaw), math.cos(yaw - pyaw)))

    def drive(self, x: float, y: float, yaw: float, hold_q, gripper) -> dict:
        """Drive to the stance (world x, y, yaw), holding the planned joints at ``hold_q`` and the gripper as it is.

        Returns what happened: ``arrived``; nav2py's final ``state`` and ``reason``; ``steps`` spent; ``off_xy`` /
        ``off_yaw`` between where the robot stands and what was asked; ``snapped``, how far the map's goal was
        moved off the stance to reach free space (0 when the stance itself was free); ``crow_m``, the straight-line
        distance the drive set out over.
        """
        from nav2py import GoalSemantics, NavigationTask, Pose2D
        from benchmarks.nav2py_driver import cap_command_to_controller_limits, make_navigator

        if self._parts is None:
            self._build()
        costmap, profile, config, limits = self._parts
        lethal = int(config.planner.lethal_cost)
        px, py, _ = self._pose()
        # A creep ends off the map's free space by construction; the planner refuses to start from such a cell.
        if float(np.hypot(*self._creep)) > 0.0 and costmap.cost_at_world(px, py) >= int(config.costmap.lethal_cost):
            back = (px - self._creep[0], py - self._creep[1])
            log.info(f"backing {100 * float(np.hypot(*self._creep)):.0f} cm out of the last creep before planning")
            self.creep(back[0], back[1], self._pose()[2], hold_q, gripper)
            px, py, _ = self._pose()
        self._creep = np.zeros(2)
        goal = (float(x), float(y))
        snapped = 0.0
        if costmap.cost_at_world(x, y) >= lethal:
            free = self._nearest_free(costmap, x, y, lethal)
            if free is None:
                result = self._result(False, "rejected", f"no free cell within {SNAP_RADIUS_M} m of the stance", 0, x, y, yaw, 0.0, 0.0)
                log.warning(f"drive to ({x:.2f}, {y:.2f}): {result['reason']}")
                return result
            snapped = math.hypot(free[0] - x, free[1] - y)
            goal = free
            log.info(f"the map calls the stance lethal; driving to the free cell {100 * snapped:.0f} cm off it and creeping the rest")
        crow = math.hypot(goal[0] - px, goal[1] - py)
        navigator = make_navigator(profile, costmap, config, clearance_is_in_costmap=True)
        navigator.submit(
            NavigationTask(
                f"stance-{self.sim.n_steps}",
                goal_pose=Pose2D(goal[0], goal[1], float(yaw)),
                goal_semantics=GoalSemantics.HEADING_REQUIRED,
            )
        )
        budget = STEP_BUDGET_MIN + int(STEP_BUDGET_PER_M * crow)
        dt = self.sim.dt
        steps, status = 0, navigator.status()
        for step in range(budget):
            now = step * dt
            command = cap_command_to_controller_limits(navigator.tick(self._state(now), now), limits)
            if command is None or command.is_stop:
                self._step(0.0, 0.0, 0.0, hold_q, gripper)
            else:
                self._step(command.velocity.vx, command.velocity.vy, command.velocity.wz, hold_q, gripper)
            steps = step + 1
            status = navigator.status()
            if status.state.value in TERMINAL:
                break
        self._stop(hold_q, gripper)
        state = status.state.value
        reason = status.reason or ("" if state == "succeeded" else f"step budget of {budget} spent in state {state}")
        arrived = state == "succeeded"
        if arrived:  # nav2py stops inside its 8 cm goal tolerance; the stance search chose a point, so close on it
            before = self._pose()
            self.creep(x, y, yaw, hold_q, gripper)
            after = self._pose()
            if snapped > 0.0:
                self._creep = np.array([after[0] - before[0], after[1] - before[1]])
        px, py, pyaw = self._pose()
        off_xy = math.hypot(x - px, y - py)
        off_yaw = abs(math.atan2(math.sin(yaw - pyaw), math.cos(yaw - pyaw)))
        result = self._result(arrived, state, reason, steps, x, y, yaw, off_xy, off_yaw, snapped=snapped, crow=crow)
        (log.info if arrived else log.warning)(
            f"drive to ({x:.2f}, {y:.2f}) yaw {math.degrees(yaw):.0f} deg over {crow:.2f} m: {state} in {steps} env "
            f"steps, standing {100 * off_xy:.0f} cm and {math.degrees(off_yaw):.0f} deg off"
            + (f", via a free cell {100 * snapped:.0f} cm off the stance" if snapped else "")
            + (f" ({reason})" if reason else "")
        )
        return result

    @staticmethod
    def _result(arrived, state, reason, steps, x, y, yaw, off_xy, off_yaw, snapped=0.0, crow=0.0) -> dict:
        return {
            "arrived": bool(arrived),
            "state": state,
            "reason": reason,
            "steps": int(steps),
            "target": [float(x), float(y), float(yaw)],
            "off_xy": float(off_xy),
            "off_yaw": float(off_yaw),
            "snapped": float(snapped),
            "crow_m": float(crow),
        }
