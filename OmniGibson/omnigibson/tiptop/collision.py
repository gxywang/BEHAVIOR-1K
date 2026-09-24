"""CPU preflight for bridge-owned joint ramps using the robot's cuRobo collision spheres.

This validates world and carried-object/robot collisions along the direct path; robot-only self-collision
checking and motion planning remain on TiPToP. No simulator poses change, and all joints enter FK.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from omnigibson.tiptop.kinematics import ArmIK

CARRIED = "carried object"  # allowed_contacts entry for the attachments' spheres
MAX_SWEEP_STEP = 0.005  # m of bounded sphere-centre travel; interval inflation is at most 2.5 mm


class JointPathCollision:
    def __init__(self, urdf_path, sphere_config, joint_names):
        root = ET.parse(urdf_path).getroot()
        # Continuous wheel joints have no position limits, which Lula cannot put in its configuration space.
        # Their collision spheres are centred on the wheel axes, so wheel spin does not change their volume.
        continuous = {joint.get("name") for joint in root.findall("joint") if joint.get("type") == "continuous"}
        self.joint_names = [name for name in joint_names if name not in continuous]
        fixed = {
            joint.get("name"): 0.0
            for joint in root.findall("joint")
            if joint.get("type") != "fixed" and joint.get("name") not in self.joint_names
        }
        self.fk = ArmIK(urdf_path, self.joint_names, fixed, frame="base_link")
        config = yaml.safe_load(Path(sphere_config).read_text())["robot_cfg"]["kinematics"]
        spheres = config["collision_spheres"]
        if isinstance(spheres, str):
            spheres = yaml.safe_load((Path(sphere_config).parent / spheres).read_text())["collision_spheres"]
        self.buffer = float(config.get("collision_sphere_buffer", 0.0))
        self.self_buffer = config.get("self_collision_buffer") or {}
        self.attachment_ignore = {}
        for arm in ("left", "right"):
            attachment = f"attached_object_{arm}_eef_link"
            ignore = config.get("self_collision_ignore") or {}
            ignored = set(ignore.get(attachment, ())) | {link for link, peers in ignore.items() if attachment in peers}
            # Same allowed grasp contacts as scripts/make_r1pro_embodiment.py. The hand/camera and distal
            # wrist are adjacent to the carried volume; the base, torso and opposite arm remain checked.
            ignored.update(f"{arm}_{suffix}" for suffix in (
                "gripper_link", "gripper_finger_link1", "gripper_finger_link2", "realsense_link", "arm_link7", "arm_link6"
            ))
            self.attachment_ignore[f"{arm}_gripper_link"] = ignored
        self.links, self.local_centres, self.radii = [], [], []
        for link, entries in spheres.items():
            if link.startswith("attached_object"):
                continue  # measured attachments are added per call
            for sphere in entries:
                if float(sphere["radius"]) <= 0:
                    continue
                self.links.append(link)
                self.local_centres.append(sphere["center"])
                self.radii.append(float(sphere["radius"]) + self.buffer)
        self.links = np.asarray(self.links)
        self.local_centres = np.asarray(self.local_centres, dtype=np.float64)
        self.radii = np.asarray(self.radii, dtype=np.float64)
        self.parents = {}
        self.joint_axes = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            offset = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            axis = joint.find("axis")
            self.joint_axes[joint.get("name")] = (
                np.fromstring(axis.get("xyz", "1 0 0"), sep=" ") if axis is not None else np.array([1.0, 0.0, 0.0])
            )
            extension = 0.0
            if joint.get("type") == "prismatic":
                limit = joint.find("limit")
                if limit is None:
                    raise ValueError(f"prismatic joint {joint.get('name')} needs limits for a conservative sweep bound")
                extension = max(abs(float(limit.get("lower"))), abs(float(limit.get("upper"))))
            self.parents[joint.find("child").get("link")] = (
                joint.find("parent").get("link"), joint.get("name"), joint.get("type"),
                float(np.linalg.norm(offset)) + extension,
            )
        self.bounds = self.motion_bounds(self.links, self.local_centres)

    def motion_bounds(self, links, centres):
        """Per-sphere Lipschitz bound in each joint: a sum of link lengths bounds distance to every ancestor axis."""
        index = {name: i for i, name in enumerate(self.joint_names)}
        bounds = np.zeros((len(links), len(index)), dtype=np.float64)
        for row, (link, centre) in enumerate(zip(links, centres)):
            radius = float(np.linalg.norm(centre))
            while link in self.parents:
                parent, joint, kind, offset = self.parents[link]
                if joint in index:
                    bounds[row, index[joint]] = 1.0 if kind == "prismatic" else radius
                radius += offset
                link = parent
        return bounds

    def centres(self, q, links, local):
        centres = np.empty_like(local)
        for link in np.unique(links):
            pos, quat = self.fk.fk(q, str(link))
            select = links == link
            centres[select] = local[select] @ Rotation.from_quat(quat).as_matrix().T + pos
        return centres

    def _finger_sweep(self, positions, joint_ranges):
        """Cover every finger opening independently of arm timing with a tight union of spheres."""
        links, local, radii = [], [], []
        for name, (start, end) in joint_ranges.items():
            positions[:, self.joint_names.index(name)] = (start + end) / 2.0
        for link, centre, radius in zip(self.links, self.local_centres, self.radii):
            parent = self.parents.get(str(link))
            if parent is None or parent[1] not in joint_ranges:
                links.append(link)
                local.append(centre)
                radii.append(radius)
                continue
            _, name, kind, _ = parent
            if kind != "prismatic":
                raise ValueError(f"finger sweep requires a prismatic joint, got {name!r}")
            start, end = joint_ranges[name]
            intervals = max(1, int(np.ceil(abs(end - start) / MAX_SWEEP_STEP)))
            # Every intermediate position is within half a sample interval of one of these sphere centres.
            margin = abs(end - start) / intervals / 2.0
            for position in np.linspace(start, end, intervals + 1):
                links.append(link)
                local.append(centre + self.joint_axes[name] * (position - (start + end) / 2.0))
                radii.append(radius + margin)
        return np.asarray(links), np.asarray(local), np.asarray(radii)

    def check(self, start, goal, world_from_base, obstacles, allowed_contacts=None, attachments=(), joint_ranges=None):
        """Check a direct path; see check_polyline for geometry and contact semantics."""
        return self.check_polyline(
            np.asarray([start, goal]), world_from_base, obstacles, allowed_contacts, attachments, joint_ranges
        )

    def check_polyline(
        self, positions, world_from_base, obstacles, allowed_contacts=None, attachments=(), joint_ranges=None,
        clearance=0.0,
    ):
        """First (robot link, obstacle, sample) intersection along every original waypoint edge, else None.

        ``obstacles`` are (name, world mesh); ``attachments`` are (gripper link, local centres, radii).
        ``joint_ranges`` covers independent finger-controller motion throughout the entire arm path. No waypoint
        corner is skipped. Only explicitly named link/object pairs can make contact; closed-component start
        containment and interval-inflated surface distances reject both deep starts and between-sample crossings.
        ``clearance`` (m) is extra room demanded from the room's obstacles, for motions that cannot back off.
        ``allowed_contacts`` may name CARRIED (the attachments' spheres, not the gripper link they ride on): what a
        carried object may keep touching, such as the support a sticky grasp just took it from.
        """
        positions = np.asarray(positions, dtype=np.float64).copy()
        if positions.ndim != 2 or len(positions) == 0 or not np.isfinite(positions).all():
            raise ValueError("motion must contain finite joint configurations")
        links, local, radii, bounds = self.links, self.local_centres, self.radii, self.bounds
        if joint_ranges:
            links, local, radii = self._finger_sweep(positions, joint_ranges)
            bounds = self.motion_bounds(links, local)
        robot_count = len(links)
        if attachments:
            links = np.concatenate([links, *[np.repeat(link, len(c)) for link, c, _ in attachments]])
            local = np.concatenate([local, *[c for _, c, _ in attachments]])
            radii = np.concatenate([radii, *[r for _, _, r in attachments]])
            bounds = self.motion_bounds(links, local)
        # Repeated configurations add no geometry; all distinct original vertices remain.
        positions = positions[np.r_[True, np.any(np.diff(positions, axis=0) != 0, axis=1)]]
        sampled, margin = [], 0.0
        for start, end in zip(positions[:-1], positions[1:]):
            travel = float((bounds @ np.abs(end - start)).max())
            intervals = max(1, int(np.ceil(travel / MAX_SWEEP_STEP)))
            margin = max(margin, travel / intervals / 2.0)
            sampled.extend(start + np.arange(intervals)[:, None] / intervals * (end - start))
        sampled.append(positions[-1])
        inflated = radii + margin + clearance
        centres = np.stack([self.centres(q, links, local) for q in sampled])
        # Check carried volume against the robot before transforming both into the room. Inflation on BOTH
        # spheres covers their relative motion between samples; finger union spheres cover independent closure.
        offset = robot_count
        for attachment_index, (link, local_attachment, attachment_radii) in enumerate(attachments):
            checked = np.flatnonzero(~np.isin(links[:robot_count], list(self.attachment_ignore[link])))
            robot_radii = radii[checked] - self.buffer + np.array([
                self.self_buffer.get(str(links[i]), 0.0) for i in checked
            ])
            limit = robot_radii[:, None] + np.asarray(attachment_radii)[None, :] + 2 * margin
            for first in range(0, len(centres), 32):
                robot = centres[first:first + 32, checked]
                held = centres[first:first + 32, offset:offset + len(local_attachment)]
                distance = np.linalg.norm(robot[:, :, None, :] - held[:, None, :, :], axis=-1)
                hits = np.argwhere(distance <= limit[None])
                if len(hits):
                    sample, sphere, _ = hits[0]
                    return f"attached_object_{link.removesuffix('_gripper_link')}", str(links[checked[sphere]]), int(first + sample)
            other_offset = offset + len(local_attachment)
            for other_link, other_local, other_radii in attachments[attachment_index + 1:]:
                limit = np.asarray(attachment_radii)[:, None] + np.asarray(other_radii)[None, :] + 2 * margin
                for first in range(0, len(centres), 32):
                    held = centres[first:first + 32, offset:offset + len(local_attachment)]
                    other = centres[first:first + 32, other_offset:other_offset + len(other_local)]
                    hits = np.argwhere(np.linalg.norm(held[:, :, None, :] - other[:, None, :, :], axis=-1) <= limit[None])
                    if len(hits):
                        return (f"attached_object_{link.removesuffix('_gripper_link')}",
                                f"attached_object_{other_link.removesuffix('_gripper_link')}", int(first + hits[0, 0]))
                other_offset += len(other_local)
            offset += len(local_attachment)
        transform = np.asarray(world_from_base, dtype=np.float64)
        centres = centres @ transform[:3, :3].T + transform[:3, 3]
        allowed_contacts = allowed_contacts or {}
        carried = np.arange(len(links)) >= robot_count

        def sphere_name(i):  # a carried object's spheres ride on the gripper link but are not the robot
            return f"attached_object_{str(links[i]).removesuffix('_gripper_link')}" if carried[i] else str(links[i])
        for name, mesh in obstacles:
            lo, hi = mesh.bounds
            possible = np.all(centres + inflated[None, :, None] >= lo, axis=-1) & np.all(
                centres - inflated[None, :, None] <= hi, axis=-1
            )
            allowed = set(allowed_contacts.get(name, ()))
            if allowed:
                possible[:, np.isin(links, list(allowed - {CARRIED})) & ~carried] = False
                if CARRIED in allowed:
                    possible[:, carried] = False
            samples, spheres = np.nonzero(possible)
            if len(samples) == 0:
                continue
            # The mesh's cached triangle BVH gives exact nearest distances without subdividing merged walls
            # into millions of tiny triangles (the image-mask proximity helper does that for depth clouds).
            distances = trimesh.proximity.closest_point(mesh, centres[samples, spheres])[1]
            hits = np.flatnonzero(distances <= inflated[spheres])
            if len(hits):
                hit = hits[np.argmin(samples[hits])]
                return sphere_name(spheres[hit]), name, int(samples[hit])
            # A sphere can start deeply inside a closed component with no surface within its radius.
            initial = np.flatnonzero(possible[0])
            if len(initial):
                if "collision_components" not in mesh.metadata:
                    mesh.metadata["collision_components"] = list(mesh.split(only_watertight=True))
                for component in mesh.metadata["collision_components"]:
                    inside = component.contains(centres[0, initial])
                    if inside.any():
                        return sphere_name(initial[np.flatnonzero(inside)[0]]), name, 0
        return None


def box_spheres(bounds, cell_size=0.04):
    """Conservative sphere cover of a held object's box; every point belongs to a covered grid cell."""
    lo, hi = np.asarray(bounds, dtype=np.float64)
    cells = np.maximum(1, np.ceil((hi - lo) / cell_size).astype(int))
    widths = (hi - lo) / cells
    axes = [lo[i] + (np.arange(cells[i]) + 0.5) * widths[i] for i in range(3)]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return centres, np.full(len(centres), np.linalg.norm(widths) / 2.0)
