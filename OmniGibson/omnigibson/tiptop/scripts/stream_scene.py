"""Load a BEHAVIOR scene and hold it, so the viewport can be inspected over WebRTC without running a plan.

Useful for checking camera framing, object placement and the stream itself without starting the planner or
M2T2. Steps forever until interrupted.

    cd <repo> && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 OMNIGIBSON_REMOTE_STREAMING=webrtc \
        ./b1k/bin/python OmniGibson/omnigibson/tiptop/scripts/stream_scene.py --activity assembling_gift_baskets

Do NOT also set OMNIGIBSON_HEADLESS: streaming already runs the app windowless, and scene.py only aims the
viewport camera while gm.HEADLESS is false.

On an RTX PRO 6000 Blackwell the stream connects but stays black unless UseRefactoredVideoEncoder=1 is also
exported; see DEPLOYMENT.md item 14.
"""

import argparse
import time

import omnigibson as og
from omnigibson.tiptop.r1pro import (
    R1ProSim,
    challenge_task_info,
    load_embodiment_meta,
    make_r1pro_env_config,
)
from omnigibson.tiptop.scene import look_at_quat_xyzw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activity", default="assembling_gift_baskets", help="BEHAVIOR challenge task to load")
    ap.add_argument("--activity-instance", type=int, default=0)
    ap.add_argument("--not-load", default="ceilings", help="comma-separated categories to leave out")
    ap.add_argument("--place", action="append", default=[], metavar="OBJ:SUPPORT[:DX,DY]")
    ap.add_argument("--stand-for", default=None, metavar="ITEM,TARGET", help="base pose reaching both")
    ap.add_argument("--steps", type=int, default=300, help="steps between liveness lines")
    args = ap.parse_args()

    scene_model, rooms = challenge_task_info(args.activity)
    print(f"[stream] {args.activity}: scene={scene_model} rooms={rooms}", flush=True)

    sim = R1ProSim(
        make_r1pro_env_config(
            scene_model=scene_model,
            load_room_types=[],
            spawn_presets=[],
            grasping_mode="sticky",
            camera="head",
            not_load_object_categories=[c for c in args.not_load.split(",") if c],
            activity=args.activity,
            activity_instance_id=args.activity_instance,
            load_room_instances=rooms,
            segmentation=False,
        ),
        camera="head",
    )
    sim.track_task_objects()
    for spec in args.place:
        parts = spec.split(":")
        dx, dy = (float(v) for v in parts[2].split(",")) if len(parts) > 2 else (0.0, 0.0)
        sim.place_on(parts[0], parts[1], dx, dy)
    if args.stand_for:
        item, target = args.stand_for.split(",")
        sim.place_robot_for(item, target)

    embodiment = load_embodiment_meta()
    sim.apply_posture(embodiment["locked_joints"], embodiment["q_home"], joint_names=embodiment["joint_names"])
    sim.mark_goal_initial()

    # Put the streamed viewport somewhere useful: behind and left of the robot, looking at its workspace.
    pos, _ = sim.robot.get_position_orientation()
    x, y = float(pos[0]), float(pos[1])
    eye = (x - 1.6, y - 1.6, 2.0)
    og.sim.viewer_camera.set_position_orientation(
        position=eye, orientation=look_at_quat_xyzw(eye, (x, y, 1.0))
    )
    print("[stream] ready -- connect the Isaac Sim WebRTC Streaming Client", flush=True)

    n = 0
    while True:
        sim.hold(args.steps, sim.OPEN)
        n += args.steps
        print(f"[stream] alive, {n} steps ({time.strftime('%H:%M:%S')})", flush=True)


if __name__ == "__main__":
    main()
