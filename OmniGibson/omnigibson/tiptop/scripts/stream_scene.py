"""Load a BEHAVIOR scene, place the robot, and hold it, so the set-up can be inspected without running a plan.

Two ways to look at it: mirrored into a running planner's Rerun view (``--state-stream localhost:8765``: the robot,
its objects and cameras appear exactly as in a live run; no planning request is made), or the Isaac viewport over
WebRTC (``OMNIGIBSON_REMOTE_STREAMING=webrtc``, and then NOT ``OMNIGIBSON_HEADLESS``: streaming already runs the app
windowless, and R1ProSim.place_robot only aims the viewport while gm.HEADLESS is false). Steps forever until
interrupted.

    cd <repo> && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 OMNIGIBSON_HEADLESS=1 \\
        ./b1k/bin/python OmniGibson/omnigibson/tiptop/scripts/stream_scene.py --activity assembling_gift_baskets \\
        --place wicker_basket.n.01_2:table.n.02_1:0.20,0.50 --torso 1.2 -1.7 -0.9 0.0 \\
        --stand-for swiss_cheese.n.01_1,wicker_basket.n.01_2 --state-stream localhost:8765
"""

import argparse
import time

import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.tiptop.r1pro import (
    R1ProSim,
    challenge_task_info,
    load_embodiment_meta,
    make_r1pro_env_config,
)
from omnigibson.tiptop.run import open_state_stream, parse_spawns, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activity", default="assembling_gift_baskets", help="BEHAVIOR challenge task to load")
    ap.add_argument("--activity-instance", type=int, default=0)
    ap.add_argument("--not-load", default="ceilings", help="comma-separated categories to leave out")
    ap.add_argument("--place", action="append", default=[], metavar="OBJ:SUPPORT[:DX,DY]")
    ap.add_argument("--torso", type=float, nargs=4, default=None, metavar=("J1", "J2", "J3", "J4"))
    ap.add_argument("--stand-for", default=None, metavar="ITEM[,ITEM...],TARGET", help="base pose reaching all")
    ap.add_argument("--state-stream", default=None, metavar="HOST:PORT", help="mirror into a tiptop-server's Rerun")
    ap.add_argument("--steps", type=int, default=300, help="steps between liveness lines")
    args = ap.parse_args()
    setup_logging()
    places = parse_spawns(args.place, flag="--place")  # validated before Isaac Sim starts

    scene_model, rooms = challenge_task_info(args.activity)
    print(f"[stream] {args.activity}: scene={scene_model} rooms={rooms}", flush=True)

    sim = R1ProSim(
        make_r1pro_env_config(
            scene_model=scene_model,
            load_room_types=[],
            not_load_object_categories=[c for c in args.not_load.split(",") if c],
            activity=args.activity,
            activity_instance_id=args.activity_instance,
            load_room_instances=rooms,
            segmentation=False,
        ),
        camera="head",
    )
    sim.track_task_objects()
    sim.track_context(*{support for _, support, _, _ in places})
    for name, support, dx, dy in places:
        sim.place_on(name, support, dx, dy)
    # the same order as run.py: the posture decides how close the head camera can see, so it comes first
    embodiment = load_embodiment_meta()
    q_home = [float(v) for v in embodiment["q_home"]]
    if args.torso:
        for joint, value in zip(embodiment["torso_joints"], args.torso):
            q_home[embodiment["joint_names"].index(joint)] = value
    sim.apply_posture(embodiment["locked_joints"], q_home, joint_names=embodiment["joint_names"])
    if args.stand_for:
        sim.place_robot_for(*[n for n in args.stand_for.split(",") if n])
    else:  # where the scene put it; place_robot aims the overview camera and the viewport at the workspace
        pos, quat = sim.robot.get_position_orientation()
        sim.place_robot(float(pos[0]), float(pos[1]), float(T.quat2euler(quat)[2]), note="as the scene put it")
    sim.hold(30, sim.OPEN)
    sim.mark_goal_initial()

    stream = open_state_stream(args.state_stream, sim)
    if gm.REMOTE_STREAMING:
        print("[stream] ready -- connect the Isaac Sim WebRTC Streaming Client", flush=True)
    elif stream is None:
        print("[stream] ready -- pass --state-stream HOST:PORT to see it in a planner's Rerun", flush=True)
    n = 0
    try:
        while True:
            sim.hold(args.steps, sim.OPEN)
            n += args.steps
            print(f"[stream] alive, {n} steps ({time.strftime('%H:%M:%S')})", flush=True)
    finally:
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
