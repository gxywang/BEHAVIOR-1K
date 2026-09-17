"""The simulator side of the BEHAVIOR-1K policy seam: what the evaluator gives, in the policy's own types.

Two modules, split by what they are allowed to know:

* ``adapter`` -- an ``omnigibson.eval.evaluator.Evaluator`` observation as a ``b1k.observation.SensorObservation``
  (nothing privileged) or ``OracleObservation`` (the simulator's truth, for bring-up). Only the oracle half
  touches a live scene.
* ``selfmask`` -- the robot's own pixels from proprioception and its URDF, replacing the privileged ``robot_mask``
  that ``omnigibson/tiptop`` reads out of ``seg_instance``. No simulator, no omnigibson import.

Nothing is imported here: ``adapter`` needs the policy package (``b1k``) on the path and ``selfmask`` needs
trimesh, and neither should be forced on a bare ``import omnigibson``.
"""
