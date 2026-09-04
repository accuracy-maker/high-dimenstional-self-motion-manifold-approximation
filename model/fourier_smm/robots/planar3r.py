import numpy as np

from realtime_smm import DHLink, TaskSpace

from ..fast_smm import FastRobot

LENGTHS = (1.0, 1.0, 1.0)

TASKSPACE = TaskSpace.X | TaskSpace.Y


def planar3r_links(lengths=LENGTHS, joint_limits=(-np.pi, np.pi)):
    lo, hi = joint_limits
    return [DHLink(a=float(length), alpha=0.0, d=0.0, theta=0.0,
                   lower_limit=lo, upper_limit=hi) for length in lengths]


def planar3r(lengths=LENGTHS, joint_limits=(-np.pi, np.pi)) -> FastRobot:
    """Planar 3R with the batched SMM backend, matching rtb.models.DH.Planar3()
    (the robot smm_3r_ode.py and flow_matching.py's fm_3r.pt checkpoint use)."""
    return FastRobot(planar3r_links(lengths, joint_limits), taskspace=TASKSPACE)
