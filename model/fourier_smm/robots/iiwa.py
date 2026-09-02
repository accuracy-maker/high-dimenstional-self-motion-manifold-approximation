"""
Standard DH table of iiwa14
"""

import numpy as np

def iiwa14_standard_dh():
    """
    Standard DH parameters for KUKA LBR iiwa14.
    Convention:
        A_i = Rz(theta_i) @ Tz(d_i) @ Tx(a_i) @ Rx(alpha_i)

    Returns
    -------
    a : (7,)
    alpha : (7,)
    d : (7,)
    theta_offset : (7,)

    Actual joint angle:
        theta_i = q_i + theta_offset[i]
    """
    a = np.zeros(7)

    alpha = np.array([
        -np.pi / 2,
        +np.pi / 2,
        +np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        +np.pi / 2,
        0.0,
    ])

    d = np.array([
        0.360,
        0.0,
        0.420,
        0.0,
        0.400,
        0.0,
        0.126,
    ])

    theta_offset = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        np.pi,
    ])

    return a, alpha, d, theta_offset

