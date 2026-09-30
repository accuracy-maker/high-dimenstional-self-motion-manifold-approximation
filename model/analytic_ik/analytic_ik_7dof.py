"""
This is the analytical IK solution for 7DoF S-R-S manipulator:
    - KUKA iiwa
    - Franka Panda

reference: 
1. "Position-based kinematics for 7-DoF serial manipulators with global 
configuration control, joint limit and singularity avoidance" by Faria et. al.

2. constraint-manifold-charts-ift github repo ny cohn
"""

import numpy as np
import math

# DH parameters (default by iiwa 14)
dh_alpha = np.array([
	-np.pi/2,
	np.pi/2,
	np.pi/2,
	-np.pi/2,
	-np.pi/2,
	np.pi/2,
	0
])

dh_d = np.array([
    0.36,
	0,
	0.42,
	0,
	0.4,
	0,
	0.081
])

dh_limits_lower = np.array([
	-2.967060,
	-2.094395,
	-2.967060,
	-2.094395,
	-2.967060,
	-2.094395,
	-3.054326
])

dh_limits_upper = np.array([
	2.967060,
	2.094395,
	2.967060,
	2.094395,
	2.967060,
	2.094395,
	3.054326
])

# help functions
def cross_product_matrix(a):
    return np.cross(a, np.identity(a.shape[0]) * -1)

def scalar_clip(val, a, b):
    """Clip val to the interval [a, b]."""
    return np.clip(val, a, b)


def safe_arccos(val, a=-1.0, b=1.0):
    """Compute arccos after clipping val to [a, b]."""
    return np.arccos(scalar_clip(val, a, b))


class Analytical_IK_7DoF():
    """
    Class for analytic forward and invese kinematics for 
    a S-R-S 7 DoF robotic manipulator
    """

    def __init__(
        self,
        alpha = dh_alpha,
        d = dh_d,
        limits_lower = dh_limits_lower,
        limits_upper = dh_limits_upper
    ):
        # other dh parameters are all zeros
        assert len(alpha) == len(d) == len(limits_lower) == len(limits_upper) == 7
        assert d[1] == d[3] == d[5] == 0

        self.alpha = alpha.copy()
        self.d = d.copy()
        
        self.d_bs, self.d_se, self.d_ew, self.d_wf = d[0], d[2], d[4], d[6]

        self.limits_lower = limits_lower.copy()
        self.limits_upper = limits_upper.copy()

        # transform matrix based on standard DH parameters
        self.Ts = [
            lambda ti, ai = ai, di=di : np.array([
                [np.cos(ti), -np.sin(ti)*np.cos(ai), np.sin(ti)*np.sin(ai), 0],
				[np.sin(ti), np.cos(ti)*np.cos(ai), -np.cos(ti)*np.sin(ai), 0],
				[0, np.sin(ai), np.cos(ai), di],
				[0, 0, 0, 1]
            ])
            for ai, di in zip(self.alpha, self.d)
        ]

    def FK(self, thetas):
        """
        Analytic Forward Kinematics
        T^0_7 = T^0_1 @ T^1_2 @...@ T^6_7
        """
        eval_Ts = [eval_T(t) for (eval_T, t) in zip(self.Ts, thetas)]
        full_mat = np.linalg.multi_dot(eval_Ts)
        return full_mat

    def GC(self, thetas):
        """
        Compute the global configurations:
            - should up/down
            - elbow up/down
            - wrist up/down
        """

        ks = [1, 3, 5]
        return np.array([1 if thetas[k] >= 0 else -1 for k in ks])

    def psi(self, thetas, return_T_vs=False):
        """
        psi is the angel of reference plane wrt arm plane
        """
        T_07 = self.FK(thetas)
        p_07 = T_07[:-1,-1]
        R_07 = T_07[:-1,:-1]
        GC2, GC4, GC6 = self.GC(thetas)

        p_02 = np.array([0, 0, self.d_bs])
        p_24 = np.array([0, self.d_se, 0])
        p_46 = np.array([0, 0, self.d_ew])
        p_67 = np.array([0, 0, self.d_wf])

        p_26 = p_07 - p_02 - (R_07 @ p_67)
        arccos_in = (np.dot(p_26, p_26) - self.d_se**2 - self.d_ew**2) / (2 * self.d_se * self.d_ew)
        theta_4v = GC4 * np.arccos(scalar_clip(arccos_in, -1, 1))

        R_01 = self.Ts[0](thetas[0])[0:3,0:3]
        cross = np.cross(p_26, R_01[:,-1])
        cond = np.dot(cross, cross) > 0
        theta_1v = np.arctan2(p_26[1], p_26[0]) if cond else 0

        arccos_in = (self.d_se**2 + np.dot(p_26, p_26) - self.d_ew**2) / (2 * self.d_se * np.linalg.norm(p_26))
        phi = np.arccos(scalar_clip(arccos_in, -1, 1))
        theta_2v = np.arctan2(np.linalg.norm(p_26[:2]), p_26[2]) + (GC4 * phi)

        theta_3v = 0
        theta_vs = [theta_1v, theta_2v, theta_3v, theta_4v]
        T_vs = [T(theta_v) for T, theta_v in zip(self.Ts[:len(theta_vs)], theta_vs)]
        T_02_v = np.linalg.multi_dot(T_vs[0:2])
        T_04_v = np.linalg.multi_dot(T_vs[0:4])
        p_02_v = T_02_v[:-1,-1]
        p_04_v = T_04_v[:-1,-1]

        Ts = [T(theta) for T, theta in zip(self.Ts, thetas)]
        T_04 = np.linalg.multi_dot(Ts[0:4])
        p_04 = T_04[:-1,-1]
        T_06 = np.linalg.multi_dot(Ts[0:6])
        p_06 = T_06[:-1,-1]
        p_06_v = p_06

        v_se_v = (p_04_v - p_02_v) / np.linalg.norm(p_04_v - p_02_v)
        v_sw_v = (p_06_v - p_02_v) / np.linalg.norm(p_06_v - p_02_v)
        v_sew_v = np.cross(v_se_v, v_sw_v)
        v_sew_v_hat = v_sew_v / np.linalg.norm(v_sew_v)

        v_se = (p_04 - p_02) / np.linalg.norm(p_04 - p_02)
        v_sw = (p_06 - p_02) / np.linalg.norm(p_06 - p_02)
        v_sew = np.cross(v_se, v_sw)
        v_sew_hat = v_sew / np.linalg.norm(v_sew)

        sg_psi = np.sign(np.dot(np.cross(v_sew_v_hat, v_sew_hat), p_26))
        psi = sg_psi * np.arccos(np.dot(v_sew_v_hat, v_sew_hat))

        if return_T_vs:
            return psi, T_vs
        else:
            return psi


    def IK(
        self,
        rigid_transform,
        GC,
        psi,
        return_unclipped_vals=False,
        return_singularity_vals=False,
        return_sw_mats=False,
        clip_stepback=1e-4
    ):
        """
        Analytic Inverse kinematics that is parameterised by:
            - global configurations
            - arm angle
        """

        clip = 1 - clip_stepback

        assert not (return_unclipped_vals and return_singularity_vals)

        thetas = np.zeros(7)
        unclipped_vals = np.zeros(4)
        singularity_vals = np.zeros(2)

        GC2, GC4, GC6 = GC

        p_02 = np.array([0, 0, self.d_bs])
        p_24 = np.array([0, self.d_se, 0])
        p_46 = np.array([0, 0, self.d_ew])
        p_67 = np.array([0, 0, self.d_wf])

        p_07 = rigid_transform[:-1,-1]
        R_07 = rigid_transform[:-1,:-1]
        p_26 = p_07 - p_02 - (R_07 @ p_67) # EQ (3)
        p_26_hat = p_26 / np.linalg.norm(p_26)

        theta_1v = np.arctan2(p_26[1], p_26[0]) # EQ (5)

        # EQ (7)
        arccos_in = (self.d_se**2 + np.dot(p_26, p_26) - self.d_ew**2) / (2 * self.d_se * np.linalg.norm(p_26))
        unclipped_vals[0] = arccos_in
        phi = safe_arccos(arccos_in, -clip, clip)
        theta_2v = np.arctan2(np.linalg.norm(p_26[:2]), p_26[2]) + (GC4 * phi)

        theta_3v = 0

        # EQ (4)
        arccos_in = (np.dot(p_26, p_26) - self.d_se**2 - self.d_ew**2) / (2 * self.d_se * self.d_ew)
        unclipped_vals[1] = arccos_in
        theta_4v = GC4 * safe_arccos(arccos_in, -clip, clip)
        thetas[3] = theta_4v

        theta_vs = [theta_1v, theta_2v, theta_3v, theta_4v]
        T_vs = [T(theta_v) for T, theta_v in zip(self.Ts[:len(theta_vs)], theta_vs)]
        T_03_v = np.linalg.multi_dot(T_vs[0:3])
        R_03_v = T_03_v[:-1,:-1]

        # EQ (15)
        cprod_p_26 = cross_product_matrix(p_26_hat)
        A_s = cprod_p_26 @ R_03_v
        B_s = -1 * cprod_p_26 @ cprod_p_26 @ R_03_v
        C_s = np.outer(p_26_hat, p_26_hat) @ R_03_v

        # EQ (17)-(19)
        thetas[0] = np.arctan2(
        GC2 * (A_s[1,1] * np.sin(psi) + B_s[1,1] * np.cos(psi) + C_s[1,1]),
        GC2 * (A_s[0,1] * np.sin(psi) + B_s[0,1] * np.cos(psi) + C_s[0,1])
        )
        arccos_in = A_s[2,1] * np.sin(psi) + B_s[2,1] * np.cos(psi) + C_s[2,1]
        unclipped_vals[2] = arccos_in
        thetas[1] = GC2 * safe_arccos(arccos_in, -clip, clip)
        thetas[2] = np.arctan2(
        GC2 * (-A_s[2,2] * np.sin(psi) - B_s[2,2] * np.cos(psi) - C_s[2,2]),
        GC2 * (-A_s[2,0] * np.sin(psi) - B_s[2,0] * np.cos(psi) - C_s[2,0])
        )

        # EQ (20)
        T_34 = T_vs[3]
        R_34 = T_34[:-1,:-1]
        A_w = R_34.T @ A_s.T @ R_07
        B_w = R_34.T @ B_s.T @ R_07
        C_w = R_34.T @ C_s.T @ R_07

        # EQ (22)-(24)
        thetas[4] = np.arctan2(
        GC6 * (A_w[1,2] * np.sin(psi) + B_w[1,2] * np.cos(psi) + C_w[1,2]),
        GC6 * (A_w[0,2] * np.sin(psi) + B_w[0,2] * np.cos(psi) + C_w[0,2])
        )
        arccos_in = A_w[2,2] * np.sin(psi) + B_w[2,2] * np.cos(psi) + C_w[2,2]
        unclipped_vals[3] = arccos_in
        thetas[5] = GC6 * safe_arccos(arccos_in, -clip, clip)
        thetas[6] = np.arctan2(
        GC6 * (A_w[2,1] * np.sin(psi) + B_w[2,1] * np.cos(psi) + C_w[2,1]),
        GC6 * (-A_w[2,0] * np.sin(psi) - B_w[2,0] * np.cos(psi) - C_w[2,0])
        )

        if return_unclipped_vals:
            return unclipped_vals
        if return_singularity_vals:
            return thetas[[3, 5]]
        else:
            if return_sw_mats:
                return np.asarray(thetas), A_s, B_s, C_s, A_w, B_w, C_w
            return thetas

