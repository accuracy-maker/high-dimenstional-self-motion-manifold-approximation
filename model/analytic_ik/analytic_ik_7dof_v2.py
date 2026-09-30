"""Analytical inverse kinematics for a non-offset 7-DoF S-R-S arm.

The implementation follows the position-based solution of Faria et al.:

    C. Faria et al., "Position-based kinematics for 7-DoF serial
    manipulators with global configuration control, joint limit and
    singularity avoidance."

The robot is described by standard (classic) DH parameters with zero link
offsets a_i.  The inverse-kinematics solution is parameterised by

    GC = (GC2, GC4, GC6),  GCk in {-1, +1}
    psi = arm angle

The code deliberately keeps the GC factors in all atan2 expressions.  In
particular, the GC6 factors in the theta_7 equation are part of the wrist
branch convention and should not be removed casually.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Default KUKA iiwa-style parameters
# ---------------------------------------------------------------------------

dh_alpha = np.array(
    [
        -np.pi / 2.0,
        +np.pi / 2.0,
        +np.pi / 2.0,
        -np.pi / 2.0,
        -np.pi / 2.0,
        +np.pi / 2.0,
        0.0,
    ],
    dtype=float,
)

dh_d = np.array(
    [
        0.36,
        0.0,
        0.42,
        0.0,
        0.40,
        0.0,
        0.081,
    ],
    dtype=float,
)

dh_limits_lower = np.array(
    [
        -2.967060,
        -2.094395,
        -2.967060,
        -2.094395,
        -2.967060,
        -2.094395,
        -3.054326,
    ],
    dtype=float,
)

dh_limits_upper = np.array(
    [
        +2.967060,
        +2.094395,
        +2.967060,
        +2.094395,
        +2.967060,
        +2.094395,
        +3.054326,
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def cross_product_matrix(a: np.ndarray) -> np.ndarray:
    """Return the 3-by-3 matrix [a]x satisfying [a]x @ b = a x b."""

    a = np.asarray(a, dtype=float)
    if a.shape != (3,):
        raise ValueError(f"Expected a with shape (3,), got {a.shape}.")

    return np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ],
        dtype=float,
    )


def scalar_clip(val, a=-1.0, b=1.0):
    """Clip a scalar or array to [a, b]."""

    return np.clip(val, a, b)


def safe_arccos(val, a=-1.0, b=1.0):
    """Evaluate arccos after clipping small floating-point violations."""

    return np.arccos(scalar_clip(val, a, b))


def wrap_to_pi(q):
    """Wrap angles to [-pi, pi]."""

    q = np.asarray(q, dtype=float)
    return (q + np.pi) % (2.0 * np.pi) - np.pi


def circular_difference(q1, q2):
    """Return q1-q2 using the shortest angular difference componentwise."""

    return np.angle(np.exp(1j * (np.asarray(q1) - np.asarray(q2))))


class Analytical_IK_7DoF:
    """Analytical FK/IK for a non-offset 7-DoF S-R-S manipulator."""

    def __init__(
        self,
        alpha=dh_alpha,
        d=dh_d,
        limits_lower=dh_limits_lower,
        limits_upper=dh_limits_upper,
        numerical_tolerance=1e-12,
    ):
        alpha = np.asarray(alpha, dtype=float)
        d = np.asarray(d, dtype=float)
        limits_lower = np.asarray(limits_lower, dtype=float)
        limits_upper = np.asarray(limits_upper, dtype=float)

        if not (
            alpha.shape == (7,)
            and d.shape == (7,)
            and limits_lower.shape == (7,)
            and limits_upper.shape == (7,)
        ):
            raise ValueError(
                "alpha, d, limits_lower and limits_upper must all have "
                "shape (7,)."
            )

        if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(d)):
            raise ValueError("DH parameters must be finite.")

        # The derivation assumes a non-offset S-R-S chain.
        if not np.allclose(d[[1, 3, 5]], 0.0):
            raise ValueError("d[1], d[3] and d[5] must be zero.")

        if np.any(limits_lower > limits_upper):
            raise ValueError("Every lower joint limit must be <= its upper limit.")

        self.alpha = alpha.copy()
        self.d = d.copy()
        self.limits_lower = limits_lower.copy()
        self.limits_upper = limits_upper.copy()
        self.numerical_tolerance = float(numerical_tolerance)

        self.d_bs = float(d[0])
        self.d_se = float(d[2])
        self.d_ew = float(d[4])
        self.d_wf = float(d[6])

        if min(self.d_se, self.d_ew) <= 0.0:
            raise ValueError("d_se and d_ew must be positive.")

        # Kept as a public attribute for compatibility with the original
        # implementation.  Each function has captured its own alpha and d.
        self.Ts = tuple(
            self._make_dh_transform(ai, di)
            for ai, di in zip(self.alpha, self.d)
        )

    @staticmethod
    def _make_dh_transform(alpha, d):
        """Create one standard-DH transform function."""

        def transform(theta):
            theta = float(theta)
            ct = np.cos(theta)
            st = np.sin(theta)
            ca = np.cos(alpha)
            sa = np.sin(alpha)

            return np.array(
                [
                    [ct, -st * ca, st * sa, 0.0],
                    [st, ct * ca, -ct * sa, 0.0],
                    [0.0, sa, ca, d],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

        return transform

    @staticmethod
    def _validate_q(thetas):
        q = np.asarray(thetas, dtype=float)
        if q.shape != (7,):
            raise ValueError(f"Expected joint vector with shape (7,), got {q.shape}.")
        if not np.all(np.isfinite(q)):
            raise ValueError("Joint vector must contain only finite values.")
        return q

    @staticmethod
    def _validate_transform(rigid_transform):
        T = np.asarray(rigid_transform, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"Expected rigid_transform with shape (4, 4), got {T.shape}.")
        if not np.all(np.isfinite(T)):
            raise ValueError("rigid_transform must contain only finite values.")
        return T

    def _unit(self, vector, name):
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm <= self.numerical_tolerance:
            raise ValueError(f"Cannot normalize near-zero vector {name}.")
        return vector / norm

    @staticmethod
    def _product(transforms):
        result = np.eye(4, dtype=float)
        for transform in transforms:
            result = result @ transform
        return result

    @staticmethod
    def _validate_gc(GC):
        gc = np.asarray(GC, dtype=int)
        if gc.shape != (3,):
            raise ValueError(f"GC must have shape (3,), got {gc.shape}.")
        if not np.all(np.isin(gc, (-1, 1))):
            raise ValueError("Every GC entry must be either -1 or +1.")
        return gc

    # ------------------------------------------------------------------
    # Forward kinematics and global configuration
    # ------------------------------------------------------------------

    def FK(self, thetas):
        """Return T^0_7 using the standard-DH convention."""

        q = self._validate_q(thetas)
        return self._product(T(qi) for T, qi in zip(self.Ts, q))

    def GC(self, thetas):
        """Return (GC2, GC4, GC6) from theta_2, theta_4 and theta_6."""

        q = self._validate_q(thetas)
        return np.where(q[[1, 3, 5]] >= 0.0, 1, -1).astype(int)

    # ------------------------------------------------------------------
    # Arm-angle computation
    # ------------------------------------------------------------------

    def psi(self, thetas, return_T_vs=False):
        """Compute the arm angle psi associated with a joint configuration."""

        q = self._validate_q(thetas)
        T_07 = self.FK(q)
        p_07 = T_07[:3, 3]
        R_07 = T_07[:3, :3]

        _, GC4, _ = self.GC(q)

        p_02 = np.array([0.0, 0.0, self.d_bs])
        p_67 = np.array([0.0, 0.0, self.d_wf])
        p_26 = p_07 - p_02 - R_07 @ p_67
        p_26_hat = self._unit(p_26, "p_26")

        # Virtual manipulator, with theta_3^v = 0.
        cos_theta4v = (
            np.dot(p_26, p_26) - self.d_se**2 - self.d_ew**2
        ) / (2.0 * self.d_se * self.d_ew)
        theta_4v = GC4 * safe_arccos(cos_theta4v)

        # For this DH convention, z_1 is the base z-axis [0, 0, 1].
        cross = np.cross(p_26, np.array([0.0, 0.0, 1.0]))
        theta_1v = (
            np.arctan2(p_26[1], p_26[0])
            if np.dot(cross, cross) > self.numerical_tolerance**2
            else 0.0
        )

        cos_phi = (
            self.d_se**2 + np.dot(p_26, p_26) - self.d_ew**2
        ) / (2.0 * self.d_se * np.linalg.norm(p_26))
        phi = safe_arccos(cos_phi)
        theta_2v = np.arctan2(
            np.linalg.norm(p_26[:2]),
            p_26[2],
        ) + GC4 * phi

        theta_vs = np.array(
            [theta_1v, theta_2v, 0.0, theta_4v],
            dtype=float,
        )

        T_vs = tuple(
            T(theta)
            for T, theta in zip(self.Ts[:4], theta_vs)
        )

        T_02_v = self._product(T_vs[:2])
        T_04_v = self._product(T_vs[:4])
        p_02_v = T_02_v[:3, 3]
        p_04_v = T_04_v[:3, 3]

        Ts = tuple(T(theta) for T, theta in zip(self.Ts, q))
        T_04 = self._product(Ts[:4])
        T_06 = self._product(Ts[:6])
        p_04 = T_04[:3, 3]
        p_06 = T_06[:3, 3]

        v_se_v = self._unit(p_04_v - p_02_v, "virtual shoulder-elbow")
        v_sw_v = self._unit(p_26, "virtual shoulder-wrist")
        v_sew_v = self._unit(np.cross(v_se_v, v_sw_v), "virtual SEW normal")

        v_se = self._unit(p_04 - p_02, "shoulder-elbow")
        v_sw = self._unit(p_06 - p_02, "shoulder-wrist")
        v_sew = self._unit(np.cross(v_se, v_sw), "SEW normal")

        # Equivalent to the paper's sign(arccos(.)) expression, but atan2
        # avoids a separate sign decision at psi = 0 or pi.
        sin_psi = np.dot(np.cross(v_sew_v, v_sew), p_26_hat)
        cos_psi = np.dot(v_sew_v, v_sew)
        psi = np.arctan2(sin_psi, scalar_clip(cos_psi, -1.0, 1.0))

        if return_T_vs:
            return float(psi), T_vs
        return float(psi)

    # ------------------------------------------------------------------
    # Analytical inverse kinematics
    # ------------------------------------------------------------------

    def IK(
        self,
        rigid_transform,
        GC,
        psi,
        return_unclipped_vals=False,
        return_singularity_vals=False,
        return_sw_mats=False,
        clip_stepback=0.0,
    ):
        """Return q for a target pose, GC branch and arm angle psi.

        Parameters
        ----------
        rigid_transform : array_like, shape (4, 4)
            Desired T^0_7.
        GC : array_like, shape (3,)
            (GC2, GC4, GC6), each equal to -1 or +1.
        psi : float
            Arm angle in radians.
        clip_stepback : float, optional
            If positive, clips acos arguments to
            [-(1-clip_stepback), +(1-clip_stepback)] to stay away from
            singularities.  The default zero gives the exact nonsingular
            analytical solution, with ordinary numerical clipping to [-1, 1].

        Returns
        -------
        ndarray, shape (7,)
            Joint angles unless a diagnostic return option is selected.
        """

        T = self._validate_transform(rigid_transform)
        gc = self._validate_gc(GC)
        psi = float(psi)

        if not np.isfinite(psi):
            raise ValueError("psi must be finite.")

        if clip_stepback < 0.0 or clip_stepback >= 1.0:
            raise ValueError("clip_stepback must satisfy 0 <= clip_stepback < 1.")

        if return_unclipped_vals and return_singularity_vals:
            raise ValueError(
                "return_unclipped_vals and return_singularity_vals are mutually exclusive."
            )

        GC2, GC4, GC6 = gc
        p_07 = T[:3, 3]
        R_07 = T[:3, :3]

        p_02 = np.array([0.0, 0.0, self.d_bs])
        p_67 = np.array([0.0, 0.0, self.d_wf])

        # Equation (3): shoulder-to-wrist vector.
        p_26 = p_07 - p_02 - R_07 @ p_67
        p_26_norm = np.linalg.norm(p_26)
        if p_26_norm <= self.numerical_tolerance:
            raise ValueError("The shoulder-wrist vector p_26 is singular.")
        p_26_hat = p_26 / p_26_norm

        # The optional stepback is only for deliberate singularity avoidance.
        if clip_stepback == 0.0:
            acos_lower, acos_upper = -1.0, 1.0
        else:
            clip = 1.0 - clip_stepback
            acos_lower, acos_upper = -clip, clip

        unclipped_vals = np.zeros(4, dtype=float)
        thetas = np.zeros(7, dtype=float)

        # Virtual elbow, equations (4)-(7).
        theta_1v = np.arctan2(p_26[1], p_26[0])

        cos_phi = (
            self.d_se**2 + np.dot(p_26, p_26) - self.d_ew**2
        ) / (2.0 * self.d_se * p_26_norm)
        unclipped_vals[0] = cos_phi
        phi = safe_arccos(cos_phi, acos_lower, acos_upper)
        theta_2v = np.arctan2(
            np.linalg.norm(p_26[:2]),
            p_26[2],
        ) + GC4 * phi

        theta_4v_input = (
            np.dot(p_26, p_26) - self.d_se**2 - self.d_ew**2
        ) / (2.0 * self.d_se * self.d_ew)
        unclipped_vals[1] = theta_4v_input
        theta_4v = GC4 * safe_arccos(
            theta_4v_input,
            acos_lower,
            acos_upper,
        )
        # theta_4 does not depend on psi.  It is easy to miss this
        # assignment because theta_4 is also needed while constructing the
        # virtual manipulator below.
        thetas[3] = theta_4v

        theta_vs = np.array(
            [theta_1v, theta_2v, 0.0, theta_4v],
            dtype=float,
        )
        T_vs = tuple(
            T_i(theta_i)
            for T_i, theta_i in zip(self.Ts[:4], theta_vs)
        )
        T_03_v = self._product(T_vs[:3])
        R_03_v = T_03_v[:3, :3]

        # Equation (15).
        cross_p26 = cross_product_matrix(p_26_hat)
        A_s = cross_p26 @ R_03_v
        B_s = -(cross_p26 @ cross_p26 @ R_03_v)
        C_s = np.outer(p_26_hat, p_26_hat) @ R_03_v

        s = np.sin(psi)
        c = np.cos(psi)

        # Equations (17)-(19).
        thetas[0] = np.arctan2(
            GC2 * (A_s[1, 1] * s + B_s[1, 1] * c + C_s[1, 1]),
            GC2 * (A_s[0, 1] * s + B_s[0, 1] * c + C_s[0, 1]),
        )

        cos_theta2 = A_s[2, 1] * s + B_s[2, 1] * c + C_s[2, 1]
        unclipped_vals[2] = cos_theta2
        thetas[1] = GC2 * safe_arccos(
            cos_theta2,
            acos_lower,
            acos_upper,
        )

        thetas[2] = np.arctan2(
            GC2 * (-A_s[2, 2] * s - B_s[2, 2] * c - C_s[2, 2]),
            GC2 * (-A_s[2, 0] * s - B_s[2, 0] * c - C_s[2, 0]),
        )

        # Equation (20): wrist rotation matrix coefficients.
        R_34 = T_vs[3][:3, :3]
        A_w = R_34.T @ A_s.T @ R_07
        B_w = R_34.T @ B_s.T @ R_07
        C_w = R_34.T @ C_s.T @ R_07

        # Equations (22)-(24).
        thetas[4] = np.arctan2(
            GC6 * (A_w[1, 2] * s + B_w[1, 2] * c + C_w[1, 2]),
            GC6 * (A_w[0, 2] * s + B_w[0, 2] * c + C_w[0, 2]),
        )

        cos_theta6 = A_w[2, 2] * s + B_w[2, 2] * c + C_w[2, 2]
        unclipped_vals[3] = cos_theta6
        thetas[5] = GC6 * safe_arccos(
            cos_theta6,
            acos_lower,
            acos_upper,
        )

        # Equation (24).  Keep GC6 on both atan2 arguments: this is what
        # selects the wrist global-configuration branch.
        thetas[6] = np.arctan2(
            GC6 * (A_w[2, 1] * s + B_w[2, 1] * c + C_w[2, 1]),
            GC6 * (-A_w[2, 0] * s - B_w[2, 0] * c - C_w[2, 0]),
        )

        if return_unclipped_vals:
            return unclipped_vals

        if return_singularity_vals:
            # Preserve the original API: theta_4 and theta_6 are the two
            # hinge-type joints whose sine vanishes at the wrist/elbow
            # singular configurations.
            return thetas[[3, 5]]

        if return_sw_mats:
            return (
                thetas,
                A_s,
                B_s,
                C_s,
                A_w,
                B_w,
                C_w,
            )

        return thetas


__all__ = [
    "Analytical_IK_7DoF",
    "cross_product_matrix",
    "scalar_clip",
    "safe_arccos",
    "wrap_to_pi",
    "circular_difference",
    "dh_alpha",
    "dh_d",
    "dh_limits_lower",
    "dh_limits_upper",
]