"""
Batched standard DH kinematics.
"""

import numpy as np

class BatchDH:
    """Batched kinematics for a standard-DH serial chain of revolute joints"""

    def __init__(self, a, alpha, d, theta=None, mask=None, tool=None):
        # (a, alpha, d, theta) is one row in DH table
        self.a = np.asarray(a, dtype=np.float64)
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.d= np.asarray(d, dtype=np.float64)
        self.theta = np.zeros_like(self.a) if theta is None else np.asarray(theta, dtype=np.float64)

        self.n = len(self.a) # how many joints
        self.mask = np.ones(6, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        self.m = int(self.mask.sum())

        self.tool = np.eye(4) if tool is None else np.asarray(tool, dtype=np.float64)

        # total length
        self.L = float(np.sum(np.abs(self.a)) + np.sum(np.abs(self.d)) + np.linalg.norm(self.tool[:3, 3]))

    def link_transforms(self, q):
        """q:(B, n) -> (B,n,4,4)"""
        q = np.asarray(q, dtype=np.float64)
        B = q.shape[0] # batch
        
        # update theta given new q
        th = q + self.theta

        # cos(theta); sin(theta)
        ct, st = np.cos(th), np.sin(th)

        # cos(alpha), sin(alpha)
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)

        # batched transform matrix
        T = np.zeros((B, self.n, 4, 4))
        T[:, :, 0, 0] = ct
        T[:, :, 0, 1] = -st * ca
        T[:, :, 0, 2] = st * sa
        T[:, :, 0, 3] = self.a * ct
        T[:, :, 1, 0] = st
        T[:, :, 1, 1] = ct * ca
        T[:, :, 1, 2] = -ct * sa
        T[:, :, 1, 3] = self.a * st
        T[:, :, 2, 1] = sa
        T[:, :, 2, 2] = ca
        T[:, :, 2, 3] = self.d
        T[:, :, 3, 3] = 1.0
        return T

    def fk_frames(self, q):
        """exclude tool frame as it's not a joint and it's not in jacobian matrix"""
        A = self.link_transforms(q)
        B = q.shape[0]
        # n + 1 because we need to add identity base frame
        out = np.empty((B, self.n + 1, 4, 4))
        out[:, 0] = np.eye(4)

        # start from base frame
        T = out[:,0]

        for i in range(self.n):
            T = T @ A[:, i]
            out[:,i + 1] = T
        return out

    def fk(self, q):
        """End-effector (tool) pose.  q:(B,n) -> (B,4,4)."""
        return self.fk_frames(q)[:, -1] @ self.tool

    def jacobian(self, q, frames=None):
        """base-frame geometric jacobian of the tool point"""
        # F: (B, n+1, 4, 4)
        F = self.fk_frames(q) if frames is None else frames

        # last joint's z-axis
        # z.shape = (B, n, 3)
        z = F[:, :-1, :3, 2] 
        # last joint coordinate based on origin-frame(base-frame)
        # o.shape = (B, n, 3)
        o = F[:, :-1, :3, 3]

        # position of end-effector
        # pe.shape = (B, 1, 3)
        pe = (F[:, -1] @ self.tool)[:, :3, 3][:, None, :]

        # J_i = [z_i-1 x (pe - o), z_i-1]^T
        J = np.empty((F.shape[0], 6, self.n))
        J[:, :3, :] = np.cross(z, pe - o).transpose(0, 2, 1)
        J[:, 3:, :] = z.transpose(0, 2, 1)

        return J
    
    def masked_jacobian(self, q, frames=None):
        return self.jacobian(q, frames)[:, self.mask, :]
    
    @staticmethod
    def rot_log(R):
        """Rotation matrix -> rotation vector: (B,3,3) -> (B,3)"""
        # Rodrigue's algorithm
        tr = np.clip((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1.0) * 0.5, -1.0, 1.0)
        ang = np.arccos(tr)
        v = np.stack([R[:, 2, 1] - R[:, 1, 2],
                      R[:, 0, 2] - R[:, 2, 0],
                      R[:, 1, 0] - R[:, 0, 1]], axis=1)
        s = np.sin(ang)
        scale = np.where(s < 1e-8, 0.5, ang / np.maximum(2.0 * s, 1e-300))
        out = v * scale[:, None]
        bad = np.abs(ang - np.pi) < 1e-4
        if np.any(bad):
            Rb = R[bad]
            w = np.sqrt(np.maximum(np.diagonal(Rb, axis1=1, axis2=2) + 1.0, 0.0) * 0.5)
            sgn = np.sign(np.stack([Rb[:, 2, 1] - Rb[:, 1, 2],
                                    Rb[:, 0, 2] - Rb[:, 2, 0],
                                    Rb[:, 1, 0] - Rb[:, 0, 1]], axis=1))
            sgn[sgn == 0] = 1.0
            out[bad] = np.pi * w * sgn
        return out

    def fk_err(self, q, T_target, frames=None):
        """compute e_p and e_o"""
        # EE transform matrix
        T = (self.fk_frames(q) if frames is None else frames)[:, -1] @ self.tool

        # error (6,) 3 for position and 3 for rotation
        e = np.empty((q.shape[0], 6))

        e[:, :3] = T_target[:, :3, 3] - T[:, :3, 3]
        e[:, 3:] = self.rot_log(T_target[:, :3, :3] @ T[:, :3, :3].transpose(0, 2, 1))
        return e[:, self.mask]

    def fk_error_pct(self, q, T_target):
        """averaged fk error, based on eq(19) on the paper"""
        T = self.fk(q)

        # normalisedd ep
        ep = np.linalg.norm(T_target[:, :3, 3] - T[:, :3, 3], axis=1) / self.L

        # normalised eo
        Rerr = T_target[:, :3, :3] @ T[:, :3, :3].transpose(0, 2, 1)
        tr = np.clip((Rerr[:, 0, 0] + Rerr[:, 1, 1] + Rerr[:, 2, 2] - 1.0) * 0.5, -1, 1)
        eo = np.arccos(tr) / np.pi

        return ep, eo

    def null_vector(self, J):
        """the tagent direction of SMM is null-vector of the Jacobian"""
        # B: batch, m: task dim, n: joint dim
        B, m, n = J.shape
        # print(f"B = {B} | m = {m} | n = {n}")
        assert n == m + 1, "eq. (6) requires exactly one degree of redundancy"

        nv = np.empty((B,n))
        idx = np.arange(n)
        for i in range(n):
            nv[:, i] = ((-1.0) ** i) * np.linalg.det(J[:, :, idx != i])
        mag = np.linalg.norm(nv, axis=1)
        return nv / np.maximum(mag, 1e-300)[:, None], mag

    def sigma_min(self, J):
        """Smallest singular value of J via eigvalsh(J J^T) -- cheaper than SVD."""
        JJt = J @ J.transpose(0, 2, 1)
        w = np.linalg.eigvalsh(JJt)[:, 0]
        return np.sqrt(np.maximum(w, 0.0))

    def dls_step(self, J, e, lam):
        """one step of dampled ik"""
        JJt = J @ J.transpose(0, 2, 1)
        JJt[:, np.arange(self.m), np.arange(self.m)] += lam ** 2
        return (J.transpose(0, 2, 1) @ np.linalg.solve(JJt, e[:, :, None]))[:, :, 0]

    def ik(self, q0, T_target, iters=200, tol=1e-8, max_step=0.4):
        """
        Batched damped least-squares IK with a decreasing dampling schedule
        q0: (B, n) T_targets: (B, 4, 4) -> q: (B,n), ok (B,)
        """
        q = np.array(q0, dtype=np.float64, copy=True)
        for k in range(iters):
            F = self.fk_frames(q)
            e = self.fk_err(q, T_target, F)
            lam = 0.3 if k < 25 else (0.05 if k < 80 else 5e-3)
            dq = self.dls_step(self.masked_jacobian(q, F), e, lam)
            nrm = np.linalg.norm(dq, axis=1, keepdims=True)
            q += dq * np.minimum(1.0, max_step / np.maximum(nrm, 1e-12))
        e = self.fk_err(q, T_target)
        return q, np.linalg.norm(e, axis=1) < tol ** 0.5

    def ik_correct(self, q, T_target, iters=3, lam=1e-3):
        """Error correction of eq. (18), used after network prediction."""
        q = np.array(q, dtype=np.float64, copy=True)
        for _ in range(iters):
            F = self.fk_frames(q)
            e = self.fk_err(q, T_target, F)
            q += self.dls_step(self.masked_jacobian(q, F), e, lam)
        return q

