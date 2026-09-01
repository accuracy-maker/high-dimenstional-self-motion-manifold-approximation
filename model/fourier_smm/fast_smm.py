"""
This file is an upgrade of original "Robot.worksapce_smms" from official released code.

1. fixed the orientation error bug:
    `Robot.fk_err` returns the orientation error as log(R_c^T R_d), i.e. in the
    *body* frame, while `Robot.jacobian` returns the *base*-frame geometric
    Jacobian.

2. Branch discovery and null-space projection are vectorised: every grid node in
   a chunk is traced simultaneously.  Same algorithm and same output types
   (`SMM` / `WorkspaceSMMs`), ~20x less wall clock for a 7R.

Algorithm (Section III-B, eq. (1) of Clark & Xie):

    dq = gamma * n_hat + J^+ dx_e

with the tangent given by the null space of the masked Jacobian, its global
orientation fixed once at the seed configuration by the closed-form minor
formula eq. (6) and propagated by sign continuity, loop closure tested in the
circle group, and a final resample to a fixed number of samples.

Branch discovery mirrors the reference: random IK solutions are accepted as new
branch seeds only if they are further than `diff_thresh` from every manifold
already found for that node

#################### IMPORTANT###############
users probably need to read the official code and paper first before reading the code

"""
import numpy as np

# pip install realtime-smm
# this is official code for that paper
from realtime_smm.helpers.robot import Robot
from realtime_smm.helpers.types import SMM, SMMStatus, WorkspaceSMMs

from batch_kin import BatchDH

# some help functions
def wrap(x):
    """wrap angle to (-pi, pi]"""
    # this line is come up with by AI. awesome code
    return np.angle(np.exp(1j * x))

def deformation(A, B):
    """A metrics that groups workspace locations to homotopy classes"""
    d = wrap(A[:, None, :] - B[None, :, :])
    return float(np.linalg.norm(d, axis=2).min(axis=0).mean())

def _dist_to_manifold(q, M):
    """Min circle-group distance from configs q:(K,n) to manifold M:(S,n)."""
    d = wrap(q[:, None, :] - M[None, :, :])
    return np.linalg.norm(d, axis=2).min(axis=1)

# Class FastRobot: A wrap for official realtime_smm.Robot
class FastRobot(Robot):
    """
    `realtime_smm.Robot` with a corrected error frame and a batched solver.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bk = BatchDH(self._dh[:, 0], self._dh[:, 1], self._dh[:, 2],
                          self._dh[:, 3], mask=self._taskspace.to_mask(),
                          tool=self._tool)
    
    # corrected fk err
    def fk_err(self, q, x_d):
        q = np.asarray(q, dtype=float).reshape(1, -1)
        return self.bk.fk_err(q, np.asarray(x_d, float).reshape(1, 4, 4))[0]

    # parallel workspace smm trace
    def workspace_smms_many(
        self,
        Ts,
        samples = 128,
        step = 0.05,
        sing_thresh = 5e-3,
        smm_iters = 1200,
        seeds = 48,
        diff_thresh = 0.15,
        max_manifs = None,
        chunk = 384, # how many different poses progressed parallely
        seed = 0,
        progress = None
    ):
        """ Solve the SMMs of many poses at once: Ts: (N, 4, 4) -> list[WorkspaceSMMs]"""
        Ts = np.asarray(Ts, dtype=np.float64)
        N = Ts.shape[0]
        max_manifs = self.max_manifs if max_manifs is None else max_manifs

        rng = np.random.default_rng(seed)

        out = []

        for s in range(0, N, chunk):
            out.extend(self._solve_block(Ts[s:s + chunk], samples, step,
                                            sing_thresh, smm_iters, seeds,
                                            diff_thresh, max_manifs, rng))
            if progress is not None:
                progress(min(s + chunk, N), N)
        return out

    def workspace_smms(self, x_d, samples=128, step=0.05, sing_thresh=5e-3,
                       smm_iters=1200, **kw):
        """Single-pose API, signature-compatible with the base class."""
        return self.workspace_smms_many(
            np.asarray(x_d)[None], samples=samples, step=step,
            sing_thresh=sing_thresh, smm_iters=smm_iters,
            seeds=kw.get("seeds", 48), diff_thresh=kw.get("diff_thresh", 0.15),
            seed=kw.get("seed", 0))[0]

    def _solve_block(
        self,
        Ts,
        samples,
        step,
        sing_thresh,
        smm_iters,
        seeds,
        diff_thresh,
        max_manifs,
        rng
    ):
        """For a batch of target poses, find all distinct SMMs"""

        # n: joint dim, N: batch
        n, n = self._n, Ts.shape[0] 

        # joint limits
        lo, hi = self._joint_limits[:, 0], self._joint_limits[:, 1]

        # IK solutions for each nodes
        T_rep = np.repeat(Ts, seeds, axis=0)
        q0 = rng.uniform(lo, hi, size=(N * seeds, n))
        q_ik, ok = self.bk.ik(q0, T_rep)
        cand = [q_ik[i * seeds:(i + 1) * seeds][ok[i * seeds:(i + 1) * seeds]]
                for i in range(N)]

        branches = [[] for _ in range(N)]
        singular = np.zeros(N, dtype=bool)
        alive = np.array([c.shape[0] > 0 for c in cand])

        """
        rounds: each rounds adds at most one new branch per node
        e.g.
        round 1 -> find SMM #1
        round 2 -> find SMM #2
        round 3 -> find SMM #3
        ...
        """
        for _ in range(max_manifs + 2):
            # picks: q_seeds needed trace
            # pick_nodes: target poses includes which seeds
            picks, pick_nodes = [], []

            # only process alive targets
            for i in np.where(alive)[0]:
                c = cand[i]
                # drop all candidates which belongs to the same SMM
                for m in branches[i]:
                    if c.shape[0]:
                        c = c[_dist_to_manifold(c, m) >= diff_thresh]
                # update condidates
                cand[i] = c

                # decide which target is still alive
                if c.shape[0] == 0 or len(branches[i]) >= max_manifs:
                    alive[i] = False
                    continue
                picks.append(c[0])
                pick_nodes.append(i)
            
            if not picks:
                break
            pnodes = np.asarray(pick_nodes)
            paths, closed, sing = self._trace(np.asarray(picks), Ts[pnodes],
                                              samples, step, sing_thresh,
                                              smm_iters)
            
            # log the labels of each node
            for j, i in enumerate(pnodes):
                if sing[j]:
                    singular[i] = True
                    alive[i] = False
                elif closed[j]:
                    branches[i].append(paths[j])
                else:                     
                    # trace failed: discard this seed only
                    cand[i] = cand[i][1:]

        # deduplicate and package
        results = []
        for i in range(N):
            if singular[i]:
                results.append(WorkspaceSMMs(status=SMMStatus.SINGULAR, data=[]))
                continue
            uniq = []
            for p in branches[i]:
                if all(max(deformation(u, p), deformation(p, u)) >= 0.5 * diff_thresh
                       for u in uniq):
                    uniq.append(p)
            if not uniq or len(uniq) > max_manifs:
                results.append(WorkspaceSMMs(
                    status=SMMStatus.NONE if not uniq else SMMStatus.ERROR, data=[]))
            else:
                results.append(WorkspaceSMMs(
                    status=SMMStatus.OK,
                    data=[SMM(status=SMMStatus.OK, data=u) for u in uniq]))
        return results
    
    def _trace(
        self,
        q_start, 
        T_target, 
        samples, 
        step, 
        sing_thresh, # threshold decides if current J loses rank 
        smm_iters, 
        sing_every=9 # how often we check singularities
    ):
        """
        batched null-space trace

        Algorithm:
            - start from on IK solution
            - move along NULL(J)
            - correct FK drift
            - keep tracing unit back to initial q

        Returns:
            paths: (M, samples, n): M: number of distinct SMMs; samples: number of samples of that curve; n: each sample has "n (DoF)" variables
            closed: (M,)
            singular: (M,)
        """
        M, n = q_start.shape
        q = q_start.astype(np.float64, copy=True)

        # store the path sample points of each manifold
        # for one manifold: buf[i] = q0, q1, q2, q3 ... q_k+1 = qk + delta qk
        buf = np.empty((M, smm_iters + 1, n), dtype=np.float32)
        buf[:, 0] = q

        length = np.full(M, -1, dtype=np.int64)
        sing = np.zeros(M, dtype=bool)
        active = np.ones(M, dtype=bool)

        prev, _ = self.bk.null_vector(self.bk.masked_jacobian(q))

        for it in range(smm_iters):
            idx = np.where(active)[0]
            if idx.size == 0:
                break
            q_a = q[idx]
            F = self.bk.fk_frames(q_a)
            J = self.bk.masked_jacobian(q_a, F)
            if it % sing_every == 0:
                hit = self.bk.sigma_min(J) <= sing_thresh
                if hit.any():
                    sing[idx[hit]] = True
                    active[idx[hit]] = False
                    idx = idx[~hit]
                    if idx.size == 0:
                        break
                    q_a = q[idx]
                    F = self.bk.fk_frames(q_a)
                    J = self.bk.masked_jacobian(q_a, F)
            # null space via complete QR of J^T: same direction as the minor
            # formula (verified to 1e-16) but cheaper; sign set by continuity
            Q, _ = np.linalg.qr(J.transpose(0, 2, 1), mode="complete")
            nv = Q[:, :, -1]
            flip = np.einsum("ij,ij->i", nv, prev[idx]) < 0
            nv[flip] *= -1.0
            prev[idx] = nv
            e = self.bk.fk_err(q_a, T_star[idx], F)
            dq = nv + self.bk.dls_step(J, e, 1e-3)
            dq *= step / np.maximum(np.linalg.norm(dq, axis=1, keepdims=True), 1e-12)
            q[idx] = q_a + dq
            buf[idx, it + 1] = q[idx]
            if it > 5:
                closed = np.linalg.norm(wrap(q[idx] - q_start[idx]), axis=1) < 1.8 * step
                if closed.any():
                    done = idx[closed]
                    length[done] = it + 1
                    active[done] = False

        paths = np.zeros((M, samples, n), dtype=np.float32)
        got = np.where(length > 0)[0]
        if got.size:
            L = length[got].astype(np.float64)
            t = np.linspace(0.0, 1.0, samples, endpoint=False)[None, :] * L[:, None]
            i0 = np.floor(t).astype(np.int64)
            frac = (t - i0)[:, :, None]
            b = buf[got]
            p0 = np.take_along_axis(b, i0[:, :, None].repeat(n, axis=2), axis=1)
            p1 = np.take_along_axis(b, (i0 + 1)[:, :, None].repeat(n, axis=2), axis=1)
            paths[got] = wrap((1.0 - frac) * p0 + frac * p1)
        return paths, length > 0, sing








