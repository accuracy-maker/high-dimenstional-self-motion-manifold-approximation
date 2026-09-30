"""
test for single target pose query

"""
from model.analytic_ik.analytic_ik_7dof import Analytical_IK_7DoF
import numpy as np

ik_solver = Analytical_IK_7DoF()

q_true = np.array([
    0.30,
    -0.40,
    0.50,
    0.70,
    -0.60,
    0.40,
    0.20,
])


# forward kinematics
T_true = ik_solver.FK(q_true)
gc_true = ik_solver.GC(q_true)
psi_true = ik_solver.psi(q_true)

print(f"T ture:\n {T_true}")
print(f"GC ture:\n {gc_true}")
print(f"psi ture:\n {psi_true}")

# inverse kinematics
q_recovered = ik_solver.IK(
    T_true,
    GC=gc_true,
    psi=psi_true
)

print(f"q recovered:\n {q_recovered}")
