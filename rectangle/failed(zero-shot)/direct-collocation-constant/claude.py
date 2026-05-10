import numpy as np
from scipy.linalg import solve
import time

def run_bem(N_per_side):
    t0 = time.time()

    N = 4 * N_per_side

    mids    = np.zeros((N, 2))
    normals = np.zeros((N, 2))
    lengths = np.zeros(N)
    bc_type = []
    bc_val  = np.zeros(N)

    # Bottom: y=0, x: 0->2, n=(0,-1), Neumann q=0
    for i in range(N_per_side):
        x = (i + 0.5) * 2.0 / N_per_side
        mids[i]    = [x, 0.0]
        normals[i] = [0.0, -1.0]
        lengths[i] = 2.0 / N_per_side
        bc_type.append('N')
        bc_val[i]  = 0.0

    # Right: x=2, y: 0->1, n=(1,0), Dirichlet u=cos(pi*y)
    for i in range(N_per_side):
        y   = (i + 0.5) / N_per_side
        idx = N_per_side + i
        mids[idx]    = [2.0, y]
        normals[idx] = [1.0, 0.0]
        lengths[idx] = 1.0 / N_per_side
        bc_type.append('D')
        bc_val[idx]  = np.cos(np.pi * y)

    # Top: y=1, x: 2->0, n=(0,1), Neumann q=0
    for i in range(N_per_side):
        x   = 2.0 - (i + 0.5) * 2.0 / N_per_side
        idx = 2 * N_per_side + i
        mids[idx]    = [x, 1.0]
        normals[idx] = [0.0, 1.0]
        lengths[idx] = 2.0 / N_per_side
        bc_type.append('N')
        bc_val[idx]  = 0.0

    # Left: x=0, y: 1->0, n=(-1,0), Dirichlet u=0
    for i in range(N_per_side):
        y   = 1.0 - (i + 0.5) / N_per_side
        idx = 3 * N_per_side + i
        mids[idx]    = [0.0, y]
        normals[idx] = [-1.0, 0.0]
        lengths[idx] = 1.0 / N_per_side
        bc_type.append('D')
        bc_val[idx]  = 0.0

    bc_type = np.array(bc_type)

    # ------------------------------------------------------------------ #
    # Assemble G and H using vectorised operations
    # ------------------------------------------------------------------ #
    t_assemble_start = time.time()

    # Difference vectors: shape (N, N, 2)  [row=colloc, col=panel]
    diff = mids[:, None, :] - mids[None, :, :]          # (N,N,2)
    dist2 = diff[:, :, 0]**2 + diff[:, :, 1]**2         # (N,N)

    # Off-diagonal G and H (midpoint quadrature * panel length)
    with np.errstate(divide='ignore', invalid='ignore'):
        logr   = np.where(dist2 > 0, np.log(np.sqrt(dist2)), 0.0)
        rdotn  = diff[:, :, 0] * normals[None, :, 0] + diff[:, :, 1] * normals[None, :, 1]
        H_kern = np.where(dist2 > 0, rdotn / dist2, 0.0)

    G = -1.0 / (2 * np.pi) * logr  * lengths[None, :]
    H = -1.0 / (2 * np.pi) * H_kern * lengths[None, :]

    # Diagonal self-terms
    h_self = lengths / 2.0
    np.fill_diagonal(G, -1.0 / (2 * np.pi) * 2 * h_self * (np.log(h_self) - 1.0))
    np.fill_diagonal(H, 0.0)

    # Jump term c_i = 1/2
    H += 0.5 * np.eye(N)

    t_assemble = time.time() - t_assemble_start

    # ------------------------------------------------------------------ #
    # Block assembly and solve
    # ------------------------------------------------------------------ #
    t_solve_start = time.time()

    dirichlet_idx = np.where(bc_type == 'D')[0]
    neumann_idx   = np.where(bc_type == 'N')[0]
    nD, nN        = len(dirichlet_idx), len(neumann_idx)

    # [H[:,neumann] | -G[:,dirichlet]] * [u_N; q_D] = G[:,neumann]*q_N - H[:,dirichlet]*u_D
    A = np.empty((N, N))
    A[:, :nN]  =  H[:, neumann_idx]
    A[:, nN:]  = -G[:, dirichlet_idx]

    rhs = (  G[:, neumann_idx]   @ bc_val[neumann_idx]
           - H[:, dirichlet_idx] @ bc_val[dirichlet_idx])

    x = solve(A, rhs, assume_a='gen')

    u_full = np.zeros(N)
    q_full = np.zeros(N)
    u_full[dirichlet_idx] = bc_val[dirichlet_idx]
    u_full[neumann_idx]   = x[:nN]
    q_full[neumann_idx]   = bc_val[neumann_idx]
    q_full[dirichlet_idx] = x[nN:]

    t_solve = time.time() - t_solve_start

    # ------------------------------------------------------------------ #
    # Interior evaluation on 40x20 grid
    # ------------------------------------------------------------------ #
    t_eval_start = time.time()

    nxg, nyg = 40, 20
    xs = np.linspace(0.05, 1.95, nxg)
    ys = np.linspace(0.05, 0.95, nyg)
    XX, YY = np.meshgrid(xs, ys)
    u_num = np.zeros_like(XX)

    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)   # (800, 2)

    # Vectorised: diff_int shape (n_pts, N, 2)
    diff_int  = pts[:, None, :] - mids[None, :, :]      # (800, N, 2)
    dist2_int = diff_int[:, :, 0]**2 + diff_int[:, :, 1]**2   # (800, N)
    logr_int  = np.log(np.sqrt(dist2_int))               # (800, N)
    rdotn_int = (diff_int[:, :, 0] * normals[None, :, 0]
               + diff_int[:, :, 1] * normals[None, :, 1])     # (800, N)

    G_int = -1.0 / (2 * np.pi) * logr_int  * lengths[None, :]  # (800, N)
    H_int = -1.0 / (2 * np.pi) * (rdotn_int / dist2_int) * lengths[None, :]

    u_vec  = H_int @ u_full - G_int @ q_full            # (800,)
    u_num  = u_vec.reshape(nyg, nxg)

    t_eval = time.time() - t_eval_start
    t_total = time.time() - t0

    # ------------------------------------------------------------------ #
    # Error
    # ------------------------------------------------------------------ #
    u_exact = (np.sinh(np.pi * XX) / np.sinh(2 * np.pi)) * np.cos(np.pi * YY)
    l2_err  = (np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact))

    return N, t_assemble, t_solve, t_eval, t_total, l2_err


# ------------------------------------------------------------------ #
# Refinement study
# ------------------------------------------------------------------ #
print(f"{'N/side':>6} {'DOFs':>6} {'Assemble(s)':>12} {'Solve(s)':>10} "
      f"{'Eval(s)':>8} {'Total(s)':>9} {'L2 err':>12} {'Rate':>6}")
print("-" * 82)

prev_err = None
for N_per_side in [20, 40, 80, 640]:
    N_dof, t_a, t_s, t_e, t_tot, err = run_bem(N_per_side)
    rate = f"{np.log2(prev_err / err):.2f}" if prev_err is not None else "  —"
    print(f"{N_per_side:>6} {N_dof:>6} {t_a:>12.4f} {t_s:>10.4f} {t_e:>8.4f} "
          f"{t_tot:>9.4f} {err:>12.2e} {rate:>6}")
    prev_err = err