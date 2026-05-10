import numpy as np
import time
from scipy.linalg import solve

def solve_bem(n_per_side):
    t_start = time.time()

    # 1. Geometry Discretization (Rectangle [0,2]x[0,1])
    L, H_dim = 2.0, 1.0
    n_elements = 4 * n_per_side

    # Define corners and generate segments
    # Order: Bottom (y=0), Right (x=2), Top (y=1), Left (x=0)
    nodes_x = np.concatenate([
        np.linspace(0, L, n_per_side, endpoint=False),
        np.full(n_per_side, L),
        np.linspace(L, 0, n_per_side, endpoint=False),
        np.zeros(n_per_side)
    ])
    nodes_y = np.concatenate([
        np.zeros(n_per_side),
        np.linspace(0, H_dim, n_per_side, endpoint=False),
        np.full(n_per_side, H_dim),
        np.linspace(H_dim, 0, n_per_side, endpoint=False)
    ])

    next_idx = (np.arange(n_elements) + 1) % n_elements
    x1, y1 = nodes_x, nodes_y
    x2, y2 = nodes_x[next_idx], nodes_y[next_idx]

    # Midpoints, Lengths, and Outward Normals
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
    lengths = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    nx = (y2 - y1) / lengths
    ny = -(x2 - x1) / lengths

    # Indices for BCs
    idx_bot, idx_rhs, idx_top, idx_lhs = np.split(np.arange(n_elements), 4)
    dir_idx = np.concatenate([idx_rhs, idx_lhs])
    neu_idx = np.concatenate([idx_bot, idx_top])

    # 2. Assembly of Influence Matrices G and H
    t_asm_start = time.time()
    # Compute distance matrices
    DX = xm[:, np.newaxis] - xm[np.newaxis, :]
    DY = ym[:, np.newaxis] - ym[np.newaxis, :]
    R2 = DX**2 + DY**2
    R = np.sqrt(R2 + np.eye(n_elements)) # Avoid log(0)

    # Single Layer Potential G
    G = -(1 / (2 * np.pi)) * np.log(R) * lengths
    # Double Layer Potential H
    H = (1 / (2 * np.pi * R2 + np.eye(n_elements))) * (DX * nx + DY * ny) * lengths

    # Self-integrals (Analytic)
    np.fill_diagonal(G, (lengths / (2 * np.pi)) * (1 - np.log(lengths / 2)))
    np.fill_diagonal(H, 0.5) # c_i jump term for smooth boundary

    t_asm = time.time() - t_asm_start

    # 3. Apply Boundary Conditions and Solve
    # Knowns
    u_bc = np.zeros(n_elements)
    q_bc = np.zeros(n_elements) # Neumann q=0 on top/bottom
    u_bc[idx_rhs] = np.cos(np.pi * ym[idx_rhs]) # u(2,y) = cos(pi*y)
    # u_bc[idx_lhs] is 0

    # Rearrange A x = B
    # Unknowns: q on Dirichlet (idx_rhs, idx_lhs), u on Neumann (idx_bot, idx_top)
    A = np.zeros((n_elements, n_elements))
    A[:, :len(dir_idx)] = G[:, dir_idx]
    A[:, len(dir_idx):] = -H[:, neu_idx]

    rhs = H[:, dir_idx] @ u_bc[dir_idx] - G[:, neu_idx] @ q_bc[neu_idx]

    t_solve_start = time.time()
    sol = solve(A, rhs)
    t_solve = time.time() - t_solve_start

    # Reconstruct full boundary vectors
    u_full, q_full = np.zeros(n_elements), np.zeros(n_elements)
    q_full[dir_idx], u_full[neu_idx] = sol[:len(dir_idx)], sol[len(dir_idx):]
    u_full[dir_idx], q_full[neu_idx] = u_bc[dir_idx], q_bc[neu_idx]

    # 4. Interior Evaluation
    t_eval_start = time.time()
    nxg, nyg = 40, 20
    xs = np.linspace(0.05, 1.95, nxg)
    ys = np.linspace(0.05, 0.95, nyg)
    XX, YY = np.meshgrid(xs, ys)
    pts_x, pts_y = XX.ravel(), YY.ravel()

    # Vectorized interior evaluation: u = integral(H*u) - integral(G*q)
    DX_int = pts_x[:, np.newaxis] - xm[np.newaxis, :]
    DY_int = pts_y[:, np.newaxis] - ym[np.newaxis, :]
    R2_int = DX_int**2 + DY_int**2

    G_int = -(1 / (2 * np.pi)) * np.log(np.sqrt(R2_int)) * lengths
    H_int = (1 / (2 * np.pi * R2_int)) * (DX_int * nx + DY_int * ny) * lengths

    u_num_flat = H_int @ u_full - G_int @ q_full
    u_num = u_num_flat.reshape(XX.shape)

    # Exact solution and error
    u_exact = (np.sinh(np.pi * XX) / np.sinh(2 * np.pi)) * np.cos(np.pi * YY)
    rel_l2 = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
    t_eval = time.time() - t_eval_start

    return {
        "dof": n_elements, "t_asm": t_asm, "t_solve": t_solve,
        "t_eval": t_eval, "t_total": time.time() - t_start, "error": rel_l2
    }

# Execution and Refinement Study
Ns = [20, 40, 80, 640]
results = []
print(f"{'DOFs':<6} | {'Asm(s)':<8} | {'Solv(s)':<8} | {'Eval(s)':<8} | {'Total(s)':<8} | {'L2 Error':<10} | {'Rate'}")
print("-" * 80)

for i, n in enumerate(Ns):
    res = solve_bem(n)
    rate = np.log(results[-1]['error']/res['error'])/np.log(Ns[i]/Ns[i-1]) if i > 0 else 0.0
    print(f"{res['dof']:<6} | {res['t_asm']:<8.3f} | {res['t_solve']:<8.3f} | {res['t_eval']:<8.3f} | "
          f"{res['t_total']:<8.3f} | {res['error']:<10.4e} | {rate:.2f}")
    results.append(res)