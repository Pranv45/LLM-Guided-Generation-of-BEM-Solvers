import numpy as np
import time
from scipy.linalg import solve
from scipy.sparse.linalg import gmres

def solve_bem_linear(m_per_side):
    """
    2D BEM Solver with Linear Elements for Laplace Equation.
    Domain: [0, 2] x [0, 1]
    """
    start_time = time.time()

    # --- 1. Geometry & Mesh ---
    L, H_dim = 2.0, 1.0
    nodes_x = np.concatenate([
        np.linspace(0, L, m_per_side, endpoint=False),
        np.full(m_per_side, L),
        np.linspace(L, 0, m_per_side, endpoint=False),
        np.zeros(m_per_side)
    ])
    nodes_y = np.concatenate([
        np.zeros(m_per_side),
        np.linspace(0, H_dim, m_per_side, endpoint=False),
        np.full(m_per_side, H_dim),
        np.linspace(H_dim, 0, m_per_side, endpoint=False)
    ])

    coords = np.stack([nodes_x, nodes_y], axis=1)
    n_nodes = len(coords)
    elements = np.stack([np.arange(n_nodes), (np.arange(n_nodes) + 1) % n_nodes], axis=1)

    # --- 2. Boundary Conditions ---
    is_dirichlet = np.zeros(n_nodes, dtype=bool)
    idx_rhs = np.arange(m_per_side, 2 * m_per_side)
    idx_lhs = np.arange(3 * m_per_side, 4 * m_per_side)
    is_dirichlet[idx_rhs] = True
    is_dirichlet[idx_lhs] = True
    # Corners: resolve mixed BCs (Dirichlet takes precedence)
    is_dirichlet[[0, m_per_side, 2*m_per_side, 3*m_per_side]] = True

    u_fixed = (np.sinh(np.pi * coords[:,0]) / np.sinh(2*np.pi)) * np.cos(np.pi * coords[:,1])
    q_fixed = np.zeros(n_nodes)

    # --- 3. Quadrature & Assembly ---
    xi, wi = np.polynomial.legendre.leggauss(8)
    phi = np.stack([(1 - xi) / 2, (1 + xi) / 2], axis=1)

    t_asm_start = time.time()
    G, H = np.zeros((n_nodes, n_nodes)), np.zeros((n_nodes, n_nodes))

    for e_idx in range(n_nodes):
        n1, n2 = elements[e_idx]
        p1, p2 = coords[n1], coords[n2]
        vec = p2 - p1
        L_e = np.linalg.norm(vec)
        detJ = L_e / 2
        n_vec = np.array([vec[1], -vec[0]]) / L_e
        pts_e = phi @ np.stack([p1, p2])

        for i in range(n_nodes):
            r_vec = pts_e - coords[i]
            r2 = np.sum(r_vec**2, axis=1)
            dist = np.sqrt(r2)

            g_ker = -(1/(2*np.pi)) * np.log(dist + 1e-18)
            h_ker = (1/(2*np.pi*(r2 + 1e-18))) * (r_vec @ n_vec)
            if i in [n1, n2]: h_ker = np.zeros_like(wi)

            G[i, n1] += np.sum(phi[:,0] * g_ker * wi) * detJ
            G[i, n2] += np.sum(phi[:,1] * g_ker * wi) * detJ
            H[i, n1] += np.sum(phi[:,0] * h_ker * wi) * detJ
            H[i, n2] += np.sum(phi[:,1] * h_ker * wi) * detJ

    np.fill_diagonal(H, np.diag(H) + 0.5)
    t_asm = time.time() - t_asm_start

    # --- 4. Matrix Setup & Solve ---
    A = np.zeros((n_nodes, n_nodes))
    rhs_vec = np.zeros(n_nodes)
    for j in range(n_nodes):
        if is_dirichlet[j]:
            A[:, j] = -G[:, j]
            rhs_vec -= H[:, j] * u_fixed[j]
        else:
            A[:, j] = H[:, j]
            rhs_vec += G[:, j] * q_fixed[j]

    t_solve_start = time.time()
    # Fixed: using atol instead of tol for modern SciPy
    sol, _ = gmres(A, rhs_vec, atol=1e-10)
    t_solve = time.time() - t_solve_start

    u_res, q_res = u_fixed.copy(), q_fixed.copy()
    u_res[~is_dirichlet] = sol[~is_dirichlet]
    q_res[is_dirichlet] = sol[is_dirichlet]

    # --- 5. Interior Evaluation ---
    t_eval_start = time.time()
    nx_g, ny_g = 40, 20
    xs = np.linspace(0.05, 1.95, nx_g)
    ys = np.linspace(0.05, 0.95, ny_g)
    XX, YY = np.meshgrid(xs, ys)
    pts_int = np.stack([XX.ravel(), YY.ravel()], axis=1)
    u_int = np.zeros(len(pts_int))

    # High-performance vectorized evaluation
    for e_idx in range(n_nodes):
        n1, n2 = elements[e_idx]
        p1, p2 = coords[n1], coords[n2]
        L_e = np.linalg.norm(p2 - p1)
        detJ = L_e / 2
        n_vec = np.array([(p2-p1)[1], -(p2-p1)[0]]) / L_e
        pts_e = phi @ np.stack([p1, p2])

        u_vals = (phi[:,0]*u_res[n1] + phi[:,1]*u_res[n2]) * (wi * detJ)
        q_vals = (phi[:,0]*q_res[n1] + phi[:,1]*q_res[n2]) * (wi * detJ)

        # Vectorized over grid points k
        for q_p in range(len(wi)):
            rv = pts_e[q_p] - pts_int
            r2 = np.sum(rv**2, axis=1)
            # H contribution: (r_vec . n) / (2*pi*r^2)
            u_int += ((rv @ n_vec) / (2*np.pi*r2)) * u_vals[q_p]
            # G contribution: -log(r) / (2*pi)
            u_int -= (-(1/(2*np.pi)) * np.log(np.sqrt(r2))) * q_vals[q_p]

    u_ex = (np.sinh(np.pi * pts_int[:,0]) / np.sinh(2*np.pi)) * np.cos(np.pi * pts_int[:,1])
    rel_err = np.linalg.norm(u_int - u_ex) / np.linalg.norm(u_ex)

    return n_nodes, t_asm, t_solve, time.time() - t_eval_start, rel_err

# --- Refinement Study ---
M_list = [10, 20, 40, 80, 160]
errors = []
print(f"{'Nodes':<8} | {'Asm(s)':<8} | {'Sol(s)':<8} | {'Eval(s)':<8} | {'L2 Error':<10} | {'Rate'}")
print("-" * 72)

for m in M_list:
    n, ta, ts, te, err = solve_bem_linear(m)
    rate = np.log(errors[-1]/err)/np.log(2) if errors else 0.0
    errors.append(err)
    print(f"{n:<8} | {ta:<8.3f} | {ts:<8.3f} | {te:<8.3f} | {err:<10.4e} | {rate:.2f}")