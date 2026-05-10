# No numba acceleration
import numpy as np
import time
from scipy.sparse.linalg import gmres, LinearOperator

def bem_setup(M, n_freq):
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    nodes = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    f = np.cos(n_freq * theta)
    idx1 = np.arange(M)
    idx2 = (idx1 + 1) % M
    elem_nodes = np.stack([nodes[idx1], nodes[idx2]], axis=1)
    # 8-point Gauss quadrature
    q_pts, q_w = np.polynomial.legendre.leggauss(8)
    return nodes, elem_nodes, f, q_pts, q_w, idx1, idx2

def bem_matvec(mu, nodes, elem_nodes, q_pts, q_w, idx1, idx2):
    M = nodes.shape[0]
    res = 0.5 * mu
    p1, p2 = elem_nodes[:, 0, :], elem_nodes[:, 1, :]
    L = np.linalg.norm(p2 - p1, axis=1)
    # Unit circle outward normals at element midpoints
    mid = (p1 + p2) / 2.0
    normals_y = mid / np.linalg.norm(mid, axis=1)[:, np.newaxis]

    phi1, phi2 = (1 - q_pts) / 2.0, (1 + q_pts) / 2.0
    nodes_sq = np.sum(nodes**2, axis=1)

    for q in range(len(q_pts)):
        # Quad point positions for all elements: (M, 2)
        y_q = p1 * phi1[q] + p2 * phi2[q]
        y_q_sq = np.sum(y_q**2, axis=1)
        mu_q = mu[idx1] * phi1[q] + mu[idx2] * phi2[q]
        jw = (L / 2.0) * q_w[q]

        # dist_sq[i, j] = |nodes[i] - y_q[j]|^2
        dist_sq = nodes_sq[:, np.newaxis] + y_q_sq[np.newaxis, :] - 2.0 * np.dot(nodes, y_q.T)

        # dot_prod[i, j] = (nodes[i] - y_q[j]) . normals_y[j]
        # (nodes . normals_y) - (y_q . normals_y)
        dot_prod = np.dot(nodes, normals_y.T) - np.sum(y_q * normals_y, axis=1)[np.newaxis, :]

        # Mask singularities (Self-element integration is analytic 0 for linear elements)
        dist_sq[dist_sq < 1e-16] = 1.0
        kernel = -1.0 / (2.0 * np.pi) * (dot_prod / dist_sq)

        # Avoid diagonal influence for self-interaction
        np.fill_diagonal(kernel, 0.0)

        res += np.dot(kernel, mu_q * jw)

    return res

def bem_evaluate_chunked(mu, nodes, elem_nodes, q_pts, q_w, eval_pts, idx1, idx2, chunk_size=2000):
    M = nodes.shape[0]
    N_eval = eval_pts.shape[0]
    u_eval = np.zeros(N_eval)
    p1, p2 = elem_nodes[:, 0, :], elem_nodes[:, 1, :]
    L = np.linalg.norm(p2 - p1, axis=1)
    mid = (p1 + p2) / 2.0
    normals_y = mid / np.linalg.norm(mid, axis=1)[:, np.newaxis]
    eval_sq = np.sum(eval_pts**2, axis=1)
    phi1, phi2 = (1 - q_pts) / 2.0, (1 + q_pts) / 2.0

    for i in range(0, N_eval, chunk_size):
        end = min(i + chunk_size, N_eval)
        chunk = eval_pts[i:end]
        c_sq = eval_sq[i:end]

        for q in range(len(q_pts)):
            y_q = p1 * phi1[q] + p2 * phi2[q]
            y_q_sq = np.sum(y_q**2, axis=1)
            mu_q = mu[idx1] * phi1[q] + mu[idx2] * phi2[q]
            jw = (L / 2.0) * q_w[q]

            dist_sq = c_sq[:, np.newaxis] + y_q_sq[np.newaxis, :] - 2.0 * np.dot(chunk, y_q.T)
            dot_prod = np.dot(chunk, normals_y.T) - np.sum(y_q * normals_y, axis=1)[np.newaxis, :]

            kernel = -1.0 / (2.0 * np.pi) * (dot_prod / dist_sq)
            u_eval[i:end] += np.dot(kernel, mu_q * jw)
    return u_eval

def run_bem(M, n_freq=3):
    t0 = time.perf_counter()
    nodes, elem_nodes, f, q_pts, q_w, idx1, idx2 = bem_setup(M, n_freq)
    setup_t = time.perf_counter() - t0

    it = {'c': 0}
    def cb(rk): it['c'] += 1

    A = LinearOperator((M, M), matvec=lambda mu: bem_matvec(mu, nodes, elem_nodes, q_pts, q_w, idx1, idx2))
    t1 = time.perf_counter()
    mu, info = gmres(A, f, atol=1e-10, restart=30, callback=cb)
    solve_t = time.perf_counter() - t1

    # Grid Evaluation
    Nx = Ny = 120
    xx, yy = np.linspace(-0.9, 0.9, Nx), np.linspace(-0.9, 0.9, Ny)
    X, Y = np.meshgrid(xx, yy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    mask = (X**2 + Y**2).ravel() < 0.9**2
    eval_pts = pts[mask]

    t2 = time.perf_counter()
    u_num = bem_evaluate_chunked(mu, nodes, elem_nodes, q_pts, q_w, eval_pts, idx1, idx2)
    eval_t = time.perf_counter() - t2

    r, th = np.sqrt(np.sum(eval_pts**2, axis=1)), np.arctan2(eval_pts[:, 1], eval_pts[:, 0])
    u_ex = (r**n_freq) * np.cos(n_freq * th)
    rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)

    return {"M": M, "iters": it['c'], "setup": setup_t, "solve": solve_t, "eval": eval_t, "l2": rel_l2}

print("No numba acceleration")
if __name__ == "__main__":
    M_list = [1000, 2000, 4000, 8000, 16000]
    print(f"{'M':>6} | {'Iters':>5} | {'Setup':>8} | {'Solve':>8} | {'Eval':>8} | {'L2 Error':>10}")
    print("-" * 65)
    for m in M_list:
        r = run_bem(m)
        print(f"{r['M']:6d} | {r['iters']:5d} | {r['setup']:8.3f} | {r['solve']:8.3f} | {r['eval']:8.3f} | {r['l2']:10.4e}")


# Numba accelerated
import numpy as np
import time
from scipy.sparse.linalg import gmres, LinearOperator
from numba import njit, prange

# -------------------------------------------------------------------------
# Numba JIT Compiled Kernels
# -------------------------------------------------------------------------

@njit(parallel=True, fastmath=True)
def bem_matvec_numba(mu, nodes, p1, p2, normals_y, L, q_pts, q_w, idx1, idx2):
    M = nodes.shape[0]
    N_q = len(q_pts)
    res = np.empty(M, dtype=np.float64)

    # Pre-calculate shape functions for quadrature points
    phi1 = (1.0 - q_pts) / 2.0
    phi2 = (1.0 + q_pts) / 2.0

    for i in prange(M):
        node_x = nodes[i, 0]
        node_y = nodes[i, 1]

        # Start with the diagonal jump term
        s = 0.5 * mu[i]

        for j in range(M):
            # Skip self-element interaction (analytic integration is 0 for linear elements)
            if i == j:
                continue

            p1x, p1y = p1[j, 0], p1[j, 1]
            p2x, p2y = p2[j, 0], p2[j, 1]
            ny_x, ny_y = normals_y[j, 0], normals_y[j, 1]

            L_j = L[j]
            mu1, mu2 = mu[idx1[j]], mu[idx2[j]]

            for q in range(N_q):
                # Interpolate geometry and unknown mu
                y_qx = p1x * phi1[q] + p2x * phi2[q]
                y_qy = p1y * phi1[q] + p2y * phi2[q]
                mu_q = mu1 * phi1[q] + mu2 * phi2[q]

                # Jacobian weight
                jw = (L_j / 2.0) * q_w[q]

                dx = node_x - y_qx
                dy = node_y - y_qy

                dist_sq = dx*dx + dy*dy
                dot_prod = dx * ny_x + dy * ny_y

                # Double layer potential kernel
                kernel = -1.0 / (2.0 * np.pi) * (dot_prod / dist_sq)
                s += kernel * mu_q * jw

        res[i] = s

    return res

@njit(parallel=True, fastmath=True)
def bem_evaluate_numba(mu, eval_pts, p1, p2, normals_y, L, q_pts, q_w, idx1, idx2):
    N_eval = eval_pts.shape[0]
    M = p1.shape[0]
    N_q = len(q_pts)
    u_eval = np.zeros(N_eval, dtype=np.float64)

    phi1 = (1.0 - q_pts) / 2.0
    phi2 = (1.0 + q_pts) / 2.0

    for i in prange(N_eval):
        pt_x = eval_pts[i, 0]
        pt_y = eval_pts[i, 1]

        s = 0.0

        for j in range(M):
            p1x, p1y = p1[j, 0], p1[j, 1]
            p2x, p2y = p2[j, 0], p2[j, 1]
            ny_x, ny_y = normals_y[j, 0], normals_y[j, 1]

            L_j = L[j]
            mu1, mu2 = mu[idx1[j]], mu[idx2[j]]

            for q in range(N_q):
                y_qx = p1x * phi1[q] + p2x * phi2[q]
                y_qy = p1y * phi1[q] + p2y * phi2[q]
                mu_q = mu1 * phi1[q] + mu2 * phi2[q]

                jw = (L_j / 2.0) * q_w[q]

                dx = pt_x - y_qx
                dy = pt_y - y_qy

                dist_sq = dx*dx + dy*dy
                dot_prod = dx * ny_x + dy * ny_y

                kernel = -1.0 / (2.0 * np.pi) * (dot_prod / dist_sq)
                s += kernel * mu_q * jw

        u_eval[i] = s

    return u_eval

# -------------------------------------------------------------------------
# Setup & Main Execution
# -------------------------------------------------------------------------

def bem_setup(M, n_freq):
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    nodes = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    f = np.cos(n_freq * theta)
    idx1 = np.arange(M)
    idx2 = (idx1 + 1) % M
    elem_nodes = np.stack([nodes[idx1], nodes[idx2]], axis=1)

    # 8-point Gauss quadrature
    q_pts, q_w = np.polynomial.legendre.leggauss(8)
    return nodes, elem_nodes, f, q_pts, q_w, idx1, idx2

def run_bem(M, n_freq=3):
    t0 = time.perf_counter()
    nodes, elem_nodes, f, q_pts, q_w, idx1, idx2 = bem_setup(M, n_freq)

    # Pre-calculate element properties outside the solver loop
    p1 = elem_nodes[:, 0, :]
    p2 = elem_nodes[:, 1, :]
    L = np.linalg.norm(p2 - p1, axis=1)
    mid = (p1 + p2) / 2.0
    normals_y = mid / np.linalg.norm(mid, axis=1)[:, np.newaxis]

    setup_t = time.perf_counter() - t0

    it = {'c': 0}
    def cb(rk):
        it['c'] += 1

    # Pass Numba kernel into LinearOperator
    A = LinearOperator(
        (M, M),
        matvec=lambda mu: bem_matvec_numba(mu, nodes, p1, p2, normals_y, L, q_pts, q_w, idx1, idx2)
    )

    t1 = time.perf_counter()
    mu, info = gmres(A, f, atol=1e-10, restart=30, callback=cb)
    solve_t = time.perf_counter() - t1

    # Grid Evaluation Setup
    Nx = Ny = 120
    xx, yy = np.linspace(-0.9, 0.9, Nx), np.linspace(-0.9, 0.9, Ny)
    X, Y = np.meshgrid(xx, yy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    mask = (X**2 + Y**2).ravel() < 0.9**2
    eval_pts = pts[mask]

    t2 = time.perf_counter()
    # Evaluate directly using the accelerated kernel (chunking is naturally handled by Numba's threading)
    u_num = bem_evaluate_numba(mu, eval_pts, p1, p2, normals_y, L, q_pts, q_w, idx1, idx2)
    eval_t = time.perf_counter() - t2

    r = np.sqrt(np.sum(eval_pts**2, axis=1))
    th = np.arctan2(eval_pts[:, 1], eval_pts[:, 0])
    u_ex = (r**n_freq) * np.cos(n_freq * th)
    rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)

    return {"M": M, "iters": it['c'], "setup": setup_t, "solve": solve_t, "eval": eval_t, "l2": rel_l2}

if __name__ == "__main__":
    # Numba Warmup: Compile the JIT kernels on a tiny problem to prevent skewing timings
    print("Compiling Numba Kernels...")
    run_bem(10, n_freq=3)
    print("Compilation complete.\n")

    M_list = [1000, 2000, 4000, 8000, 16000]
    print(f"{'M':>6} | {'Iters':>5} | {'Setup (s)':>10} | {'Solve (s)':>10} | {'Eval (s)':>10} | {'L2 Error':>10}")
    print("-" * 75)
    for m in M_list:
        r = run_bem(m)
        print(f"{r['M']:6d} | {r['iters']:5d} | {r['setup']:10.3f} | {r['solve']:10.3f} | {r['eval']:10.3f} | {r['l2']:10.4e}")