import numpy as np
import time
from scipy.sparse.linalg import gmres
from numpy.polynomial.legendre import leggauss
from numba import njit, prange

# ============================================================
# Numba kernels
# ============================================================

@njit(cache=True, fastmath=True)
def assemble_matrix_numba(nodes, normals, elem_nodes, quad_pts, quad_w, phi1, phi2, elem_len, f):
    M = nodes.shape[0]
    A = np.zeros((M, M), dtype=np.float64)
    b = np.zeros(M, dtype=np.float64)

    # Mass matrix assembly (1/2 term)
    for e in range(M):
        n0 = elem_nodes[e, 0]
        n1 = elem_nodes[e, 1]
        Le = elem_len[e]

        Me00 = (Le / 6.0) * 2.0
        Me01 = (Le / 6.0) * 1.0
        Me10 = Me01
        Me11 = Me00

        A[n0, n0] += 0.5 * Me00
        A[n0, n1] += 0.5 * Me01
        A[n1, n0] += 0.5 * Me10
        A[n1, n1] += 0.5 * Me11

    # Double-layer operator assembly
    c = -(1.0 / (2.0 * np.pi))
    Q = phi1.shape[0]

    for ex in range(M):
        nx0 = elem_nodes[ex, 0]
        nx1 = elem_nodes[ex, 1]

        for ey in range(M):
            ny0 = elem_nodes[ey, 0]
            ny1 = elem_nodes[ey, 1]

            # Four basis combinations on the two linear shape functions
            contrib00 = 0.0
            contrib01 = 0.0
            contrib10 = 0.0
            contrib11 = 0.0

            for qx in range(Q):
                wx = quad_w[ex, qx]
                px0 = quad_pts[ex, qx, 0]
                px1 = quad_pts[ex, qx, 1]
                phix0 = phi1[qx]
                phix1 = phi2[qx]

                for qy in range(Q):
                    wy = quad_w[ey, qy]
                    py0 = quad_pts[ey, qy, 0]
                    py1 = quad_pts[ey, qy, 1]

                    dx = px0 - py0
                    dy = px1 - py1
                    r2 = dx * dx + dy * dy

                    if r2 > 1e-14:
                        dot = dx * py0 + dy * py1
                        kernel = c * dot / r2
                        wxy = wx * wy

                        phiy0 = phi1[qy]
                        phiy1 = phi2[qy]

                        contrib00 += phix0 * kernel * phiy0 * wxy
                        contrib01 += phix0 * kernel * phiy1 * wxy
                        contrib10 += phix1 * kernel * phiy0 * wxy
                        contrib11 += phix1 * kernel * phiy1 * wxy

            A[nx0, ny0] += contrib00
            A[nx0, ny1] += contrib01
            A[nx1, ny0] += contrib10
            A[nx1, ny1] += contrib11

    # RHS assembly
    for e in range(M):
        n0 = elem_nodes[e, 0]
        n1 = elem_nodes[e, 1]
        Le = elem_len[e]

        be0 = (Le / 6.0) * (2.0 * f[n0] + f[n1])
        be1 = (Le / 6.0) * (f[n0] + 2.0 * f[n1])

        b[n0] += be0
        b[n1] += be1

    return A, b


@njit(cache=True, fastmath=True, parallel=True)
def eval_solution_numba(mu, elem_nodes, quad_pts, quad_w, phi1, phi2, eval_pts):
    n_eval = eval_pts.shape[0]
    M = elem_nodes.shape[0]
    Q = phi1.shape[0]
    u = np.zeros(n_eval, dtype=np.float64)
    c = -(1.0 / (2.0 * np.pi))

    # Precompute mu at quadrature points element-wise
    mu_q = np.zeros((M, Q), dtype=np.float64)
    for e in range(M):
        n0 = elem_nodes[e, 0]
        n1 = elem_nodes[e, 1]
        for q in range(Q):
            mu_q[e, q] = phi1[q] * mu[n0] + phi2[q] * mu[n1]

    for k in prange(n_eval):
        x = eval_pts[k, 0]
        y = eval_pts[k, 1]
        acc = 0.0

        for e in range(M):
            for q in range(Q):
                dx = x - quad_pts[e, q, 0]
                dy = y - quad_pts[e, q, 1]
                r2 = dx * dx + dy * dy

                if r2 > 1e-14:
                    dot = dx * quad_pts[e, q, 0] + dy * quad_pts[e, q, 1]
                    kernel = c * dot / r2
                    acc += kernel * mu_q[e, q] * quad_w[e, q]

        u[k] = acc

    return u

# ============================================================
# BEM Setup
# ============================================================

def bem_setup(M, quad_order=8):
    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    nodes = np.column_stack((np.cos(theta), np.sin(theta)))
    normals = nodes.copy()

    elem_nodes = np.column_stack((np.arange(M), (np.arange(M)+1) % M))
    x0 = nodes[elem_nodes[:, 0]]
    x1 = nodes[elem_nodes[:, 1]]
    elem_vec = x1 - x0
    elem_len = np.linalg.norm(elem_vec, axis=1)

    s, w = leggauss(quad_order)
    phi1 = (1 - s) / 2
    phi2 = (1 + s) / 2

    quad_pts = x0[:, None, :] + ((s + 1) / 2)[None, :, None] * elem_vec[:, None, :]
    quad_w = w[None, :] * (elem_len / 2)[:, None]

    f = np.cos(3 * theta)

    return {
        "nodes": nodes,
        "normals": normals,
        "elem_nodes": elem_nodes,
        "quad_pts": quad_pts,
        "quad_w": quad_w,
        "phi1": phi1,
        "phi2": phi2,
        "elem_len": elem_len,
        "f": f
    }

# ============================================================
# Assembly (Galerkin)
# ============================================================

def assemble_matrix(data):
    t0 = time.perf_counter()

    A, b = assemble_matrix_numba(
        data["nodes"],
        data["normals"],
        data["elem_nodes"],
        data["quad_pts"],
        data["quad_w"],
        data["phi1"],
        data["phi2"],
        data["elem_len"],
        data["f"]
    )

    setup_time = time.perf_counter() - t0
    return A, b, setup_time

# ============================================================
# Solve
# ============================================================

def solve_system(A, b):
    t0 = time.perf_counter()
    iter_count = [0]

    def callback(res):
        iter_count[0] += 1

    mu, _ = gmres(A, b, atol=1e-10, callback=callback)
    solve_time = time.perf_counter() - t0
    return mu, iter_count[0], solve_time

# ============================================================
# Interior Evaluation
# ============================================================

def bem_evaluate_chunked(mu, data, chunk=2000):
    t0 = time.perf_counter()

    Nx = Ny = 120
    xx = np.linspace(-0.9, 0.9, Nx)
    yy = np.linspace(-0.9, 0.9, Ny)
    X, Y = np.meshgrid(xx, yy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    mask = (X**2 + Y**2) < 0.9**2
    eval_pts = pts[mask.ravel()]

    u_num = np.zeros(len(eval_pts), dtype=np.float64)

    # chunking kept, but each chunk is evaluated by Numba
    for i in range(0, len(eval_pts), chunk):
        xe = eval_pts[i:i+chunk]
        u_num[i:i+chunk] = eval_solution_numba(
            mu,
            data["elem_nodes"],
            data["quad_pts"],
            data["quad_w"],
            data["phi1"],
            data["phi2"],
            xe
        )

    x = eval_pts[:, 0]
    y = eval_pts[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    u_exact = r**3 * np.cos(3 * theta)

    rel_L2 = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
    Linf = np.max(np.abs(u_num - u_exact))

    eval_time = time.perf_counter() - t0
    return rel_L2, Linf, eval_time

# ============================================================
# Run
# ============================================================

def run_bem(M):
    data = bem_setup(M)
    A, b, setup_time = assemble_matrix(data)
    mu, iters, solve_time = solve_system(A, b)
    rel_L2, Linf, eval_time = bem_evaluate_chunked(mu, data)

    total_time = setup_time + solve_time + eval_time

    return {
        "M": M,
        "iterations": iters,
        "setup_time": setup_time,
        "solve_time": solve_time,
        "eval_time": eval_time,
        "total_time": total_time,
        "relative_L2_error": rel_L2,
        "Linf_error": Linf
    }

# ============================================================
# Main Sweep
# ============================================================

if __name__ == "__main__":
    Ms = [200, 400, 800, 1600]

    print(" M      Iter    Setup(s)  Solve(s)  Eval(s)   Total(s)   RelL2       Linf")
    print("-"*80)

    for M in Ms:
        res = run_bem(M)
        print(f"{res['M']:6d}  {res['iterations']:6d}  "
              f"{res['setup_time']:8.3f}  {res['solve_time']:8.3f}  "
              f"{res['eval_time']:8.3f}  {res['total_time']:9.3f}  "
              f"{res['relative_L2_error']:9.2e}  {res['Linf_error']:9.2e}")