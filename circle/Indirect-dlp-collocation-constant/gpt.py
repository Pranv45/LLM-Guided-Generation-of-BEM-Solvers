import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

# ---------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------
MODE_N = 5
GRID_NX = 120
GRID_NY = 120
GRID_RADIUS = 0.9

M_VALUES = [4000, 8000, 16000, 32000, 64000]

GMRES_RTOL = 1e-12
GMRES_ATOL = 0.0
GMRES_RESTART = 50
GMRES_MAXITER = 300

# Chunk sizes chosen to limit memory use in matrix-free matvec / evaluation.
TARGET_CHUNK = 64
SOURCE_CHUNK = 4096
EVAL_TARGET_CHUNK = 1024
EVAL_SOURCE_CHUNK = 4096


# ---------------------------------------------------------------------
# Exact solution and boundary data
# ---------------------------------------------------------------------
def exact_solution(x, y, n=MODE_N):
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return (r ** n) * np.cos(n * theta)


def boundary_data(theta, n=MODE_N):
    return np.cos(n * theta)


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
def build_unit_circle_geometry(m):
    """
    Nyström discretization on the unit circle using M equally spaced points.
    Collocation/source points are the same.
    """
    theta = (2.0 * np.pi / m) * np.arange(m, dtype=np.float64) + (np.pi / m)
    x = np.cos(theta)
    y = np.sin(theta)

    points = np.column_stack((x, y)).astype(np.float64, copy=False)
    normals = points.copy()  # outward normal on unit circle
    weights = np.full(m, 2.0 * np.pi / m, dtype=np.float64)

    return theta, points, normals, weights


# ---------------------------------------------------------------------
# Matrix-free operator: (1/2 I + K) mu
# K(x_i, y_j) = -(1/(2π)) * ((x_i - y_j)·n_j)/|x_i - y_j|^2
# ---------------------------------------------------------------------
def make_dlp_operator(points, normals, weights,
                      target_chunk=TARGET_CHUNK, source_chunk=SOURCE_CHUNK):
    xsrc = np.ascontiguousarray(points[:, 0])
    ysrc = np.ascontiguousarray(points[:, 1])
    nx = np.ascontiguousarray(normals[:, 0])
    ny = np.ascontiguousarray(normals[:, 1])
    w = np.ascontiguousarray(weights)
    m = points.shape[0]

    def matvec(mu):
        mu = np.asarray(mu, dtype=np.float64)
        out = 0.5 * mu.copy()

        for i0 in range(0, m, target_chunk):
            i1 = min(i0 + target_chunk, m)
            xi = xsrc[i0:i1][:, None]
            yi = ysrc[i0:i1][:, None]

            acc = np.zeros(i1 - i0, dtype=np.float64)

            for j0 in range(0, m, source_chunk):
                j1 = min(j0 + source_chunk, m)

                dx = xi - xsrc[j0:j1][None, :]
                dy = yi - ysrc[j0:j1][None, :]

                r2 = dx * dx + dy * dy
                dot = dx * nx[j0:j1][None, :] + dy * ny[j0:j1][None, :]

                inv_r2 = np.zeros_like(r2)
                np.divide(1.0, r2, out=inv_r2, where=(r2 > 0.0))

                kern = -(1.0 / (2.0 * np.pi)) * dot * inv_r2
                kern *= w[j0:j1][None, :]

                # Remove self-interaction (principal value handled by +1/2 I)
                if i0 < j1 and j0 < i1:
                    s0 = max(i0, j0)
                    s1 = min(i1, j1)
                    diag_idx = np.arange(s0, s1)
                    kern[diag_idx - i0, diag_idx - j0] = 0.0

                acc += kern @ mu[j0:j1]

            out[i0:i1] += acc

        return out

    return LinearOperator((m, m), matvec=matvec, dtype=np.float64)


# ---------------------------------------------------------------------
# Interior evaluation: u(x) = ∫ K(x,y) mu(y) ds_y
# ---------------------------------------------------------------------
def evaluate_dlp(mu, eval_pts, src_points, src_normals, src_weights,
                 target_chunk=EVAL_TARGET_CHUNK, source_chunk=EVAL_SOURCE_CHUNK):
    mu = np.asarray(mu, dtype=np.float64)
    ex = np.ascontiguousarray(eval_pts[:, 0])
    ey = np.ascontiguousarray(eval_pts[:, 1])

    xsrc = np.ascontiguousarray(src_points[:, 0])
    ysrc = np.ascontiguousarray(src_points[:, 1])
    nx = np.ascontiguousarray(src_normals[:, 0])
    ny = np.ascontiguousarray(src_normals[:, 1])
    w = np.ascontiguousarray(src_weights)

    n_eval = eval_pts.shape[0]
    out = np.zeros(n_eval, dtype=np.float64)

    for i0 in range(0, n_eval, target_chunk):
        i1 = min(i0 + target_chunk, n_eval)
        xi = ex[i0:i1][:, None]
        yi = ey[i0:i1][:, None]

        acc = np.zeros(i1 - i0, dtype=np.float64)

        for j0 in range(0, xsrc.size, source_chunk):
            j1 = min(j0 + source_chunk, xsrc.size)

            dx = xi - xsrc[j0:j1][None, :]
            dy = yi - ysrc[j0:j1][None, :]

            r2 = dx * dx + dy * dy
            dot = dx * nx[j0:j1][None, :] + dy * ny[j0:j1][None, :]

            inv_r2 = np.zeros_like(r2)
            np.divide(1.0, r2, out=inv_r2, where=(r2 > 0.0))

            kern = -(1.0 / (2.0 * np.pi)) * dot * inv_r2
            kern *= w[j0:j1][None, :]

            acc += kern @ mu[j0:j1]

        out[i0:i1] = acc

    return out


# ---------------------------------------------------------------------
# Interior grid
# ---------------------------------------------------------------------
def build_interior_grid(nx=GRID_NX, ny=GRID_NY, radius=GRID_RADIUS):
    x = np.linspace(-radius, radius, nx, dtype=np.float64)
    y = np.linspace(-radius, radius, ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    mask = (xx * xx + yy * yy) < (radius * radius)
    pts = np.column_stack((xx[mask], yy[mask])).astype(np.float64, copy=False)
    u_exact = exact_solution(pts[:, 0], pts[:, 1], MODE_N).astype(np.float64, copy=False)
    return pts, u_exact


# ---------------------------------------------------------------------
# Solve one refinement level
# ---------------------------------------------------------------------
def run_case(m, eval_pts, u_exact):
    t0 = time.perf_counter()

    theta, points, normals, weights = build_unit_circle_geometry(m)
    rhs = boundary_data(theta, MODE_N).astype(np.float64, copy=False)

    operator = make_dlp_operator(points, normals, weights)

    setup_time = time.perf_counter() - t0

    it_counter = {"count": 0}

    def callback(_residual_norm):
        it_counter["count"] += 1

    t1 = time.perf_counter()
    mu, info = gmres(
        operator,
        rhs,
        x0=np.zeros_like(rhs),
        rtol=GMRES_RTOL,
        atol=GMRES_ATOL,
        restart=GMRES_RESTART,
        maxiter=GMRES_MAXITER,
        callback=callback,
        callback_type="pr_norm",
    )
    solve_time = time.perf_counter() - t1

    if info != 0:
        raise RuntimeError(f"GMRES failed to converge for M={m}, info={info}")

    t2 = time.perf_counter()
    u_num = evaluate_dlp(mu, eval_pts, points, normals, weights)
    rel_l2 = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
    linf = np.max(np.abs(u_num - u_exact))
    eval_time = time.perf_counter() - t2

    total_time = setup_time + solve_time + eval_time

    return {
        "M": m,
        "iters": it_counter["count"],
        "rel_l2": rel_l2,
        "linf": linf,
        "setup": setup_time,
        "solve": solve_time,
        "eval": eval_time,
        "total": total_time,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    eval_pts, u_exact = build_interior_grid()

    print(
        f"{'M':>8} {'GMRES':>8} {'Rel L2 Error':>16} {'Linf Error':>16} "
        f"{'Setup(s)':>12} {'Solve(s)':>12} {'Eval(s)':>12} {'Total(s)':>12}"
    )

    for m in M_VALUES:
        result = run_case(m, eval_pts, u_exact)
        print(
            f"{result['M']:8d} {result['iters']:8d} "
            f"{result['rel_l2']:16.8e} {result['linf']:16.8e} "
            f"{result['setup']:12.4f} {result['solve']:12.4f} "
            f"{result['eval']:12.4f} {result['total']:12.4f}"
        )


if __name__ == "__main__":
    main()

#numba
import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, gmres
from numpy.polynomial.legendre import leggauss
from numba import njit, prange

# ============================================================
# Numba-accelerated kernels
# ============================================================

@njit(cache=True, fastmath=True, parallel=True)
def bem_matvec_numba(mu, nodes, normals, elem_nodes, quad_pts, quad_w, phi1, phi2):
    M = nodes.shape[0]
    Q = phi1.shape[0]
    result = np.empty(M, dtype=np.float64)

    c = -(1.0 / (2.0 * np.pi))

    for i in prange(M):
        acc = 0.5 * mu[i]

        for e in range(M):
            n0 = elem_nodes[e, 0]
            n1 = elem_nodes[e, 1]

            nx = normals[n0, 0]
            ny = normals[n0, 1]

            for q in range(Q):
                mu_q = phi1[q] * mu[n0] + phi2[q] * mu[n1]

                dx = nodes[i, 0] - quad_pts[e, q, 0]
                dy = nodes[i, 1] - quad_pts[e, q, 1]
                r2 = dx * dx + dy * dy

                if r2 > 1e-14:
                    dot = dx * nx + dy * ny
                    acc += c * dot / r2 * mu_q * quad_w[e, q]

        result[i] = acc

    return result


@njit(cache=True, fastmath=True, parallel=True)
def bem_eval_chunk_numba(eval_pts, mu_all, quad_pts_all, quad_w_all):
    n_eval = eval_pts.shape[0]
    n_src = quad_pts_all.shape[0]
    u_num = np.empty(n_eval, dtype=np.float64)

    c = -(1.0 / (2.0 * np.pi))

    for i in prange(n_eval):
        xi = eval_pts[i, 0]
        yi = eval_pts[i, 1]
        acc = 0.0

        for j in range(n_src):
            dx = xi - quad_pts_all[j, 0]
            dy = yi - quad_pts_all[j, 1]
            r2 = dx * dx + dy * dy

            # Kept exactly in the same spirit as your current code:
            # kernel = -(1/(2*pi)) * dot / r2
            dot = dx * quad_pts_all[j, 0] + dy * quad_pts_all[j, 1]
            kernel = c * dot / r2
            acc += kernel * mu_all[j] * quad_w_all[j]

        u_num[i] = acc

    return u_num

# ============================================================
# BEM Setup
# ============================================================

def bem_setup(M, n_mode, quad_order=8):
    t0 = time.perf_counter()

    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    nodes = np.column_stack((np.cos(theta), np.sin(theta)))
    normals = nodes.copy()

    elem_nodes = np.column_stack((np.arange(M), (np.arange(M)+1)%M))
    x0 = nodes[elem_nodes[:,0]]
    x1 = nodes[elem_nodes[:,1]]
    elem_vec = x1 - x0
    elem_len = np.linalg.norm(elem_vec, axis=1)

    s, w = leggauss(quad_order)
    phi1 = (1 - s)/2
    phi2 = (1 + s)/2

    quad_pts = x0[:,None,:] + ((s+1)/2)[None,:,None] * elem_vec[:,None,:]
    quad_w = w[None,:] * (elem_len/2)[:,None]

    f = np.cos(n_mode * theta)

    setup_time = time.perf_counter() - t0

    return {
        "nodes": nodes,
        "normals": normals,
        "elem_nodes": elem_nodes,
        "quad_pts": quad_pts,
        "quad_w": quad_w,
        "phi1": phi1,
        "phi2": phi2,
        "f": f,
        "setup_time": setup_time
    }

# ============================================================
# Matrix-free matvec
# ============================================================

def bem_matvec(mu, data):
    return bem_matvec_numba(
        mu,
        data["nodes"],
        data["normals"],
        data["elem_nodes"],
        data["quad_pts"],
        data["quad_w"],
        data["phi1"],
        data["phi2"]
    )

# ============================================================
# Interior evaluation (Cartesian masked grid)
# ============================================================

def bem_evaluate_chunked(mu, data, n_mode, chunk=2000):
    t0 = time.perf_counter()

    # ---- Cartesian evaluation grid (your version) ----
    Nx = Ny = 120
    xx = np.linspace(-0.9, 0.9, Nx)
    yy = np.linspace(-0.9, 0.9, Ny)
    X, Y = np.meshgrid(xx, yy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    mask = (X**2 + Y**2) < 0.9**2
    eval_pts = pts[mask.ravel()]
    # ---------------------------------------------------

    elem_nodes = data["elem_nodes"]
    quad_pts = data["quad_pts"]
    quad_w = data["quad_w"]
    phi1 = data["phi1"]
    phi2 = data["phi2"]

    # Interpolate μ to quadrature points (same as your code)
    mu_elem = phi1[None,:]*mu[elem_nodes[:,0],None] + \
              phi2[None,:]*mu[elem_nodes[:,1],None]

    quad_pts_all = quad_pts.reshape(-1,2)
    quad_w_all = quad_w.reshape(-1)
    mu_all = mu_elem.reshape(-1)

    u_num = np.zeros(len(eval_pts))

    # Chunking retained; each chunk is computed by Numba
    for i in range(0, len(eval_pts), chunk):
        xe = eval_pts[i:i+chunk]
        u_num[i:i+chunk] = bem_eval_chunk_numba(xe, mu_all, quad_pts_all, quad_w_all)

    # ---- Exact solution in Cartesian coordinates ----
    x = eval_pts[:,0]
    y = eval_pts[:,1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    u_exact = (r**n_mode) * np.cos(n_mode*theta)

    rel_L2 = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
    Linf = np.max(np.abs(u_num - u_exact))

    eval_time = time.perf_counter() - t0

    return rel_L2, Linf, eval_time

# ============================================================
# Run BEM
# ============================================================

def run_bem(M, n_mode):
    data = bem_setup(M, n_mode)
    f = data["f"]

    t0 = time.perf_counter()

    iter_count = [0]
    def callback(res):
        iter_count[0] += 1

    A = LinearOperator((M,M), matvec=lambda x: bem_matvec(x, data))
    mu, _ = gmres(A, f, atol=1e-8, callback=callback)

    solve_time = time.perf_counter() - t0

    rel_L2, Linf, eval_time = bem_evaluate_chunked(mu, data, n_mode)

    total_time = data["setup_time"] + solve_time + eval_time

    return {
        "M": M,
        "iterations": iter_count[0],
        "setup_time": data["setup_time"],
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
    n_mode = 3
    Ms = [1000, 2000, 4000, 8000, 16000]

    print(" M      Iter    Setup(s)  Solve(s)  Eval(s)   Total(s)   RelL2       Linf")
    print("-"*80)

    for M in Ms:
        res = run_bem(M, n_mode)
        print(f"{res['M']:6d}  {res['iterations']:6d}  "
              f"{res['setup_time']:8.3f}  {res['solve_time']:8.3f}  "
              f"{res['eval_time']:8.3f}  {res['total_time']:9.3f}  "
              f"{res['relative_L2_error']:9.2e}  {res['Linf_error']:9.2e}")