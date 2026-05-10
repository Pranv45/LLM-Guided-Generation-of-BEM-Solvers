#no numba
import numpy as np
import time
from scipy.sparse.linalg import gmres, LinearOperator

def bem_setup(M, n):
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    nodes = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    normals = nodes.copy()
    f = np.cos(n * theta)
    w = 2 * np.pi / M
    return nodes, normals, f, w

def bem_matvec_chunked(mu, nodes, normals, w, chunk_size=1000):
    """
    Memory-optimized matvec.
    Uses |x-y|^2 = |x|^2 + |y|^2 - 2(x.y) to avoid 3D array allocations.
    """
    M = nodes.shape[0]
    result = 0.5 * mu

    # Precompute norms and self-dot products
    nodes_norm_sq = np.sum(nodes**2, axis=1) # Constant 1.0 for unit circle
    y_dot_ny = np.sum(nodes * normals, axis=1) # Constant 1.0 for unit circle

    # Pre-multiply mu by weights to save ops in the loop
    mu_w = mu * w

    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        target_nodes = nodes[i:end]

        # 1. Compute dot product matrix: (chunk, 2) @ (2, M) -> (chunk, M)
        dot_xy = np.dot(target_nodes, nodes.T)

        # 2. Distance squared: |x|^2 + |y|^2 - 2(x.y)
        dist_sq = nodes_norm_sq[i:end, np.newaxis] + nodes_norm_sq[np.newaxis, :] - 2 * dot_xy

        # 3. (x-y).n_y = x.n_y - y.n_y
        dot_prod = np.dot(target_nodes, normals.T) - y_dot_ny[np.newaxis, :]

        # Handle self-interaction (diagonal)
        diag_idx = np.arange(i, end)
        dist_sq[np.arange(end-i), diag_idx] = 1.0

        kernel = -1.0 / (2 * np.pi) * (dot_prod / dist_sq)
        kernel[np.arange(end-i), diag_idx] = 0.0

        result[i:end] += np.dot(kernel, mu_w)

    return result

def bem_evaluate_chunked(mu, nodes, normals, w, eval_pts, chunk_size=1000):
    """Vectorized interior evaluation with strict memory limits."""
    M = nodes.shape[0]
    N_eval = eval_pts.shape[0]
    u_eval = np.zeros(N_eval)

    nodes_norm_sq = np.sum(nodes**2, axis=1)
    eval_norm_sq = np.sum(eval_pts**2, axis=1)
    y_dot_ny = np.sum(nodes * normals, axis=1)
    mu_w = mu * w

    for i in range(0, N_eval, chunk_size):
        end = min(i + chunk_size, N_eval)
        target_pts = eval_pts[i:end]

        dist_sq = eval_norm_sq[i:end, np.newaxis] + nodes_norm_sq[np.newaxis, :] - 2 * np.dot(target_pts, nodes.T)
        dot_prod = np.dot(target_pts, normals.T) - y_dot_ny[np.newaxis, :]

        kernel = -1.0 / (2 * np.pi) * (dot_prod / dist_sq)
        u_eval[i:end] = np.dot(kernel, mu_w)

    return u_eval

def run_bem(M, n_freq=2):
    metrics = {"M": M}

    # 1. Setup
    t_start = time.perf_counter()
    nodes, normals, f, w = bem_setup(M, n_freq)
    metrics["setup_time"] = time.perf_counter() - t_start

    # 2. Solve
    iter_count = 0
    def callback(rk):
        nonlocal iter_count
        iter_count += 1

    # LinearOperator with the optimized matvec
    A_op = LinearOperator((M, M), matvec=lambda mu: bem_matvec_chunked(mu, nodes, normals, w))

    t_solve_start = time.perf_counter()
    # atol=1e-9 provides a good balance of precision and speed
    # restart=30 prevents GMRES from consuming too much memory for the Krylov subspace
    mu, info = gmres(A_op, f, atol=1e-9, restart=30, callback=callback)
    metrics["solve_time"] = time.perf_counter() - t_solve_start
    metrics["iterations"] = iter_count

    # 3. Grid Evaluation (Masked Interior)
    Nx = Ny = 120
    xx = np.linspace(-0.9, 0.9, Nx)
    yy = np.linspace(-0.9, 0.9, Ny)
    X, Y = np.meshgrid(xx, yy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    mask = (X**2 + Y**2) < 0.9**2
    eval_pts = pts[mask.ravel()]

    t_eval_start = time.perf_counter()
    u_num = bem_evaluate_chunked(mu, nodes, normals, w, eval_pts)
    metrics["eval_time"] = time.perf_counter() - t_eval_start

    # 4. Error Metrics
    r = np.sqrt(eval_pts[:, 0]**2 + eval_pts[:, 1]**2)
    theta = np.arctan2(eval_pts[:, 1], eval_pts[:, 0])
    u_exact = (r**n_freq) * np.cos(n_freq * theta)

    metrics["relative_L2_error"] = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
    metrics["Linf_error"] = np.max(np.abs(u_num - u_exact))
    metrics["total_time"] = metrics["setup_time"] + metrics["solve_time"] + metrics["eval_time"]

    return metrics

if __name__ == "__main__":
    # Now supports up to 128,000 without MemoryError
    M_list = [4000, 8000, 16000, 32000, 64000]
    n_freq = 3

    print(f"{'M':>8} | {'Iters':>5} | {'Setup(s)':>8} | {'Solve(s)':>9} | {'Eval(s)':>8} | {'L2 Error':>10}")
    print("-" * 75)

    for M in M_list:
        res = run_bem(M, n_freq)
        print(f"{res['M']:8d} | {res['iterations']:5d} | {res['setup_time']:8.3f} | "
              f"{res['solve_time']:9.2f} | {res['eval_time']:8.3f} | {res['relative_L2_error']:10.4e}")

# numba
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange
import time


# ── 1. Setup ──────────────────────────────────────────────────────────────────

def bem_setup(M: int, n: int = 3):
    theta   = 2.0 * np.pi * np.arange(M) / M
    nodes   = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    normals = nodes.copy()
    w       = 2.0 * np.pi / M
    f       = np.cos(n * theta)
    return nodes, normals, w, f, theta


# ── 2. Numba kernel: matvec chunk ─────────────────────────────────────────────

@njit(cache=True, parallel=True)
def _matvec_kernel(mu, nodes, normals, w, start, end):
    """
    Compute  0.5*mu[start:end]  +  K_block @ mu
    where K_block[i,m] = (-w/2π) * dot(xi-xm, nm) / |xi-xm|²
    with the diagonal (i==m in global index) zeroed out.
    """
    inv2pi  = 1.0 / (2.0 * np.pi)
    M       = nodes.shape[0]
    chunk   = end - start
    result  = np.empty(chunk, dtype=np.float64)

    for i in prange(chunk):
        gi  = start + i          # global row index
        xi0 = nodes[gi, 0]
        xi1 = nodes[gi, 1]
        acc = 0.0
        for m in range(M):
            d0    = xi0 - nodes[m, 0]
            d1    = xi1 - nodes[m, 1]
            dist2 = d0 * d0 + d1 * d1
            dot   = d0 * normals[m, 0] + d1 * normals[m, 1]
            if m == gi:
                # diagonal → zero contribution
                pass
            else:
                acc += (-w * inv2pi) * dot / dist2 * mu[m]
        result[i] = 0.5 * mu[gi] + acc

    return result


def bem_matvec(mu, nodes, normals, w, chunk_size=512):
    M      = nodes.shape[0]
    result = np.empty(M, dtype=np.float64)
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        result[start:end] = _matvec_kernel(mu, nodes, normals, w, start, end)
    return result


# ── 3. Numba kernel: interior evaluation chunk ────────────────────────────────

@njit(cache=True, parallel=True)
def _eval_kernel(mu, nodes, normals, w, eval_pts, start, end):
    """
    u[i] = sum_m  (-w/2π) * dot(xp_i - xm, nm) / |xp_i - xm|²  *  mu[m]
    No diagonal removal needed (eval_pts are strictly interior).
    """
    inv2pi = 1.0 / (2.0 * np.pi)
    M      = nodes.shape[0]
    chunk  = end - start
    result = np.empty(chunk, dtype=np.float64)

    for i in prange(chunk):
        xp0 = eval_pts[start + i, 0]
        xp1 = eval_pts[start + i, 1]
        acc = 0.0
        for m in range(M):
            d0    = xp0 - nodes[m, 0]
            d1    = xp1 - nodes[m, 1]
            dist2 = d0 * d0 + d1 * d1
            dot   = d0 * normals[m, 0] + d1 * normals[m, 1]
            acc  += (-w * inv2pi) * dot / dist2 * mu[m]
        result[i] = acc

    return result


def bem_evaluate_chunked(mu, nodes, normals, w, eval_pts, chunk_size=512):
    N      = eval_pts.shape[0]
    result = np.empty(N, dtype=np.float64)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        result[start:end] = _eval_kernel(mu, nodes, normals, w, eval_pts, start, end)
    return result


# ── 4. JIT warm-up (call once at import time so timings are clean) ─────────────

def _warmup():
    nodes_s, normals_s, w_s, _, _ = bem_setup(8)
    mu_s  = np.ones(8, dtype=np.float64)
    eps   = np.array([[0.1, 0.1]], dtype=np.float64)
    _matvec_kernel(mu_s, nodes_s, normals_s, w_s, 0, 4)
    _eval_kernel(mu_s, nodes_s, normals_s, w_s, eps, 0, 1)

_warmup()


# ── 5. Full BEM run ────────────────────────────────────────────────────────────

def run_bem(M: int, n: int = 3,
            gmres_tol: float  = 1e-10,
            matvec_chunk: int = 512,
            eval_chunk: int   = 512):

    t_start = time.perf_counter()

    # Setup
    t0 = time.perf_counter()
    nodes, normals, w, f, theta = bem_setup(M, n)
    setup_time = time.perf_counter() - t0

    # GMRES
    iters = [0]
    def _callback(pr_norm): iters[0] += 1
    def _matvec(mu): return bem_matvec(mu, nodes, normals, w, chunk_size=matvec_chunk)

    A = LinearOperator((M, M), matvec=_matvec, dtype=np.float64)

    t2 = time.perf_counter()
    mu, info = gmres(A, f, atol=gmres_tol, callback=_callback, callback_type='legacy')
    solve_time = time.perf_counter() - t2

    if info != 0:
        print(f"  [M={M}] GMRES warning: info={info}")

    # ---------- Evaluation grid (fixed, interior) ----------
    Nx = Ny = 120
    xx = np.linspace(-0.9, 0.9, Nx)
    yy = np.linspace(-0.9, 0.9, Ny)
    X, Y = np.meshgrid(xx, yy)
    pts  = np.column_stack([X.ravel(), Y.ravel()])
    mask = (X**2 + Y**2) < 0.9**2
    eval_pts = pts[mask.ravel()]

    # ---------- Evaluation ----------
    t0 = time.perf_counter()
    u_num = bem_evaluate_chunked(mu, nodes, normals, w, eval_pts, chunk_size=eval_chunk)
    eval_time = time.perf_counter() - t0

    # Exact solution
    r_pts   = np.sqrt(eval_pts[:, 0]**2 + eval_pts[:, 1]**2)
    th_pts  = np.arctan2(eval_pts[:, 1], eval_pts[:, 0])
    u_exact = (r_pts ** n) * np.cos(n * th_pts)

    err = u_num - u_exact
    return dict(
        M                 = M,
        iterations        = iters[0],
        setup_time        = setup_time,
        solve_time        = solve_time,
        eval_time         = eval_time,
        total_time        = time.perf_counter() - t_start,
        relative_L2_error = np.linalg.norm(err) / np.linalg.norm(u_exact),
        Linf_error        = np.max(np.abs(err)),
        u_num_grid        = u_num,
        eval_pts          = eval_pts,
        mask              = mask,
        X                 = X,
        Y                 = Y,
    )


# ── 6. Convergence sweep ───────────────────────────────────────────────────────

M_values = [4_000, 8_000, 16_000, 32_000, 64_000]

HDR = (f"{'M':>8}  {'Iters':>5}  {'Setup(s)':>9}  {'Solve(s)':>9}  "
       f"{'Eval(s)':>8}  {'Total(s)':>9}  {'Rel L2':>10}  {'Linf':>10}")
print(HDR)
print("-" * len(HDR))

results = []
for M in M_values:
    r = run_bem(M)
    results.append(r)
    print(f"{r['M']:>8,}  {r['iterations']:>5}  "
          f"{r['setup_time']:>9.3f}  {r['solve_time']:>9.3f}  "
          f"{r['eval_time']:>8.3f}  {r['total_time']:>9.3f}  "
          f"{r['relative_L2_error']:>10.3e}  {r['Linf_error']:>10.3e}")

print("-" * len(HDR))