#no numba acceleration
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
import time

# ── 1. Setup ─────────────────────────────────────────────────────────────────

def bem_setup(M: int, n: int = 3):
    theta   = 2.0 * np.pi * np.arange(M) / M
    nodes   = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    normals = nodes.copy()
    w       = 2.0 * np.pi / M
    f       = np.cos(n * theta)
    return nodes, normals, w, f, theta


# ── 2. Chunked matvec ─────────────────────────────────────────────────────────

def bem_matvec(mu, nodes, normals, w, chunk_size=512):
    M      = nodes.shape[0]
    result = np.empty(M, dtype=np.float64)
    inv2pi = 1.0 / (2.0 * np.pi)

    for start in range(0, M, chunk_size):
        end  = min(start + chunk_size, M)
        xi   = nodes[start:end]
        diff  = xi[:, None, :] - nodes[None, :, :]
        dist2 = np.einsum('cmk,cmk->cm', diff, diff)
        dot   = np.einsum('cmk,mk->cm', diff, normals)
        idx   = np.arange(start, end) - start
        glob  = np.arange(start, end)
        dist2[idx, glob] = 1.0
        dot  [idx, glob] = 0.0
        K_block = (-w * inv2pi) * (dot / dist2)
        result[start:end] = 0.5 * mu[start:end] + K_block @ mu

    return result


# ── 3. Chunked interior evaluation ───────────────────────────────────────────

def bem_evaluate_chunked(mu, nodes, normals, w, eval_pts, chunk_size=512):
    N      = eval_pts.shape[0]
    result = np.empty(N, dtype=np.float64)
    inv2pi = 1.0 / (2.0 * np.pi)

    for start in range(0, N, chunk_size):
        end  = min(start + chunk_size, N)
        xp   = eval_pts[start:end]
        diff  = xp[:, None, :] - nodes[None, :, :]
        dist2 = np.einsum('cmk,cmk->cm', diff, diff)
        dot   = np.einsum('cmk,mk->cm', diff, normals)
        result[start:end] = ((-w * inv2pi) * (dot / dist2)) @ mu

    return result


# ── 4. Full BEM run ───────────────────────────────────────────────────────────

def run_bem(M: int, n: int = 3,
            gmres_tol: float = 1e-10,
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

    # ---------- Vectorized evaluation ----------
    t0 = time.perf_counter()
    u_num = bem_evaluate_chunked(mu, nodes, normals, w, eval_pts, chunk_size=eval_chunk)
    eval_time = time.perf_counter() - t0

    # Exact solution at the same masked interior points
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
        # grids returned for optional plotting
        u_num_grid        = u_num,
        eval_pts          = eval_pts,
        mask              = mask,
        X                 = X,
        Y                 = Y,
    )


# ── 5. Convergence sweep ──────────────────────────────────────────────────────
print("no numba acceleration")
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

#numba acceleration
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
print("numba acceleration")
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