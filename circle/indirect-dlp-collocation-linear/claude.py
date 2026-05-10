#no numba acceleration
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
import time

# ── Gaussian quadrature on [-1,1], order 8 ───────────────────────────────────
def _gauss_quad(order=8):
    return np.polynomial.legendre.leggauss(order)

_GQ_PTS, _GQ_WTS = _gauss_quad(8)          # shape (8,) each, module-level cache

# ── 1. Setup ──────────────────────────────────────────────────────────────────
def bem_setup(M: int, n: int = 3):
    """
    M nodes on the unit circle → M linear elements (periodic).
    Returns
    -------
    nodes    : (M,2)   node coordinates
    normals  : (M,2)   outward normals at nodes
    f        : (M,)    Dirichlet data cos(nθ) at nodes
    elems    : (M,2)   element connectivity  [i, i+1 mod M]
    elem_len : (M,)    physical length of each element
    """
    theta   = 2.0 * np.pi * np.arange(M) / M
    nodes   = np.stack([np.cos(theta), np.sin(theta)], axis=1)   # (M,2)
    normals = nodes.copy()                                         # unit outward
    f       = np.cos(n * theta)

    i0      = np.arange(M)
    i1      = (i0 + 1) % M
    elems   = np.stack([i0, i1], axis=1)                          # (M,2)

    diff        = nodes[i1] - nodes[i0]                           # (M,2)
    elem_len    = np.linalg.norm(diff, axis=1)                    # (M,)

    return nodes, normals, f, elems, elem_len


# ── 2. Matrix-free matvec (½I + K)μ ──────────────────────────────────────────
def bem_matvec(mu, nodes, normals, elems, elem_len):
    """
    (½ I + K) μ  via element-wise Gauss quadrature.
    No (M,M) matrix is formed.
    """
    M      = nodes.shape[0]
    result = 0.5 * mu.copy()                       # identity contribution
    inv2pi = 1.0 / (2.0 * np.pi)
    gs, gw = _GQ_PTS, _GQ_WTS                     # (Q,)

    # shape functions at all quadrature pts:  (Q,) each
    phi1 = 0.5 * (1.0 - gs)    # φ1(s)
    phi2 = 0.5 * (1.0 + gs)    # φ2(s)

    # precompute interpolated normals & positions along every element
    # n_a, n_b : global node indices for each element
    na = elems[:, 0]            # (M,)
    nb = elems[:, 1]            # (M,)

    # physical quadrature points on every element: (M, Q, 2)
    # y(s) = phi1(s)*node_a + phi2(s)*node_b
    y_all = (nodes[na][:, None, :] * phi1[None, :, None] +
             nodes[nb][:, None, :] * phi2[None, :, None])     # (M,Q,2)

    # normal at quadrature point by linear interpolation (already unit for circle,
    # but we interpolate consistently with the basis)
    ny_all = (normals[na][:, None, :] * phi1[None, :, None] +
              normals[nb][:, None, :] * phi2[None, :, None])  # (M,Q,2)

    # Jacobian: |dy/ds| = elem_len/2  (constant per element)
    jac = 0.5 * elem_len                                       # (M,)

    # interpolated μ at quadrature points: (M, Q)
    mu_q = mu[na][:, None] * phi1[None, :] + mu[nb][:, None] * phi2[None, :]

    # loop over collocation nodes i
    for i in range(M):
        xi = nodes[i]                      # (2,)

        # diff = xi - y  for all elements and all quadrature pts: (M, Q, 2)
        diff  = xi[None, None, :] - y_all

        # |xi - y|²  (M, Q)
        dist2 = np.einsum('mqk,mqk->mq', diff, diff)

        # (xi - y) · n_y  (M, Q)
        dot   = np.einsum('mqk,mqk->mq', diff, ny_all)

        # kernel value  K(xi, y)  (M, Q)
        K_val = (-inv2pi) * dot / np.where(dist2 == 0.0, 1.0, dist2)

        # zero out singular quadrature pts (shouldn't occur for well-separated colloc)
        K_val = np.where(dist2 == 0.0, 0.0, K_val)

        # integrand = K * μ(y) * Jacobian  summed over quadrature  → (M,)
        contrib = np.einsum('mq,mq,q,m->m', K_val, mu_q,  gw, jac)  # wrong shape
        # correct: sum over q for each element m
        contrib = (K_val * mu_q * gw[None, :] * jac[:, None]).sum(axis=1)  # (M,)

        result[i] += contrib.sum()

    return result


# ── 3. Chunked interior evaluation ───────────────────────────────────────────
def bem_evaluate_chunked(mu, nodes, normals, elems, elem_len, eval_pts,
                         chunk_size=256):
    """
    u(x) = ∫_Γ K(x,y) μ(y) ds   for interior x, via linear elements + Gauss quad.
    Outer loop over chunks of eval_pts; inner loop is fully vectorised over
    elements and quadrature points.
    """
    N      = eval_pts.shape[0]
    result = np.zeros(N, dtype=np.float64)
    inv2pi = 1.0 / (2.0 * np.pi)
    gs, gw = _GQ_PTS, _GQ_WTS

    phi1 = 0.5 * (1.0 - gs)
    phi2 = 0.5 * (1.0 + gs)

    na = elems[:, 0]
    nb = elems[:, 1]

    # precompute element quadrature geometry  (M, Q, 2)
    y_all  = (nodes[na][:, None, :] * phi1[None, :, None] +
              nodes[nb][:, None, :] * phi2[None, :, None])
    ny_all = (normals[na][:, None, :] * phi1[None, :, None] +
              normals[nb][:, None, :] * phi2[None, :, None])
    mu_q   = mu[na][:, None] * phi1[None, :] + mu[nb][:, None] * phi2[None, :]
    jac    = 0.5 * elem_len                                      # (M,)

    # weight × jacobian × μ(y_q)  collapsed to (M, Q)
    wjmu   = gw[None, :] * jac[:, None] * mu_q                  # (M, Q)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        xp  = eval_pts[start:end]                                # (C, 2)

        # diff[c, m, q, k] = xp_c - y_{m,q}
        # Build as  (C,1,1,2) - (1,M,Q,2)  → (C,M,Q,2)
        diff  = (xp[:, None, None, :] -
                 y_all[None, :, :, :])                           # (C,M,Q,2)

        dist2 = np.einsum('cmqk,cmqk->cmq', diff, diff)         # (C,M,Q)
        dot   = np.einsum('cmqk,mqk->cmq', diff, ny_all)        # (C,M,Q)

        K_val = (-inv2pi) * dot / dist2                          # (C,M,Q)

        # integrate: sum over q then over m
        result[start:end] = np.einsum('cmq,mq->c', K_val, wjmu) # (C,)

    return result


# ── 4. Full BEM run ───────────────────────────────────────────────────────────
def run_bem(M: int, n: int = 3,
            gmres_tol: float = 1e-10,
            matvec_chunk: int = 512,
            eval_chunk: int   = 64):

    t_start = time.perf_counter()

    # Setup
    t0 = time.perf_counter()
    nodes, normals, f, elems, elem_len = bem_setup(M, n)
    setup_time = time.perf_counter() - t0

    # GMRES
    iters = [0]
    def _callback(rk): iters[0] += 1
    def _matvec(mu):
        return bem_matvec(mu, nodes, normals, elems, elem_len)

    A = LinearOperator((M, M), matvec=_matvec, dtype=np.float64)

    t2 = time.perf_counter()
    mu, info = gmres(A, f, atol=gmres_tol, callback=_callback,
                     callback_type='legacy')
    solve_time = time.perf_counter() - t2
    if info != 0:
        print(f"  [M={M}] GMRES warning: info={info}")

    # Evaluation grid
    Nx = Ny = 120
    xx = np.linspace(-0.9, 0.9, Nx)
    yy = np.linspace(-0.9, 0.9, Ny)
    X, Y   = np.meshgrid(xx, yy)
    pts    = np.column_stack([X.ravel(), Y.ravel()])
    mask   = (X**2 + Y**2) < 0.9**2
    eval_pts = pts[mask.ravel()]

    t4 = time.perf_counter()
    u_num = bem_evaluate_chunked(mu, nodes, normals, elems, elem_len,
                                 eval_pts, chunk_size=eval_chunk)
    eval_time = time.perf_counter() - t4

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
        u_num             = u_num,
        eval_pts          = eval_pts,
        mask              = mask,
        X=X, Y=Y,
    )

print("no numba acceleration")
# ── 5. Convergence sweep ──────────────────────────────────────────────────────
M_values = [1_000, 2_000, 4_000, 8_000, 16_000]

HDR = (f"{'M':>7}  {'Iters':>5}  {'Setup(s)':>9}  {'Solve(s)':>9}  "
       f"{'Eval(s)':>8}  {'Total(s)':>9}  {'Rel L2':>10}  {'Linf':>10}")
print(HDR)
print("-" * len(HDR))

results = []
for M in M_values:
    r = run_bem(M)
    results.append(r)
    print(f"{r['M']:>7,}  {r['iterations']:>5}  "
          f"{r['setup_time']:>9.3f}  {r['solve_time']:>9.3f}  "
          f"{r['eval_time']:>8.3f}  {r['total_time']:>9.3f}  "
          f"{r['relative_L2_error']:>10.3e}  {r['Linf_error']:>10.3e}")
print("-" * len(HDR))


#numba accelerated
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange
import time

# ── Gaussian quadrature on [-1,1], order 8 ───────────────────────────────────
def _gauss_quad(order=8):
    return np.polynomial.legendre.leggauss(order)

_GQ_PTS, _GQ_WTS = _gauss_quad(8)   # shape (8,) each, module-level cache


# ── 1. Setup ──────────────────────────────────────────────────────────────────
def bem_setup(M: int, n: int = 3):
    theta    = 2.0 * np.pi * np.arange(M) / M
    nodes    = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    normals  = nodes.copy()
    f        = np.cos(n * theta)
    i0       = np.arange(M)
    i1       = (i0 + 1) % M
    elems    = np.stack([i0, i1], axis=1)
    diff     = nodes[i1] - nodes[i0]
    elem_len = np.linalg.norm(diff, axis=1)
    return nodes, normals, f, elems, elem_len


# ── 2. Numba kernel: matvec ───────────────────────────────────────────────────

@njit(cache=True, parallel=True)
def _matvec_kernel(mu, nodes, normals, elems, elem_len, gs, gw):
    """
    Computes (½I + K)μ entirely in scalar loops.
    Outer loop over collocation nodes i is parallelised.
    Inner loops: elements m, quadrature points q.
    """
    M      = nodes.shape[0]
    Q      = gs.shape[0]
    inv2pi = 1.0 / (2.0 * np.pi)
    result = np.empty(M, dtype=np.float64)

    for i in prange(M):
        xi0 = nodes[i, 0]
        xi1 = nodes[i, 1]
        acc = 0.0

        for m in range(M):
            na   = elems[m, 0]
            nb   = elems[m, 1]
            jac  = 0.5 * elem_len[m]   # Jacobian |dy/ds| = L/2

            # node coords
            xa0 = nodes[na, 0];  xa1 = nodes[na, 1]
            xb0 = nodes[nb, 0];  xb1 = nodes[nb, 1]
            # node normals
            na0 = normals[na, 0]; na1 = normals[na, 1]
            nb0 = normals[nb, 0]; nb1 = normals[nb, 1]
            # node mu values
            mu_a = mu[na];  mu_b = mu[nb]

            for q in range(Q):
                s    = gs[q]
                phi1 = 0.5 * (1.0 - s)
                phi2 = 0.5 * (1.0 + s)

                # quadrature point position
                y0 = phi1 * xa0 + phi2 * xb0
                y1 = phi1 * xa1 + phi2 * xb1

                # interpolated normal
                ny0 = phi1 * na0 + phi2 * nb0
                ny1 = phi1 * na1 + phi2 * nb1

                # interpolated mu
                mu_y = phi1 * mu_a + phi2 * mu_b

                d0    = xi0 - y0
                d1    = xi1 - y1
                dist2 = d0 * d0 + d1 * d1

                if dist2 == 0.0:
                    continue

                dot  = d0 * ny0 + d1 * ny1
                Kval = (-inv2pi) * dot / dist2

                acc += Kval * mu_y * gw[q] * jac

        result[i] = 0.5 * mu[i] + acc

    return result


def bem_matvec(mu, nodes, normals, elems, elem_len):
    return _matvec_kernel(mu, nodes, normals, elems, elem_len, _GQ_PTS, _GQ_WTS)


# ── 3. Numba kernel: interior evaluation ─────────────────────────────────────

@njit(cache=True, parallel=True)
def _eval_kernel(mu, nodes, normals, elems, elem_len, eval_pts, gs, gw,
                 start, end):
    """
    u(xp) = ∫_Γ K(xp, y) μ(y) ds  for each eval point in [start, end).
    Outer loop over eval points is parallelised.
    """
    Q      = gs.shape[0]
    M      = nodes.shape[0]
    inv2pi = 1.0 / (2.0 * np.pi)
    chunk  = end - start
    result = np.empty(chunk, dtype=np.float64)

    for ci in prange(chunk):
        xp0 = eval_pts[start + ci, 0]
        xp1 = eval_pts[start + ci, 1]
        acc = 0.0

        for m in range(M):
            na   = elems[m, 0]
            nb   = elems[m, 1]
            jac  = 0.5 * elem_len[m]

            xa0 = nodes[na, 0];  xa1 = nodes[na, 1]
            xb0 = nodes[nb, 0];  xb1 = nodes[nb, 1]
            na0 = normals[na, 0]; na1 = normals[na, 1]
            nb0 = normals[nb, 0]; nb1 = normals[nb, 1]
            mu_a = mu[na];  mu_b = mu[nb]

            for q in range(Q):
                s    = gs[q]
                phi1 = 0.5 * (1.0 - s)
                phi2 = 0.5 * (1.0 + s)

                y0 = phi1 * xa0 + phi2 * xb0
                y1 = phi1 * xa1 + phi2 * xb1

                ny0 = phi1 * na0 + phi2 * nb0
                ny1 = phi1 * na1 + phi2 * nb1

                mu_y = phi1 * mu_a + phi2 * mu_b

                d0    = xp0 - y0
                d1    = xp1 - y1
                dist2 = d0 * d0 + d1 * d1

                if dist2 == 0.0:
                    continue

                dot  = d0 * ny0 + d1 * ny1
                Kval = (-inv2pi) * dot / dist2

                acc += Kval * mu_y * gw[q] * jac

        result[ci] = acc

    return result


def bem_evaluate_chunked(mu, nodes, normals, elems, elem_len, eval_pts,
                         chunk_size=256):
    N      = eval_pts.shape[0]
    result = np.empty(N, dtype=np.float64)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        result[start:end] = _eval_kernel(
            mu, nodes, normals, elems, elem_len,
            eval_pts, _GQ_PTS, _GQ_WTS, start, end
        )
    return result


# ── 4. JIT warm-up (runs once at import; keeps timed sweep clean) ─────────────
def _warmup():
    nodes_s, normals_s, _, elems_s, elen_s = bem_setup(16)
    mu_s  = np.ones(16, dtype=np.float64)
    eps   = np.array([[0.1, 0.1]], dtype=np.float64)
    _matvec_kernel(mu_s, nodes_s, normals_s, elems_s, elen_s, _GQ_PTS, _GQ_WTS)
    _eval_kernel(mu_s, nodes_s, normals_s, elems_s, elen_s,
                 eps, _GQ_PTS, _GQ_WTS, 0, 1)

_warmup()


# ── 5. Full BEM run ───────────────────────────────────────────────────────────
def run_bem(M: int, n: int = 3,
            gmres_tol: float = 1e-10,
            eval_chunk: int  = 256):

    t_start = time.perf_counter()

    t0 = time.perf_counter()
    nodes, normals, f, elems, elem_len = bem_setup(M, n)
    setup_time = time.perf_counter() - t0

    iters = [0]
    def _callback(rk): iters[0] += 1
    def _mv(mu): return bem_matvec(mu, nodes, normals, elems, elem_len)

    A = LinearOperator((M, M), matvec=_mv, dtype=np.float64)

    t2 = time.perf_counter()
    mu, info = gmres(A, f, atol=gmres_tol, callback=_callback,
                     callback_type='legacy')
    solve_time = time.perf_counter() - t2

    if info != 0:
        print(f"  [M={M}] GMRES warning: info={info}")

    Nx = Ny = 120
    xx = np.linspace(-0.9, 0.9, Nx)
    yy = np.linspace(-0.9, 0.9, Ny)
    X, Y     = np.meshgrid(xx, yy)
    pts      = np.column_stack([X.ravel(), Y.ravel()])
    mask     = (X**2 + Y**2) < 0.9**2
    eval_pts = pts[mask.ravel()]

    t4 = time.perf_counter()
    u_num = bem_evaluate_chunked(mu, nodes, normals, elems, elem_len,
                                 eval_pts, chunk_size=eval_chunk)
    eval_time = time.perf_counter() - t4

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
        u_num             = u_num,
        eval_pts          = eval_pts,
        mask              = mask,
        X=X, Y=Y,
    )

print("numba acceleration")
# ── 6. Convergence sweep ──────────────────────────────────────────────────────
M_values = [4000, 8000, 16000]

HDR = (f"{'M':>7}  {'Iters':>5}  {'Setup(s)':>9}  {'Solve(s)':>9}  "
       f"{'Eval(s)':>8}  {'Total(s)':>9}  {'Rel L2':>10}  {'Linf':>10}")
print(HDR)
print("-" * len(HDR))

results = []
for M in M_values:
    r = run_bem(M)
    results.append(r)
    print(f"{r['M']:>7,}  {r['iterations']:>5}  "
          f"{r['setup_time']:>9.3f}  {r['solve_time']:>9.3f}  "
          f"{r['eval_time']:>8.3f}  {r['total_time']:>9.3f}  "
          f"{r['relative_L2_error']:>10.3e}  {r['Linf_error']:>10.3e}")
print("-" * len(HDR))