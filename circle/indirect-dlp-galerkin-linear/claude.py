import numpy as np
from scipy.sparse.linalg import gmres
from numba import njit, prange
import time

# ── Gauss quadrature ──────────────────────────────────────────────────────────
_GS,   _GW   = np.polynomial.legendre.leggauss(8)
_GS16, _GW16 = np.polynomial.legendre.leggauss(16)
Q   = len(_GS)
Q16 = len(_GS16)

_PHI1   = 0.5 * (1.0 - _GS)
_PHI2   = 0.5 * (1.0 + _GS)
_PHI1_16 = 0.5 * (1.0 - _GS16)
_PHI2_16 = 0.5 * (1.0 + _GS16)


# ── 1. Setup ──────────────────────────────────────────────────────────────────
def bem_setup(M: int, n: int = 3):
    theta    = 2.0 * np.pi * np.arange(M) / M
    nodes    = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    normals  = nodes.copy()
    f_nodal  = np.cos(n * theta)
    i0       = np.arange(M)
    i1       = (i0 + 1) % M
    elems    = np.stack([i0, i1], axis=1)
    elem_len = np.linalg.norm(nodes[i1] - nodes[i0], axis=1)
    return nodes, normals, f_nodal, elems, elem_len


# ── 2a. Numba kernel: mass matrix contribution ────────────────────────────────
@njit(cache=True, parallel=True)
def _assemble_mass(na, nb, jac, M_dof):
    """
    A_mass[r, c] += 0.5 * jac[e] * [[2/3, 1/3],[1/3, 2/3]]
    Returns a flat (M_dof, M_dof) array.
    """
    NE = na.shape[0]
    A  = np.zeros((M_dof, M_dof), dtype=np.float64)
    # mass matrix local entries (exact)
    m00 = 2.0 / 3.0;  m01 = 1.0 / 3.0
    m10 = 1.0 / 3.0;  m11 = 2.0 / 3.0
    for e in prange(NE):
        j   = jac[e]
        ra  = na[e];  rb = nb[e]
        A[ra, ra] += 0.5 * j * m00
        A[ra, rb] += 0.5 * j * m01
        A[rb, ra] += 0.5 * j * m10
        A[rb, rb] += 0.5 * j * m11
    return A


# ── 2b. Numba kernel: off-diagonal double-layer block ─────────────────────────
@njit(cache=True, parallel=True)
def _assemble_offdiag_block(
        p_start, p_end, q_start, q_end,
        nodes, normals, na, nb, jac,
        y_eq, ny_eq,           # (NE, Q, 2)
        gs, gw, phi1, phi2):
    """
    Returns local_K (Tp, Tq, 2, 2):
    local_K[p,q,a,b] = jac_p * jac_q
                       * Σ_qx Σ_qy φ_a(qx) K(x_{p,qx}, y_{q,qy}) φ_b(qy) w_qx w_qy
    Diagonal pairs (p_global == q_global) are zeroed → handled separately.
    """
    inv2pi = 1.0 / (2.0 * np.pi)
    Q      = gs.shape[0]
    Tp     = p_end - p_start
    Tq     = q_end - q_start

    # output: (Tp, Tq, 2, 2)
    out = np.zeros((Tp, Tq, 2, 2), dtype=np.float64)

    for lp in prange(Tp):
        gp   = p_start + lp
        jp   = jac[gp]

        for lq in range(Tq):
            gq = q_start + lq
            if gp == gq:
                continue           # diagonal → skip, handled by _assemble_diag
            jq = jac[gq]

            # local 2×2 accumulator
            loc = np.zeros((2, 2), dtype=np.float64)

            for qx in range(Q):
                wx  = gw[qx]
                p1x = phi1[qx];  p2x = phi2[qx]
                x0  = y_eq[gp, qx, 0]
                x1  = y_eq[gp, qx, 1]

                for qy in range(Q):
                    wy   = gw[qy]
                    p1y  = phi1[qy];  p2y  = phi2[qy]
                    y0   = y_eq[gq, qy, 0]
                    y1   = y_eq[gq, qy, 1]
                    ny0  = ny_eq[gq, qy, 0]
                    ny1  = ny_eq[gq, qy, 1]

                    d0    = x0 - y0
                    d1    = x1 - y1
                    dist2 = d0 * d0 + d1 * d1
                    if dist2 == 0.0:
                        continue

                    dotn = d0 * ny0 + d1 * ny1
                    Kval = (-inv2pi) * dotn / dist2
                    w    = wx * wy * Kval * jp * jq

                    loc[0, 0] += w * p1x * p1y
                    loc[0, 1] += w * p1x * p2y
                    loc[1, 0] += w * p2x * p1y
                    loc[1, 1] += w * p2x * p2y

            out[lp, lq, 0, 0] = loc[0, 0]
            out[lp, lq, 0, 1] = loc[0, 1]
            out[lp, lq, 1, 0] = loc[1, 0]
            out[lp, lq, 1, 1] = loc[1, 1]

    return out


# ── 2c. Numba kernel: diagonal elements (16-pt rule) ─────────────────────────
@njit(cache=True, parallel=True)
def _assemble_diag(na, nb, jac, y_eq16, ny_eq16, gw16, phi1_16, phi2_16):
    """
    For each element e, compute the 2×2 self-interaction matrix using the
    16-pt × 16-pt quadrature rule and accumulate into A.
    Returns contribution array (NE, 2, 2).
    """
    NE     = na.shape[0]
    inv2pi = 1.0 / (2.0 * np.pi)
    Q16    = gw16.shape[0]
    out    = np.zeros((NE, 2, 2), dtype=np.float64)

    for e in prange(NE):
        jp = jac[e]
        loc = np.zeros((2, 2), dtype=np.float64)
        for qx in range(Q16):
            wx  = gw16[qx]
            p1x = phi1_16[qx];  p2x = phi2_16[qx]
            x0  = y_eq16[e, qx, 0]
            x1  = y_eq16[e, qx, 1]
            for qy in range(Q16):
                wy   = gw16[qy]
                p1y  = phi1_16[qy];  p2y  = phi2_16[qy]
                y0   = y_eq16[e, qy, 0]
                y1   = y_eq16[e, qy, 1]
                ny0  = ny_eq16[e, qy, 0]
                ny1  = ny_eq16[e, qy, 1]

                d0    = x0 - y0
                d1    = x1 - y1
                dist2 = d0 * d0 + d1 * d1
                if dist2 == 0.0:
                    continue

                dotn = d0 * ny0 + d1 * ny1
                Kval = (-inv2pi) * dotn / dist2
                w    = wx * wy * Kval * jp * jp

                loc[0, 0] += w * p1x * p1y
                loc[0, 1] += w * p1x * p2y
                loc[1, 0] += w * p2x * p1y
                loc[1, 1] += w * p2x * p2y

        out[e, 0, 0] = loc[0, 0]
        out[e, 0, 1] = loc[0, 1]
        out[e, 1, 0] = loc[1, 0]
        out[e, 1, 1] = loc[1, 1]

    return out


# ── 2. assemble_matrix (Python orchestrator) ──────────────────────────────────
def assemble_matrix(nodes, normals, elems, elem_len, tile=64):
    M_dof  = nodes.shape[0]
    NE     = len(elems)
    na     = elems[:, 0].astype(np.int64)
    nb     = elems[:, 1].astype(np.int64)
    jac    = 0.5 * elem_len

    # precompute quadrature geometry for all elements
    y_eq   = (nodes[na][:, None, :] * _PHI1[None, :, None] +
              nodes[nb][:, None, :] * _PHI2[None, :, None])    # (NE, Q, 2)
    ny_eq  = (normals[na][:, None, :] * _PHI1[None, :, None] +
              normals[nb][:, None, :] * _PHI2[None, :, None])

    y_eq16  = (nodes[na][:, None, :] * _PHI1_16[None, :, None] +
               nodes[nb][:, None, :] * _PHI2_16[None, :, None])
    ny_eq16 = (normals[na][:, None, :] * _PHI1_16[None, :, None] +
               normals[nb][:, None, :] * _PHI2_16[None, :, None])

    # Mass matrix
    A = _assemble_mass(na, nb, jac, M_dof)

    # Diagonal self-interaction
    diag_loc = _assemble_diag(na, nb, jac, y_eq16, ny_eq16,
                               _GW16, _PHI1_16, _PHI2_16)   # (NE, 2, 2)
    for e in range(NE):
        for a, ra in enumerate([na[e], nb[e]]):
            for b, cb in enumerate([na[e], nb[e]]):
                A[ra, cb] += diag_loc[e, a, b]

    # Off-diagonal tiled assembly
    for p_start in range(0, NE, tile):
        p_end = min(p_start + tile, NE)
        for q_start in range(0, NE, tile):
            q_end = min(q_start + tile, NE)

            loc_K = _assemble_offdiag_block(
                p_start, p_end, q_start, q_end,
                nodes, normals, na, nb, jac,
                y_eq, ny_eq,
                _GS, _GW, _PHI1, _PHI2)              # (Tp, Tq, 2, 2)

            Tp = p_end - p_start
            Tq = q_end - q_start
            na_p = na[p_start:p_end]
            nb_p = nb[p_start:p_end]
            na_q = na[q_start:q_end]
            nb_q = nb[q_start:q_end]

            for a, rows in enumerate([na_p, nb_p]):
                for b, cols in enumerate([na_q, nb_q]):
                    np.add.at(A,
                              (rows[:, None], cols[None, :]),
                              loc_K[:, :, a, b])

    return A


# ── 3. Solve ──────────────────────────────────────────────────────────────────
def solve_system(A, nodes, normals, f_nodal, elems, elem_len, gmres_tol=1e-10):
    M_dof = nodes.shape[0]
    na    = elems[:, 0]
    nb    = elems[:, 1]
    jac   = 0.5 * elem_len

    b    = np.zeros(M_dof)
    f_q  = (f_nodal[na][:, None] * _PHI1[None, :] +
            f_nodal[nb][:, None] * _PHI2[None, :])
    for a, phi_a in enumerate([_PHI1, _PHI2]):
        dofs = na if a == 0 else nb
        np.add.at(b, dofs,
                  (phi_a[None, :] * f_q * _GW[None, :] * jac[:, None]).sum(axis=1))

    iters = [0]
    def _cb(rk): iters[0] += 1
    mu, info = gmres(A, b, atol=gmres_tol, callback=_cb, callback_type='legacy')
    if info != 0:
        print(f"  GMRES warning: info={info}")
    return mu, iters[0]


# ── 4. Numba kernel: interior evaluation ─────────────────────────────────────
@njit(cache=True, parallel=True)
def _eval_kernel(eval_pts, y_all, ny_all, wjmu, start, end):
    """
    u[ci] = Σ_{m,q}  K(xp_ci, y_{m,q}) * wjmu[m,q]
    Outer loop over eval points is parallelised.
    """
    inv2pi = 1.0 / (2.0 * np.pi)
    NE     = y_all.shape[0]
    Q      = y_all.shape[1]
    chunk  = end - start
    result = np.empty(chunk, dtype=np.float64)

    for ci in prange(chunk):
        xp0 = eval_pts[start + ci, 0]
        xp1 = eval_pts[start + ci, 1]
        acc = 0.0
        for m in range(NE):
            for q in range(Q):
                d0    = xp0 - y_all[m, q, 0]
                d1    = xp1 - y_all[m, q, 1]
                dist2 = d0 * d0 + d1 * d1
                if dist2 == 0.0:
                    continue
                dotn = d0 * ny_all[m, q, 0] + d1 * ny_all[m, q, 1]
                acc += (-inv2pi) * dotn / dist2 * wjmu[m, q]
        result[ci] = acc

    return result


def bem_evaluate_chunked(mu, nodes, normals, elems, elem_len,
                         eval_pts, chunk_size=256):
    N      = eval_pts.shape[0]
    result = np.empty(N, dtype=np.float64)
    na, nb = elems[:, 0], elems[:, 1]
    jac    = 0.5 * elem_len

    y_all  = (nodes[na][:, None, :] * _PHI1[None, :, None] +
              nodes[nb][:, None, :] * _PHI2[None, :, None])
    ny_all = (normals[na][:, None, :] * _PHI1[None, :, None] +
              normals[nb][:, None, :] * _PHI2[None, :, None])
    mu_q   = mu[na][:, None] * _PHI1[None, :] + mu[nb][:, None] * _PHI2[None, :]
    wjmu   = _GW[None, :] * jac[:, None] * mu_q    # (NE, Q)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        result[start:end] = _eval_kernel(eval_pts, y_all, ny_all, wjmu, start, end)

    return result


# ── 5. Warm-up (JIT compile before timed sweep) ───────────────────────────────
def _warmup():
    nodes_s, normals_s, _, elems_s, elen_s = bem_setup(16)
    na_s = elems_s[:, 0].astype(np.int64)
    nb_s = elems_s[:, 1].astype(np.int64)
    jac_s = 0.5 * elen_s

    y_eq_s   = (nodes_s[na_s][:, None, :] * _PHI1[None, :, None] +
                nodes_s[nb_s][:, None, :] * _PHI2[None, :, None])
    ny_eq_s  = (normals_s[na_s][:, None, :] * _PHI1[None, :, None] +
                normals_s[nb_s][:, None, :] * _PHI2[None, :, None])
    y_eq16_s = (nodes_s[na_s][:, None, :] * _PHI1_16[None, :, None] +
                nodes_s[nb_s][:, None, :] * _PHI2_16[None, :, None])
    ny_eq16_s= (normals_s[na_s][:, None, :] * _PHI1_16[None, :, None] +
                normals_s[nb_s][:, None, :] * _PHI2_16[None, :, None])

    _assemble_mass(na_s, nb_s, jac_s, 16)
    _assemble_offdiag_block(0, 8, 8, 16,
        nodes_s, normals_s, na_s, nb_s, jac_s,
        y_eq_s, ny_eq_s, _GS, _GW, _PHI1, _PHI2)
    _assemble_diag(na_s, nb_s, jac_s, y_eq16_s, ny_eq16_s, _GW16, _PHI1_16, _PHI2_16)

    mu_s  = np.ones(16, dtype=np.float64)
    eps   = np.array([[0.1, 0.1]], dtype=np.float64)
    wjmu_s = np.ones((16, Q), dtype=np.float64)
    _eval_kernel(eps, y_eq_s, ny_eq_s, wjmu_s, 0, 1)

_warmup()


# ── 6. Full BEM run ───────────────────────────────────────────────────────────
def run_bem(M: int, n: int = 3, gmres_tol: float = 1e-10,
            tile: int = 64, eval_chunk: int = 256):
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    nodes, normals, f_nodal, elems, elem_len = bem_setup(M, n)
    setup_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    A  = assemble_matrix(nodes, normals, elems, elem_len, tile=tile)
    assembly_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    mu, iters = solve_system(A, nodes, normals, f_nodal, elems, elem_len, gmres_tol)
    solve_time = time.perf_counter() - t2

    Nx = Ny = 120
    xx, yy   = np.linspace(-0.9, 0.9, Nx), np.linspace(-0.9, 0.9, Ny)
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
        iterations        = iters,
        setup_time        = setup_time + assembly_time,
        solve_time        = solve_time,
        eval_time         = eval_time,
        total_time        = time.perf_counter() - t_start,
        relative_L2_error = np.linalg.norm(err) / np.linalg.norm(u_exact),
        Linf_error        = np.max(np.abs(err)),
    )


# ── 7. Convergence sweep ──────────────────────────────────────────────────────
M_values = [200, 400, 800, 1600, 3200]

HDR = (f"{'M':>6}  {'Iters':>5}  {'Setup(s)':>9}  {'Solve(s)':>9}  "
       f"{'Eval(s)':>8}  {'Total(s)':>9}  {'Rel L2':>10}  {'Linf':>10}  {'Rate':>6}")
print(HDR)
print("-" * len(HDR))

results, prev_err = [], None
for M in M_values:
    r = run_bem(M)
    results.append(r)
    rate = ""
    if prev_err is not None and prev_err > 0:
        rate = f"{np.log2(prev_err / r['relative_L2_error']):>6.2f}"
    prev_err = r['relative_L2_error']
    print(f"{r['M']:>6,}  {r['iterations']:>5}  "
          f"{r['setup_time']:>9.3f}  {r['solve_time']:>9.3f}  "
          f"{r['eval_time']:>8.3f}  {r['total_time']:>9.3f}  "
          f"{r['relative_L2_error']:>10.3e}  {r['Linf_error']:>10.3e}  {rate}")
print("-" * len(HDR))