import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange
import time

# ---------------------------------------------------------------------------
# Precompute quadrature geometry arrays (called once in setup)
# ---------------------------------------------------------------------------
def _precompute_quad_geom(x1, y1, x2, y2, el_len, xi, w):
    """
    Returns:
      qx   : (M, Ng)  quadrature x-coords on each element
      qy   : (M, Ng)  quadrature y-coords on each element
      phi1 : (Ng,)    shape function 1 at each quad point
      phi2 : (Ng,)    shape function 2 at each quad point
      jac  : (M,)     element jacobians = len/2
      w    : (Ng,)    quadrature weights (returned as-is)
    """
    M  = x1.shape[0]
    Ng = xi.shape[0]
    phi1 = 0.5*(1.0 - xi)          # (Ng,)
    phi2 = 0.5*(1.0 + xi)          # (Ng,)
    # qx[e,g] = phi1[g]*x1[e] + phi2[g]*x2[e]
    qx   = np.outer(x1, phi1) + np.outer(x2, phi2)   # (M, Ng)
    qy   = np.outer(y1, phi1) + np.outer(y2, phi2)   # (M, Ng)
    jac  = 0.5 * el_len                               # (M,)
    return qx, qy, phi1, phi2, jac, w


# ---------------------------------------------------------------------------
# Galerkin matvec — parallelised over TEST elements
# Precomputed geometry passed in to avoid recomputation each call.
# ---------------------------------------------------------------------------
@njit(parallel=True)
def _galerkin_matvec_fast(
        # test-element precomputed quad coords
        qx_o, qy_o, phi1_o, phi2_o, jac_o, w_o,
        # trial-element precomputed quad coords
        qx_i, qy_i, phi1_i, phi2_i, jac_i, w_i,
        # trial normals
        nx_f, ny_f,
        # nodal densities
        sigma,
        # output (M,)
        out):
    M  = qx_o.shape[0]
    No = phi1_o.shape[0]
    Ni = phi1_i.shape[0]
    inv2pi = 1.0 / (2.0 * np.pi)

    for e in prange(M):
        acc0 = 0.0   # accumulates to node e
        acc1 = 0.0   # accumulates to node (e+1)%M

        for go in range(No):
            xo   = qx_o[e, go]
            yo   = qy_o[e, go]
            wo   = w_o[go] * jac_o[e]
            p1o  = phi1_o[go]
            p2o  = phi2_o[go]

            inner = 0.0   # ∫ K(xo, y) σ(y) ds_y

            for f in range(M):
                if f == e:
                    continue          # (xo-y).n_y == 0 on same straight element
                nxf  = nx_f[f]
                nyf  = ny_f[f]
                jf   = jac_i[f]
                s0   = sigma[f]
                s1   = sigma[(f + 1) % M]

                for gi in range(Ni):
                    dx    = xo - qx_i[f, gi]
                    dy    = yo - qy_i[f, gi]
                    r2    = dx*dx + dy*dy
                    if r2 < 1.0e-28:
                        continue
                    dot   = dx*nxf + dy*nyf
                    sig_q = phi1_i[gi]*s0 + phi2_i[gi]*s1
                    inner += w_i[gi] * (-dot * inv2pi / r2) * sig_q * jf

            acc0 += wo * p1o * inner
            acc1 += wo * p2o * inner

        out[e]           += acc0
        out[(e + 1) % M] += acc1


# ---------------------------------------------------------------------------
# Mass matrix matvec — race-condition-free via local accumulation
# M_ij = ∫ φ_i φ_j ds,  assembled analytically for linear elements.
# Each element contributes len/6 * [[2,1],[1,2]] to its two nodes.
# We parallelise over elements and write results to a (M,2) thread-safe array,
# then reduce — no two elements share the same pair (e, (e+1)%M) atomically.
# Instead we accumulate per-element contributions and do a single serial pass.
# ---------------------------------------------------------------------------
@njit(parallel=True)
def _mass_matvec_fast(sigma, el_len, M):
    # Each prange worker writes to its own element slot → no race condition.
    contrib = np.zeros((M, 2), dtype=np.float64)
    for e in prange(M):
        L  = el_len[e]
        s0 = sigma[e]
        s1 = sigma[(e + 1) % M]
        contrib[e, 0] = L / 6.0 * (2.0*s0 + s1)   # to node e
        contrib[e, 1] = L / 6.0 * (s0 + 2.0*s1)   # to node e+1
    # serial reduction (cheap, M additions)
    out = np.zeros(M, dtype=np.float64)
    for e in range(M):
        out[e]           += contrib[e, 0]
        out[(e + 1) % M] += contrib[e, 1]
    return out


# ---------------------------------------------------------------------------
# Interior evaluation kernel — parallelised over interior points
# ---------------------------------------------------------------------------
@njit(parallel=True)
def _evaluate_kernel(ex, ey, nx_f, ny_f,
                     qx_ev, qy_ev, phi1_ev, phi2_ev, jac_ev, w_ev,
                     sigma, out, M):
    N      = ex.shape[0]
    Ng     = phi1_ev.shape[0]
    inv2pi = 1.0 / (2.0 * np.pi)
    for i in prange(N):
        s = 0.0
        xi_pt = ex[i]
        yi_pt = ey[i]
        for f in range(M):
            nxf  = nx_f[f]
            nyf  = ny_f[f]
            jf   = jac_ev[f]
            s0   = sigma[f]
            s1   = sigma[(f + 1) % M]
            for g in range(Ng):
                dx    = xi_pt - qx_ev[f, g]
                dy    = yi_pt - qy_ev[f, g]
                r2    = dx*dx + dy*dy
                if r2 < 1.0e-28:
                    continue
                dot   = dx*nxf + dy*nyf
                sig_q = phi1_ev[g]*s0 + phi2_ev[g]*s1
                s    += w_ev[g] * (-dot * inv2pi / r2) * sig_q * jf
        out[i] = s


# ---------------------------------------------------------------------------
def bem_setup(M, a, b):
    t0    = time.perf_counter()

    theta = np.linspace(0.0, 2.0*np.pi, M, endpoint=False)
    xn    = a * np.cos(theta)
    yn    = b * np.sin(theta)

    x1 = xn.copy()
    y1 = yn.copy()
    x2 = np.roll(xn, -1)
    y2 = np.roll(yn, -1)

    tx     = x2 - x1;  ty = y2 - y1
    el_len = np.sqrt(tx**2 + ty**2)
    nx_el  =  ty / el_len
    ny_el  = -tx / el_len

    # ensure outward normals
    mx_el = 0.5*(x1+x2);  my_el = 0.5*(y1+y2)
    if np.mean(nx_el*mx_el + ny_el*my_el) < 0.0:
        nx_el = -nx_el;  ny_el = -ny_el

    # Gauss rules
    xi_o_raw, w_o_raw = np.polynomial.legendre.leggauss(4)   # outer (test)
    xi_i_raw, w_i_raw = np.polynomial.legendre.leggauss(6)   # inner (trial)
    xi_ev_raw, w_ev_raw = np.polynomial.legendre.leggauss(8) # evaluation
    xi_o  = xi_o_raw.astype(np.float64);  w_o  = w_o_raw.astype(np.float64)
    xi_i  = xi_i_raw.astype(np.float64);  w_i  = w_i_raw.astype(np.float64)
    xi_ev = xi_ev_raw.astype(np.float64); w_ev = w_ev_raw.astype(np.float64)

    # precompute quadrature geometry (done ONCE, reused every matvec)
    qx_o,  qy_o,  phi1_o,  phi2_o,  jac_o,  w_o  = _precompute_quad_geom(x1, y1, x2, y2, el_len, xi_o,  w_o)
    qx_i,  qy_i,  phi1_i,  phi2_i,  jac_i,  w_i  = _precompute_quad_geom(x1, y1, x2, y2, el_len, xi_i,  w_i)
    qx_ev, qy_ev, phi1_ev, phi2_ev, jac_ev, w_ev  = _precompute_quad_geom(x1, y1, x2, y2, el_len, xi_ev, w_ev)

    # ensure C-contiguous float64 for Numba
    def _c64(a): return np.ascontiguousarray(a, dtype=np.float64)
    qx_o  = _c64(qx_o);  qy_o  = _c64(qy_o)
    qx_i  = _c64(qx_i);  qy_i  = _c64(qy_i)
    qx_ev = _c64(qx_ev); qy_ev = _c64(qy_ev)
    phi1_o = _c64(phi1_o); phi2_o = _c64(phi2_o)
    phi1_i = _c64(phi1_i); phi2_i = _c64(phi2_i)
    phi1_ev= _c64(phi1_ev);phi2_ev= _c64(phi2_ev)

    # RHS: F_i = ∫ φ_i(x) f(x) ds_x  — fully vectorised
    # f(x) = x^2 - y^2 evaluated at all (M, No) outer quad points
    fval = qx_o**2 - qy_o**2                        # (M, No)
    # contribution of element e to node e:   Σ_g w_g * phi1_g * f_eg * jac_e
    # contribution of element e to node e+1: Σ_g w_g * phi2_g * f_eg * jac_e
    w_phi1 = w_o * phi1_o                            # (No,)
    w_phi2 = w_o * phi2_o                            # (No,)
    contrib_node0 = (fval * w_phi1).sum(axis=1) * jac_o   # (M,) -> node e
    contrib_node1 = (fval * w_phi2).sum(axis=1) * jac_o   # (M,) -> node e+1
    F = np.zeros(M)
    np.add.at(F, np.arange(M),       contrib_node0)
    np.add.at(F, (np.arange(M)+1)%M, contrib_node1)

    setup_time = time.perf_counter() - t0

    geom = dict(
        x1=x1, y1=y1, x2=x2, y2=y2,
        nx_el=nx_el, ny_el=ny_el, el_len=el_len,
        qx_o=qx_o, qy_o=qy_o, phi1_o=phi1_o, phi2_o=phi2_o, jac_o=jac_o, w_o=w_o,
        qx_i=qx_i, qy_i=qy_i, phi1_i=phi1_i, phi2_i=phi2_i, jac_i=jac_i, w_i=w_i,
        qx_ev=qx_ev, qy_ev=qy_ev, phi1_ev=phi1_ev, phi2_ev=phi2_ev,
        jac_ev=jac_ev, w_ev=w_ev,
    )
    return geom, F, setup_time


# ---------------------------------------------------------------------------
def bem_matvec(sigma, geom):
    M   = geom['x1'].shape[0]
    out = _mass_matvec_fast(sigma, geom['el_len'], M) * 0.5

    k_out = np.zeros(M, dtype=np.float64)
    _galerkin_matvec_fast(
        geom['qx_o'],  geom['qy_o'],
        geom['phi1_o'], geom['phi2_o'], geom['jac_o'], geom['w_o'],
        geom['qx_i'],  geom['qy_i'],
        geom['phi1_i'], geom['phi2_i'], geom['jac_i'], geom['w_i'],
        geom['nx_el'], geom['ny_el'],
        sigma, k_out,
    )
    return out + k_out


# ---------------------------------------------------------------------------
def bem_evaluate_chunked(ex, ey, geom, sigma, chunk_size=4096):
    M      = geom['x1'].shape[0]
    N      = ex.shape[0]
    result = np.zeros(N, dtype=np.float64)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        tmp = np.zeros(end - start, dtype=np.float64)
        _evaluate_kernel(
            ex[start:end], ey[start:end],
            geom['nx_el'], geom['ny_el'],
            geom['qx_ev'], geom['qy_ev'],
            geom['phi1_ev'], geom['phi2_ev'],
            geom['jac_ev'], geom['w_ev'],
            sigma, tmp, M,
        )
        result[start:end] = tmp
    return result


# ---------------------------------------------------------------------------
def run_bem(M, a=2.0, b=1.0, grid_n=200):
    geom, F, setup_time = bem_setup(M, a, b)

    iters = [0]
    def callback(r): iters[0] += 1

    op = LinearOperator(
        (M, M),
        matvec=lambda s: bem_matvec(s, geom),
        dtype=np.float64,
    )

    t1 = time.perf_counter()
    sigma, info = gmres(op, F, rtol=1e-10, restart=100, maxiter=500,
                        callback=callback, callback_type='pr_norm')
    solve_time = time.perf_counter() - t1

    t2  = time.perf_counter()
    gx  = np.linspace(-a, a, grid_n)
    gy  = np.linspace(-b, b, grid_n)
    GX, GY   = np.meshgrid(gx, gy)
    GXf, GYf = GX.ravel(), GY.ravel()
    mask     = (GXf/a)**2 + (GYf/b)**2 < 0.95
    ex_int, ey_int = GXf[mask], GYf[mask]

    u_num  = bem_evaluate_chunked(ex_int, ey_int, geom, sigma)
    u_ex   = ex_int**2 - ey_int**2
    rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)

    eval_time  = time.perf_counter() - t2
    total_time = setup_time + solve_time + eval_time

    return dict(M=M, iterations=iters[0],
                setup_time=setup_time, solve_time=solve_time,
                eval_time=eval_time, total_time=total_time,
                relative_L2_error=rel_l2)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    a, b   = 2.0, 1.0
    M_list = [4000, 8000, 16000, 32000]

    print("Warming up Numba JIT ...")
    run_bem(32, a, b, grid_n=20)
    print("JIT warm-up done.\n")

    hdr = (f"{'M':>8}  {'Iters':>6}  {'Setup(s)':>9}  {'Solve(s)':>9}"
           f"  {'Eval(s)':>8}  {'Total(s)':>9}  {'Rel L2 Err':>12}")
    print(hdr)
    print("-" * len(hdr))

    for M in M_list:
        r = run_bem(M, a, b)
        print(f"{r['M']:>8}  {r['iterations']:>6}  {r['setup_time']:>9.3f}"
              f"  {r['solve_time']:>9.3f}  {r['eval_time']:>8.3f}"
              f"  {r['total_time']:>9.3f}  {r['relative_L2_error']:>12.3e}")