import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit
import time

# ── Gaussian quadrature ───────────────────────────────────────────────────────
_GQ_PTS, _GQ_WTS = np.polynomial.legendre.leggauss(8)
GQ_PTS = np.ascontiguousarray(_GQ_PTS)
GQ_WTS = np.ascontiguousarray(_GQ_WTS)

# Higher order for singular self-interactions
_GQ_PTS16, _GQ_WTS16 = np.polynomial.legendre.leggauss(16)
GQ_PTS16 = np.ascontiguousarray(_GQ_PTS16)
GQ_WTS16 = np.ascontiguousarray(_GQ_WTS16)

# ── Boundary geometry ─────────────────────────────────────────────────────────
def r_func(t):  return 1.0 + 0.3 * np.cos(5.0 * t)
def dr_func(t): return -1.5 * np.sin(5.0 * t)

def exact_u(x, y): return x**3 - 3.0 * x * y**2

# ── Interior grid ─────────────────────────────────────────────────────────────
def build_interior_grid():
    ngrid = 60
    _grid = np.linspace(-1.5, 1.5, ngrid)
    gx, gy = np.meshgrid(_grid, _grid)
    gx = gx.flatten(); gy = gy.flatten()
    r_val     = np.sqrt(gx**2 + gy**2)
    theta_val = np.arctan2(gy, gx)
    r_bound   = r_func(theta_val)
    mask      = r_val < r_bound - 0.1
    return gx[mask], gy[mask]

# ── Boundary discretization ───────────────────────────────────────────────────
def make_boundary(N, gp, gw, gp16, gw16):
    nq     = len(gp)
    nq16   = len(gp16)
    dtheta = 2.0 * np.pi / N
    theta  = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)

    r_n  = r_func(theta); dr_n = dr_func(theta)
    nodes = np.stack([r_n * np.cos(theta), r_n * np.sin(theta)], axis=1)

    # Connectivity: element e -> nodes [e, (e+1)%N]
    conn = np.stack([np.arange(N), (np.arange(N) + 1) % N], axis=1).astype(np.int64)

    # Per-element quadrature arrays (nq points)
    # Each element spans theta[e] to theta[e]+dtheta
    def elem_quad(tc, nq_, gp_, gw_):
        th_half = 0.5 * dtheta
        xq = np.zeros(nq_); yq = np.zeros(nq_)
        nxq = np.zeros(nq_); nyq = np.zeros(nq_)
        wjq = np.zeros(nq_); ph1 = np.zeros(nq_); ph2 = np.zeros(nq_)
        for q in range(nq_):
            tq   = tc + th_half * gp_[q]
            r_q  = r_func(tq); dr_q = dr_func(tq)
            xq[q]  = r_q * np.cos(tq); yq[q]  = r_q * np.sin(tq)
            tx_ = dr_q * np.cos(tq) - r_q * np.sin(tq)
            ty_ = dr_q * np.sin(tq) + r_q * np.cos(tq)
            spd = np.sqrt(tx_**2 + ty_**2)
            tx_ /= spd; ty_ /= spd
            nxq[q] = ty_; nyq[q] = -tx_
            if nxq[q]*xq[q] + nyq[q]*yq[q] < 0.0:
                nxq[q] = -nxq[q]; nyq[q] = -nyq[q]
            wjq[q] = gw_[q] * spd * th_half
            ph1[q] = 0.5*(1.0 - gp_[q])
            ph2[q] = 0.5*(1.0 + gp_[q])
        return xq, yq, nxq, nyq, wjq, ph1, ph2

    # Standard quadrature (nq=8)
    qx  = np.zeros((N, nq)); qy  = np.zeros((N, nq))
    qnx = np.zeros((N, nq)); qny = np.zeros((N, nq))
    qwj = np.zeros((N, nq)); qph1 = np.zeros((N, nq)); qph2 = np.zeros((N, nq))

    # High-order quadrature (nq16=16) for self-element integrals
    qx16  = np.zeros((N, nq16)); qy16  = np.zeros((N, nq16))
    qnx16 = np.zeros((N, nq16)); qny16 = np.zeros((N, nq16))
    qwj16 = np.zeros((N, nq16)); qph116 = np.zeros((N, nq16)); qph216 = np.zeros((N, nq16))

    for e in range(N):
        tc = theta[e] + 0.5*dtheta
        (qx[e],  qy[e],  qnx[e],  qny[e],  qwj[e],  qph1[e],  qph2[e])  = elem_quad(tc, nq,   gp,   gw)
        (qx16[e],qy16[e],qnx16[e],qny16[e],qwj16[e],qph116[e],qph216[e]) = elem_quad(tc, nq16, gp16, gw16)

    return (nodes, conn,
            np.ascontiguousarray(qx),   np.ascontiguousarray(qy),
            np.ascontiguousarray(qnx),  np.ascontiguousarray(qny),
            np.ascontiguousarray(qwj),  np.ascontiguousarray(qph1),
            np.ascontiguousarray(qph2),
            np.ascontiguousarray(qx16), np.ascontiguousarray(qy16),
            np.ascontiguousarray(qnx16),np.ascontiguousarray(qny16),
            np.ascontiguousarray(qwj16),np.ascontiguousarray(qph116),
            np.ascontiguousarray(qph216))

# ── Galerkin matvec ───────────────────────────────────────────────────────────
# A[i,j] = (-1/2) M[i,j]  +  K[i,j]
# Interior limit: (-1/2 + K)[mu] = f  =>  Galerkin: (-1/2)*mass + K_galerkin = rhs
# K[i,j] = integral_Gamma integral_Gamma phi_i(x) T(x,y) phi_j(y) ds_y ds_x
# T(x,y) = 1/(2pi) (x-y).n_y / |x-y|^2
#
# Matrix-free: for given mu, compute (A mu)[i] for each test node i.
# Node i is shared by elements (i-1) and i (periodic).
# For test function phi_i: non-zero on elements ei_left=(i-1)%N and ei_right=i.

@njit(cache=True)
def matvec_galerkin(mu, conn, N, nq, nq16,
                    qx, qy, qnx, qny, qwj, qph1, qph2,
                    qx16, qy16, qnx16, qny16, qwj16, qph116, qph216):
    result = np.zeros(N)

    for i in range(N):
        # Test node i has support on elements: ei_L = (i-1)%N, ei_R = i
        # On ei_L: node i corresponds to phi2 (s=+1 end), local index 1
        # On ei_R: node i corresponds to phi1 (s=-1 start), local index 0
        ei_L = (i - 1) % N
        ei_R = i

        Av = 0.0

        # ── Mass term: (-1/2) ∫ phi_i(x) phi_j(x) ds_x ────────────────────
        # Only diagonal j==i contributes appreciably (phi_i * phi_i on support)
        # sum over j: (-1/2) sum_j mu_j ∫ phi_i phi_j ds_x
        # phi_i, phi_j both nonzero only on shared support

        # Contribution from ei_L: test=phi2, trial=phi1(n1) and phi2(n2)
        n1L = conn[ei_L, 0]; n2L = conn[ei_L, 1]  # n2L == i
        for q in range(nq):
            phi_test = qph2[ei_L, q]  # test function phi_i on ei_L
            ds = qwj[ei_L, q]
            # phi_n1L * mu[n1L] + phi_n2L * mu[n2L]
            mu_x = qph1[ei_L, q] * mu[n1L] + qph2[ei_L, q] * mu[n2L]
            Av += -0.5 * phi_test * mu_x * ds

        # Contribution from ei_R: test=phi1, trial=phi1(n1) and phi2(n2)
        n1R = conn[ei_R, 0]; n2R = conn[ei_R, 1]  # n1R == i
        for q in range(nq):
            phi_test = qph1[ei_R, q]  # test function phi_i on ei_R
            ds = qwj[ei_R, q]
            mu_x = qph1[ei_R, q] * mu[n1R] + qph2[ei_R, q] * mu[n2R]
            Av += -0.5 * phi_test * mu_x * ds

        # ── Double-layer K term ─────────────────────────────────────────────
        # K contribution: sum over source elements ej, for each test element ei

        # Test on ei_L (phi_test = phi2)
        for ej in range(N):
            n1j = conn[ej, 0]; n2j = conn[ej, 1]
            same = (ej == ei_L)
            # choose quadrature order
            nq_s  = nq16   if same else nq
            qx_s  = qx16   if same else qx
            qy_s  = qy16   if same else qy
            qnx_s = qnx16  if same else qnx
            qny_s = qny16  if same else qny
            qwj_s = qwj16  if same else qwj
            qp1_s = qph116 if same else qph1
            qp2_s = qph216 if same else qph2

            for qx_ in range(nq):
                # test quad point on ei_L
                xi_x = qx[ei_L, qx_]
                xi_y = qy[ei_L, qx_]
                phi_test = qph2[ei_L, qx_]
                wjx = qwj[ei_L, qx_]

                inner = 0.0
                for qy_ in range(nq_s):
                    rx = xi_x - qx_s[ej, qy_]
                    ry = xi_y - qy_s[ej, qy_]
                    r2 = rx*rx + ry*ry
                    if r2 < 1e-28:
                        continue
                    T    = (rx*qnx_s[ej,qy_] + ry*qny_s[ej,qy_]) / (2.0*np.pi*r2)
                    mu_y = qp1_s[ej,qy_]*mu[n1j] + qp2_s[ej,qy_]*mu[n2j]
                    inner += T * mu_y * qwj_s[ej, qy_]

                Av += phi_test * inner * wjx

        # Test on ei_R (phi_test = phi1)
        for ej in range(N):
            n1j = conn[ej, 0]; n2j = conn[ej, 1]
            same = (ej == ei_R)
            nq_s  = nq16   if same else nq
            qx_s  = qx16   if same else qx
            qy_s  = qy16   if same else qy
            qnx_s = qnx16  if same else qnx
            qny_s = qny16  if same else qny
            qwj_s = qwj16  if same else qwj
            qp1_s = qph116 if same else qph1
            qp2_s = qph216 if same else qph2

            for qx_ in range(nq):
                xi_x = qx[ei_R, qx_]
                xi_y = qy[ei_R, qx_]
                phi_test = qph1[ei_R, qx_]
                wjx = qwj[ei_R, qx_]

                inner = 0.0
                for qy_ in range(nq_s):
                    rx = xi_x - qx_s[ej, qy_]
                    ry = xi_y - qy_s[ej, qy_]
                    r2 = rx*rx + ry*ry
                    if r2 < 1e-28:
                        continue
                    T    = (rx*qnx_s[ej,qy_] + ry*qny_s[ej,qy_]) / (2.0*np.pi*r2)
                    mu_y = qp1_s[ej,qy_]*mu[n1j] + qp2_s[ej,qy_]*mu[n2j]
                    inner += T * mu_y * qwj_s[ej, qy_]

                Av += phi_test * inner * wjx

        result[i] = Av

    return result

# ── RHS assembly ──────────────────────────────────────────────────────────────
@njit(cache=True)
def assemble_rhs(conn, N, nq, qx, qy, qwj, qph1, qph2, f_vals_n):
    """b_i = integral phi_i(x) f(x) ds_x, f evaluated at quad pts via nodal values"""
    rhs = np.zeros(N)
    for e in range(N):
        n1 = conn[e, 0]; n2 = conn[e, 1]
        for q in range(nq):
            f_q = qph1[e,q]*f_vals_n[n1] + qph2[e,q]*f_vals_n[n2]
            # test phi1 -> node n1
            rhs[n1] += qph1[e,q] * f_q * qwj[e,q]
            # test phi2 -> node n2
            rhs[n2] += qph2[e,q] * f_q * qwj[e,q]
    return rhs

# ── Interior evaluation ───────────────────────────────────────────────────────
@njit(cache=True)
def eval_interior(xpts, ypts, mu, conn, N, nq, qx, qy, qnx, qny, qwj, qph1, qph2):
    npts = len(xpts)
    u = np.zeros(npts)
    for p in range(npts):
        cx = xpts[p]; cy = ypts[p]
        val = 0.0
        for e in range(N):
            n1 = conn[e,0]; n2 = conn[e,1]
            for q in range(nq):
                rx = cx - qx[e,q]; ry = cy - qy[e,q]
                r2 = rx*rx + ry*ry
                if r2 < 1e-28: continue
                T    = (rx*qnx[e,q] + ry*qny[e,q]) / (2.0*np.pi*r2)
                mu_q = qph1[e,q]*mu[n1] + qph2[e,q]*mu[n2]
                val += T * mu_q * qwj[e,q]
        u[p] = val
    return u

# ── BEM solve ─────────────────────────────────────────────────────────────────
def run_bem(N, GQ_PTS, GQ_WTS, GQ_PTS16, GQ_WTS16):
    t0 = time.time()

    (nodes, conn,
     qx, qy, qnx, qny, qwj, qph1, qph2,
     qx16, qy16, qnx16, qny16, qwj16, qph116, qph216) = make_boundary(
         N, GQ_PTS, GQ_WTS, GQ_PTS16, GQ_WTS16)

    nq   = len(GQ_PTS)
    nq16 = len(GQ_PTS16)

    # RHS: f at nodes
    f_nodes = exact_u(nodes[:,0], nodes[:,1])
    rhs = assemble_rhs(conn, N, nq, qx, qy, qwj, qph1, qph2,
                       np.ascontiguousarray(f_nodes))

    t_setup = time.time() - t0

    # Warm up Numba
    _ = matvec_galerkin(np.zeros(N), conn, N, nq, nq16,
                        qx, qy, qnx, qny, qwj, qph1, qph2,
                        qx16, qy16, qnx16, qny16, qwj16, qph116, qph216)

    iters = [0]
    def callback(pr_norm): iters[0] += 1

    def mv(mu_vec):
        return matvec_galerkin(mu_vec, conn, N, nq, nq16,
                               qx, qy, qnx, qny, qwj, qph1, qph2,
                               qx16, qy16, qnx16, qny16, qwj16, qph116, qph216)

    op = LinearOperator((N, N), matvec=mv, dtype=float)

    t1 = time.time()
    mu, info = gmres(op, rhs, rtol=1e-12, atol=1e-12, maxiter=1000,
                     callback=callback, callback_type='pr_norm')
    t_solve = time.time() - t1

    xi, yi = build_interior_grid()
    xi = np.ascontiguousarray(xi); yi = np.ascontiguousarray(yi)

    t2 = time.time()
    u_num = eval_interior(xi, yi, mu, conn, N, nq, qx, qy, qnx, qny, qwj, qph1, qph2)
    t_eval = time.time() - t2

    u_ex    = exact_u(xi, yi)
    rel_err = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)
    t_total = time.time() - t0

    return N, iters[0], rel_err, t_setup, t_solve, t_eval, t_total

# ── Refinement study ──────────────────────────────────────────────────────────
print("Warming up Numba JIT...")
run_bem(10, GQ_PTS, GQ_WTS, GQ_PTS16, GQ_WTS16)
print("JIT warm-up done.\n")

N_values = [160, 320, 640, 1280, 2560, 5120]
results  = []
for N in N_values:
    res = run_bem(N, GQ_PTS, GQ_WTS, GQ_PTS16, GQ_WTS16)
    results.append(res)

print(f"{'N':<5} {'Unknowns':<12} {'GMRES':<8} {'Rel L2 Error':<14} {'Conv Rate':<12} "
      f"{'Setup':<10} {'Solve':<10} {'Eval':<10} {'Total':<10}")
print("-" * 95)

log_h = []; log_e = []
for k, (N, niters, rel_err, ts, tsol, tev, ttot) in enumerate(results):
    h = 2.0 * np.pi / N
    log_h.append(np.log(h)); log_e.append(np.log(rel_err))
    if k == 0:
        rate_str = "   -   "
    else:
        prev_err = results[k-1][2]; prev_N = results[k-1][0]
        rate     = np.log(prev_err / rel_err) / np.log(N / prev_N)
        rate_str = f"{rate:.2f}"
    print(f"{N:<5} {N:<12} {niters:<8} {rel_err:<14.4e} {rate_str:<12} "
          f"{ts:<10.3f} {tsol:<10.3f} {tev:<10.3f} {ttot:<10.3f}")

coeffs = np.polyfit(log_h, log_e, 1)
print(f"\nFinal observed convergence order (least squares fit): {coeffs[0]:.2f}")