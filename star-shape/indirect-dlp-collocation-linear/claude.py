import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit
import time

# ── Gaussian quadrature ───────────────────────────────────────────────────────
_GQ_PTS, _GQ_WTS = np.polynomial.legendre.leggauss(8)
GQ_PTS = np.ascontiguousarray(_GQ_PTS)
GQ_WTS = np.ascontiguousarray(_GQ_WTS)

# ── Star boundary ─────────────────────────────────────────────────────────────
def r_func(t):  return 1.0 + 0.3 * np.cos(5.0 * t)
def dr_func(t): return -1.5 * np.sin(5.0 * t)

# ── Interior grid ─────────────────────────────────────────────────────────────
def build_interior_grid():
    ngrid = 60
    _grid = np.linspace(-1.5, 1.5, ngrid)
    gx, gy = np.meshgrid(_grid, _grid)
    gx = gx.flatten()
    gy = gy.flatten()
    r_val     = np.sqrt(gx**2 + gy**2)
    theta_val = np.arctan2(gy, gx)
    r_bound   = r_func(theta_val)
    mask      = r_val < r_bound - 0.1
    return gx[mask], gy[mask]

# ── Exact solution ────────────────────────────────────────────────────────────
def exact_u(x, y): return x**3 - 3.0 * x * y**2

# ── Boundary discretization (linear elements, node-based DOFs) ───────────────
def make_boundary(N, gp, gw):
    """
    N nodes, N elements (periodic).
    Returns:
        nodes  (N,2)   node coordinates
        normals (N,2)  outward unit normals at nodes (averaged from adjacent elements)
        qx,qy  (N,nq)  quadrature coords per element
        qnx,qny (N,nq) outward normals at quad pts
        qwj    (N,nq)  weight * jacobian
        conn   (N,2)   element connectivity [n1, n2]
    """
    nq     = len(gp)
    dtheta = 2.0 * np.pi / N
    theta  = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)

    r_n  = r_func(theta);  dr_n = dr_func(theta)
    nx_c = r_n * np.cos(theta);  ny_c = r_n * np.sin(theta)
    nodes = np.stack([nx_c, ny_c], axis=1)

    # Element connectivity: element e connects node e -> node (e+1)%N
    conn = np.stack([np.arange(N), (np.arange(N) + 1) % N], axis=1)

    # Quadrature points per element
    qx  = np.zeros((N, nq)); qy  = np.zeros((N, nq))
    qnx = np.zeros((N, nq)); qny = np.zeros((N, nq))
    qwj = np.zeros((N, nq))

    th_half = 0.5 * dtheta
    for e in range(N):
        tc = theta[e] + th_half          # midpoint theta of element e
        for q in range(nq):
            tq   = tc + th_half * gp[q]
            r_q  = r_func(tq);  dr_q = dr_func(tq)
            xq   = r_q * np.cos(tq);  yq = r_q * np.sin(tq)
            tx_  = dr_q * np.cos(tq) - r_q * np.sin(tq)
            ty_  = dr_q * np.sin(tq) + r_q * np.cos(tq)
            spd  = np.sqrt(tx_**2 + ty_**2)
            tx_ /= spd;  ty_ /= spd
            nxq  =  ty_;  nyq = -tx_
            if nxq * xq + nyq * yq < 0.0:
                nxq = -nxq;  nyq = -nyq
            qx[e, q]  = xq;  qy[e, q]  = yq
            qnx[e, q] = nxq; qny[e, q] = nyq
            qwj[e, q] = gw[q] * spd * th_half

    # Node normals: average of adjacent element normals
    node_nx = np.zeros(N); node_ny = np.zeros(N)
    for e in range(N):
        n1, n2 = conn[e]
        enx = qnx[e].mean(); eny = qny[e].mean()
        node_nx[n1] += enx; node_ny[n1] += eny
        node_nx[n2] += enx; node_ny[n2] += eny
    nrm = np.sqrt(node_nx**2 + node_ny**2)
    node_nx /= nrm; node_ny /= nrm

    return (nodes,
            np.stack([node_nx, node_ny], axis=1),
            np.ascontiguousarray(qx),
            np.ascontiguousarray(qy),
            np.ascontiguousarray(qnx),
            np.ascontiguousarray(qny),
            np.ascontiguousarray(qwj),
            conn.astype(np.int64))

# ── Numba matvec: (-1/2 I + K)[μ] ────────────────────────────────────────────
# Interior limit of DLP with outward normal: u(x) = (-1/2 + K)[μ](x)
# qph1[q] = (1-s)/2, qph2[q] = (1+s)/2 at Gauss point q (precomputed)
@njit(cache=True)
def matvec_linear(mu, node_x, node_y,
                  qx, qy, qnx, qny, qwj, qph1, qph2,
                  conn, N, nq):
    result = np.zeros(N)
    for i in range(N):
        cx = node_x[i];  cy = node_y[i]
        Kval = 0.0
        for e in range(N):
            n1 = conn[e, 0];  n2 = conn[e, 1]
            mu1 = mu[n1];     mu2 = mu[n2]
            for q in range(nq):
                rx = cx - qx[e, q]
                ry = cy - qy[e, q]
                r2 = rx * rx + ry * ry
                if r2 < 1e-28:
                    continue
                T   = (rx * qnx[e, q] + ry * qny[e, q]) / (2.0 * np.pi * r2)
                mu_q = mu1 * qph1[q] + mu2 * qph2[q]
                Kval += T * mu_q * qwj[e, q]
        result[i] = -0.5 * mu[i] + Kval
    return result

# ── Interior evaluation ───────────────────────────────────────────────────────
@njit(cache=True)
def eval_interior_numba(xpts, ypts, mu,
                         qx, qy, qnx, qny, qwj, qph1, qph2,
                         conn, N, nq):
    npts = len(xpts)
    u    = np.zeros(npts)
    for p in range(npts):
        cx = xpts[p];  cy = ypts[p]
        val = 0.0
        for e in range(N):
            n1 = conn[e, 0];  n2 = conn[e, 1]
            mu1 = mu[n1];     mu2 = mu[n2]
            for q in range(nq):
                rx = cx - qx[e, q]
                ry = cy - qy[e, q]
                r2 = rx * rx + ry * ry
                if r2 < 1e-28:
                    continue
                T    = (rx * qnx[e, q] + ry * qny[e, q]) / (2.0 * np.pi * r2)
                mu_q = mu1 * qph1[q] + mu2 * qph2[q]
                val += T * mu_q * qwj[e, q]
        u[p] = val
    return u

# ── Precompute shape function values at Gauss points ─────────────────────────
def make_shape_vals(gp):
    qph1 = np.ascontiguousarray(0.5 * (1.0 - gp))   # phi1 at each quad pt
    qph2 = np.ascontiguousarray(0.5 * (1.0 + gp))   # phi2 at each quad pt
    return qph1, qph2

# ── BEM solve ─────────────────────────────────────────────────────────────────
def run_bem(N, GQ_PTS, GQ_WTS):
    t0 = time.time()

    qph1, qph2 = make_shape_vals(GQ_PTS)

    (nodes, node_norms,
     qx, qy, qnx, qny, qwj, conn) = make_boundary(N, GQ_PTS, GQ_WTS)

    nq    = len(GQ_PTS)
    nx_c  = np.ascontiguousarray(nodes[:, 0])
    ny_c  = np.ascontiguousarray(nodes[:, 1])

    t_setup = time.time() - t0

    # Numba warm-up
    _ = matvec_linear(np.zeros(N), nx_c, ny_c,
                      qx, qy, qnx, qny, qwj, qph1, qph2, conn, N, nq)

    rhs = exact_u(nx_c, ny_c)

    iters = [0]
    def callback(pr_norm):
        iters[0] += 1

    def mv(mu_vec):
        return matvec_linear(mu_vec, nx_c, ny_c,
                             qx, qy, qnx, qny, qwj, qph1, qph2, conn, N, nq)

    op = LinearOperator((N, N), matvec=mv, dtype=float)

    t1 = time.time()
    mu, info = gmres(op, rhs, rtol=1e-12, atol=1e-12, maxiter=1000,
                     callback=callback, callback_type='pr_norm')
    t_solve = time.time() - t1

    xi, yi = build_interior_grid()
    xi = np.ascontiguousarray(xi)
    yi = np.ascontiguousarray(yi)

    t2    = time.time()
    u_num = eval_interior_numba(xi, yi, mu,
                                 qx, qy, qnx, qny, qwj, qph1, qph2,
                                 conn, N, nq)
    t_eval = time.time() - t2

    u_ex    = exact_u(xi, yi)
    rel_err = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)
    t_total = time.time() - t0

    return N, iters[0], rel_err, t_setup, t_solve, t_eval, t_total

# ── Refinement study ──────────────────────────────────────────────────────────
print("Warming up Numba JIT...")
run_bem(10, GQ_PTS, GQ_WTS)
print("JIT warm-up done.\n")

N_values = [ 320, 640, 1280, 2560, 5120]
results  = []
for N in N_values:
    res = run_bem(N, GQ_PTS, GQ_WTS)
    results.append(res)

# ── Table ─────────────────────────────────────────────────────────────────────
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