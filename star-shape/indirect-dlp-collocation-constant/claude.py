import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit
import time

# ── Gaussian quadrature on [-1,1] ────────────────────────────────────────────
_GQ_PTS, _GQ_WTS = np.polynomial.legendre.leggauss(8)
GQ_PTS  = np.ascontiguousarray(_GQ_PTS)
GQ_WTS  = np.ascontiguousarray(_GQ_WTS)

# ── Star-shaped boundary ──────────────────────────────────────────────────────
def r_star(t):   return 1.0 + 0.3 * np.cos(5.0 * t)
def dr_star(t):  return -1.5 * np.sin(5.0 * t)

# ── Boundary discretization (constant elements) ───────────────────────────────
def make_boundary(N, gp, gw):
    nq      = len(gp)
    dtheta  = 2.0 * np.pi / N
    th_mid  = np.linspace(dtheta / 2.0, 2.0 * np.pi - dtheta / 2.0, N)

    r_m  = r_star(th_mid);  dr_m = dr_star(th_mid)
    mx   = r_m * np.cos(th_mid);   my   = r_m * np.sin(th_mid)
    tx_m = dr_m * np.cos(th_mid) - r_m * np.sin(th_mid)
    ty_m = dr_m * np.sin(th_mid) + r_m * np.cos(th_mid)
    t_len_m = np.sqrt(tx_m**2 + ty_m**2)
    tx_m /= t_len_m;  ty_m /= t_len_m
    nx_m =  ty_m;   ny_m = -tx_m
    if np.mean(nx_m * mx + ny_m * my) < 0.0:
        nx_m = -nx_m;  ny_m = -ny_m

    mids  = np.stack([mx, my], axis=1)
    norms = np.stack([nx_m, ny_m], axis=1)

    qx  = np.zeros((N, nq));  qy  = np.zeros((N, nq))
    qnx = np.zeros((N, nq));  qny = np.zeros((N, nq))
    qwj = np.zeros((N, nq));  ds  = np.zeros(N)

    th_half = 0.5 * dtheta
    for e in range(N):
        tc = th_mid[e]
        for q in range(nq):
            tq   = tc + th_half * gp[q]
            r_q  = r_star(tq);  dr_q = dr_star(tq)
            xq   = r_q * np.cos(tq);   yq   = r_q * np.sin(tq)
            tx_  = dr_q * np.cos(tq) - r_q * np.sin(tq)
            ty_  = dr_q * np.sin(tq) + r_q * np.cos(tq)
            spd  = np.sqrt(tx_**2 + ty_**2)
            tx_ /= spd;  ty_ /= spd
            nxq  =  ty_;  nyq = -tx_
            if nxq * xq + nyq * yq < 0.0:
                nxq = -nxq;  nyq = -nyq
            qx[e, q]  = xq;  qy[e, q]  = yq
            qnx[e, q] = nxq; qny[e, q] = nyq
            wj = gw[q] * spd * th_half
            qwj[e, q] = wj
            ds[e]    += wj

    return (mids, norms, ds,
            np.ascontiguousarray(qx),  np.ascontiguousarray(qy),
            np.ascontiguousarray(qnx), np.ascontiguousarray(qny),
            np.ascontiguousarray(qwj))

# ── Numba matvec: (-1/2 I + K) μ = f  ────────────────────────────────────────
# Interior limit of DLP with outward normal: u(x) = (-1/2 + K)[μ](x)
@njit(cache=True)
def matvec_numba(mu, mx, my, qx, qy, qnx, qny, qwj, N, nq):
    result = np.zeros(N)
    for i in range(N):
        cx = mx[i];  cy = my[i]
        Kval = 0.0
        for e in range(N):
            s = 0.0
            for q in range(nq):
                rx  = cx - qx[e, q]
                ry  = cy - qy[e, q]
                r2  = rx * rx + ry * ry
                if r2 < 1e-28:
                    continue
                T    = (rx * qnx[e, q] + ry * qny[e, q]) / (2.0 * np.pi * r2)
                s   += T * qwj[e, q]
            Kval += s * mu[e]
        result[i] = -0.5 * mu[i] + Kval
    return result

# ── Interior evaluation ───────────────────────────────────────────────────────
@njit(cache=True)
def eval_interior_numba(xpts, ypts, mu, qx, qy, qnx, qny, qwj, N, nq):
    npts = len(xpts)
    u    = np.zeros(npts)
    for p in range(npts):
        cx  = xpts[p];  cy = ypts[p]
        val = 0.0
        for e in range(N):
            s = 0.0
            for q in range(nq):
                rx = cx - qx[e, q]
                ry = cy - qy[e, q]
                r2 = rx * rx + ry * ry
                if r2 < 1e-28:
                    continue
                T  = (rx * qnx[e, q] + ry * qny[e, q]) / (2.0 * np.pi * r2)
                s += T * qwj[e, q]
            val += s * mu[e]
        u[p] = val
    return u

# ── Exact solution & interior mask ───────────────────────────────────────────
def exact_u(x, y):    return x**3 - 3.0 * x * y**2

def inside_star(x, y):
    r     = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r < r_star(theta) - 0.1

# ── BEM solve for given N ─────────────────────────────────────────────────────
def run_bem(N, GQ_PTS, GQ_WTS):
    t0 = time.time()

    (mids, norms, ds,
     qx, qy, qnx, qny, qwj) = make_boundary(N, GQ_PTS, GQ_WTS)

    nq   = len(GQ_PTS)
    mx_c = np.ascontiguousarray(mids[:, 0])
    my_c = np.ascontiguousarray(mids[:, 1])

    t_setup = time.time() - t0

    _ = matvec_numba(np.zeros(N), mx_c, my_c, qx, qy, qnx, qny, qwj, N, nq)

    rhs = exact_u(mx_c, my_c)

    iters = [0]
    def callback(pr_norm):
        iters[0] += 1

    def mv(mu_vec):
        return matvec_numba(mu_vec, mx_c, my_c, qx, qy, qnx, qny, qwj, N, nq)

    op = LinearOperator((N, N), matvec=mv, dtype=float)

    t1   = time.time()
    mu, info = gmres(op, rhs, rtol=1e-12, atol=1e-12, maxiter=1000,
                     callback=callback, callback_type='pr_norm')
    t_solve = time.time() - t1

    ngrid = 60
    xg    = np.linspace(-1.5, 1.5, ngrid)
    yg    = np.linspace(-1.5, 1.5, ngrid)
    XX, YY = np.meshgrid(xg, yg)
    xf    = XX.ravel();  yf = YY.ravel()
    mask  = inside_star(xf, yf)
    xi    = np.ascontiguousarray(xf[mask])
    yi    = np.ascontiguousarray(yf[mask])

    t2    = time.time()
    u_num = eval_interior_numba(xi, yi, mu, qx, qy, qnx, qny, qwj, N, nq)
    t_eval = time.time() - t2

    u_ex    = exact_u(xi, yi)
    rel_err = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)
    t_total = time.time() - t0

    return N, iters[0], rel_err, t_setup, t_solve, t_eval, t_total

# ── Refinement study ──────────────────────────────────────────────────────────
print("Warming up Numba JIT...")
run_bem(10, GQ_PTS, GQ_WTS)
print("JIT warm-up done.\n")

N_values = [ 160, 320, 640, 1280, 2560, 5120]
results  = []

for N in N_values:
    res = run_bem(N, GQ_PTS, GQ_WTS)
    results.append(res)

print(f"{'N':<5} {'Unknowns':<12} {'GMRES':<8} {'Rel L2 Error':<14} {'Conv Rate':<12} "
      f"{'Setup':<10} {'Solve':<10} {'Eval':<10} {'Total':<10}")
print("-" * 95)

log_h = [];  log_e = []

for k, (N, niters, rel_err, ts, tsol, tev, ttot) in enumerate(results):
    h = 2.0 * np.pi / N
    log_h.append(np.log(h));  log_e.append(np.log(rel_err))

    if k == 0:
        rate_str = "   -   "
    else:
        prev_err = results[k-1][2];  prev_N = results[k-1][0]
        rate     = np.log(prev_err / rel_err) / np.log(N / prev_N)
        rate_str = f"{rate:.2f}"

    print(f"{N:<5} {N:<12} {niters:<8} {rel_err:<14.4e} {rate_str:<12} "
          f"{ts:<10.3f} {tsol:<10.3f} {tev:<10.3f} {ttot:<10.3f}")

coeffs = np.polyfit(log_h, log_e, 1)
print(f"\nFinal observed convergence order (least squares fit): {coeffs[0]:.2f}")