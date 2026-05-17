# gpt code for indirect after one iteration
import numpy as np
import time
from numba import njit, prange
from scipy.sparse.linalg import gmres, LinearOperator

# ============================================================
# Exact solution
# ============================================================

def u_exact(x, y):
    return x**3 - 3.0 * x * y**2


# ============================================================
# Star-shaped boundary sampled by nodes, then flat panels
# ============================================================

def boundary_nodes(N):
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    r = 1.0 + 0.3 * np.cos(5.0 * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, theta

def build_flat_geometry(N):
    x_nodes, y_nodes, theta_nodes = boundary_nodes(N)

    x1 = x_nodes
    y1 = y_nodes
    x2 = np.roll(x_nodes, -1)
    y2 = np.roll(y_nodes, -1)

    dx = x2 - x1
    dy = y2 - y1
    lengths = np.sqrt(dx * dx + dy * dy)

    xmid = 0.5 * (x1 + x2)
    ymid = 0.5 * (y1 + y2)

    tx = dx / lengths
    ty = dy / lengths

    # CCW orientation => outward normal is right-hand normal
    nx = ty
    ny = -tx

    return x_nodes, y_nodes, x1, y1, x2, y2, xmid, ymid, nx, ny, lengths


# ============================================================
# Quadrature on flat panels
# ============================================================

def build_flat_quadrature(x1, y1, x2, y2, nx, ny, lengths, ngl=16):
    glx, glw = np.polynomial.legendre.leggauss(ngl)
    glx = glx.astype(np.float64)
    glw = glw.astype(np.float64)

    N = len(x1)
    xq = np.empty((N, ngl), dtype=np.float64)
    yq = np.empty((N, ngl), dtype=np.float64)
    nxq = np.empty((N, ngl), dtype=np.float64)
    nyq = np.empty((N, ngl), dtype=np.float64)
    wq = np.empty((N, ngl), dtype=np.float64)

    for j in range(N):
        dx = x2[j] - x1[j]
        dy = y2[j] - y1[j]
        L = lengths[j]
        for q in range(ngl):
            t = glx[q]
            s = 0.5 * (1.0 + t)
            xq[j, q] = x1[j] + s * dx
            yq[j, q] = y1[j] + s * dy
            nxq[j, q] = nx[j]
            nyq[j, q] = ny[j]
            wq[j, q] = 0.5 * L * glw[q]

    return xq, yq, nxq, nyq, wq


# ============================================================
# Numba kernels
# ============================================================

@njit(cache=True, fastmath=True, parallel=True)
def matvec_kernel(mu, xmid, ymid, xq, yq, nxq, nyq, wq, self_term):
    N = mu.shape[0]
    nq = xq.shape[1]
    out = np.empty(N, dtype=np.float64)
    inv2pi = 1.0 / (2.0 * np.pi)

    for i in prange(N):
        xt = xmid[i]
        yt = ymid[i]
        acc = -0.5 * mu[i] + self_term[i] * mu[i]  # self_term is explicitly zero

        for j in range(N):
            if j == i:
                continue  # flat self-panel DLP kernel is exactly zero
            s = 0.0
            for q in range(nq):
                dx = xt - xq[j, q]
                dy = yt - yq[j, q]
                r2 = dx * dx + dy * dy
                if r2 > 0.0:
                    kern = inv2pi * (dx * nxq[j, q] + dy * nyq[j, q]) / r2
                    s += kern * wq[j, q]
            acc += mu[j] * s

        out[i] = acc

    return out


@njit(cache=True, fastmath=True, parallel=True)
def eval_interior_kernel(xp, yp, mu, xq, yq, nxq, nyq, wq):
    M = xp.shape[0]
    N = mu.shape[0]
    nq = xq.shape[1]
    out = np.empty(M, dtype=np.float64)
    inv2pi = 1.0 / (2.0 * np.pi)

    for p in prange(M):
        xt = xp[p]
        yt = yp[p]
        acc = 0.0
        for j in range(N):
            s = 0.0
            for q in range(nq):
                dx = xt - xq[j, q]
                dy = yt - yq[j, q]
                r2 = dx * dx + dy * dy
                if r2 > 0.0:
                    kern = inv2pi * (dx * nxq[j, q] + dy * nyq[j, q]) / r2
                    s += kern * wq[j, q]
            acc += mu[j] * s
        out[p] = acc

    return out


# ============================================================
# Numba warm-up
# ============================================================

def warmup_numba():
    mu = np.ones(2, dtype=np.float64)
    xmid = np.array([0.0, 1.0], dtype=np.float64)
    ymid = np.array([0.0, 0.0], dtype=np.float64)
    xq = np.array([[0.5, 0.6], [1.5, 1.6]], dtype=np.float64)
    yq = np.array([[0.0, 0.1], [0.0, -0.1]], dtype=np.float64)
    nxq = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    nyq = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    wq = np.array([[0.2, 0.2], [0.2, 0.2]], dtype=np.float64)
    self_term = np.array([0.0, 0.0], dtype=np.float64)
    _ = matvec_kernel(mu, xmid, ymid, xq, yq, nxq, nyq, wq, self_term)
    xp = np.array([0.1], dtype=np.float64)
    yp = np.array([0.1], dtype=np.float64)
    _ = eval_interior_kernel(xp, yp, mu, xq, yq, nxq, nyq, wq)

warmup_numba()


# ============================================================
# Fixed interior grid
# ============================================================

ngrid = 60
xg = np.linspace(-1.5, 1.5, ngrid)
yg = np.linspace(-1.5, 1.5, ngrid)
Xg, Yg = np.meshgrid(xg, yg, indexing="xy")
Rg = np.sqrt(Xg * Xg + Yg * Yg)
Thetag = np.mod(np.arctan2(Yg, Xg), 2.0 * np.pi)
Rbdry = 1.0 + 0.3 * np.cos(5.0 * Thetag)
inside = Rg < (Rbdry - 0.1)

xp_all = Xg[inside].astype(np.float64).ravel()
yp_all = Yg[inside].astype(np.float64).ravel()
uex_all = u_exact(xp_all, yp_all)
uex_norm = np.linalg.norm(uex_all)


# ============================================================
# One BEM solve
# ============================================================

def run_case(N):
    t0 = time.perf_counter()

    x_nodes, y_nodes, x1, y1, x2, y2, xmid, ymid, nx, ny, lengths = build_flat_geometry(N)

    ngl = 16
    xq, yq, nxq, nyq, wq = build_flat_quadrature(x1, y1, x2, y2, nx, ny, lengths, ngl=ngl)

    self_term = np.zeros(N, dtype=np.float64)

    rhs = u_exact(xmid, ymid)

    def mv(v):
        v = np.asarray(v, dtype=np.float64)
        return matvec_kernel(v, xmid, ymid, xq, yq, nxq, nyq, wq, self_term)

    A = LinearOperator((N, N), matvec=mv, dtype=np.float64)

    setup_time = time.perf_counter() - t0

    iter_count = [0]
    def cb(residual_norm):
        iter_count[0] += 1

    t1 = time.perf_counter()
    mu, info = gmres(
        A,
        rhs,
        rtol=1e-10,
        atol=1e-12,
        restart=min(50, N),
        maxiter=1000,
        callback=cb,
        callback_type="pr_norm",
    )
    solve_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    u_num = eval_interior_kernel(xp_all, yp_all, mu, xq, yq, nxq, nyq, wq)
    rel_l2 = np.linalg.norm(u_num - uex_all) / uex_norm
    eval_time = time.perf_counter() - t2

    total_time = setup_time + solve_time + eval_time

    return {
        "N": N,
        "unknowns": N,
        "gmres": int(iter_count[0]),
        "rel_err": float(rel_l2),
        "setup": float(setup_time),
        "solve": float(solve_time),
        "eval": float(eval_time),
        "total": float(total_time),
        "info": int(info),
    }


# ============================================================
# Refinement study
# ============================================================

N_values = [1280, 2560, 5120]
results = []

for N in N_values:
    results.append(run_case(N))

errors = np.array([r["rel_err"] for r in results], dtype=np.float64)
hs = np.array([1.0 / r["N"] for r in results], dtype=np.float64)
order = np.polyfit(np.log(hs), np.log(errors), 1)[0]

conv_rates = [None]
for k in range(1, len(results)):
    conv_rates.append(np.log(results[k - 1]["rel_err"] / results[k]["rel_err"]) / np.log(2.0))

print("N   Unknowns    GMRES    Rel L2 Error  Conv Rate     Setup     Solve      Eval     Total")
for k, r in enumerate(results):
    conv_str = "-" if conv_rates[k] is None else f"{conv_rates[k]:.2f}"
    print(
        f"{r['N']:>4d}"
        f"{r['unknowns']:>11d}"
        f"{r['gmres']:>9d}"
        f"{r['rel_err']:>15.6e}"
        f"{conv_str:>10}"
        f"{r['setup']:>10.2f}"
        f"{r['solve']:>10.2f}"
        f"{r['eval']:>10.2f}"
        f"{r['total']:>10.2f}"
    )

print(f"Observed order: {order:.6f}")