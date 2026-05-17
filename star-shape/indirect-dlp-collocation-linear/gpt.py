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
# Boundary definition
# ============================================================

def r_func(theta):
    return 1.0 + 0.3 * np.cos(5.0 * theta)


def build_linear_geometry(N):
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    r = r_func(theta)
    x_nodes = r * np.cos(theta)
    y_nodes = r * np.sin(theta)

    # Ensure CCW orientation
    signed_area = 0.5 * np.sum(x_nodes * np.roll(y_nodes, -1) - np.roll(x_nodes, -1) * y_nodes)
    if signed_area < 0.0:
        x_nodes = x_nodes[::-1].copy()
        y_nodes = y_nodes[::-1].copy()

    x1 = x_nodes.copy()
    y1 = y_nodes.copy()
    x2 = np.roll(x_nodes, -1)
    y2 = np.roll(y_nodes, -1)

    dx = x2 - x1
    dy = y2 - y1
    lengths = np.sqrt(dx * dx + dy * dy)

    xcol = 0.5 * (x1 + x2)
    ycol = 0.5 * (y1 + y2)

    tx = dx / lengths
    ty = dy / lengths

    # CCW polygon => outward normal is the right-hand normal
    nx = ty
    ny = -tx

    return x_nodes, y_nodes, x1, y1, x2, y2, dx, dy, xcol, ycol, nx, ny, lengths


def build_quadrature_ref(ngl=16):
    xi, wi = np.polynomial.legendre.leggauss(ngl)
    xi = xi.astype(np.float64)
    wi = wi.astype(np.float64)

    # map from [-1,1] -> [0,1]
    t = 0.5 * (xi + 1.0)
    w = 0.5 * wi

    phi1 = 1.0 - t
    phi2 = t
    return t, phi1, phi2, w


# ============================================================
# Numba kernels
# ============================================================

@njit(cache=True, fastmath=True, parallel=True)
def matvec_kernel(mu, xcol, ycol, x1, y1, x2, y2, dx, dy, nx, ny, lengths, t_ref, phi1_ref, phi2_ref, w_ref):
    N = mu.shape[0]
    nq = t_ref.shape[0]
    out = np.empty(N, dtype=np.float64)
    inv2pi = 1.0 / (2.0 * np.pi)

    for i in prange(N):
        xt = xcol[i]
        yt = ycol[i]

        ip1 = i + 1
        if ip1 == N:
            ip1 = 0

        # Interior jump term evaluated at panel midpoint:
        # -1/2 * mu(x_i) with mu(x_i) = 0.5*(mu_i + mu_{i+1})
        acc = -0.25 * (mu[i] + mu[ip1])

        for j in range(N):
            if j == i:
                # Flat self-panel contribution is exactly zero
                continue

            s = 0.0
            for q in range(nq):
                t = t_ref[q]
                ph1 = phi1_ref[q]
                ph2 = phi2_ref[q]

                xs = x1[j] + t * dx[j]
                ys = y1[j] + t * dy[j]

                rx = xt - xs
                ry = yt - ys
                r2 = rx * rx + ry * ry
                if r2 > 0.0:
                    kern = inv2pi * (rx * nx[j] + ry * ny[j]) / r2
                    mu_src = ph1 * mu[j] + ph2 * mu[(j + 1) % N]
                    s += kern * mu_src * lengths[j] * w_ref[q]

            acc += s

        out[i] = acc

    return out


@njit(cache=True, fastmath=True, parallel=True)
def eval_interior_kernel(xp, yp, mu, x1, y1, x2, y2, dx, dy, nx, ny, lengths, t_ref, phi1_ref, phi2_ref, w_ref):
    M = xp.shape[0]
    N = mu.shape[0]
    nq = t_ref.shape[0]
    out = np.empty(M, dtype=np.float64)
    inv2pi = 1.0 / (2.0 * np.pi)

    for p in prange(M):
        xt = xp[p]
        yt = yp[p]
        acc = 0.0

        for j in range(N):
            s = 0.0
            for q in range(nq):
                t = t_ref[q]
                ph1 = phi1_ref[q]
                ph2 = phi2_ref[q]

                xs = x1[j] + t * dx[j]
                ys = y1[j] + t * dy[j]

                rx = xt - xs
                ry = yt - ys
                r2 = rx * rx + ry * ry
                if r2 > 0.0:
                    kern = inv2pi * (rx * nx[j] + ry * ny[j]) / r2
                    mu_src = ph1 * mu[j] + ph2 * mu[(j + 1) % N]
                    s += kern * mu_src * lengths[j] * w_ref[q]

            acc += s

        out[p] = acc

    return out


# ============================================================
# Numba warm-up
# ============================================================

def warmup_numba():
    mu = np.ones(4, dtype=np.float64)
    xcol = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    ycol = np.array([0.0, 0.1, 0.0, -0.1], dtype=np.float64)

    x1 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y1 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    x2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y2 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dx = x2 - x1
    dy = y2 - y1
    nx = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ny = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    lengths = np.ones(4, dtype=np.float64)

    t_ref = np.array([0.25, 0.75], dtype=np.float64)
    phi1_ref = 1.0 - t_ref
    phi2_ref = t_ref
    w_ref = np.array([0.5, 0.5], dtype=np.float64)

    _ = matvec_kernel(mu, xcol, ycol, x1, y1, x2, y2, dx, dy, nx, ny, lengths, t_ref, phi1_ref, phi2_ref, w_ref)
    xp = np.array([0.1], dtype=np.float64)
    yp = np.array([0.2], dtype=np.float64)
    _ = eval_interior_kernel(xp, yp, mu, x1, y1, x2, y2, dx, dy, nx, ny, lengths, t_ref, phi1_ref, phi2_ref, w_ref)

warmup_numba()


# ============================================================
# Interior grid (must use exactly this function)
# ============================================================

def build_interior_grid():
    ngrid = 60
    _grid = np.linspace(-1.5, 1.5, ngrid)
    gx, gy = np.meshgrid(_grid, _grid)

    gx = gx.flatten()
    gy = gy.flatten()

    r_val = np.sqrt(gx**2 + gy**2)
    theta_val = np.arctan2(gy, gx)
    r_bound = r_func(theta_val)

    mask = r_val < r_bound - 0.1

    return gx[mask], gy[mask]


xp_all, yp_all = build_interior_grid()
u_exact_all = u_exact(xp_all, yp_all)
u_exact_norm = np.linalg.norm(u_exact_all)

# Quadrature references
NGQ = 16
t_ref, phi1_ref, phi2_ref, w_ref = build_quadrature_ref(NGQ)


# ============================================================
# One solve
# ============================================================

def run_case(N):
    t0 = time.perf_counter()

    x_nodes, y_nodes, x1, y1, x2, y2, dx, dy, xcol, ycol, nx, ny, lengths = build_linear_geometry(N)

    rhs = u_exact(xcol, ycol)

    def mv(v):
        v = np.asarray(v, dtype=np.float64)
        return matvec_kernel(v, xcol, ycol, x1, y1, x2, y2, dx, dy, nx, ny, lengths, t_ref, phi1_ref, phi2_ref, w_ref)

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
        callback_type='pr_norm',
    )
    solve_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    u_num = eval_interior_kernel(xp_all, yp_all, mu, x1, y1, x2, y2, dx, dy, nx, ny, lengths, t_ref, phi1_ref, phi2_ref, w_ref)
    rel_l2 = np.linalg.norm(u_num - u_exact_all) / u_exact_norm
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