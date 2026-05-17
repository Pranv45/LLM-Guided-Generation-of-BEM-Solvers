import numpy as np
import time
import math
from numba import njit
from scipy.sparse.linalg import LinearOperator, gmres


def r_func(theta):
    return 1.0 + 0.3 * np.cos(5.0 * theta)


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


@njit(fastmath=True)
def u_exact_numba(x, y):
    return x * x * x - 3.0 * x * y * y


@njit(fastmath=True)
def build_geometry(N):
    x = np.empty(N, dtype=np.float64)
    y = np.empty(N, dtype=np.float64)

    for i in range(N):
        th = 2.0 * np.pi * i / N
        r = 1.0 + 0.3 * np.cos(5.0 * th)
        x[i] = r * np.cos(th)
        y[i] = r * np.sin(th)

    area2 = 0.0
    for i in range(N):
        j = 0 if i == N - 1 else i + 1
        area2 += x[i] * y[j] - x[j] * y[i]

    if area2 < 0.0:
        xr = np.empty(N, dtype=np.float64)
        yr = np.empty(N, dtype=np.float64)
        for i in range(N):
            xr[i] = x[N - 1 - i]
            yr[i] = y[N - 1 - i]
        x = xr
        y = yr

    x0 = np.empty(N, dtype=np.float64)
    y0 = np.empty(N, dtype=np.float64)
    dx = np.empty(N, dtype=np.float64)
    dy = np.empty(N, dtype=np.float64)
    L = np.empty(N, dtype=np.float64)
    nx = np.empty(N, dtype=np.float64)
    ny = np.empty(N, dtype=np.float64)

    for e in range(N):
        j = 0 if e == N - 1 else e + 1
        x0[e] = x[e]
        y0[e] = y[e]
        dx[e] = x[j] - x[e]
        dy[e] = y[j] - y[e]
        Le = math.sqrt(dx[e] * dx[e] + dy[e] * dy[e])
        L[e] = Le
        tx = dx[e] / Le
        ty = dy[e] / Le
        nx[e] = ty
        ny[e] = -tx

    return x0, y0, dx, dy, L, nx, ny


def gauss01(qorder=8):
    xi, wi = np.polynomial.legendre.leggauss(qorder)
    s = 0.5 * (xi + 1.0)
    w = 0.5 * wi
    phi0 = 1.0 - s
    phi1 = s
    return s.astype(np.float64), w.astype(np.float64), phi0.astype(np.float64), phi1.astype(np.float64)


@njit(fastmath=True)
def pair_block_standard(x0t, y0t, dxt, dyt, Lt,
                        x0s, y0s, dxs, dys, Ls, nxs, nys,
                        s, w, phi0, phi1):
    b00 = 0.0
    b01 = 0.0
    b10 = 0.0
    b11 = 0.0
    nqp = s.shape[0]
    cker = -0.15915494309189535  # -1/(2*pi)

    for i in range(nqp):
        st = s[i]
        xt = x0t + dxt * st
        yt = y0t + dyt * st
        pt0 = phi0[i]
        pt1 = phi1[i]
        wi = w[i]

        for j in range(nqp):
            ss = s[j]
            xs = x0s + dxs * ss
            ys = y0s + dys * ss
            rx = xt - xs
            ry = yt - ys
            r2 = rx * rx + ry * ry
            kern = cker * ((rx * nxs + ry * nys) / r2)
            wgt = wi * w[j] * Lt * Ls * kern
            ps0 = phi0[j]
            ps1 = phi1[j]

            b00 += pt0 * ps0 * wgt
            b01 += pt0 * ps1 * wgt
            b10 += pt1 * ps0 * wgt
            b11 += pt1 * ps1 * wgt

    return b00, b01, b10, b11


@njit(fastmath=True)
def pair_block_adj_forward(x0t, y0t, dxt, dyt, Lt,
                           x0s, y0s, dxs, dys, Ls, nxs, nys,
                           s, w):
    bx = x0t + dxt
    by = y0t + dyt

    b00 = 0.0
    b01 = 0.0
    b10 = 0.0
    b11 = 0.0
    nqp = s.shape[0]
    cker = -0.15915494309189535

    for i in range(nqp):
        xi = s[i]
        wi = w[i]

        a = xi
        xt = bx - a * dxt
        yt = by - a * dyt
        pt0 = a
        pt1 = 1.0 - a

        for j in range(nqp):
            eta = s[j]
            b = xi * eta
            xs = bx + b * dxs
            ys = by + b * dys
            rx = xt - xs
            ry = yt - ys
            r2 = rx * rx + ry * ry
            kern = cker * ((rx * nxs + ry * nys) / r2)
            wgt = wi * w[j] * xi * Lt * Ls * kern
            ps0 = 1.0 - b
            ps1 = b

            b00 += pt0 * ps0 * wgt
            b01 += pt0 * ps1 * wgt
            b10 += pt1 * ps0 * wgt
            b11 += pt1 * ps1 * wgt

        b = xi
        xs = bx + b * dxs
        ys = by + b * dys
        ps0 = 1.0 - b
        ps1 = b

        for j in range(nqp):
            eta = s[j]
            a = xi * eta
            xt = bx - a * dxt
            yt = by - a * dyt
            pt0 = a
            pt1 = 1.0 - a
            rx = xt - xs
            ry = yt - ys
            r2 = rx * rx + ry * ry
            kern = cker * ((rx * nxs + ry * nys) / r2)
            wgt = wi * w[j] * xi * Lt * Ls * kern

            b00 += pt0 * ps0 * wgt
            b01 += pt0 * ps1 * wgt
            b10 += pt1 * ps0 * wgt
            b11 += pt1 * ps1 * wgt

    return b00, b01, b10, b11


@njit(fastmath=True)
def pair_block_adj_backward(x0t, y0t, dxt, dyt, Lt,
                            x0s, y0s, dxs, dys, Ls, nxs, nys,
                            s, w):
    jx = x0t
    jy = y0t

    b00 = 0.0
    b01 = 0.0
    b10 = 0.0
    b11 = 0.0
    nqp = s.shape[0]
    cker = -0.15915494309189535

    for i in range(nqp):
        xi = s[i]
        wi = w[i]

        a = xi
        xt = jx + a * dxt
        yt = jy + a * dyt
        pt0 = 1.0 - a
        pt1 = a

        for j in range(nqp):
            eta = s[j]
            b = xi * eta
            xs = jx - b * dxs
            ys = jy - b * dys
            rx = xt - xs
            ry = yt - ys
            r2 = rx * rx + ry * ry
            kern = cker * ((rx * nxs + ry * nys) / r2)
            wgt = wi * w[j] * xi * Lt * Ls * kern
            ps0 = b
            ps1 = 1.0 - b

            b00 += pt0 * ps0 * wgt
            b01 += pt0 * ps1 * wgt
            b10 += pt1 * ps0 * wgt
            b11 += pt1 * ps1 * wgt

        b = xi
        xs = jx - b * dxs
        ys = jy - b * dys
        ps0 = b
        ps1 = 1.0 - b

        for j in range(nqp):
            eta = s[j]
            a = xi * eta
            xt = jx + a * dxt
            yt = jy + a * dyt
            pt0 = 1.0 - a
            pt1 = a
            rx = xt - xs
            ry = yt - ys
            r2 = rx * rx + ry * ry
            kern = cker * ((rx * nxs + ry * nys) / r2)
            wgt = wi * w[j] * xi * Lt * Ls * kern

            b00 += pt0 * ps0 * wgt
            b01 += pt0 * ps1 * wgt
            b10 += pt1 * ps0 * wgt
            b11 += pt1 * ps1 * wgt

    return b00, b01, b10, b11


@njit(fastmath=True)
def assemble_rhs(N, x0, y0, dx, dy, L, s, w, phi0, phi1):
    b = np.zeros(N, dtype=np.float64)
    nqp = s.shape[0]

    for e in range(N):
        j = 0 if e == N - 1 else e + 1
        for k in range(nqp):
            ss = s[k]
            x = x0[e] + dx[e] * ss
            y = y0[e] + dy[e] * ss
            g = u_exact_numba(x, y)
            wk = w[k] * L[e] * g
            b[e] += phi0[k] * wk
            b[j] += phi1[k] * wk

    return b


@njit(fastmath=True)
def apply_operator(c, x0, y0, dx, dy, L, nx, ny, s, w, phi0, phi1):
    N = c.shape[0]
    yout = np.zeros(N, dtype=np.float64)

    for p in range(N):
        ip = p
        jp = 0 if p == N - 1 else p + 1

        for q in range(N):
            if q == p:
                continue

            if q == (p + 1) % N:
                b00, b01, b10, b11 = pair_block_adj_forward(
                    x0[p], y0[p], dx[p], dy[p], L[p],
                    x0[q], y0[q], dx[q], dy[q], L[q], nx[q], ny[q],
                    s, w
                )
            elif q == (p - 1) % N:
                b00, b01, b10, b11 = pair_block_adj_backward(
                    x0[p], y0[p], dx[p], dy[p], L[p],
                    x0[q], y0[q], dx[q], dy[q], L[q], nx[q], ny[q],
                    s, w
                )
            else:
                b00, b01, b10, b11 = pair_block_standard(
                    x0[p], y0[p], dx[p], dy[p], L[p],
                    x0[q], y0[q], dx[q], dy[q], L[q], nx[q], ny[q],
                    s, w, phi0, phi1
                )

            cq0 = c[q]
            cq1 = c[0 if q == N - 1 else q + 1]
            yout[ip] += b00 * cq0 + b01 * cq1
            yout[jp] += b10 * cq0 + b11 * cq1

    for p in range(N):
        ip = p
        jp = 0 if p == N - 1 else p + 1
        fac = L[p] / 12.0
        ci = c[ip]
        cj = c[jp]
        yout[ip] += fac * (2.0 * ci + cj)
        yout[jp] += fac * (ci + 2.0 * cj)

    return yout


@njit(fastmath=True)
def eval_potential(points_x, points_y, c, x0, y0, dx, dy, L, nx, ny, s, w, phi0, phi1):
    m = points_x.shape[0]
    N = c.shape[0]
    out = np.zeros(m, dtype=np.float64)
    nqp = s.shape[0]
    cker = -0.15915494309189535

    for p in range(m):
        xp = points_x[p]
        yp = points_y[p]
        acc = 0.0

        for e in range(N):
            j = 0 if e == N - 1 else e + 1
            ce0 = c[e]
            ce1 = c[j]

            for k in range(nqp):
                ss = s[k]
                x = x0[e] + dx[e] * ss
                y = y0[e] + dy[e] * ss
                mu = phi0[k] * ce0 + phi1[k] * ce1
                rx = xp - x
                ry = yp - y
                r2 = rx * rx + ry * ry
                kern = cker * ((rx * nx[e] + ry * ny[e]) / r2)
                acc += mu * kern * w[k] * L[e]

        out[p] = acc

    return out


def run_case(N, qorder=8):
    t0 = time.perf_counter()

    s, w, phi0, phi1 = gauss01(qorder)
    x0, y0, dx, dy, L, nx, ny = build_geometry(N)
    b = assemble_rhs(N, x0, y0, dx, dy, L, s, w, phi0, phi1)

    setup_time = time.perf_counter() - t0

    def mv(v):
        return apply_operator(v, x0, y0, dx, dy, L, nx, ny, s, w, phi0, phi1)

    A = LinearOperator((N, N), matvec=mv, dtype=np.float64)

    it_count = [0]

    def cb(_residual):
        it_count[0] += 1

    t1 = time.perf_counter()
    sol, info = gmres(
        A, b,
        rtol=1e-10,
        atol=0.0,
        restart=50,
        maxiter=200,
        callback=cb,
        callback_type='pr_norm'
    )
    solve_time = time.perf_counter() - t1

    ptsx, ptsy = build_interior_grid()
    t2 = time.perf_counter()
    u_num = eval_potential(ptsx, ptsy, sol, x0, y0, dx, dy, L, nx, ny, s, w, phi0, phi1)
    u_ex = ptsx**3 - 3.0 * ptsx * ptsy**2
    rel_l2 = np.sqrt(np.sum((u_num - u_ex)**2) / np.sum(u_ex**2))
    eval_time = time.perf_counter() - t2

    total_time = time.perf_counter() - t0

    return {
        "N": N,
        "unk": N,
        "iters": it_count[0],
        "err": rel_l2,
        "setup": setup_time,
        "solve": solve_time,
        "eval": eval_time,
        "total": total_time,
        "h": np.mean(L),
    }


# warm-up compilation
_ = run_case(8)

N_values = [1280, 2560, 5120]
results = [run_case(N) for N in N_values]

hs = np.array([r["h"] for r in results], dtype=np.float64)
errs = np.array([r["err"] for r in results], dtype=np.float64)
order = np.polyfit(np.log(hs), np.log(errs), 1)[0]

print(f"{'N':>6} {'Unknowns':>10} {'GMRES':>8} {'Rel L2 Error':>14} {'Conv Rate':>10} {'Setup':>10} {'Solve':>10} {'Eval':>10} {'Total':>10}")
prev_h = None
prev_e = None
for r in results:
    if prev_h is None:
        rate = "-"
    else:
        rate = f"{np.log(prev_e / r['err']) / np.log(prev_h / r['h']):.4f}"
    print(
        f"{r['N']:6d} {r['unk']:10d} {r['iters']:8d} {r['err']:14.6e} {rate:>10} "
        f"{r['setup']:10.6f} {r['solve']:10.6f} {r['eval']:10.6f} {r['total']:10.6f}"
    )
    prev_h = r["h"]
    prev_e = r["err"]

print(f"Final observed order: {order:.6f}")