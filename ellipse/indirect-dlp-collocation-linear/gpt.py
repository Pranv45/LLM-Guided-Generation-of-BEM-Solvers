import time
import math
import sys
import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, gmres

TWO_PI_INV = 1.0 / (2.0 * math.pi)


def u_exact(x, y):
    return x * x - y * y


def _interior_angles(x_nodes, y_nodes):
    M = x_nodes.size
    ang = np.empty(M, dtype=np.float64)
    for i in range(M):
        im1 = (i - 1) % M
        ip1 = (i + 1) % M

        v1x = x_nodes[im1] - x_nodes[i]
        v1y = y_nodes[im1] - y_nodes[i]
        v2x = x_nodes[ip1] - x_nodes[i]
        v2y = y_nodes[ip1] - y_nodes[i]

        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        c = (v1x * v2x + v1y * v2y) / (n1 * n2)
        c = max(-1.0, min(1.0, c))
        ang[i] = math.acos(c)
    return ang


def bem_setup(a, b, M, quad_order=8):
    theta = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False)
    x_nodes = a * np.cos(theta)
    y_nodes = b * np.sin(theta)

    x_next = np.roll(x_nodes, -1)
    y_next = np.roll(y_nodes, -1)

    cx = x_nodes.copy()
    cy = y_nodes.copy()

    ex = x_next - x_nodes
    ey = y_next - y_nodes
    length = np.hypot(ex, ey)

    # CCW ordering -> outward normal
    nx_elem = ey / length
    ny_elem = -ex / length

    # Corner/vertex free-term coefficient for polygonal approximation
    # Smooth limit -> 1/2
    interior_angle = _interior_angles(x_nodes, y_nodes)
    c = interior_angle / (2.0 * np.pi)

    # Gauss-Legendre quadrature on each element
    xi, wi = np.polynomial.legendre.leggauss(quad_order)
    phi1 = 0.5 * (1.0 - xi)
    phi2 = 0.5 * (1.0 + xi)

    nq = int(quad_order)
    nsrc = M * nq

    src_x = np.empty(nsrc, dtype=np.float64)
    src_y = np.empty(nsrc, dtype=np.float64)
    src_nx = np.empty(nsrc, dtype=np.float64)
    src_ny = np.empty(nsrc, dtype=np.float64)
    src_w = np.empty(nsrc, dtype=np.float64)
    src_phi1 = np.empty(nsrc, dtype=np.float64)
    src_phi2 = np.empty(nsrc, dtype=np.float64)
    src_left = np.empty(nsrc, dtype=np.int64)
    src_right = np.empty(nsrc, dtype=np.int64)
    src_elem = np.empty(nsrc, dtype=np.int64)

    k = 0
    for j in range(M):
        x0 = x_nodes[j]
        y0 = y_nodes[j]
        dx = x_next[j] - x0
        dy = y_next[j] - y0
        halfL = 0.5 * length[j]
        for q in range(nq):
            t = 0.5 * (xi[q] + 1.0)
            src_x[k] = x0 + t * dx
            src_y[k] = y0 + t * dy
            src_nx[k] = nx_elem[j]
            src_ny[k] = ny_elem[j]
            src_w[k] = wi[q] * halfL
            src_phi1[k] = phi1[q]
            src_phi2[k] = phi2[q]
            src_left[k] = j
            src_right[k] = (j + 1) % M
            src_elem[k] = j
            k += 1

    rhs = u_exact(cx, cy)

    return {
        "a": float(a),
        "b": float(b),
        "M": int(M),
        "quad_order": nq,
        "x_nodes": np.ascontiguousarray(x_nodes, dtype=np.float64),
        "y_nodes": np.ascontiguousarray(y_nodes, dtype=np.float64),
        "cx": np.ascontiguousarray(cx, dtype=np.float64),
        "cy": np.ascontiguousarray(cy, dtype=np.float64),
        "nx_elem": np.ascontiguousarray(nx_elem, dtype=np.float64),
        "ny_elem": np.ascontiguousarray(ny_elem, dtype=np.float64),
        "length": np.ascontiguousarray(length, dtype=np.float64),
        "c": np.ascontiguousarray(c, dtype=np.float64),
        "src_x": np.ascontiguousarray(src_x, dtype=np.float64),
        "src_y": np.ascontiguousarray(src_y, dtype=np.float64),
        "src_nx": np.ascontiguousarray(src_nx, dtype=np.float64),
        "src_ny": np.ascontiguousarray(src_ny, dtype=np.float64),
        "src_w": np.ascontiguousarray(src_w, dtype=np.float64),
        "src_phi1": np.ascontiguousarray(src_phi1, dtype=np.float64),
        "src_phi2": np.ascontiguousarray(src_phi2, dtype=np.float64),
        "src_left": np.ascontiguousarray(src_left, dtype=np.int64),
        "src_right": np.ascontiguousarray(src_right, dtype=np.int64),
        "src_elem": np.ascontiguousarray(src_elem, dtype=np.int64),
        "rhs": np.ascontiguousarray(rhs, dtype=np.float64),
    }


@njit(parallel=True, fastmath=True, cache=True)
def bem_matvec_numba(
    sigma,
    cx,
    cy,
    c,
    src_x,
    src_y,
    src_nx,
    src_ny,
    src_w,
    src_phi1,
    src_phi2,
    src_left,
    src_right,
    src_elem,
):
    M = sigma.shape[0]
    nsrc = src_x.shape[0]
    out = np.empty(M, dtype=np.float64)

    for i in prange(M):
        xi = cx[i]
        yi = cy[i]
        acc = c[i] * sigma[i]
        im1 = M - 1 if i == 0 else i - 1

        for s in range(nsrc):
            e = src_elem[s]
            if e == i or e == im1:
                continue

            dens = src_phi1[s] * sigma[src_left[s]] + src_phi2[s] * sigma[src_right[s]]
            dx = xi - src_x[s]
            dy = yi - src_y[s]
            r2 = dx * dx + dy * dy
            kern = -TWO_PI_INV * ((dx * src_nx[s] + dy * src_ny[s]) / r2)
            acc += kern * dens * src_w[s]

        out[i] = acc

    return out


def bem_matvec(mu, geom):
    sigma = np.ascontiguousarray(mu, dtype=np.float64)
    return bem_matvec_numba(
        sigma,
        geom["cx"],
        geom["cy"],
        geom["c"],
        geom["src_x"],
        geom["src_y"],
        geom["src_nx"],
        geom["src_ny"],
        geom["src_w"],
        geom["src_phi1"],
        geom["src_phi2"],
        geom["src_left"],
        geom["src_right"],
        geom["src_elem"],
    )


def bem_evaluate_chunked(sigma, geom, x_eval, y_eval, point_chunk=16, source_block=4096):
    sigma = np.ascontiguousarray(sigma, dtype=np.float64)

    src_density = (
        geom["src_phi1"] * sigma[geom["src_left"]]
        + geom["src_phi2"] * sigma[geom["src_right"]]
    )
    weighted = src_density * geom["src_w"]

    pts_x = np.ascontiguousarray(np.ravel(x_eval), dtype=np.float64)
    pts_y = np.ascontiguousarray(np.ravel(y_eval), dtype=np.float64)
    out = np.empty_like(pts_x)

    src_x = geom["src_x"]
    src_y = geom["src_y"]
    src_nx = geom["src_nx"]
    src_ny = geom["src_ny"]

    npts = pts_x.shape[0]
    nsrc = src_x.shape[0]
    point_chunk = max(1, int(point_chunk))
    source_block = max(1, int(source_block))

    for p0 in range(0, npts, point_chunk):
        p1 = min(p0 + point_chunk, npts)
        xs = pts_x[p0:p1][:, None]
        ys = pts_y[p0:p1][:, None]
        acc = np.zeros(p1 - p0, dtype=np.float64)

        for s0 in range(0, nsrc, source_block):
            s1 = min(s0 + source_block, nsrc)
            dx = xs - src_x[s0:s1][None, :]
            dy = ys - src_y[s0:s1][None, :]
            r2 = dx * dx + dy * dy
            kern = -TWO_PI_INV * (
                (dx * src_nx[s0:s1][None, :] + dy * src_ny[s0:s1][None, :]) / r2
            )
            acc += kern @ weighted[s0:s1]

        out[p0:p1] = acc

    return out.reshape(np.shape(x_eval))


def run_bem(
    a,
    b,
    M,
    grid_n=33,
    quad_order=8,
    point_chunk=16,
    source_block=4096,
    rtol=1e-8,
    restart=50,
    maxiter=200,
):
    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    geom = bem_setup(a, b, M, quad_order=quad_order)
    rhs = geom["rhs"].copy()
    A = LinearOperator(
        shape=(M, M),
        matvec=lambda v: bem_matvec(v, geom),
        dtype=np.float64,
    )
    setup_time = time.perf_counter() - t0

    iter_count = {"n": 0}

    def cb(_resid):
        iter_count["n"] += 1

    t1 = time.perf_counter()
    sigma, info = gmres(
        A,
        rhs,
        rtol=rtol,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
        callback=cb,
        callback_type="pr_norm",
    )
    solve_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    x = np.linspace(-a, a, grid_n)
    y = np.linspace(-b, b, grid_n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    mask = ((X / a) ** 2 + (Y / b) ** 2) < 0.95
    X_in = X[mask]
    Y_in = Y[mask]

    u_num = bem_evaluate_chunked(
        sigma,
        geom,
        X_in,
        Y_in,
        point_chunk=point_chunk,
        source_block=source_block,
    ).ravel()
    u_ex = u_exact(X_in, Y_in)
    relative_L2_error = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)
    eval_time = time.perf_counter() - t2

    total_time = time.perf_counter() - t_total0

    return {
        "M": int(M),
        "iterations": int(iter_count["n"]),
        "setup_time": float(setup_time),
        "solve_time": float(solve_time),
        "eval_time": float(eval_time),
        "total_time": float(total_time),
        "relative_L2_error": float(relative_L2_error),
        "gmres_info": int(info),
    }


if __name__ == "__main__":
    # Numba warm-up
    _diag = np.array([0.5, 0.5], dtype=np.float64)
    _sigma = np.array([1.0, 2.0], dtype=np.float64)
    _cx = np.array([0.0, 1.0], dtype=np.float64)
    _cy = np.array([0.0, 0.0], dtype=np.float64)
    _src_x = np.array([0.5, 1.5], dtype=np.float64)
    _src_y = np.array([1.0, 1.0], dtype=np.float64)
    _src_nx = np.array([1.0, 1.0], dtype=np.float64)
    _src_ny = np.array([0.0, 0.0], dtype=np.float64)
    _src_w = np.array([1.0, 1.0], dtype=np.float64)
    _phi1 = np.array([0.5, 0.5], dtype=np.float64)
    _phi2 = np.array([0.5, 0.5], dtype=np.float64)
    _src_left = np.array([0, 0], dtype=np.int64)
    _src_right = np.array([1, 1], dtype=np.int64)
    _src_elem = np.array([0, 1], dtype=np.int64)
    _ = bem_matvec_numba(
        _sigma,
        _cx,
        _cy,
        _diag,
        _src_x,
        _src_y,
        _src_nx,
        _src_ny,
        _src_w,
        _phi1,
        _phi2,
        _src_left,
        _src_right,
        _src_elem,
    )

    a = 2.0
    b = 1.0
    M_list = [4000, 8000, 16000, 32000, 64000]

    header = (
        f"{'M':>8} | {'Iterations':>10} | {'Setup (s)':>10} | {'Solve (s)':>10} | "
        f"{'Eval (s)':>10} | {'Total (s)':>10} | {'Rel L2 Error':>14}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for M in M_list:
        res = run_bem(a, b, M)
        print(
            f"{res['M']:8d} | {res['iterations']:10d} | {res['setup_time']:10.4f} | "
            f"{res['solve_time']:10.4f} | {res['eval_time']:10.4f} | {res['total_time']:10.4f} | "
            f"{res['relative_L2_error']:14.6e}",
            flush=True,
        )