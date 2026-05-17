import time
import math
import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, gmres

TWO_PI_INV = 1.0 / (2.0 * math.pi)


def u_exact(x, y):
    return x * x - y * y


def bem_setup(a, b, M, quad_order=4):
    theta = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False)
    x_nodes = a * np.cos(theta)
    y_nodes = b * np.sin(theta)

    x_next = np.roll(x_nodes, -1)
    y_next = np.roll(y_nodes, -1)

    ex = x_next - x_nodes
    ey = y_next - y_nodes
    length = np.hypot(ex, ey)

    # CCW orientation -> outward normal
    nx_elem = ey / length
    ny_elem = -ex / length

    # Gauss-Legendre quadrature on each element
    xi, wi = np.polynomial.legendre.leggauss(quad_order)
    phi1 = np.ascontiguousarray(0.5 * (1.0 - xi), dtype=np.float64)
    phi2 = np.ascontiguousarray(0.5 * (1.0 + xi), dtype=np.float64)

    nq = int(quad_order)
    xq = np.empty((M, nq), dtype=np.float64)
    yq = np.empty((M, nq), dtype=np.float64)
    wq = np.empty((M, nq), dtype=np.float64)

    for e in range(M):
        x0 = x_nodes[e]
        y0 = y_nodes[e]
        dx = x_next[e] - x0
        dy = y_next[e] - y0
        halfL = 0.5 * length[e]
        for q in range(nq):
            t = 0.5 * (xi[q] + 1.0)
            xq[e, q] = x0 + t * dx
            yq[e, q] = y0 + t * dy
            wq[e, q] = wi[q] * halfL

    # Flattened source quadrature data for fast matvec/evaluation
    src_x = np.ascontiguousarray(xq.reshape(-1), dtype=np.float64)
    src_y = np.ascontiguousarray(yq.reshape(-1), dtype=np.float64)
    src_w = np.ascontiguousarray(wq.reshape(-1), dtype=np.float64)
    src_nx = np.ascontiguousarray(np.repeat(nx_elem, nq), dtype=np.float64)
    src_ny = np.ascontiguousarray(np.repeat(ny_elem, nq), dtype=np.float64)
    elem_start = np.ascontiguousarray(np.arange(M, dtype=np.int64) * nq)

    # Test quadrature on the two support elements of each nodal basis function
    # Node i is supported on elements (i-1) and i
    test_x = np.empty((M, 2, nq), dtype=np.float64)
    test_y = np.empty((M, 2, nq), dtype=np.float64)
    test_w = np.empty((M, 2, nq), dtype=np.float64)
    test_elem_id = np.empty((M, 2), dtype=np.int64)
    rhs = np.zeros(M, dtype=np.float64)

    for i in range(M):
        eL = M - 1 if i == 0 else i - 1
        eR = i

        test_elem_id[i, 0] = eL
        test_elem_id[i, 1] = eR

        for q in range(nq):
            # Left support element: basis value = phi2
            test_x[i, 0, q] = xq[eL, q]
            test_y[i, 0, q] = yq[eL, q]
            test_w[i, 0, q] = wq[eL, q] * phi2[q]

            # Right support element: basis value = phi1
            test_x[i, 1, q] = xq[eR, q]
            test_y[i, 1, q] = yq[eR, q]
            test_w[i, 1, q] = wq[eR, q] * phi1[q]

            rhs[i] += test_w[i, 0, q] * u_exact(test_x[i, 0, q], test_y[i, 0, q])
            rhs[i] += test_w[i, 1, q] * u_exact(test_x[i, 1, q], test_y[i, 1, q])

    return {
        "a": float(a),
        "b": float(b),
        "M": int(M),
        "quad_order": int(nq),
        "x_nodes": np.ascontiguousarray(x_nodes, dtype=np.float64),
        "y_nodes": np.ascontiguousarray(y_nodes, dtype=np.float64),
        "x_next": np.ascontiguousarray(x_next, dtype=np.float64),
        "y_next": np.ascontiguousarray(y_next, dtype=np.float64),
        "length": np.ascontiguousarray(length, dtype=np.float64),
        "nx_elem": np.ascontiguousarray(nx_elem, dtype=np.float64),
        "ny_elem": np.ascontiguousarray(ny_elem, dtype=np.float64),
        "phi1": phi1,
        "phi2": phi2,
        "xq": np.ascontiguousarray(xq, dtype=np.float64),
        "yq": np.ascontiguousarray(yq, dtype=np.float64),
        "wq": np.ascontiguousarray(wq, dtype=np.float64),
        "src_x": src_x,
        "src_y": src_y,
        "src_w": src_w,
        "src_nx": src_nx,
        "src_ny": src_ny,
        "elem_start": elem_start,
        "test_x": np.ascontiguousarray(test_x, dtype=np.float64),
        "test_y": np.ascontiguousarray(test_y, dtype=np.float64),
        "test_w": np.ascontiguousarray(test_w, dtype=np.float64),
        "test_elem_id": np.ascontiguousarray(test_elem_id, dtype=np.int64),
        "rhs": np.ascontiguousarray(rhs, dtype=np.float64),
    }


@njit(parallel=True, cache=True)
def bem_matvec_numba(
    sigma,
    length,
    elem_start,
    src_x,
    src_y,
    src_w,
    src_nx,
    src_ny,
    phi1,
    phi2,
    test_x,
    test_y,
    test_w,
    test_elem_id,
):
    M = sigma.shape[0]
    nq = phi1.shape[0]
    out = np.empty(M, dtype=np.float64)

    # Precompute source density * quadrature weight once per matvec
    src_weighted = np.empty(src_x.shape[0], dtype=np.float64)
    for e in range(M):
        se = sigma[e]
        sep1 = sigma[0] if e == M - 1 else sigma[e + 1]
        start = elem_start[e]
        for q in range(nq):
            s = start + q
            src_weighted[s] = (phi1[q] * se + phi2[q] * sep1) * src_w[s]

    for i in prange(M):
        im = M - 1 if i == 0 else i - 1
        ip = 0 if i == M - 1 else i + 1

        # Exact mass action for continuous linear elements on a periodic mesh
        mass_action = (
            length[im] * (sigma[im] + 2.0 * sigma[i])
            + length[i] * (2.0 * sigma[i] + sigma[ip])
        ) / 6.0

        acc = 0.5 * mass_action

        # Galerkin double-layer contribution
        for side in range(2):
            e_x = test_elem_id[i, side]

            for qx in range(nq):
                x = test_x[i, side, qx]
                y = test_y[i, side, qx]
                wx = test_w[i, side, qx]

                # elements before excluded element
                for e in range(e_x):
                    start = elem_start[e]
                    nx = src_nx[start]
                    ny = src_ny[start]

                    for qy in range(nq):
                        s = start + qy
                        dx = x - src_x[s]
                        dy = y - src_y[s]
                        r2 = dx * dx + dy * dy
                        kern = -TWO_PI_INV * ((dx * nx + dy * ny) / r2)
                        acc += wx * kern * src_weighted[s]

                # elements after excluded element
                for e in range(e_x + 1, M):
                    start = elem_start[e]
                    nx = src_nx[start]
                    ny = src_ny[start]

                    for qy in range(nq):
                        s = start + qy
                        dx = x - src_x[s]
                        dy = y - src_y[s]
                        r2 = dx * dx + dy * dy
                        kern = -TWO_PI_INV * ((dx * nx + dy * ny) / r2)
                        acc += wx * kern * src_weighted[s]

        out[i] = acc

    return out


def bem_matvec(mu, geom):
    sigma = np.ascontiguousarray(mu, dtype=np.float64)
    return bem_matvec_numba(
        sigma,
        geom["length"],
        geom["elem_start"],
        geom["src_x"],
        geom["src_y"],
        geom["src_w"],
        geom["src_nx"],
        geom["src_ny"],
        geom["phi1"],
        geom["phi2"],
        geom["test_x"],
        geom["test_y"],
        geom["test_w"],
        geom["test_elem_id"],
    )


def bem_evaluate_chunked(
    sigma,
    geom,
    x_eval,
    y_eval,
    point_chunk=16,
    source_block=4096,
    src_density_flat=None,
):
    sigma = np.ascontiguousarray(sigma, dtype=np.float64)

    if src_density_flat is None:
        sigma_next = np.empty_like(sigma)
        sigma_next[:-1] = sigma[1:]
        sigma_next[-1] = sigma[0]
        src_density_flat = (
            sigma[:, None] * geom["phi1"][None, :]
            + sigma_next[:, None] * geom["phi2"][None, :]
        ).reshape(-1)

    weighted = np.ascontiguousarray(src_density_flat * geom["src_w"], dtype=np.float64)

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
    quad_order=4,
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

    sigma_next = np.empty_like(sigma)
    sigma_next[:-1] = sigma[1:]
    sigma_next[-1] = sigma[0]
    src_density_flat = (
        sigma[:, None] * geom["phi1"][None, :]
        + sigma_next[:, None] * geom["phi2"][None, :]
    ).reshape(-1)

    u_num = bem_evaluate_chunked(
        sigma,
        geom,
        X_in,
        Y_in,
        point_chunk=point_chunk,
        source_block=source_block,
        src_density_flat=src_density_flat,
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
    _geom = bem_setup(2.0, 1.0, 8, quad_order=4)
    _sigma = np.ones(8, dtype=np.float64)
    _ = bem_matvec(_sigma, _geom)

    a = 2.0
    b = 1.0
    M_list = [4000, 8000, 16000, 32000]

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