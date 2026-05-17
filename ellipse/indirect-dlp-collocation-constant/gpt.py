import time
import math
import sys
import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, gmres

TWO_PI_INV = 1.0 / (2.0 * math.pi)


def u_exact(x, y):
    return x * x - y * y


def bem_setup(a, b, M):
    theta = np.linspace(0.0, 2.0 * math.pi, M + 1, endpoint=True)

    x_nodes = a * np.cos(theta)
    y_nodes = b * np.sin(theta)

    x0 = x_nodes[:-1]
    y0 = y_nodes[:-1]
    x1 = x_nodes[1:]
    y1 = y_nodes[1:]

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    tx = x1 - x0
    ty = y1 - y0
    length = np.sqrt(tx * tx + ty * ty)

    # CCW-oriented polygon => outward normal = (ty, -tx)/|t|
    nx = ty / length
    ny = -tx / length

    rhs = u_exact(cx, cy)

    return {
        "a": float(a),
        "b": float(b),
        "M": int(M),
        "x_nodes": np.ascontiguousarray(x_nodes, dtype=np.float64),
        "y_nodes": np.ascontiguousarray(y_nodes, dtype=np.float64),
        "cx": np.ascontiguousarray(cx, dtype=np.float64),
        "cy": np.ascontiguousarray(cy, dtype=np.float64),
        "nx": np.ascontiguousarray(nx, dtype=np.float64),
        "ny": np.ascontiguousarray(ny, dtype=np.float64),
        "length": np.ascontiguousarray(length, dtype=np.float64),
        "rhs": np.ascontiguousarray(rhs, dtype=np.float64),
    }


@njit(parallel=True, fastmath=True, cache=True)
def bem_matvec_numba(mu, cx, cy, nx, ny, length):
    M = mu.shape[0]
    out = np.empty(M, dtype=np.float64)

    for i in prange(M):
        xi = cx[i]
        yi = cy[i]
        acc = 0.0
        for j in range(M):
            if i == j:
                continue
            dx = xi - cx[j]
            dy = yi - cy[j]
            r2 = dx * dx + dy * dy
            # K(x, y) = -(1 / 2pi) * ((x-y) · n_y) / |x-y|^2
            kern = -TWO_PI_INV * ((dx * nx[j] + dy * ny[j]) / r2)
            acc += kern * mu[j] * length[j]
        out[i] = 0.5 * mu[i] + acc

    return out


def bem_matvec(mu, geom):
    mu = np.ascontiguousarray(mu, dtype=np.float64)
    return bem_matvec_numba(
        mu,
        geom["cx"],
        geom["cy"],
        geom["nx"],
        geom["ny"],
        geom["length"],
    )


def bem_evaluate_chunked(mu, geom, x_eval, y_eval, chunk_size=16):
    cx = geom["cx"]
    cy = geom["cy"]
    nx = geom["nx"]
    ny = geom["ny"]
    length = geom["length"]

    weighted = np.ascontiguousarray(mu * length, dtype=np.float64)

    pts_x = np.ascontiguousarray(x_eval.ravel(), dtype=np.float64)
    pts_y = np.ascontiguousarray(y_eval.ravel(), dtype=np.float64)
    out = np.empty_like(pts_x)

    npts = pts_x.shape[0]
    for start in range(0, npts, chunk_size):
        end = min(start + chunk_size, npts)
        xs = pts_x[start:end]
        ys = pts_y[start:end]

        dx = xs[:, None] - cx[None, :]
        dy = ys[:, None] - cy[None, :]
        r2 = dx * dx + dy * dy

        kern = -TWO_PI_INV * ((dx * nx[None, :] + dy * ny[None, :]) / r2)
        out[start:end] = kern @ weighted

    return out.reshape(x_eval.shape)


def run_bem(a, b, M, grid_n=65, chunk_size=16, rtol=1e-8, restart=50, maxiter=200):
    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    geom = bem_setup(a, b, M)
    rhs = geom["rhs"].copy()

    A = LinearOperator(
        shape=(M, M),
        matvec=lambda v: bem_matvec(v, geom),
        dtype=np.float64,
    )
    setup_time = time.perf_counter() - t0

    iterations = {"n": 0}

    def cb(_residual_norm):
        iterations["n"] += 1

    t1 = time.perf_counter()
    mu, info = gmres(
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
        mu,
        geom,
        X_in.reshape(-1, 1),
        Y_in.reshape(-1, 1),
        chunk_size=chunk_size,
    ).ravel()
    u_ex = u_exact(X_in, Y_in)
    relative_L2_error = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)
    eval_time = time.perf_counter() - t2

    total_time = time.perf_counter() - t_total0

    return {
        "M": int(M),
        "iterations": int(iterations["n"]),
        "setup_time": float(setup_time),
        "solve_time": float(solve_time),
        "eval_time": float(eval_time),
        "total_time": float(total_time),
        "relative_L2_error": float(relative_L2_error),
        "gmres_info": int(info),
    }


if __name__ == "__main__":
    # Numba warm-up so compilation time does not pollute the sweep timings
    _mu0 = np.ones(4, dtype=np.float64)
    _x0 = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64)
    _y0 = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float64)
    _n0x = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float64)
    _n0y = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float64)
    _l0 = np.ones(4, dtype=np.float64)
    _ = bem_matvec_numba(_mu0, _x0, _y0, _n0x, _n0y, _l0)

    a = 2.0
    b = 1.0
    M_list = [4000, 8000, 16000, 32000, 64000]

    header = (
        f"{'M':>8} | {'Iterations':>10} | {'Setup (s)':>10} | {'Solve (s)':>10} | "
        f"{'Eval (s)':>10} | {'Total (s)':>10} | {'Rel L2 Error':>14}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    sys.stdout.flush()

    for M in M_list:
        res = run_bem(a, b, M)
        print(
            f"{res['M']:8d} | {res['iterations']:10d} | {res['setup_time']:10.4f} | "
            f"{res['solve_time']:10.4f} | {res['eval_time']:10.4f} | {res['total_time']:10.4f} | "
            f"{res['relative_L2_error']:14.6e}",
            flush=True,
        )
        sys.stdout.flush()