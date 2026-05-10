import time
import math
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit

PI = np.pi
INV2PI = 1.0 / (2.0 * np.pi)
INV4PI = 1.0 / (4.0 * np.pi)

ETA = 1.0
Q_ORDER = 16
N_VALUES = [100, 200, 400, 800]

GMRES_RTL = 1e-10
GMRES_ATL = 1e-12
GMRES_RESTART = 80
GMRES_MAXITER = 200


def exact_solution(x, y):
    return np.sin(PI * x) * np.cosh(PI * y)


def cosine_edge_points(p0, p1, N):
    j = np.arange(N + 1, dtype=np.float64)
    t = 0.5 * (1.0 - np.cos(PI * j / N))
    pts = np.empty((N + 1, 2), dtype=np.float64)
    pts[:, 0] = (1.0 - t) * p0[0] + t * p1[0]
    pts[:, 1] = (1.0 - t) * p0[1] + t * p1[1]
    return pts


def build_geometry(N, nq):
    total_nodes = 3 * N
    total_elems = 3 * N

    s_ref, w_ref = np.polynomial.legendre.leggauss(nq)
    phi1 = 0.5 * (1.0 - s_ref)
    phi2 = 0.5 * (1.0 + s_ref)

    nodes_x = np.empty(total_nodes, dtype=np.float64)
    nodes_y = np.empty(total_nodes, dtype=np.float64)

    elem_a = np.empty(total_elems, dtype=np.int64)
    elem_b = np.empty(total_elems, dtype=np.int64)

    elem_ax = np.empty(total_elems, dtype=np.float64)
    elem_ay = np.empty(total_elems, dtype=np.float64)
    elem_bx = np.empty(total_elems, dtype=np.float64)
    elem_by = np.empty(total_elems, dtype=np.float64)

    elem_len = np.empty(total_elems, dtype=np.float64)
    elem_loglen = np.empty(total_elems, dtype=np.float64)
    elem_nx = np.empty(total_elems, dtype=np.float64)
    elem_ny = np.empty(total_elems, dtype=np.float64)

    qx = np.empty((total_elems, nq), dtype=np.float64)
    qy = np.empty((total_elems, nq), dtype=np.float64)
    qnx = np.empty((total_elems, nq), dtype=np.float64)
    qny = np.empty((total_elems, nq), dtype=np.float64)
    wq = np.empty((total_elems, nq), dtype=np.float64)

    v0 = np.array([0.0, 0.0], dtype=np.float64)
    v1 = np.array([1.0, 0.0], dtype=np.float64)
    v2 = np.array([0.0, 1.0], dtype=np.float64)

    e1 = cosine_edge_points(v0, v1, N)
    e2 = cosine_edge_points(v1, v2, N)
    e3 = cosine_edge_points(v2, v0, N)

    for j in range(N + 1):
        nodes_x[j] = e1[j, 0]
        nodes_y[j] = e1[j, 1]

    for j in range(N + 1):
        nodes_x[N + j] = e2[j, 0]
        nodes_y[N + j] = e2[j, 1]

    for j in range(N):
        nodes_x[2 * N + j] = e3[j, 0]
        nodes_y[2 * N + j] = e3[j, 1]

    # Edge 1: v0 -> v1
    for k in range(N):
        e = k
        a = k
        b = k + 1
        elem_a[e] = a
        elem_b[e] = b

    # Edge 2: v1 -> v2
    for k in range(N):
        e = N + k
        a = N + k
        b = N + k + 1
        elem_a[e] = a
        elem_b[e] = b

    # Edge 3: v2 -> v0
    for k in range(N):
        e = 2 * N + k
        a = 2 * N + k
        b = 0 if k == N - 1 else 2 * N + k + 1
        elem_a[e] = a
        elem_b[e] = b

    for e in range(total_elems):
        a = elem_a[e]
        b = elem_b[e]
        ax = nodes_x[a]
        ay = nodes_y[a]
        bx = nodes_x[b]
        by = nodes_y[b]
        elem_ax[e] = ax
        elem_ay[e] = ay
        elem_bx[e] = bx
        elem_by[e] = by
        dx = bx - ax
        dy = by - ay
        L = math.hypot(dx, dy)
        elem_len[e] = L
        elem_loglen[e] = math.log(L)
        tx = dx / L
        ty = dy / L
        nx = ty
        ny = -tx
        elem_nx[e] = nx
        elem_ny[e] = ny
        for q in range(nq):
            t = 0.5 * (1.0 + s_ref[q])
            qx[e, q] = ax + t * dx
            qy[e, q] = ay + t * dy
            qnx[e, q] = nx
            qny[e, q] = ny
            wq[e, q] = 0.5 * L * w_ref[q]

    return (
        nodes_x,
        nodes_y,
        elem_a,
        elem_b,
        elem_ax,
        elem_ay,
        elem_bx,
        elem_by,
        elem_len,
        elem_loglen,
        elem_nx,
        elem_ny,
        qx,
        qy,
        qnx,
        qny,
        wq,
        phi1,
        phi2,
        total_nodes,
        total_elems,
    )


@njit(cache=True, fastmath=True)
def apply_operator_core(
    sigma,
    nodes_x,
    nodes_y,
    elem_a,
    elem_b,
    elem_len,
    elem_loglen,
    qx,
    qy,
    qnx,
    qny,
    wq,
    phi1,
    phi2,
    eta,
):
    n_nodes = sigma.shape[0]
    n_elems = elem_a.shape[0]
    nq = phi1.shape[0]
    out = np.zeros(n_nodes, dtype=np.float64)

    for i in range(n_nodes):
        xt = nodes_x[i]
        yt = nodes_y[i]
        acc = 0.0

        for e in range(n_elems):
            a = elem_a[e]
            b = elem_b[e]
            sa = sigma[a]
            sb = sigma[b]

            if i == a or i == b:
                half_logL = 0.5 * elem_loglen[e]
                if i == a:
                    c0 = -elem_len[e] * INV2PI * (half_logL - 0.75)
                    c1 = -elem_len[e] * INV2PI * (half_logL - 0.25)
                else:
                    c0 = -elem_len[e] * INV2PI * (half_logL - 0.25)
                    c1 = -elem_len[e] * INV2PI * (half_logL - 0.75)
                acc += c0 * sa + c1 * sb
            else:
                for q in range(nq):
                    muq = phi1[q] * sa + phi2[q] * sb
                    dx = qx[e, q] - xt
                    dy = qy[e, q] - yt
                    r2 = dx * dx + dy * dy
                    acc += -INV4PI * math.log(r2) * muq * wq[e, q]
                    acc += eta * (-INV2PI * (dx * qnx[e, q] + dy * qny[e, q]) / r2) * muq * wq[e, q]

        acc += -0.5 * eta * sigma[i]
        out[i] = acc

    return out


@njit(cache=True, fastmath=True)
def evaluate_potential_core(
    sigma,
    pts_x,
    pts_y,
    elem_a,
    elem_b,
    qx,
    qy,
    qnx,
    qny,
    wq,
    phi1,
    phi2,
    eta,
):
    n_pts = pts_x.shape[0]
    n_elems = elem_a.shape[0]
    nq = phi1.shape[0]
    out = np.zeros(n_pts, dtype=np.float64)

    for i in range(n_pts):
        xt = pts_x[i]
        yt = pts_y[i]
        acc = 0.0

        for e in range(n_elems):
            a = elem_a[e]
            b = elem_b[e]
            sa = sigma[a]
            sb = sigma[b]
            for q in range(nq):
                muq = phi1[q] * sa + phi2[q] * sb
                dx = qx[e, q] - xt
                dy = qy[e, q] - yt
                r2 = dx * dx + dy * dy
                acc += -INV4PI * math.log(r2) * muq * wq[e, q]
                acc += eta * (-INV2PI * (dx * qnx[e, q] + dy * qny[e, q]) / r2) * muq * wq[e, q]

        out[i] = acc

    return out


def warmup_numba():
    dummy_N = 4
    dummy = build_geometry(dummy_N, Q_ORDER)
    (
        nodes_x,
        nodes_y,
        elem_a,
        elem_b,
        elem_ax,
        elem_ay,
        elem_bx,
        elem_by,
        elem_len,
        elem_loglen,
        elem_nx,
        elem_ny,
        qx,
        qy,
        qnx,
        qny,
        wq,
        phi1,
        phi2,
        total_nodes,
        total_elems,
    ) = dummy
    sigma = np.zeros(total_nodes, dtype=np.float64)
    pts_x = np.array([0.2, 0.25], dtype=np.float64)
    pts_y = np.array([0.2, 0.25], dtype=np.float64)
    _ = apply_operator_core(
        sigma,
        nodes_x,
        nodes_y,
        elem_a,
        elem_b,
        elem_len,
        elem_loglen,
        qx,
        qy,
        qnx,
        qny,
        wq,
        phi1,
        phi2,
        ETA,
    )
    _ = evaluate_potential_core(
        sigma,
        pts_x,
        pts_y,
        elem_a,
        elem_b,
        qx,
        qy,
        qnx,
        qny,
        wq,
        phi1,
        phi2,
        ETA,
    )


def solve_case(N, pts_x, pts_y, u_exact_pts, u_exact_norm):
    t0 = time.perf_counter()
    (
        nodes_x,
        nodes_y,
        elem_a,
        elem_b,
        elem_ax,
        elem_ay,
        elem_bx,
        elem_by,
        elem_len,
        elem_loglen,
        elem_nx,
        elem_ny,
        qx,
        qy,
        qnx,
        qny,
        wq,
        phi1,
        phi2,
        total_nodes,
        total_elems,
    ) = build_geometry(N, Q_ORDER)

    rhs = exact_solution(nodes_x, nodes_y)
    setup_time = time.perf_counter() - t0

    sigma0 = np.zeros(total_nodes, dtype=np.float64)
    _ = apply_operator_core(
        sigma0,
        nodes_x,
        nodes_y,
        elem_a,
        elem_b,
        elem_len,
        elem_loglen,
        qx,
        qy,
        qnx,
        qny,
        wq,
        phi1,
        phi2,
        ETA,
    )

    def matvec(sigma):
        return apply_operator_core(
            sigma,
            nodes_x,
            nodes_y,
            elem_a,
            elem_b,
            elem_len,
            elem_loglen,
            qx,
            qy,
            qnx,
            qny,
            wq,
            phi1,
            phi2,
            ETA,
        )

    A = LinearOperator((total_nodes, total_nodes), matvec=matvec, dtype=np.float64)

    iter_counter = {"n": 0}

    def callback(_residual_norm):
        iter_counter["n"] += 1

    t1 = time.perf_counter()
    sigma, info = gmres(
        A,
        rhs,
        rtol=GMRES_RTL,
        atol=GMRES_ATL,
        restart=min(GMRES_RESTART, total_nodes),
        maxiter=GMRES_MAXITER,
        callback=callback,
        callback_type="pr_norm",
    )
    solve_time = time.perf_counter() - t1
    if info != 0:
        raise RuntimeError(f"GMRES failed for N={N} with info={info}")

    t2 = time.perf_counter()
    u_num = evaluate_potential_core(
        sigma,
        pts_x,
        pts_y,
        elem_a,
        elem_b,
        qx,
        qy,
        qnx,
        qny,
        wq,
        phi1,
        phi2,
        ETA,
    )
    eval_time = time.perf_counter() - t2

    rel_l2 = np.linalg.norm(u_num - u_exact_pts) / u_exact_norm
    total_time = setup_time + solve_time + eval_time

    return {
        "N": N,
        "unknowns": total_nodes,
        "gmres": iter_counter["n"],
        "rel_err": rel_l2,
        "setup": setup_time,
        "solve": solve_time,
        "eval": eval_time,
        "total": total_time,
    }


ngrid = 60
x = np.linspace(0.02, 0.98, ngrid)
y = np.linspace(0.02, 0.98, ngrid)
X, Y = np.meshgrid(x, y, indexing="xy")
mask = (X + Y) < 1.0
pts_x = X[mask].ravel()
pts_y = Y[mask].ravel()
u_exact_pts = exact_solution(pts_x, pts_y)
u_exact_norm = np.linalg.norm(u_exact_pts)

warmup_numba()

results = []
for N in N_VALUES:
    results.append(solve_case(N, pts_x, pts_y, u_exact_pts, u_exact_norm))

print(f"{'N':>5} {'Unknowns':>10} {'GMRES':>8} {'Rel L2 Error':>14} {'Setup':>10} {'Solve':>10} {'Eval':>10} {'Total':>10}")
for r in results:
    print(
        f"{r['N']:5d} {r['unknowns']:10d} {r['gmres']:8d} {r['rel_err']:14.3e} "
        f"{r['setup']:10.3f} {r['solve']:10.3f} {r['eval']:10.3f} {r['total']:10.3f}"
    )