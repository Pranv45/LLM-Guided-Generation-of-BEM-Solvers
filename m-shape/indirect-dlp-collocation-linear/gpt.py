import math
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, gmres

N_VALUES = [400, 800, 1600, 3200, 6400]
NQ = 8

VERTICES = np.array(
    [
        (-1.5,  1.0),
        (-0.75, 1.0),
        ( 0.0,  0.4),
        ( 0.75, 1.0),
        ( 1.5,  1.0),
        ( 1.5,  0.0),
        ( 1.5, -1.0),
        ( 0.75,-1.0),
        ( 0.75, 0.3),
        ( 0.0, -0.3),
        (-0.75, 0.3),
        (-0.75,-1.0),
        (-1.5,-1.0),
        (-1.5, 0.0),
    ],
    dtype=np.float64,
)

INV_2PI = 1.0 / (2.0 * np.pi)


def u_exact(x, y):
    return x**3 - 3.0 * x * y**2


def ensure_ccw(vertices):
    v = np.array(vertices, dtype=np.float64, copy=True)
    area2 = np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
    if area2 < 0.0:
        v = v[::-1].copy()
    return v


def allocate_edge_counts(lengths, n_total):
    lengths = np.asarray(lengths, dtype=np.float64)
    m = len(lengths)
    if n_total < m:
        raise ValueError("N must be at least the number of boundary edges.")
    remaining = n_total - m
    weights = lengths / np.sum(lengths)
    raw = remaining * weights
    extra = np.floor(raw).astype(np.int32)
    counts = np.ones(m, dtype=np.int32) + extra
    leftover = int(remaining - int(extra.sum()))
    if leftover > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        counts[order[:leftover]] += 1
    return counts


def build_discretization(vertices, n_total, nq=NQ):
    V = ensure_ccw(vertices)
    n_edges = len(V)

    edge_vecs = np.roll(V, -1, axis=0) - V
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)
    perimeter = float(np.sum(edge_lengths))

    counts = allocate_edge_counts(edge_lengths, n_total)
    if int(np.sum(counts)) != n_total:
        raise RuntimeError("Node allocation failed.")

    nodes = np.empty((n_total, 2), dtype=np.float64)
    node_edge = np.empty(n_total, dtype=np.int32)
    edge_start = np.empty(n_edges + 1, dtype=np.int32)

    idx = 0
    for e in range(n_edges):
        edge_start[e] = idx
        m = int(counts[e])
        a = V[e]
        b = V[(e + 1) % n_edges]
        j = np.arange(m, dtype=np.float64)
        t = 0.5 * (1.0 - np.cos(np.pi * j / m))
        pts = (1.0 - t)[:, None] * a + t[:, None] * b
        nodes[idx : idx + m] = pts
        node_edge[idx : idx + m] = e
        idx += m
    edge_start[n_edges] = idx

    elem_n0 = np.arange(n_total, dtype=np.int32)
    elem_n1 = (elem_n0 + 1) % n_total
    elements = np.column_stack((elem_n0, elem_n1)).astype(np.int32, copy=False)

    elem_len = np.empty(n_total, dtype=np.float64)
    nx = np.empty(n_total, dtype=np.float64)
    ny = np.empty(n_total, dtype=np.float64)
    qx = np.empty((n_total, nq), dtype=np.float64)
    qy = np.empty((n_total, nq), dtype=np.float64)
    wphi1 = np.empty((n_total, nq), dtype=np.float64)
    wphi2 = np.empty((n_total, nq), dtype=np.float64)

    xi, wi = leggauss(nq)
    phi1 = 0.5 * (1.0 - xi)
    phi2 = 0.5 * (1.0 + xi)

    for k in range(n_total):
        a = nodes[k]
        b = nodes[(k + 1) % n_total]
        d = b - a
        L = float(np.hypot(d[0], d[1]))
        elem_len[k] = L
        nx[k] = d[1] / L
        ny[k] = -d[0] / L
        for q in range(nq):
            lam = 0.5 * (1.0 + xi[q])
            qx[k, q] = a[0] + lam * d[0]
            qy[k, q] = a[1] + lam * d[1]
            w = 0.5 * L * wi[q]
            wphi1[k, q] = w * phi1[q]
            wphi2[k, q] = w * phi2[q]

    c_diag = np.full(n_total, -0.5, dtype=np.float64)
    for e in range(n_edges):
        i = edge_start[e]
        prev_v = V[(e - 1) % n_edges]
        curr_v = V[e]
        next_v = V[(e + 1) % n_edges]
        tin = curr_v - prev_v
        tout = next_v - curr_v
        turn = math.atan2(
            tin[0] * tout[1] - tin[1] * tout[0],
            float(np.dot(tin, tout)),
        )
        theta = np.pi - turn
        c_diag[i] = -theta / (2.0 * np.pi)

    skip_edge1 = node_edge.copy()
    skip_edge2 = np.full(n_total, -1, dtype=np.int32)
    vertex_ids = edge_start[:-1]
    skip_edge2[vertex_ids] = (np.arange(n_edges, dtype=np.int32) - 1) % n_edges

    normals = np.column_stack((nx, ny))

    return {
        "V": V,
        "nodes": nodes,
        "elements": elements,
        "lengths": elem_len,
        "normals": normals,
        "edge_start": edge_start,
        "node_edge": node_edge,
        "skip_edge1": skip_edge1,
        "skip_edge2": skip_edge2,
        "elem_n0": elem_n0,
        "elem_n1": elem_n1,
        "c_diag": c_diag,
        "qx": qx,
        "qy": qy,
        "nx": nx,
        "ny": ny,
        "wphi1": wphi1,
        "wphi2": wphi2,
        "perimeter": perimeter,
        "n_edges": n_edges,
        "n_total": n_total,
        "nq": nq,
    }


def points_in_polygon(points, poly):
    x = points[:, 0]
    y = points[:, 1]
    x0 = poly[:, 0]
    y0 = poly[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    inside = np.zeros(points.shape[0], dtype=bool)
    for i in range(len(poly)):
        cond = ((y0[i] > y) != (y1[i] > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            xinters = (x1[i] - x0[i]) * (y - y0[i]) / (y1[i] - y0[i]) + x0[i]
        inside ^= cond & (x < xinters)
    return inside


def min_dist_to_boundary(points, poly):
    p = np.asarray(points, dtype=np.float64)
    d2min = np.full(p.shape[0], np.inf, dtype=np.float64)
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        v = b - a
        vv = float(np.dot(v, v))
        w = p - a
        t = (w[:, 0] * v[0] + w[:, 1] * v[1]) / vv
        t = np.clip(t, 0.0, 1.0)
        proj = a + t[:, None] * v
        d2 = np.sum((p - proj) ** 2, axis=1)
        d2min = np.minimum(d2min, d2)
    return np.sqrt(d2min)


def build_interior_grid(vertices, n_values):
    ngrid = 200
    xs = np.linspace(-1.5, 1.5, ngrid)
    ys = np.linspace(-1.0, 1.0, ngrid)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])

    interior_mask = points_in_polygon(all_pts, vertices)
    interior = all_pts[interior_mask]

    n_min = min(n_values)
    perimeter = float(np.sum(np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)))
    h_coarse = perimeter / n_min
    delta = 2.0 * h_coarse

    dist = min_dist_to_boundary(interior, vertices)
    grid_pts = interior[dist > delta]
    if grid_pts.shape[0] == 0:
        raise RuntimeError("No interior evaluation points survived the near-boundary filter.")
    return grid_pts


@njit(parallel=True, fastmath=True)
def apply_boundary_operator(mu, nodes, skip_edge1, skip_edge2, edge_start, c_diag, qx, qy, nx, ny, wphi1, wphi2, elem_n0, elem_n1):
    n_total = mu.shape[0]
    n_edges = edge_start.shape[0] - 1
    nq = qx.shape[1]
    out = np.empty(n_total, dtype=np.float64)

    for i in prange(n_total):
        x0 = nodes[i, 0]
        x1 = nodes[i, 1]
        s = c_diag[i] * mu[i]
        s1 = skip_edge1[i]
        s2 = skip_edge2[i]

        for e in range(n_edges):
            if e == s1 or e == s2:
                continue
            a = edge_start[e]
            b = edge_start[e + 1]
            for k in range(a, b):
                mu0 = mu[elem_n0[k]]
                mu1 = mu[elem_n1[k]]
                nxk = nx[k]
                nyk = ny[k]
                for q in range(nq):
                    dx = x0 - qx[k, q]
                    dy = x1 - qy[k, q]
                    r2 = dx * dx + dy * dy
                    ker = INV_2PI * (dx * nxk + dy * nyk) / r2
                    s += ker * (wphi1[k, q] * mu0 + wphi2[k, q] * mu1)

        out[i] = s

    return out


@njit(parallel=True, fastmath=True)
def evaluate_potential(pts, mu, edge_start, qx, qy, nx, ny, wphi1, wphi2, elem_n0, elem_n1):
    n_pts = pts.shape[0]
    n_edges = edge_start.shape[0] - 1
    nq = qx.shape[1]
    out = np.empty(n_pts, dtype=np.float64)

    for p in prange(n_pts):
        x0 = pts[p, 0]
        x1 = pts[p, 1]
        s = 0.0

        for e in range(n_edges):
            a = edge_start[e]
            b = edge_start[e + 1]
            for k in range(a, b):
                mu0 = mu[elem_n0[k]]
                mu1 = mu[elem_n1[k]]
                nxk = nx[k]
                nyk = ny[k]
                for q in range(nq):
                    dx = x0 - qx[k, q]
                    dy = x1 - qy[k, q]
                    r2 = dx * dx + dy * dy
                    ker = INV_2PI * (dx * nxk + dy * nyk) / r2
                    s += ker * (wphi1[k, q] * mu0 + wphi2[k, q] * mu1)

        out[p] = s

    return out


def main():
    V = ensure_ccw(VERTICES)
    grid_pts = build_interior_grid(V, N_VALUES)
    u_exact_grid = u_exact(grid_pts[:, 0], grid_pts[:, 1])
    norm_exact = float(np.linalg.norm(u_exact_grid))

    dummy_geo = build_discretization(V, len(V), nq=NQ)
    dummy_mu = np.zeros(dummy_geo["n_total"], dtype=np.float64)
    _ = apply_boundary_operator(
        dummy_mu,
        dummy_geo["nodes"],
        dummy_geo["skip_edge1"],
        dummy_geo["skip_edge2"],
        dummy_geo["edge_start"],
        dummy_geo["c_diag"],
        dummy_geo["qx"],
        dummy_geo["qy"],
        dummy_geo["nx"],
        dummy_geo["ny"],
        dummy_geo["wphi1"],
        dummy_geo["wphi2"],
        dummy_geo["elem_n0"],
        dummy_geo["elem_n1"],
    )
    _ = evaluate_potential(
        grid_pts[:1],
        dummy_mu,
        dummy_geo["edge_start"],
        dummy_geo["qx"],
        dummy_geo["qy"],
        dummy_geo["nx"],
        dummy_geo["ny"],
        dummy_geo["wphi1"],
        dummy_geo["wphi2"],
        dummy_geo["elem_n0"],
        dummy_geo["elem_n1"],
    )

    print(f"{'N':>6} {'Unknowns':>9} {'GMRES':>7} {'Rel L2 Error':>15} {'Setup':>9} {'Solve':>9} {'Eval':>9} {'Total':>9}")

    errors = []

    for N in N_VALUES:
        t0 = perf_counter()
        geo = build_discretization(V, N, nq=NQ)
        nodes = geo["nodes"]
        rhs = u_exact(nodes[:, 0], nodes[:, 1])

        def mv(v, geo=geo):
            return apply_boundary_operator(
                v,
                geo["nodes"],
                geo["skip_edge1"],
                geo["skip_edge2"],
                geo["edge_start"],
                geo["c_diag"],
                geo["qx"],
                geo["qy"],
                geo["nx"],
                geo["ny"],
                geo["wphi1"],
                geo["wphi2"],
                geo["elem_n0"],
                geo["elem_n1"],
            )

        A = LinearOperator((N, N), matvec=mv, dtype=np.float64)
        setup_time = perf_counter() - t0

        iters = [0]

        def cb(_):
            iters[0] += 1

        t1 = perf_counter()
        mu, info = gmres(
            A,
            rhs,
            rtol=1e-10,
            atol=1e-10,
            restart=N,
            callback=cb,
            callback_type="pr_norm",
        )
        solve_time = perf_counter() - t1

        t2 = perf_counter()
        u_num = evaluate_potential(
            grid_pts,
            mu,
            geo["edge_start"],
            geo["qx"],
            geo["qy"],
            geo["nx"],
            geo["ny"],
            geo["wphi1"],
            geo["wphi2"],
            geo["elem_n0"],
            geo["elem_n1"],
        )
        eval_time = perf_counter() - t2

        rel_l2 = float(np.linalg.norm(u_num - u_exact_grid) / norm_exact)
        errors.append(rel_l2)

        total_time = setup_time + solve_time + eval_time
        print(
            f"{N:6d} {N:9d} {iters[0]:7d} {rel_l2:15.6e} "
            f"{setup_time:9.2f} {solve_time:9.2f} {eval_time:9.2f} {total_time:9.2f}"
        )

    h = 1.0 / np.array(N_VALUES, dtype=np.float64)
    log_h = np.log(h)
    log_err = np.log(np.array(errors, dtype=np.float64))
    slope, _ = np.polyfit(log_h, log_err, 1)
    print(f"Estimated convergence order = {slope:.6f}")


if __name__ == "__main__":
    main()