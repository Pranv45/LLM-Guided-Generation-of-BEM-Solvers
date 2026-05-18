import gc
import math
import time

import numpy as np
from numba import njit, prange

INV_4PI = 1.0 / (4.0 * math.pi)
SQRT2 = math.sqrt(2.0)

DUNAVANT_BARY = np.array(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.059715871789770, 0.470142064105115, 0.470142064105115],
        [0.470142064105115, 0.059715871789770, 0.470142064105115],
        [0.470142064105115, 0.470142064105115, 0.059715871789770],
        [0.797426985353087, 0.101286507323456, 0.101286507323456],
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
    ],
    dtype=np.float64,
)

DUNAVANT_W = np.array(
    [
        0.225000000000000,
        0.132394152788506,
        0.132394152788506,
        0.132394152788506,
        0.125939180544827,
        0.125939180544827,
        0.125939180544827,
    ],
    dtype=np.float64,
)

GAUSS3_X = np.array(
    [0.1127016653792583, 0.5000000000000000, 0.8872983346207417],
    dtype=np.float64,
)

GAUSS3_W = np.array(
    [5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0],
    dtype=np.float64,
)

CENTER_PHYS = np.array([0.75, 0.75, 0.50], dtype=np.float64)


def shear_map(x, y, z):
    return x + 0.5 * y, y + 0.5 * z, z


def exact_u(x, y, z):
    return np.sinh(SQRT2 * x) * np.sin(y) * np.cos(z)


def build_face(base, u_dir, v_dir, N):
    s = np.linspace(0.0, 1.0, N + 1, dtype=np.float64)
    t = np.linspace(0.0, 1.0, N + 1, dtype=np.float64)
    S, T = np.meshgrid(s, t, indexing="ij")

    X = base[0] + S * u_dir[0] + T * v_dir[0]
    Y = base[1] + S * u_dir[1] + T * v_dir[1]
    Z = base[2] + S * u_dir[2] + T * v_dir[2]
    X, Y, Z = shear_map(X, Y, Z)

    nodes = np.column_stack((X.ravel(), Y.ravel(), Z.ravel())).astype(np.float64)

    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    n00 = (ii * (N + 1) + jj).ravel()
    n10 = ((ii + 1) * (N + 1) + jj).ravel()
    n01 = (ii * (N + 1) + (jj + 1)).ravel()
    n11 = ((ii + 1) * (N + 1) + (jj + 1)).ravel()

    tri1 = np.column_stack((n00, n10, n11))
    tri2 = np.column_stack((n00, n11, n01))
    elems = np.vstack((tri1, tri2)).astype(np.int64)

    p0 = nodes[elems[:, 0]]
    p1 = nodes[elems[:, 1]]
    p2 = nodes[elems[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    centroids = (p0 + p1 + p2) / 3.0
    sign = np.sum(cross * (centroids - CENTER_PHYS), axis=1)

    flip = sign < 0.0
    if np.any(flip):
        tmp = elems[flip, 1].copy()
        elems[flip, 1] = elems[flip, 2]
        elems[flip, 2] = tmp

    return nodes, elems


def generate_mesh(N):
    faces = [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ]

    nodes_all = []
    elems_all = []
    node_offset = 0

    for base, u_dir, v_dir in faces:
        nodes_f, elems_f = build_face(base, u_dir, v_dir, N)
        nodes_all.append(nodes_f)
        elems_all.append(elems_f + node_offset)
        node_offset += nodes_f.shape[0]

    nodes = np.ascontiguousarray(np.vstack(nodes_all), dtype=np.float64)
    elems = np.ascontiguousarray(np.vstack(elems_all), dtype=np.int64)
    return nodes, elems


def compute_geometry(nodes, elems):
    p0 = nodes[elems[:, 0]]
    p1 = nodes[elems[:, 1]]
    p2 = nodes[elems[:, 2]]

    e1 = p1 - p0
    e2 = p2 - p0
    cross = np.cross(e1, e2)
    cross_norm = np.sqrt(np.sum(cross * cross, axis=1))

    areas = 0.5 * cross_norm
    normals = cross / cross_norm[:, None]
    centroids = (p0 + p1 + p2) / 3.0

    return areas, normals, centroids


def assign_bcs(centroids):
    return exact_u(centroids[:, 0], centroids[:, 1], centroids[:, 2]).astype(np.float64)


def precompute_regular_quadrature(verts, areas):
    ne = verts.shape[0]
    qx = np.empty((ne, 7), dtype=np.float64)
    qy = np.empty((ne, 7), dtype=np.float64)
    qz = np.empty((ne, 7), dtype=np.float64)
    qw = np.empty((ne, 7), dtype=np.float64)

    for e in range(ne):
        p0x, p0y, p0z = verts[e, 0, 0], verts[e, 0, 1], verts[e, 0, 2]
        p1x, p1y, p1z = verts[e, 1, 0], verts[e, 1, 1], verts[e, 1, 2]
        p2x, p2y, p2z = verts[e, 2, 0], verts[e, 2, 1], verts[e, 2, 2]

        for k in range(7):
            l1 = DUNAVANT_BARY[k, 0]
            l2 = DUNAVANT_BARY[k, 1]
            l3 = DUNAVANT_BARY[k, 2]

            qx[e, k] = l1 * p0x + l2 * p1x + l3 * p2x
            qy[e, k] = l1 * p0y + l2 * p1y + l3 * p2y
            qz[e, k] = l1 * p0z + l2 * p1z + l3 * p2z
            qw[e, k] = areas[e] * DUNAVANT_W[k]

    return (
        np.ascontiguousarray(qx),
        np.ascontiguousarray(qy),
        np.ascontiguousarray(qz),
        np.ascontiguousarray(qw),
    )


def make_interior_points():
    ref = np.linspace(0.2, 0.8, 5, dtype=np.float64)
    xx, yy, zz = np.meshgrid(ref, ref, ref, indexing="ij")

    x = xx.ravel()
    y = yy.ravel()
    z = zz.ravel()

    xp, yp, zp = shear_map(x, y, z)
    pts = np.column_stack((xp, yp, zp)).astype(np.float64)
    uex = exact_u(pts[:, 0], pts[:, 1], pts[:, 2]).astype(np.float64)
    return pts, uex


@njit(fastmath=True, cache=True, inline="always")
def regular_dlp_integral(xi, yi, zi, nxj, nyj, nzj, j, qx, qy, qz, qw):
    acc = 0.0
    for k in range(7):
        dx = xi - qx[j, k]
        dy = yi - qy[j, k]
        dz = zi - qz[j, k]

        r2 = dx * dx + dy * dy + dz * dz
        inv_r = 1.0 / math.sqrt(r2)
        dot = dx * nxj + dy * nyj + dz * nzj
        acc += qw[j, k] * dot * inv_r * inv_r * inv_r

    return INV_4PI * acc


@njit(parallel=True, fastmath=True, cache=True)
def assemble_h_matrix(cx, cy, cz, nx, ny, nz, qx, qy, qz, qw):
    ne = cx.shape[0]
    H = np.empty((ne, ne), dtype=np.float64)

    for i in prange(ne):
        xi = cx[i]
        yi = cy[i]
        zi = cz[i]

        for j in range(ne):
            if i == j:
                H[i, j] = -0.5
            else:
                H[i, j] = regular_dlp_integral(
                    xi, yi, zi, nx[j], ny[j], nz[j], j, qx, qy, qz, qw
                )

    return H


@njit(parallel=True, fastmath=True, cache=True)
def evaluate_interior_points(tx, ty, tz, mu, nx, ny, nz, qx, qy, qz, qw):
    npnt = tx.shape[0]
    ne = mu.shape[0]
    vals = np.empty(npnt, dtype=np.float64)

    for p in prange(npnt):
        xp = tx[p]
        yp = ty[p]
        zp = tz[p]
        val = 0.0

        for j in range(ne):
            val += mu[j] * regular_dlp_integral(
                xp, yp, zp, nx[j], ny[j], nz[j], j, qx, qy, qz, qw
            )

        vals[p] = val

    return vals


def warmup_numba():
    nodes, elems = generate_mesh(1)
    verts = np.ascontiguousarray(nodes[elems], dtype=np.float64)
    areas, normals, centroids = compute_geometry(nodes, elems)
    u_bc = assign_bcs(centroids)
    qx, qy, qz, qw = precompute_regular_quadrature(verts, areas)

    cx = np.ascontiguousarray(centroids[:, 0], dtype=np.float64)
    cy = np.ascontiguousarray(centroids[:, 1], dtype=np.float64)
    cz = np.ascontiguousarray(centroids[:, 2], dtype=np.float64)
    nx = np.ascontiguousarray(normals[:, 0], dtype=np.float64)
    ny = np.ascontiguousarray(normals[:, 1], dtype=np.float64)
    nz = np.ascontiguousarray(normals[:, 2], dtype=np.float64)

    H = assemble_h_matrix(cx, cy, cz, nx, ny, nz, qx, qy, qz, qw)
    mu = np.linalg.solve(H, u_bc)

    pts, _ = make_interior_points()
    tx = np.ascontiguousarray(pts[:1, 0], dtype=np.float64)
    ty = np.ascontiguousarray(pts[:1, 1], dtype=np.float64)
    tz = np.ascontiguousarray(pts[:1, 2], dtype=np.float64)
    _ = evaluate_interior_points(tx, ty, tz, mu, nx, ny, nz, qx, qy, qz, qw)


def run_case(N):
    t0 = time.perf_counter()

    nodes, elems = generate_mesh(N)
    verts = np.ascontiguousarray(nodes[elems], dtype=np.float64)
    areas, normals, centroids = compute_geometry(nodes, elems)
    u_bc = assign_bcs(centroids)

    qx, qy, qz, qw = precompute_regular_quadrature(verts, areas)

    cx = np.ascontiguousarray(centroids[:, 0], dtype=np.float64)
    cy = np.ascontiguousarray(centroids[:, 1], dtype=np.float64)
    cz = np.ascontiguousarray(centroids[:, 2], dtype=np.float64)
    nx = np.ascontiguousarray(normals[:, 0], dtype=np.float64)
    ny = np.ascontiguousarray(normals[:, 1], dtype=np.float64)
    nz = np.ascontiguousarray(normals[:, 2], dtype=np.float64)

    H = assemble_h_matrix(cx, cy, cz, nx, ny, nz, qx, qy, qz, qw)
    setup_t = time.perf_counter() - t0

    t1 = time.perf_counter()
    mu = np.linalg.solve(H, u_bc)
    solve_t = time.perf_counter() - t1

    del H
    gc.collect()

    pts, uex = make_interior_points()
    tx = np.ascontiguousarray(pts[:, 0], dtype=np.float64)
    ty = np.ascontiguousarray(pts[:, 1], dtype=np.float64)
    tz = np.ascontiguousarray(pts[:, 2], dtype=np.float64)

    t2 = time.perf_counter()
    unum = evaluate_interior_points(tx, ty, tz, mu, nx, ny, nz, qx, qy, qz, qw)
    eval_t = time.perf_counter() - t2

    rel_err = np.linalg.norm(unum - uex) / np.linalg.norm(uex)
    total_t = setup_t + solve_t + eval_t

    return elems.shape[0], rel_err, setup_t, solve_t, eval_t, total_t


def main():
    warmup_numba()

    Ns = [8, 16, 32]
    results = []

    for N in Ns:
        results.append((N,) + run_case(N))

    slope = (
        math.log(results[-1][2] / results[0][2])
        / math.log((1.0 / results[-1][0]) / (1.0 / results[0][0]))
    )

    print("N    | Ne      | Rel L2 Error   | Setup (s) | Solve (s) | Eval (s) | Total (s)")
    for N, Ne, err, setup_t, solve_t, eval_t, total_t in results:
        print(
            f"{N:<4d} | {Ne:<7d} | {err:<14.6e} | {setup_t:<8.4f} | {solve_t:<8.4f} | {eval_t:<8.4f} | {total_t:<8.4f}"
        )
    print("Convergence Analysis:")
    print(f"Computed Slope: {slope:.4f}")
    print("Expected Slope: ~1.0000 (O(h) for constant elements)")


if __name__ == "__main__":
    main()