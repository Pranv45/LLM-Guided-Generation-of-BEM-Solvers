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

CENTER_PHYS = np.array([0.75, 0.75, 0.50], dtype=np.float64)


def shear_map(x, y, z):
    return x + 0.5 * y, y + 0.5 * z, z


def exact_u(x, y, z):
    return np.sinh(SQRT2 * x) * np.sin(y) * np.cos(z)


def _rounded_key(x, y, z):
    return (round(float(x), 6), round(float(y), 6), round(float(z), 6))


def build_face(base, u_dir, v_dir, N, face_id, node_map, nodes, node_mask, elems, elem_face):
    # Face corner geometry in physical coordinates
    p00 = shear_map(base[0], base[1], base[2])
    p10 = shear_map(base[0] + u_dir[0], base[1] + u_dir[1], base[2] + u_dir[2])
    p01 = shear_map(base[0] + v_dir[0], base[1] + v_dir[1], base[2] + v_dir[2])

    e1x = p10[0] - p00[0]
    e1y = p10[1] - p00[1]
    e1z = p10[2] - p00[2]
    e2x = p01[0] - p00[0]
    e2y = p01[1] - p00[1]
    e2z = p01[2] - p00[2]

    fnx = e1y * e2z - e1z * e2y
    fny = e1z * e2x - e1x * e2z
    fnz = e1x * e2y - e1y * e2x

    fn_norm = math.sqrt(fnx * fnx + fny * fny + fnz * fnz)
    fnx /= fn_norm
    fny /= fn_norm
    fnz /= fn_norm

    face_center_x = (p00[0] + p10[0] + p01[0]) / 3.0
    face_center_y = (p00[1] + p10[1] + p01[1]) / 3.0
    face_center_z = (p00[2] + p10[2] + p01[2]) / 3.0

    sx = face_center_x - CENTER_PHYS[0]
    sy = face_center_y - CENTER_PHYS[1]
    sz = face_center_z - CENTER_PHYS[2]

    if fnx * sx + fny * sy + fnz * sz < 0.0:
        fnx = -fnx
        fny = -fny
        fnz = -fnz

    local_ids = np.empty((N + 1, N + 1), dtype=np.int64)

    for i in range(N + 1):
        s = i / N
        for j in range(N + 1):
            t = j / N

            x = base[0] + s * u_dir[0] + t * v_dir[0]
            y = base[1] + s * u_dir[1] + t * v_dir[1]
            z = base[2] + s * u_dir[2] + t * v_dir[2]
            X, Y, Z = shear_map(x, y, z)

            key = _rounded_key(X, Y, Z)
            if key in node_map:
                idx = node_map[key]
            else:
                idx = len(nodes)
                node_map[key] = idx
                nodes.append((X, Y, Z))
                node_mask.append(0)

            node_mask[idx] |= (1 << face_id)
            local_ids[i, j] = idx

    for i in range(N):
        for j in range(N):
            n00 = local_ids[i, j]
            n10 = local_ids[i + 1, j]
            n01 = local_ids[i, j + 1]
            n11 = local_ids[i + 1, j + 1]

            elems.append((n00, n10, n11))
            elem_face.append(face_id)

            elems.append((n00, n11, n01))
            elem_face.append(face_id)

    return (fnx, fny, fnz)


def generate_mesh(N):
    faces = [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),  # x = 0
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # x = 1
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # y = 0
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),  # y = 1
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),  # z = 0
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # z = 1
    ]

    node_map = {}
    nodes = []
    node_mask = []
    elems = []
    elem_face = []
    face_normals = np.zeros((6, 3), dtype=np.float64)

    for face_id, (base, u_dir, v_dir) in enumerate(faces):
        fnx, fny, fnz = build_face(
            base, u_dir, v_dir, N, face_id, node_map, nodes, node_mask, elems, elem_face
        )
        face_normals[face_id, 0] = fnx
        face_normals[face_id, 1] = fny
        face_normals[face_id, 2] = fnz

    nodes = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(elems, dtype=np.int64)
    elem_face = np.asarray(elem_face, dtype=np.int64)
    node_mask = np.asarray(node_mask, dtype=np.int64)

    # Ensure outward orientation of each triangle.
    p0 = nodes[elems[:, 0]]
    p1 = nodes[elems[:, 1]]
    p2 = nodes[elems[:, 2]]

    cx = (p0[:, 0] + p1[:, 0] + p2[:, 0]) / 3.0
    cy = (p0[:, 1] + p1[:, 1] + p2[:, 1]) / 3.0
    cz = (p0[:, 2] + p1[:, 2] + p2[:, 2]) / 3.0

    nx = (p1[:, 1] - p0[:, 1]) * (p2[:, 2] - p0[:, 2]) - (p1[:, 2] - p0[:, 2]) * (p2[:, 1] - p0[:, 1])
    ny = (p1[:, 2] - p0[:, 2]) * (p2[:, 0] - p0[:, 0]) - (p1[:, 0] - p0[:, 0]) * (p2[:, 2] - p0[:, 2])
    nz = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])

    fn = face_normals[elem_face]
    sign = nx * fn[:, 0] + ny * fn[:, 1] + nz * fn[:, 2]
    flip = sign < 0.0
    if np.any(flip):
        tmp = elems[flip, 1].copy()
        elems[flip, 1] = elems[flip, 2]
        elems[flip, 2] = tmp

    return nodes, elems, elem_face, node_mask, face_normals


def compute_geometry(nodes, elems):
    p0 = nodes[elems[:, 0]]
    p1 = nodes[elems[:, 1]]
    p2 = nodes[elems[:, 2]]

    e1x = p1[:, 0] - p0[:, 0]
    e1y = p1[:, 1] - p0[:, 1]
    e1z = p1[:, 2] - p0[:, 2]

    e2x = p2[:, 0] - p0[:, 0]
    e2y = p2[:, 1] - p0[:, 1]
    e2z = p2[:, 2] - p0[:, 2]

    cx = e1y * e2z - e1z * e2y
    cy = e1z * e2x - e1x * e2z
    cz = e1x * e2y - e1y * e2x

    cn = np.sqrt(cx * cx + cy * cy + cz * cz)
    areas = 0.5 * cn
    normals = np.column_stack((cx / cn, cy / cn, cz / cn)).astype(np.float64)
    centroids = ((p0 + p1 + p2) / 3.0).astype(np.float64)
    return areas.astype(np.float64), normals, centroids


def assign_bcs(nodes):
    return exact_u(nodes[:, 0], nodes[:, 1], nodes[:, 2]).astype(np.float64)


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
            l0 = DUNAVANT_BARY[k, 0]
            l1 = DUNAVANT_BARY[k, 1]
            l2 = DUNAVANT_BARY[k, 2]

            qx[e, k] = l0 * p0x + l1 * p1x + l2 * p2x
            qy[e, k] = l0 * p0y + l1 * p1y + l2 * p2y
            qz[e, k] = l0 * p0z + l1 * p1z + l2 * p2z
            qw[e, k] = areas[e] * DUNAVANT_W[k]

    return (
        np.ascontiguousarray(qx),
        np.ascontiguousarray(qy),
        np.ascontiguousarray(qz),
        np.ascontiguousarray(qw),
    )


def compute_jump_coefficients(node_mask, face_normals):
    nnodes = node_mask.shape[0]
    jump = np.empty(nnodes, dtype=np.float64)

    for i in range(nnodes):
        mask = int(node_mask[i])
        faces = []
        for f in range(6):
            if mask & (1 << f):
                faces.append(f)

        nfaces = len(faces)
        if nfaces == 1:
            c = 0.5
        elif nfaces == 2:
            n1 = face_normals[faces[0]]
            n2 = face_normals[faces[1]]
            dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]
            if dot > 1.0:
                dot = 1.0
            if dot < -1.0:
                dot = -1.0
            alpha = math.pi - math.acos(dot)
            c = alpha / (2.0 * math.pi)
        elif nfaces == 3:
            n1 = face_normals[faces[0]]
            n2 = face_normals[faces[1]]
            n3 = face_normals[faces[2]]
            det = abs(
                n1[0] * (n2[1] * n3[2] - n2[2] * n3[1])
                - n1[1] * (n2[0] * n3[2] - n2[2] * n3[0])
                + n1[2] * (n2[0] * n3[1] - n2[1] * n3[0])
            )
            d12 = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]
            d23 = n2[0] * n3[0] + n2[1] * n3[1] + n2[2] * n3[2]
            d31 = n3[0] * n1[0] + n3[1] * n1[1] + n3[2] * n1[2]
            denom = 1.0 + d12 + d23 + d31
            omega = 2.0 * math.atan2(det, denom)
            c = omega / (4.0 * math.pi)
        else:
            # Should not occur for a parallelepiped mesh
            c = 0.5

        jump[i] = -c

    return np.ascontiguousarray(jump, dtype=np.float64)


def make_interior_points():
    ref = np.linspace(0.2, 0.8, 5, dtype=np.float64)
    xx, yy, zz = np.meshgrid(ref, ref, ref, indexing="ij")
    x = xx.ravel()
    y = yy.ravel()
    z = zz.ravel()

    X, Y, Z = shear_map(x, y, z)
    pts = np.column_stack((X, Y, Z)).astype(np.float64)
    uex = exact_u(pts[:, 0], pts[:, 1], pts[:, 2]).astype(np.float64)
    return pts, uex


@njit(fastmath=True, cache=True, inline="always")
def regular_element_contrib_with_normal(xi, yi, zi, nxj, nyj, nzj, e, qx, qy, qz, qw):
    i0 = 0.0
    i1 = 0.0
    i2 = 0.0

    for k in range(7):
        dx = xi - qx[e, k]
        dy = yi - qy[e, k]
        dz = zi - qz[e, k]

        r2 = dx * dx + dy * dy + dz * dz
        if r2 < 1e-30:
            r2 = 1e-30

        inv_r = 1.0 / math.sqrt(r2)
        inv_r3 = inv_r * inv_r * inv_r
        dot = dx * nxj + dy * nyj + dz * nzj
        ker = INV_4PI * dot * inv_r3

        w = qw[e, k]
        l0 = DUNAVANT_BARY[k, 0]
        l1 = DUNAVANT_BARY[k, 1]
        l2 = DUNAVANT_BARY[k, 2]

        i0 += w * l0 * ker
        i1 += w * l1 * ker
        i2 += w * l2 * ker

    return i0, i1, i2


@njit(parallel=True, fastmath=True, cache=True)
def assemble_h_matrix(
    nodes_x,
    nodes_y,
    nodes_z,
    conn0,
    conn1,
    conn2,
    elem_face_bit,
    nx,
    ny,
    nz,
    qx,
    qy,
    qz,
    qw,
    jump,
    node_mask,
):
    nnodes = nodes_x.shape[0]
    ne = conn0.shape[0]
    H = np.zeros((nnodes, nnodes), dtype=np.float64)

    for i in prange(nnodes):
        xi = nodes_x[i]
        yi = nodes_y[i]
        zi = nodes_z[i]
        mask_i = node_mask[i]

        H[i, i] = jump[i]

        for e in range(ne):
            if mask_i & elem_face_bit[e]:
                continue

            n0 = conn0[e]
            n1 = conn1[e]
            n2 = conn2[e]

            c0, c1, c2 = regular_element_contrib_with_normal(
                xi, yi, zi, nx[e], ny[e], nz[e], e, qx, qy, qz, qw
            )

            H[i, n0] += c0
            H[i, n1] += c1
            H[i, n2] += c2

    return H


@njit(parallel=True, fastmath=True, cache=True)
def evaluate_interior_points(
    tx,
    ty,
    tz,
    mu,
    conn0,
    conn1,
    conn2,
    nx,
    ny,
    nz,
    qx,
    qy,
    qz,
    qw,
):
    npnt = tx.shape[0]
    ne = conn0.shape[0]
    vals = np.empty(npnt, dtype=np.float64)

    for p in prange(npnt):
        xp = tx[p]
        yp = ty[p]
        zp = tz[p]
        acc = 0.0

        for e in range(ne):
            n0 = conn0[e]
            n1 = conn1[e]
            n2 = conn2[e]

            c0, c1, c2 = regular_element_contrib_with_normal(
                xp, yp, zp, nx[e], ny[e], nz[e], e, qx, qy, qz, qw
            )

            acc += mu[n0] * c0 + mu[n1] * c1 + mu[n2] * c2

        vals[p] = acc

    return vals


def warmup_numba():
    nodes, elems, elem_face, node_mask, face_normals = generate_mesh(1)
    verts = np.ascontiguousarray(nodes[elems], dtype=np.float64)
    areas, normals, centroids = compute_geometry(nodes, elems)
    qx, qy, qz, qw = precompute_regular_quadrature(verts, areas)
    u_bc = assign_bcs(nodes)
    jump = compute_jump_coefficients(node_mask, face_normals)

    conn0 = np.ascontiguousarray(elems[:, 0], dtype=np.int64)
    conn1 = np.ascontiguousarray(elems[:, 1], dtype=np.int64)
    conn2 = np.ascontiguousarray(elems[:, 2], dtype=np.int64)
    elem_face_bit = np.ascontiguousarray(np.left_shift(np.int64(1), elem_face), dtype=np.int64)

    H = assemble_h_matrix(
        np.ascontiguousarray(nodes[:, 0], dtype=np.float64),
        np.ascontiguousarray(nodes[:, 1], dtype=np.float64),
        np.ascontiguousarray(nodes[:, 2], dtype=np.float64),
        conn0,
        conn1,
        conn2,
        elem_face_bit,
        np.ascontiguousarray(normals[:, 0], dtype=np.float64),
        np.ascontiguousarray(normals[:, 1], dtype=np.float64),
        np.ascontiguousarray(normals[:, 2], dtype=np.float64),
        qx,
        qy,
        qz,
        qw,
        jump,
        np.ascontiguousarray(node_mask, dtype=np.int64),
    )
    mu = np.linalg.solve(H, u_bc)

    pts, _ = make_interior_points()
    _ = evaluate_interior_points(
        np.ascontiguousarray(pts[:1, 0], dtype=np.float64),
        np.ascontiguousarray(pts[:1, 1], dtype=np.float64),
        np.ascontiguousarray(pts[:1, 2], dtype=np.float64),
        mu,
        conn0,
        conn1,
        conn2,
        np.ascontiguousarray(normals[:, 0], dtype=np.float64),
        np.ascontiguousarray(normals[:, 1], dtype=np.float64),
        np.ascontiguousarray(normals[:, 2], dtype=np.float64),
        qx,
        qy,
        qz,
        qw,
    )


def run_case(N):
    t0 = time.perf_counter()

    nodes, elems, elem_face, node_mask, face_normals = generate_mesh(N)
    verts = np.ascontiguousarray(nodes[elems], dtype=np.float64)
    areas, normals, centroids = compute_geometry(nodes, elems)
    u_bc = assign_bcs(nodes)

    qx, qy, qz, qw = precompute_regular_quadrature(verts, areas)
    jump = compute_jump_coefficients(node_mask, face_normals)

    nodes_x = np.ascontiguousarray(nodes[:, 0], dtype=np.float64)
    nodes_y = np.ascontiguousarray(nodes[:, 1], dtype=np.float64)
    nodes_z = np.ascontiguousarray(nodes[:, 2], dtype=np.float64)

    conn0 = np.ascontiguousarray(elems[:, 0], dtype=np.int64)
    conn1 = np.ascontiguousarray(elems[:, 1], dtype=np.int64)
    conn2 = np.ascontiguousarray(elems[:, 2], dtype=np.int64)

    elem_face_bit = np.ascontiguousarray(np.left_shift(np.int64(1), elem_face), dtype=np.int64)

    nx = np.ascontiguousarray(normals[:, 0], dtype=np.float64)
    ny = np.ascontiguousarray(normals[:, 1], dtype=np.float64)
    nz = np.ascontiguousarray(normals[:, 2], dtype=np.float64)

    setup_t0 = time.perf_counter()
    H = assemble_h_matrix(
        nodes_x,
        nodes_y,
        nodes_z,
        conn0,
        conn1,
        conn2,
        elem_face_bit,
        nx,
        ny,
        nz,
        qx,
        qy,
        qz,
        qw,
        jump,
        np.ascontiguousarray(node_mask, dtype=np.int64),
    )
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
    unum = evaluate_interior_points(
        tx,
        ty,
        tz,
        mu,
        conn0,
        conn1,
        conn2,
        nx,
        ny,
        nz,
        qx,
        qy,
        qz,
        qw,
    )
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
    print("Expected Slope: ~2.0000 (O(h^2) for linear elements)")


if __name__ == "__main__":
    main()