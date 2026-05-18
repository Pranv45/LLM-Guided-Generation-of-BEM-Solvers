import gc
import os
import tempfile
import time

import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, gmres

# -----------------------------------------------------------------------------
# Global quadrature data
# -----------------------------------------------------------------------------
PI4 = 4.0 * np.pi
INV_4PI = 1.0 / PI4

# 7-point Dunavant rule (weights sum to 1; scale by triangle area)
DUNAVANT_L1 = np.array(
    [
        1.0 / 3.0,
        0.059715871789770,
        0.470142064105115,
        0.470142064105115,
        0.797426985353087,
        0.101286507323456,
        0.101286507323456,
    ],
    dtype=np.float64,
)
DUNAVANT_L2 = np.array(
    [
        1.0 / 3.0,
        0.470142064105115,
        0.059715871789770,
        0.470142064105115,
        0.101286507323456,
        0.797426985353087,
        0.101286507323456,
    ],
    dtype=np.float64,
)
DUNAVANT_L3 = np.array(
    [
        1.0 / 3.0,
        0.470142064105115,
        0.470142064105115,
        0.059715871789770,
        0.101286507323456,
        0.101286507323456,
        0.797426985353087,
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
NQ7 = 7

GL12_X, GL12_W = np.polynomial.legendre.leggauss(12)
GL12_X = np.asarray(GL12_X, dtype=np.float64)
GL12_W = np.asarray(GL12_W, dtype=np.float64)
NQ12 = 12 * 12

NEAR_RATIO = 20.0
GMRES_RESTART = 50
GMRES_MAXITER = 200
GMRES_RTOL = 1.0e-8

# -----------------------------------------------------------------------------
# Manufactured solution and gradient
# -----------------------------------------------------------------------------
def exact_u(X, Y, Z):
    return np.sinh(X) * np.sin(Y) + np.cosh(Y) * np.cos(Z)


def exact_grad(X, Y, Z):
    ux = np.cosh(X) * np.sin(Y)
    uy = np.sinh(X) * np.cos(Y) + np.sinh(Y) * np.cos(Z)
    uz = -np.cosh(Y) * np.sin(Z)
    return ux, uy, uz


# -----------------------------------------------------------------------------
# Mesh generation: cubed-sphere + bump radius
# -----------------------------------------------------------------------------
def _face_cube_coords(face, uu, vv):
    if face == 0:   # x = -1
        return -np.ones_like(uu), uu, vv
    if face == 1:   # x = +1
        return np.ones_like(uu), uu, vv
    if face == 2:   # y = -1
        return uu, -np.ones_like(uu), vv
    if face == 3:   # y = +1
        return uu, np.ones_like(uu), vv
    if face == 4:   # z = -1
        return uu, vv, -np.ones_like(uu)
    if face == 5:   # z = +1
        return uu, vv, np.ones_like(uu)
    raise ValueError("Invalid face index")


def generate_bumpy_mesh(N):
    s = np.linspace(-1.0, 1.0, N + 1)
    uu, vv = np.meshgrid(s, s, indexing="ij")

    nodes_list = []
    elems_list = []
    offset = 0
    face_node_count = (N + 1) * (N + 1)

    for face in range(6):
        x, y, z = _face_cube_coords(face, uu, vv)

        rcube = np.sqrt(x * x + y * y + z * z)
        xs = x / rcube
        ys = y / rcube
        zs = z / rcube

        theta = np.arccos(np.clip(zs, -1.0, 1.0))
        phi = np.arctan2(ys, xs)
        r = 1.5 + 0.3 * np.sin(4.0 * theta) * np.cos(5.0 * phi)

        X = r * xs
        Y = r * ys
        Z = r * zs

        nodes_face = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        nodes_list.append(nodes_face)

        idx = np.arange(face_node_count, dtype=np.int64).reshape(N + 1, N + 1) + offset
        ll = idx[:-1, :-1].ravel()
        lr = idx[:-1, 1:].ravel()
        ul = idx[1:, :-1].ravel()
        ur = idx[1:, 1:].ravel()

        tri1 = np.column_stack((ll, ul, ur))
        tri2 = np.column_stack((ll, ur, lr))

        elems_face = np.empty((2 * N * N, 3), dtype=np.int64)
        elems_face[0::2] = tri1
        elems_face[1::2] = tri2
        elems_list.append(elems_face)

        offset += face_node_count

    nodes = np.vstack(nodes_list)
    elems = np.vstack(elems_list)
    return nodes, elems


def compute_geometry(nodes, elems):
    tris = nodes[elems].astype(np.float64, copy=True)
    centroids = tris.mean(axis=1)

    v1 = tris[:, 1] - tris[:, 0]
    v2 = tris[:, 2] - tris[:, 0]
    cross = np.cross(v1, v2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (2.0 * areas[:, None])

    interior = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    flip = np.einsum("ij,ij->i", normals, centroids - interior) < 0.0

    if np.any(flip):
        tmp = tris[flip, 1].copy()
        tris[flip, 1] = tris[flip, 2]
        tris[flip, 2] = tmp

        v1 = tris[:, 1] - tris[:, 0]
        v2 = tris[:, 2] - tris[:, 0]
        cross = np.cross(v1, v2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        normals = cross / (2.0 * areas[:, None])
        centroids = tris.mean(axis=1)

    return areas, normals, centroids, tris


def assign_mixed_bcs(centroids, normals):
    x = centroids[:, 0]
    y = centroids[:, 1]
    z = centroids[:, 2]
    ux, uy, uz = exact_grad(x, y, z)
    u_exact = exact_u(x, y, z)
    q_exact = ux * normals[:, 0] + uy * normals[:, 1] + uz * normals[:, 2]

    is_dirichlet = x > 0.0
    u_known = np.zeros_like(u_exact)
    q_known = np.zeros_like(q_exact)
    u_known[is_dirichlet] = u_exact[is_dirichlet]
    q_known[~is_dirichlet] = q_exact[~is_dirichlet]
    return is_dirichlet, u_known, q_known, u_exact, q_exact


# -----------------------------------------------------------------------------
# Precompute quadrature mappings
# -----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def precompute_dunavant(tris, areas, qpts, qwts):
    ne = tris.shape[0]
    for e in prange(ne):
        x0 = tris[e, 0, 0]
        y0 = tris[e, 0, 1]
        z0 = tris[e, 0, 2]
        x1 = tris[e, 1, 0]
        y1 = tris[e, 1, 1]
        z1 = tris[e, 1, 2]
        x2 = tris[e, 2, 0]
        y2 = tris[e, 2, 1]
        z2 = tris[e, 2, 2]
        area = areas[e]
        for k in range(NQ7):
            l1 = DUNAVANT_L1[k]
            l2 = DUNAVANT_L2[k]
            l3 = DUNAVANT_L3[k]
            qpts[e, k, 0] = l1 * x0 + l2 * x1 + l3 * x2
            qpts[e, k, 1] = l1 * y0 + l2 * y1 + l3 * y2
            qpts[e, k, 2] = l1 * z0 + l2 * z1 + l3 * z2
            qwts[e, k] = area * DUNAVANT_W[k]


@njit(parallel=True, fastmath=True)
def precompute_dense_duffy(tris, qpts, qwts):
    ne = tris.shape[0]
    for e in prange(ne):
        a0 = tris[e, 0, 0]
        a1 = tris[e, 0, 1]
        a2 = tris[e, 0, 2]
        b0 = tris[e, 1, 0]
        b1 = tris[e, 1, 1]
        b2 = tris[e, 1, 2]
        c0 = tris[e, 2, 0]
        c1 = tris[e, 2, 1]
        c2 = tris[e, 2, 2]

        e1x = b0 - a0
        e1y = b1 - a1
        e1z = b2 - a2
        e2x = c0 - a0
        e2y = c1 - a1
        e2z = c2 - a2

        cx = e1y * e2z - e1z * e2y
        cy = e1z * e2x - e1x * e2z
        cz = e1x * e2y - e1y * e2x
        crossnorm = np.sqrt(cx * cx + cy * cy + cz * cz)

        idx = 0
        for iu in range(12):
            u = 0.5 * (GL12_X[iu] + 1.0)
            wu = 0.5 * GL12_W[iu]
            for iv in range(12):
                v = 0.5 * (GL12_X[iv] + 1.0)
                wv = 0.5 * GL12_W[iv]

                a = u * (1.0 - v)
                b = u * v

                qpts[e, idx, 0] = a0 + a * e1x + b * e2x
                qpts[e, idx, 1] = a1 + a * e1y + b * e2y
                qpts[e, idx, 2] = a2 + a * e1z + b * e2z
                qwts[e, idx] = wu * wv * crossnorm * u
                idx += 1


# -----------------------------------------------------------------------------
# Quadrature kernels
# -----------------------------------------------------------------------------
@njit(fastmath=True)
def pair_integrals_from_rule(xi, yi, zi, nx, ny, nz, qpts_j, qwts_j):
    h = 0.0
    g = 0.0
    nq = qpts_j.shape[0]
    for k in range(nq):
        dx = xi - qpts_j[k, 0]
        dy = yi - qpts_j[k, 1]
        dz = zi - qpts_j[k, 2]
        r2 = dx * dx + dy * dy + dz * dz
        r = np.sqrt(r2)
        w = qwts_j[k]
        rinv = 1.0 / r
        g += w * rinv
        h += w * (dx * nx + dy * ny + dz * nz) * rinv * rinv * rinv
    return h * INV_4PI, g * INV_4PI


@njit(fastmath=True)
def self_g_duffy(xi, yi, zi, tri):
    # Split triangle into 3 subtriangles around centroid singular vertex (xi, yi, zi)
    g = 0.0

    # vertices
    v0x = tri[0, 0]
    v0y = tri[0, 1]
    v0z = tri[0, 2]
    v1x = tri[1, 0]
    v1y = tri[1, 1]
    v1z = tri[1, 2]
    v2x = tri[2, 0]
    v2y = tri[2, 1]
    v2z = tri[2, 2]

    for s in range(3):
        if s == 0:
            ax, ay, az = xi, yi, zi
            bx, by, bz = v0x, v0y, v0z
            cx, cy, cz = v1x, v1y, v1z
        elif s == 1:
            ax, ay, az = xi, yi, zi
            bx, by, bz = v1x, v1y, v1z
            cx, cy, cz = v2x, v2y, v2z
        else:
            ax, ay, az = xi, yi, zi
            bx, by, bz = v2x, v2y, v2z
            cx, cy, cz = v0x, v0y, v0z

        e1x = bx - ax
        e1y = by - ay
        e1z = bz - az
        e2x = cx - ax
        e2y = cy - ay
        e2z = cz - az

        cxr = e1y * e2z - e1z * e2y
        cyr = e1z * e2x - e1x * e2z
        czr = e1x * e2y - e1y * e2x
        crossnorm = np.sqrt(cxr * cxr + cyr * cyr + czr * czr)

        for iu in range(12):
            u = 0.5 * (GL12_X[iu] + 1.0)
            wu = 0.5 * GL12_W[iu]
            for iv in range(12):
                v = 0.5 * (GL12_X[iv] + 1.0)
                wv = 0.5 * GL12_W[iv]

                a = u * (1.0 - v)
                b = u * v
                px = ax + a * e1x + b * e2x
                py = ay + a * e1y + b * e2y
                pz = az + a * e1z + b * e2z

                dx = xi - px
                dy = yi - py
                dz = zi - pz
                r = np.sqrt(dx * dx + dy * dy + dz * dz)

                g += wu * wv * crossnorm * u * (INV_4PI / r)

    return 0.0, g


@njit(parallel=True, fastmath=True)
def assemble_block(
    start_row,
    end_row,
    centroids,
    normals,
    tris,
    areas,
    q7,
    w7,
    q12,
    w12,
    is_dirichlet,
    u_known,
    q_known,
    A_block,
    b_block,
    diag_block,
):
    ne = centroids.shape[0]
    for lr in prange(end_row - start_row):
        i = start_row + lr
        xi = centroids[i, 0]
        yi = centroids[i, 1]
        zi = centroids[i, 2]

        bi = 0.0
        if is_dirichlet[i]:
            bi -= 0.5 * u_known[i]

        diag_val = 0.0

        for j in range(ne):
            if i == j:
                h, g = self_g_duffy(xi, yi, zi, tris[j])
            else:
                dx = xi - centroids[j, 0]
                dy = yi - centroids[j, 1]
                dz = zi - centroids[j, 2]
                d2 = dx * dx + dy * dy + dz * dz

                if d2 < NEAR_RATIO * areas[j]:
                    h, g = pair_integrals_from_rule(
                        xi,
                        yi,
                        zi,
                        normals[j, 0],
                        normals[j, 1],
                        normals[j, 2],
                        q12[j],
                        w12[j],
                    )
                else:
                    h, g = pair_integrals_from_rule(
                        xi,
                        yi,
                        zi,
                        normals[j, 0],
                        normals[j, 1],
                        normals[j, 2],
                        q7[j],
                        w7[j],
                    )

            if is_dirichlet[j]:
                coeff = -g
            else:
                coeff = h
                if i == j:
                    coeff += 0.5

            A_block[lr, j] = coeff
            if i == j:
                diag_val = coeff

            bi += -h * u_known[j] + g * q_known[j]

        b_block[lr] = bi
        diag_block[lr] = diag_val


# -----------------------------------------------------------------------------
# Interior evaluation
# -----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def evaluate_interior(points, u_all, q_all, centroids, normals, q7, w7):
    npnt = points.shape[0]
    ne = centroids.shape[0]
    out = np.empty(npnt, dtype=np.float64)

    for p in prange(npnt):
        x = points[p, 0]
        y = points[p, 1]
        z = points[p, 2]
        val = 0.0

        for j in range(ne):
            h, g = pair_integrals_from_rule(
                x,
                y,
                z,
                normals[j, 0],
                normals[j, 1],
                normals[j, 2],
                q7[j],
                w7[j],
            )
            val += g * q_all[j] - h * u_all[j]

        out[p] = val

    return out


def make_interior_points():
    g = np.linspace(-0.5, 0.5, 5)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


def warmup_numba():
    tris = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=np.float64,
    )
    areas = np.array([0.5], dtype=np.float64)
    normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    centroids = np.array([[1.0 / 3.0, 1.0 / 3.0, 0.0]], dtype=np.float64)

    q7 = np.empty((1, NQ7, 3), dtype=np.float64)
    w7 = np.empty((1, NQ7), dtype=np.float64)
    q12 = np.empty((1, NQ12, 3), dtype=np.float64)
    w12 = np.empty((1, NQ12), dtype=np.float64)

    precompute_dunavant(tris, areas, q7, w7)
    precompute_dense_duffy(tris, q12, w12)

    is_dir = np.array([True], dtype=np.bool_)
    u_known = np.array([1.0], dtype=np.float64)
    q_known = np.array([0.0], dtype=np.float64)

    A_block = np.empty((1, 1), dtype=np.float64)
    b_block = np.empty(1, dtype=np.float64)
    d_block = np.empty(1, dtype=np.float64)
    assemble_block(
        0,
        1,
        centroids,
        normals,
        tris,
        areas,
        q7,
        w7,
        q12,
        w12,
        is_dir,
        u_known,
        q_known,
        A_block,
        b_block,
        d_block,
    )

    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    u_all = np.array([1.0], dtype=np.float64)
    q_all = np.array([0.0], dtype=np.float64)
    _ = evaluate_interior(pts, u_all, q_all, centroids, normals, q7, w7)


def run_case(N):
    t0 = time.perf_counter()

    nodes, elems = generate_bumpy_mesh(N)
    areas, normals, centroids, tris = compute_geometry(nodes, elems)
    is_dirichlet, u_known, q_known, _, _ = assign_mixed_bcs(centroids, normals)

    ne = elems.shape[0]

    q7 = np.empty((ne, NQ7, 3), dtype=np.float64)
    w7 = np.empty((ne, NQ7), dtype=np.float64)
    q12 = np.empty((ne, NQ12, 3), dtype=np.float64)
    w12 = np.empty((ne, NQ12), dtype=np.float64)

    precompute_dunavant(tris, areas, q7, w7)
    precompute_dense_duffy(tris, q12, w12)

    setup_time = time.perf_counter() - t0

    fd, mm_path = tempfile.mkstemp(prefix=f"bem_bumpy_{N}_", suffix=".dat")
    os.close(fd)

    A_mm = None
    x_unknown = None
    info = 0

    try:
        A_mm = np.memmap(mm_path, dtype=np.float64, mode="w+", shape=(ne, ne))
        b = np.empty(ne, dtype=np.float64)
        diag = np.empty(ne, dtype=np.float64)

        block_rows = 8 if ne >= 4096 else 16
        for s in range(0, ne, block_rows):
            e = min(ne, s + block_rows)
            nrows = e - s
            A_block = np.empty((nrows, ne), dtype=np.float64)
            b_block = np.empty(nrows, dtype=np.float64)
            d_block = np.empty(nrows, dtype=np.float64)

            assemble_block(
                s,
                e,
                centroids,
                normals,
                tris,
                areas,
                q7,
                w7,
                q12,
                w12,
                is_dirichlet,
                u_known,
                q_known,
                A_block,
                b_block,
                d_block,
            )

            A_mm[s:e, :] = A_block
            b[s:e] = b_block
            diag[s:e] = d_block

        A_mm.flush()

        diag_safe = np.where(np.abs(diag) > 1.0e-14, diag, 1.0)
        Minv = 1.0 / diag_safe

        chunk_rows = 32

        def matvec(v):
            v = np.asarray(v, dtype=np.float64)
            y = np.empty(ne, dtype=np.float64)
            for s in range(0, ne, chunk_rows):
                e = min(ne, s + chunk_rows)
                y[s:e] = A_mm[s:e, :] @ v
            return y

        Aop = LinearOperator((ne, ne), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((ne, ne), matvec=lambda v: Minv * v, dtype=np.float64)

        t1 = time.perf_counter()
        x_unknown, info = gmres(
            Aop,
            b,
            M=Mop,
            restart=GMRES_RESTART,
            maxiter=GMRES_MAXITER,
            rtol=GMRES_RTOL,
            atol=0.0,
        )
        solve_time = time.perf_counter() - t1

        if info != 0:
            # Keep the best iterate returned by GMRES and continue.
            pass

        u_all = u_known.copy()
        q_all = q_known.copy()
        u_all[~is_dirichlet] = x_unknown[~is_dirichlet]
        q_all[is_dirichlet] = x_unknown[is_dirichlet]

        points = make_interior_points()
        t2 = time.perf_counter()
        u_num = evaluate_interior(points, u_all, q_all, centroids, normals, q7, w7)
        u_ex = exact_u(points[:, 0], points[:, 1], points[:, 2])
        eval_time = time.perf_counter() - t2

        total_time = time.perf_counter() - t0
        rel_err = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)

        return rel_err, setup_time, solve_time, eval_time, total_time, ne

    finally:
        try:
            if A_mm is not None:
                del A_mm
        except Exception:
            pass
        gc.collect()
        if os.path.exists(mm_path):
            try:
                os.remove(mm_path)
            except Exception:
                pass


def main():
    warmup_numba()

    Ns = [8, 16, 32]
    results = []

    print("N    | Ne      | Rel L2 Error   | Setup (s) | Solve (s) | Eval (s) | Total (s)")
    for N in Ns:
        rel_err, setup_t, solve_t, eval_t, total_t, ne = run_case(N)
        results.append((N, ne, rel_err))
        print(
            f"{N:<4d} | {ne:<7d} | {rel_err:.6e}   | "
            f"{setup_t:8.4f} | {solve_t:8.4f} | {eval_t:8.4f} | {total_t:8.4f}"
        )

    slope = np.log(results[0][2] / results[-1][2]) / np.log(results[-1][0] / results[0][0])

    print("Convergence Analysis:")
    print(f"Computed Slope: {slope:.4f}")
    print("Expected Slope: ~1.0000 (O(h) for constant elements)")


if __name__ == "__main__":
    main()