import numpy as np
from time import perf_counter
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, gmres

INV_4PI = 1.0 / (4.0 * np.pi)

# 7-point Dunavant rule on reference triangle (area = 1/2)
TRI_BARY = np.array(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.059715871789770, 0.470142064105115, 0.470142064105115],
        [0.470142064105115, 0.059715871789770, 0.470142064105115],
        [0.470142064105115, 0.470142064105115, 0.059715871789770],
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        [0.797426985353087, 0.101286507323456, 0.101286507323456],
    ],
    dtype=np.float64,
)

TRI_WREF = np.array(
    [
        0.112500000000000,
        0.066197076394253,
        0.066197076394253,
        0.066197076394253,
        0.062969590272414,
        0.062969590272414,
        0.062969590272414,
    ],
    dtype=np.float64,
)

# ------------------------------------------------------------
# Exact solution
# ------------------------------------------------------------
def u_exact_xyz(x, y, z):
    return x * x - y * y


# ------------------------------------------------------------
# Icosahedron / icosphere
# ------------------------------------------------------------
def orient_faces_outward(vertices, faces):
    faces = faces.copy()
    for i in range(faces.shape[0]):
        a, b, c = faces[i]
        v0 = vertices[a]
        v1 = vertices[b]
        v2 = vertices[c]
        n = np.cross(v1 - v0, v2 - v0)
        if np.dot(n, v0 + v1 + v2) < 0.0:
            faces[i, 1], faces[i, 2] = faces[i, 2], faces[i, 1]
    return faces


def create_icosahedron():
    t = (1.0 + np.sqrt(5.0)) / 2.0

    vertices = np.array(
        [
            [-1.0,  t,  0.0],
            [ 1.0,  t,  0.0],
            [-1.0, -t,  0.0],
            [ 1.0, -t,  0.0],
            [ 0.0, -1.0,  t],
            [ 0.0,  1.0,  t],
            [ 0.0, -1.0, -t],
            [ 0.0,  1.0, -t],
            [ t,  0.0, -1.0],
            [ t,  0.0,  1.0],
            [-t,  0.0, -1.0],
            [-t,  0.0,  1.0],
        ],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]

    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    faces = orient_faces_outward(vertices, faces)
    return vertices, faces


def subdivide(vertices, faces):
    verts = vertices.tolist()
    midpoint_cache = {}

    def midpoint(i, j):
        key = (i, j) if i < j else (j, i)
        if key in midpoint_cache:
            return midpoint_cache[key]
        vi = np.array(verts[i], dtype=np.float64)
        vj = np.array(verts[j], dtype=np.float64)
        m = 0.5 * (vi + vj)
        m /= np.linalg.norm(m)
        idx = len(verts)
        verts.append(m.tolist())
        midpoint_cache[key] = idx
        return idx

    new_faces = []
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])

        a = midpoint(i0, i1)
        b = midpoint(i1, i2)
        c = midpoint(i2, i0)

        new_faces.append([i0, a, c])
        new_faces.append([i1, b, a])
        new_faces.append([i2, c, b])
        new_faces.append([a, b, c])

    vertices = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(new_faces, dtype=np.int32)
    faces = orient_faces_outward(vertices, faces)
    return vertices, faces


def build_icosphere(subdivisions):
    vertices, faces = create_icosahedron()
    for _ in range(subdivisions):
        vertices, faces = subdivide(vertices, faces)
    faces = orient_faces_outward(vertices, faces)
    return vertices, faces


# ------------------------------------------------------------
# Geometry preprocessing
# ------------------------------------------------------------
def precompute_geometry(vertices, faces):
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)

    ne = faces.shape[0]
    centroids = np.empty((ne, 3), dtype=np.float64)
    areas = np.empty(ne, dtype=np.float64)
    normals = np.empty((ne, 3), dtype=np.float64)
    qpts = np.empty((ne, TRI_BARY.shape[0], 3), dtype=np.float64)
    qwts = np.empty((ne, TRI_BARY.shape[0]), dtype=np.float64)

    for e in range(ne):
        i0, i1, i2 = faces[e]
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]

        n_raw = np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(n_raw)
        normal = n_raw / (2.0 * area)

        centroid = (v0 + v1 + v2) / 3.0

        if np.dot(normal, centroid) < 0.0:
            faces[e, 1], faces[e, 2] = faces[e, 2], faces[e, 1]
            i0, i1, i2 = faces[e]
            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]
            n_raw = np.cross(v1 - v0, v2 - v0)
            area = 0.5 * np.linalg.norm(n_raw)
            normal = n_raw / (2.0 * area)
            centroid = (v0 + v1 + v2) / 3.0

        centroids[e] = centroid
        areas[e] = area
        normals[e] = normal

        for q in range(TRI_BARY.shape[0]):
            l0, l1, l2 = TRI_BARY[q]
            qpts[e, q, 0] = l0 * v0[0] + l1 * v1[0] + l2 * v2[0]
            qpts[e, q, 1] = l0 * v0[1] + l1 * v1[1] + l2 * v2[1]
            qpts[e, q, 2] = l0 * v0[2] + l1 * v1[2] + l2 * v2[2]
            qwts[e, q] = 2.0 * area * TRI_WREF[q]

    return faces, centroids, areas, normals, qpts, qwts


# ------------------------------------------------------------
# Matrix-free kernels
# ------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def compute_c_diagonal(nodes, faces, qpts, qwts, normals):
    n_nodes = nodes.shape[0]
    ne = faces.shape[0]
    nq = qwts.shape[1]
    cdiag = np.empty(n_nodes, dtype=np.float64)

    for i in prange(n_nodes):
        xi = nodes[i, 0]
        yi = nodes[i, 1]
        zi = nodes[i, 2]

        ksum = 0.0
        for e in range(ne):
            n0 = faces[e, 0]
            n1 = faces[e, 1]
            n2 = faces[e, 2]
            if i == n0 or i == n1 or i == n2:
                continue

            nx = normals[e, 0]
            ny = normals[e, 1]
            nz = normals[e, 2]

            for q in range(nq):
                dx = xi - qpts[e, q, 0]
                dy = yi - qpts[e, q, 1]
                dz = zi - qpts[e, q, 2]
                r2 = dx * dx + dy * dy + dz * dz
                r = np.sqrt(r2)
                ker = INV_4PI * (dx * nx + dy * ny + dz * nz) / (r2 * r)
                ksum += ker * qwts[e, q]

        cdiag[i] = -1.0 - ksum

    return cdiag


@njit(parallel=True, fastmath=True)
def apply_dlp_operator(mu, nodes, faces, qpts, qwts, normals, cdiag):
    n_nodes = nodes.shape[0]
    ne = faces.shape[0]
    nq = qwts.shape[1]
    out = np.empty(n_nodes, dtype=np.float64)

    for i in prange(n_nodes):
        xi = nodes[i, 0]
        yi = nodes[i, 1]
        zi = nodes[i, 2]

        s = cdiag[i] * mu[i]

        for e in range(ne):
            n0 = faces[e, 0]
            n1 = faces[e, 1]
            n2 = faces[e, 2]
            if i == n0 or i == n1 or i == n2:
                continue

            mu0 = mu[n0]
            mu1 = mu[n1]
            mu2 = mu[n2]

            nx = normals[e, 0]
            ny = normals[e, 1]
            nz = normals[e, 2]

            for q in range(nq):
                l0 = TRI_BARY[q, 0]
                l1 = TRI_BARY[q, 1]
                l2 = TRI_BARY[q, 2]

                muq = l0 * mu0 + l1 * mu1 + l2 * mu2

                dx = xi - qpts[e, q, 0]
                dy = yi - qpts[e, q, 1]
                dz = zi - qpts[e, q, 2]
                r2 = dx * dx + dy * dy + dz * dz
                r = np.sqrt(r2)
                ker = INV_4PI * (dx * nx + dy * ny + dz * nz) / (r2 * r)

                s += ker * qwts[e, q] * muq

        out[i] = s

    return out


@njit(parallel=True, fastmath=True)
def evaluate_potential(points, mu, faces, qpts, qwts, normals):
    npnts = points.shape[0]
    ne = faces.shape[0]
    nq = qwts.shape[1]
    out = np.empty(npnts, dtype=np.float64)

    for p in prange(npnts):
        xp = points[p, 0]
        yp = points[p, 1]
        zp = points[p, 2]

        s = 0.0

        for e in range(ne):
            n0 = faces[e, 0]
            n1 = faces[e, 1]
            n2 = faces[e, 2]

            mu0 = mu[n0]
            mu1 = mu[n1]
            mu2 = mu[n2]

            nx = normals[e, 0]
            ny = normals[e, 1]
            nz = normals[e, 2]

            for q in range(nq):
                l0 = TRI_BARY[q, 0]
                l1 = TRI_BARY[q, 1]
                l2 = TRI_BARY[q, 2]

                muq = l0 * mu0 + l1 * mu1 + l2 * mu2

                dx = xp - qpts[e, q, 0]
                dy = yp - qpts[e, q, 1]
                dz = zp - qpts[e, q, 2]
                r2 = dx * dx + dy * dy + dz * dz
                r = np.sqrt(r2)
                ker = INV_4PI * (dx * nx + dy * ny + dz * nz) / (r2 * r)

                s += ker * qwts[e, q] * muq

        out[p] = s

    return out


# ------------------------------------------------------------
# Interior evaluation grid
# ------------------------------------------------------------
def build_grid_points():
    xs = np.linspace(-0.8, 0.8, 30)
    ys = np.linspace(-0.8, 0.8, 30)
    zs = np.linspace(-0.8, 0.8, 30)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    r = np.linalg.norm(pts, axis=1)
    mask = (r > 0.1) & (r < 0.8)
    return pts[mask]


# ------------------------------------------------------------
# Warm-up for Numba compilation
# ------------------------------------------------------------
def warmup_numba():
    v, f = create_icosahedron()
    f, centroids, areas, normals, qpts, qwts = precompute_geometry(v, f)
    cdiag = compute_c_diagonal(v, f, qpts, qwts, normals)
    mu = np.ones(v.shape[0], dtype=np.float64)
    _ = apply_dlp_operator(mu, v, f, qpts, qwts, normals, cdiag)
    _ = evaluate_potential(v[:1], mu, f, qpts, qwts, normals)


# ------------------------------------------------------------
# Single case
# ------------------------------------------------------------
def run_case(subdivisions, grid_pts):
    t0 = perf_counter()

    nodes, faces = build_icosphere(subdivisions)
    faces, centroids, areas, normals, qpts, qwts = precompute_geometry(nodes, faces)
    cdiag = compute_c_diagonal(nodes, faces, qpts, qwts, normals)

    rhs = u_exact_xyz(nodes[:, 0], nodes[:, 1], nodes[:, 2]).astype(np.float64)
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    qpts = np.ascontiguousarray(qpts, dtype=np.float64)
    qwts = np.ascontiguousarray(qwts, dtype=np.float64)
    cdiag = np.ascontiguousarray(cdiag, dtype=np.float64)
    rhs = np.ascontiguousarray(rhs, dtype=np.float64)

    n_nodes = nodes.shape[0]
    n_elem = faces.shape[0]

    setup_time = perf_counter() - t0

    def mv(v):
        return apply_dlp_operator(v, nodes, faces, qpts, qwts, normals, cdiag)

    A = LinearOperator((n_nodes, n_nodes), matvec=mv, dtype=np.float64)

    iters = [0]

    def cb(_):
        iters[0] += 1

    t1 = perf_counter()
    mu, info = gmres(
        A,
        rhs,
        rtol=1e-8,
        atol=1e-8,
        restart=n_nodes,
        callback=cb,
        callback_type="pr_norm",
    )
    solve_time = perf_counter() - t1

    t2 = perf_counter()
    u_num = evaluate_potential(grid_pts, mu, faces, qpts, qwts, normals)
    u_ex = u_exact_xyz(grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2])
    rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)
    eval_time = perf_counter() - t2

    total_time = setup_time + solve_time + eval_time

    return {
        "N": n_nodes,
        "Ne": n_elem,
        "iters": iters[0],
        "rel_l2": rel_l2,
        "setup": setup_time,
        "solve": solve_time,
        "eval": eval_time,
        "total": total_time,
        "info": info,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    warmup_numba()
    grid_pts = build_grid_points()

    subdivisions_list = [3, 4, 5]
    results = []

    print(f"{'N':>8} {'Ne':>8} {'GMRES':>8} {'Rel L2 Error':>15} {'Setup':>10} {'Solve':>10} {'Eval':>10} {'Total':>10}")
    for s in subdivisions_list:
        res = run_case(s, grid_pts)
        results.append(res)
        print(
            f"{res['N']:8d} {res['Ne']:8d} {res['iters']:8d} {res['rel_l2']:15.6e} "
            f"{res['setup']:10.4f} {res['solve']:10.4f} {res['eval']:10.4f} {res['total']:10.4f}"
        )

    n_arr = np.array([r["N"] for r in results], dtype=np.float64)
    err_arr = np.array([r["rel_l2"] for r in results], dtype=np.float64)
    h = 1.0 / np.sqrt(n_arr)
    slope, _ = np.polyfit(np.log(h), np.log(err_arr), 1)

    print(f"order {slope:.6f}")


if __name__ == "__main__":
    main()