import numpy as np
from time import perf_counter
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, gmres

INV_4PI = 1.0 / (4.0 * np.pi)

# ------------------------------------------------------------
# Exact solution
# ------------------------------------------------------------
def u_exact(x, y, z):
    return x * x - y * y


# ------------------------------------------------------------
# Icosahedron / Icosphere
# ------------------------------------------------------------
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
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int32,
    )

    return vertices, faces


def orient_faces_outward(vertices, faces):
    faces = faces.copy()
    for e in range(faces.shape[0]):
        a, b, c = faces[e]
        v0, v1, v2 = vertices[a], vertices[b], vertices[c]
        n = np.cross(v1 - v0, v2 - v0)
        if np.dot(n, v0 + v1 + v2) < 0.0:
            faces[e, 1], faces[e, 2] = faces[e, 2], faces[e, 1]
    return faces


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
# Geometry
# ------------------------------------------------------------
def precompute_geometry(vertices, faces):
    elem_verts = np.ascontiguousarray(vertices[faces], dtype=np.float64)
    ne = faces.shape[0]
    areas = np.empty(ne, dtype=np.float64)
    normals = np.empty((ne, 3), dtype=np.float64)
    centroids = np.empty((ne, 3), dtype=np.float64)

    for e in range(ne):
        v0, v1, v2 = elem_verts[e, 0], elem_verts[e, 1], elem_verts[e, 2]
        n = np.cross(v1 - v0, v2 - v0)
        a = 0.5 * np.linalg.norm(n)
        areas[e] = a
        normals[e] = n / (2.0 * a)
        centroids[e] = (v0 + v1 + v2) / 3.0

    return elem_verts, areas, normals, centroids


def build_node_support(faces, n_nodes):
    counts = np.zeros(n_nodes, dtype=np.int32)
    ne = faces.shape[0]
    for e in range(ne):
        counts[faces[e, 0]] += 1
        counts[faces[e, 1]] += 1
        counts[faces[e, 2]] += 1

    ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    ptr[1:] = np.cumsum(counts)

    elem_ids = np.empty(3 * ne, dtype=np.int32)
    loc_ids = np.empty(3 * ne, dtype=np.int32)
    cursor = ptr.copy()

    for e in range(ne):
        for k in range(3):
            n = faces[e, k]
            idx = cursor[n]
            elem_ids[idx] = e
            loc_ids[idx] = k
            cursor[n] += 1

    return ptr, elem_ids, loc_ids


def build_element_adjacency(faces, node_ptr, node_elems):
    ne = faces.shape[0]
    adj = np.zeros((ne, ne), dtype=np.uint8)

    for n in range(node_ptr.shape[0] - 1):
        s = node_ptr[n]
        t = node_ptr[n + 1]
        elems = node_elems[s:t]
        m = elems.shape[0]
        for a in range(m):
            e1 = elems[a]
            for b in range(a + 1, m):
                e2 = elems[b]
                adj[e1, e2] = 1
                adj[e2, e1] = 1

    np.fill_diagonal(adj, 0)
    return adj


# ------------------------------------------------------------
# Triangle quadrature
# ------------------------------------------------------------
REF_BARY = np.array(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.059715871789770, 0.470142064105115, 0.470142064105115],
        [0.470142064105115, 0.059715871789770, 0.470142064105115],
        [0.470142064105115, 0.470142064105115, 0.059715871789770],
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        [0.797426507323456, 0.101286507323456, 0.101286507323456],
    ],
    dtype=np.float64,
)

REF_W = np.array(
    [
        0.1125,
        0.066197076394253,
        0.066197076394253,
        0.066197076394253,
        0.062969590272414,
        0.062969590272414,
        0.062969590272414,
    ],
    dtype=np.float64,
)


def refine_reference_triangle(level):
    tris = [
        np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    ]
    for _ in range(level):
        new_tris = []
        for T in tris:
            a, b, c = T[0], T[1], T[2]
            ab = 0.5 * (a + b)
            bc = 0.5 * (b + c)
            ca = 0.5 * (c + a)
            new_tris.append(np.array([a, ab, ca], dtype=np.float64))
            new_tris.append(np.array([ab, b, bc], dtype=np.float64))
            new_tris.append(np.array([ca, bc, c], dtype=np.float64))
            new_tris.append(np.array([ab, bc, ca], dtype=np.float64))
        tris = new_tris
    return np.asarray(tris, dtype=np.float64)


def precompute_quadrature_data(elem_verts, areas, level):
    subtris = refine_reference_triangle(level)
    nsub = subtris.shape[0]
    ne = elem_verts.shape[0]
    nq = nsub * REF_BARY.shape[0]

    qpts = np.empty((ne, nq, 3), dtype=np.float64)
    qwts = np.empty((ne, nq), dtype=np.float64)
    qphi = np.empty((ne, nq, 3), dtype=np.float64)

    for e in range(ne):
        v0, v1, v2 = elem_verts[e, 0], elem_verts[e, 1], elem_verts[e, 2]
        factor = (2.0 * areas[e]) / float(nsub)
        idx = 0
        for s in range(nsub):
            T = subtris[s]
            for q in range(REF_BARY.shape[0]):
                lam = (
                    REF_BARY[q, 0] * T[0]
                    + REF_BARY[q, 1] * T[1]
                    + REF_BARY[q, 2] * T[2]
                )
                qphi[e, idx] = lam
                qpts[e, idx] = lam[0] * v0 + lam[1] * v1 + lam[2] * v2
                qwts[e, idx] = REF_W[q] * factor
                idx += 1

    return qpts, qwts, qphi


# ------------------------------------------------------------
# Matrix-free pair integration for Galerkin double layer
# ------------------------------------------------------------
@njit(fastmath=True, inline='always')
def pair_integral_three(out, ex, ey, lt,
                        qpts_x, qwts_x, qphi_x,
                        qpts_y, qwts_y, qphi_y):
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 0.0

    qnx = qpts_x.shape[1]
    qny = qpts_y.shape[1]

    for ix in range(qnx):
        wx = qwts_x[ex, ix]
        phix = qphi_x[ex, ix, lt]
        x0 = qpts_x[ex, ix, 0]
        x1 = qpts_x[ex, ix, 1]
        x2 = qpts_x[ex, ix, 2]

        for iy in range(qny):
            dx = x0 - qpts_y[ey, iy, 0]
            dy = x1 - qpts_y[ey, iy, 1]
            dz = x2 - qpts_y[ey, iy, 2]
            r2 = dx * dx + dy * dy + dz * dz
            r = np.sqrt(r2)
            ker = INV_4PI * ((dx) * 0.0 + (dy) * 0.0 + (dz) * 0.0)  # placeholder

            # normal is constant on each flat element; the dot product is exact
            # we recover it from the element vertices via the geometry arrays
            # by passing the normals through qphi arrays is unnecessary, so
            # we compute it outside this kernel in a scalar way by a trick:
            # the unit normals are injected through the y-side qphi arrays
            # after precomputation, which is handled below in the assembly loops.

            # This function is kept for compatibility; the actual kernel is
            # evaluated in the assembly loops with the element normals provided.

    return


@njit(fastmath=True, inline='always')
def pair_integral_three_normals(out, ex, ey, lt,
                                qpts_x, qwts_x, qphi_x, nx_x, ny_x, nz_x,
                                qpts_y, qwts_y, qphi_y, nx_y, ny_y, nz_y):
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 0.0

    qnx = qpts_x.shape[1]
    qny = qpts_y.shape[1]

    for ix in range(qnx):
        wx = qwts_x[ex, ix]
        phix = qphi_x[ex, ix, lt]
        x0 = qpts_x[ex, ix, 0]
        x1 = qpts_x[ex, ix, 1]
        x2 = qpts_x[ex, ix, 2]

        for iy in range(qny):
            dy0 = x0 - qpts_y[ey, iy, 0]
            dy1 = x1 - qpts_y[ey, iy, 1]
            dy2 = x2 - qpts_y[ey, iy, 2]
            r2 = dy0 * dy0 + dy1 * dy1 + dy2 * dy2
            r = np.sqrt(r2)

            dotn = dy0 * nx_y[ey] + dy1 * ny_y[ey] + dy2 * nz_y[ey]
            ker = INV_4PI * dotn / (r2 * r)

            coeff = wx * qwts_y[ey, iy] * phix * ker
            out[0] += coeff * qphi_y[ey, iy, 0]
            out[1] += coeff * qphi_y[ey, iy, 1]
            out[2] += coeff * qphi_y[ey, iy, 2]

    return


# ------------------------------------------------------------
# System assembly
# ------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def assemble_system(nodes, faces, areas,
                    node_ptr, node_elems, node_locs,
                    adj,
                    qpts0, qwts0, qphi0,
                    qpts1, qwts1, qphi1,
                    nx, ny, nz):
    n_nodes = nodes.shape[0]
    ne = faces.shape[0]
    A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    b = np.zeros(n_nodes, dtype=np.float64)
    tmp = np.empty(3, dtype=np.float64)

    for i in prange(n_nodes):
        bi = 0.0

        for idx in range(node_ptr[i], node_ptr[i + 1]):
            ex = node_elems[idx]
            lt = node_locs[idx]
            area = areas[ex]

            # RHS: b_i = ∫ phi_i f ds
            for q in range(qpts0.shape[1]):
                x = qpts0[ex, q, 0]
                y = qpts0[ex, q, 1]
                z = qpts0[ex, q, 2]
                fval = x * x - y * y
                bi += qwts0[ex, q] * qphi0[ex, q, lt] * fval

            # Exact local mass matrix contribution
            for m in range(3):
                mval = area / 12.0 * (2.0 if m == lt else 1.0)
                A[i, faces[ex, m]] += -0.5 * mval

            # Double-layer Galerkin double integral
            for ey in range(ne):
                if ey == ex:
                    continue

                if adj[ex, ey] != 0:
                    pair_integral_three_normals(
                        tmp, ex, ey, lt,
                        qpts1, qwts1, qphi1, nx, ny, nz,
                        qpts1, qwts1, qphi1, nx, ny, nz
                    )
                else:
                    pair_integral_three_normals(
                        tmp, ex, ey, lt,
                        qpts0, qwts0, qphi0, nx, ny, nz,
                        qpts0, qwts0, qphi0, nx, ny, nz
                    )

                A[i, faces[ey, 0]] += tmp[0]
                A[i, faces[ey, 1]] += tmp[1]
                A[i, faces[ey, 2]] += tmp[2]

        b[i] = bi

    return A, b


# ------------------------------------------------------------
# Interior evaluation
# ------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def evaluate_potential(points, mu, faces, qpts0, qwts0, qphi0, nx, ny, nz):
    npnts = points.shape[0]
    ne = faces.shape[0]
    nq = qpts0.shape[1]
    out = np.empty(npnts, dtype=np.float64)

    for p in prange(npnts):
        x0 = points[p, 0]
        x1 = points[p, 1]
        x2 = points[p, 2]
        s = 0.0

        for e in range(ne):
            mu0 = mu[faces[e, 0]]
            mu1 = mu[faces[e, 1]]
            mu2 = mu[faces[e, 2]]

            for q in range(nq):
                lam0 = qphi0[e, q, 0]
                lam1 = qphi0[e, q, 1]
                lam2 = qphi0[e, q, 2]
                muq = lam0 * mu0 + lam1 * mu1 + lam2 * mu2

                dx = x0 - qpts0[e, q, 0]
                dy = x1 - qpts0[e, q, 1]
                dz = x2 - qpts0[e, q, 2]
                r2 = dx * dx + dy * dy + dz * dz
                r = np.sqrt(r2)
                ker = INV_4PI * ((dx * nx[e]) + (dy * ny[e]) + (dz * nz[e])) / (r2 * r)

                s += qwts0[e, q] * muq * ker

        out[p] = s

    return out


# ------------------------------------------------------------
# Grid and warmup
# ------------------------------------------------------------
def build_grid_points():
    xs = np.linspace(-0.8, 0.8, 30)
    ys = np.linspace(-0.8, 0.8, 30)
    zs = np.linspace(-0.8, 0.8, 30)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    r = np.linalg.norm(pts, axis=1)
    return pts[(r > 0.1) & (r < 0.8)]


def warmup_numba():
    V, F = create_icosahedron()
    F = orient_faces_outward(V, F)
    elem_verts, areas, normals, centroids = precompute_geometry(V, F)
    node_ptr, node_elems, node_locs = build_node_support(F, V.shape[0])
    adj = build_element_adjacency(F, node_ptr, node_elems)
    qpts0, qwts0, qphi0 = precompute_quadrature_data(elem_verts, areas, level=0)
    qpts1, qwts1, qphi1 = precompute_quadrature_data(elem_verts, areas, level=1)
    A, b = assemble_system(
        V, F, areas,
        node_ptr, node_elems, node_locs,
        adj,
        qpts0, qwts0, qphi0,
        qpts1, qwts1, qphi1,
        normals[:, 0], normals[:, 1], normals[:, 2]
    )
    _ = evaluate_potential(V[:2], np.ones(V.shape[0]), F, qpts0, qwts0, qphi0, normals[:, 0], normals[:, 1], normals[:, 2])
    return A, b


# ------------------------------------------------------------
# Solve one refinement level
# ------------------------------------------------------------
def run_case(subdivisions, grid_pts):
    t0 = perf_counter()

    vertices, faces = build_icosphere(subdivisions)
    elem_verts, areas, normals, centroids = precompute_geometry(vertices, faces)
    node_ptr, node_elems, node_locs = build_node_support(faces, vertices.shape[0])
    adj = build_element_adjacency(faces, node_ptr, node_elems)

    qpts0, qwts0, qphi0 = precompute_quadrature_data(elem_verts, areas, level=0)
    qpts1, qwts1, qphi1 = precompute_quadrature_data(elem_verts, areas, level=1)

    assembly_t0 = perf_counter()
    A, b = assemble_system(
        vertices, faces, areas,
        node_ptr, node_elems, node_locs,
        adj,
        qpts0, qwts0, qphi0,
        qpts1, qwts1, qphi1,
        normals[:, 0], normals[:, 1], normals[:, 2]
    )
    setup_assembly_time = perf_counter() - assembly_t0

    n = vertices.shape[0]
    Aop = LinearOperator((n, n), matvec=lambda x: A @ x, dtype=np.float64)

    iters = [0]

    def cb(_):
        iters[0] += 1

    solve_t0 = perf_counter()
    mu, info = gmres(
        Aop,
        b,
        rtol=1e-8,
        atol=1e-8,
        restart=n,
        callback=cb,
        callback_type="pr_norm",
    )
    solve_time = perf_counter() - solve_t0

    eval_t0 = perf_counter()
    u_num = evaluate_potential(grid_pts, mu, faces, qpts0, qwts0, qphi0, normals[:, 0], normals[:, 1], normals[:, 2])
    u_ex = u_exact(grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2])
    rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)
    eval_time = perf_counter() - eval_t0

    total_time = perf_counter() - t0

    return {
        "N": n,
        "Ne": faces.shape[0],
        "iters": iters[0],
        "rel_l2": rel_l2,
        "setup_asm": setup_assembly_time,
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
    levels = [3, 4, 5]
    results = []

    print(f"{'N':>6} {'Ne':>6} {'GMRES':>8} {'Rel L2 Error':>15} {'Setup+Asm':>11} {'Solve':>10} {'Eval':>10} {'Total':>10}")
    for lev in levels:
        res = run_case(lev, grid_pts)
        results.append(res)
        print(
            f"{res['N']:6d} {res['Ne']:6d} {res['iters']:8d} {res['rel_l2']:15.6e} "
            f"{res['setup_asm']:11.4f} {res['solve']:10.4f} {res['eval']:10.4f} {res['total']:10.4f}"
        )

    h = 1.0 / np.sqrt(np.array([r["N"] for r in results], dtype=np.float64))
    err = np.array([r["rel_l2"] for r in results], dtype=np.float64)
    slope, _ = np.polyfit(np.log(h), np.log(err), 1)
    print(f"{slope:.6f}")


if __name__ == "__main__":
    main()