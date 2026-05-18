import numpy as np
from numba import njit, prange
from time import perf_counter

PI = np.pi
INV4PI = 1.0 / (4.0 * PI)

# 7-point Dunavant rule on the reference triangle (weights sum to 1)
BARY_L1 = np.array([
    1.0 / 3.0,
    0.059715871789770,
    0.470142064105115,
    0.470142064105115,
    0.797426985353087,
    0.101286507323456,
    0.101286507323456,
], dtype=np.float64)

BARY_L2 = np.array([
    1.0 / 3.0,
    0.470142064105115,
    0.059715871789770,
    0.470142064105115,
    0.101286507323456,
    0.797426985353087,
    0.101286507323456,
], dtype=np.float64)

BARY_L3 = np.array([
    1.0 / 3.0,
    0.470142064105115,
    0.470142064105115,
    0.059715871789770,
    0.101286507323456,
    0.101286507323456,
    0.797426985353087,
], dtype=np.float64)

BARY_W = np.array([
    0.225000000000000,
    0.132394152788506,
    0.132394152788506,
    0.132394152788506,
    0.125939180544827,
    0.125939180544827,
    0.125939180544827,
], dtype=np.float64)

# Gauss-Legendre points on [0,1] for Duffy self-term integration
g_nodes, g_weights = np.polynomial.legendre.leggauss(6)
DUFFY_S = 0.5 * (g_nodes + 1.0)
DUFFY_W = 0.5 * g_weights


def u_exact(x, y, z):
    return x * y + y * z + z * x


def grad_u_exact(x, y, z):
    return np.array([y + z, x + z, x + y], dtype=np.float64)


def build_cube_mesh(N):
    coords = np.linspace(0.0, 1.0, N + 1, dtype=np.float64)

    nodes = []
    node_map = {}
    elements = []

    def add_node(pt):
        key = (float(pt[0]), float(pt[1]), float(pt[2]))
        idx = node_map.get(key)
        if idx is None:
            idx = len(nodes)
            node_map[key] = idx
            nodes.append(key)
        return idx

    def add_point(face, fixed_val, a, b):
        p = [0.0, 0.0, 0.0]
        p[face] = fixed_val
        other = [0, 1, 2]
        other.remove(face)
        p[other[0]] = a
        p[other[1]] = b
        return np.array(p, dtype=np.float64)

    def add_triangle(p0, p1, p2, expected_normal):
        n = np.cross(p1 - p0, p2 - p0)
        if np.dot(n, expected_normal) < 0.0:
            p1, p2 = p2, p1
        i0 = add_node(p0)
        i1 = add_node(p1)
        i2 = add_node(p2)
        elements.append((i0, i1, i2))

    faces = [
        (0, 0.0, np.array([-1.0, 0.0, 0.0], dtype=np.float64)),
        (0, 1.0, np.array([ 1.0, 0.0, 0.0], dtype=np.float64)),
        (1, 0.0, np.array([0.0, -1.0, 0.0], dtype=np.float64)),
        (1, 1.0, np.array([0.0,  1.0, 0.0], dtype=np.float64)),
        (2, 0.0, np.array([0.0, 0.0, -1.0], dtype=np.float64)),
        (2, 1.0, np.array([0.0, 0.0,  1.0], dtype=np.float64)),
    ]

    for face_axis, fixed_val, expected_normal in faces:
        for i in range(N):
            for j in range(N):
                a0 = coords[i]
                a1 = coords[i + 1]
                b0 = coords[j]
                b1 = coords[j + 1]

                p00 = add_point(face_axis, fixed_val, a0, b0)
                p10 = add_point(face_axis, fixed_val, a1, b0)
                p11 = add_point(face_axis, fixed_val, a1, b1)
                p01 = add_point(face_axis, fixed_val, a0, b1)

                add_triangle(p00, p10, p11, expected_normal)
                add_triangle(p00, p11, p01, expected_normal)

    nodes = np.array(nodes, dtype=np.float64)
    elements = np.array(elements, dtype=np.int64)

    tri_verts = nodes[elements]
    centroids = tri_verts.mean(axis=1)

    e1 = tri_verts[:, 1, :] - tri_verts[:, 0, :]
    e2 = tri_verts[:, 2, :] - tri_verts[:, 0, :]
    cr = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cr, axis=1)
    normals = cr / (2.0 * areas[:, None])

    tol = 1e-12
    elem_bcs = np.empty(elements.shape[0], dtype=np.int64)

    for i in range(elements.shape[0]):
        x, y, z = centroids[i]
        if abs(x - 0.0) < tol or abs(y - 0.0) < tol or abs(z - 0.0) < tol:
            elem_bcs[i] = 0  # Dirichlet
        else:
            elem_bcs[i] = 1  # Neumann

    return nodes, elements, tri_verts, centroids, areas, normals, elem_bcs


@njit(fastmath=True)
def pair_integrals(x0, x1, x2, tri, nx, ny, nz, area):
    v0x, v0y, v0z = tri[0, 0], tri[0, 1], tri[0, 2]
    v1x, v1y, v1z = tri[1, 0], tri[1, 1], tri[1, 2]
    v2x, v2y, v2z = tri[2, 0], tri[2, 1], tri[2, 2]

    h = 0.0
    g = 0.0

    for k in range(7):
        l1 = BARY_L1[k]
        l2 = BARY_L2[k]
        l3 = BARY_L3[k]
        w = BARY_W[k]

        yx = l1 * v0x + l2 * v1x + l3 * v2x
        yy = l1 * v0y + l2 * v1y + l3 * v2y
        yz = l1 * v0z + l2 * v1z + l3 * v2z

        rx = x0 - yx
        ry = x1 - yy
        rz = x2 - yz

        r2 = rx * rx + ry * ry + rz * rz
        r = np.sqrt(r2)
        invr = 1.0 / r
        invr3 = invr * invr * invr

        dot = rx * nx + ry * ny + rz * nz
        coeff = area * w * INV4PI

        g += coeff * invr
        h += coeff * dot * invr3

    return h, g


@njit(fastmath=True)
def self_G_duffy(x0, x1, x2, tri):
    v0x, v0y, v0z = tri[0, 0], tri[0, 1], tri[0, 2]
    v1x, v1y, v1z = tri[1, 0], tri[1, 1], tri[1, 2]
    v2x, v2y, v2z = tri[2, 0], tri[2, 1], tri[2, 2]

    g = 0.0

    for sidx in range(3):
        if sidx == 0:
            bx1, by1, bz1 = v0x - x0, v0y - x1, v0z - x2
            bx2, by2, bz2 = v1x - x0, v1y - x1, v1z - x2
        elif sidx == 1:
            bx1, by1, bz1 = v1x - x0, v1y - x1, v1z - x2
            bx2, by2, bz2 = v2x - x0, v2y - x1, v2z - x2
        else:
            bx1, by1, bz1 = v2x - x0, v2y - x1, v2z - x2
            bx2, by2, bz2 = v0x - x0, v0y - x1, v0z - x2

        cx = by1 * bz2 - bz1 * by2
        cy = bz1 * bx2 - bx1 * bz2
        cz = bx1 * by2 - by1 * bx2
        crossmag = np.sqrt(cx * cx + cy * cy + cz * cz)

        for i in range(DUFFY_S.shape[0]):
            s = DUFFY_S[i]
            ws = DUFFY_W[i]
            for j in range(DUFFY_S.shape[0]):
                t = DUFFY_S[j]
                wt = DUFFY_W[j]

                lx = (1.0 - t) * bx1 + t * bx2
                ly = (1.0 - t) * by1 + t * by2
                lz = (1.0 - t) * bz1 + t * bz2

                yx = x0 + s * lx
                yy = x1 + s * ly
                yz = x2 + s * lz

                rx = x0 - yx
                ry = x1 - yy
                rz = x2 - yz

                r = np.sqrt(rx * rx + ry * ry + rz * rz)
                jac = s * crossmag

                g += ws * wt * jac * INV4PI / r

    return g


@njit(parallel=True, fastmath=True)
def assemble_HG(centroids, tri_verts, tri_normals, tri_areas):
    ne = tri_verts.shape[0]
    H = np.zeros((ne, ne), dtype=np.float64)
    G = np.zeros((ne, ne), dtype=np.float64)

    for i in prange(ne):
        x0, x1, x2 = centroids[i, 0], centroids[i, 1], centroids[i, 2]
        row_sum = 0.0

        for j in range(ne):
            if i == j:
                G[i, j] = self_G_duffy(x0, x1, x2, tri_verts[j])
                H[i, j] = 0.0
            else:
                h, g = pair_integrals(
                    x0, x1, x2,
                    tri_verts[j],
                    tri_normals[j, 0], tri_normals[j, 1], tri_normals[j, 2],
                    tri_areas[j]
                )
                H[i, j] = h
                G[i, j] = g
                row_sum += h

        H[i, i] = -row_sum

    return H, G


@njit(parallel=True, fastmath=True)
def evaluate_interior(grid_pts, tri_verts, tri_normals, tri_areas, u_b, q_b):
    m = grid_pts.shape[0]
    ne = tri_verts.shape[0]
    vals = np.zeros(m, dtype=np.float64)

    for p in prange(m):
        x0, x1, x2 = grid_pts[p, 0], grid_pts[p, 1], grid_pts[p, 2]
        s = 0.0

        for j in range(ne):
            h, g = pair_integrals(
                x0, x1, x2,
                tri_verts[j],
                tri_normals[j, 0], tri_normals[j, 1], tri_normals[j, 2],
                tri_areas[j]
            )
            s += g * q_b[j] - h * u_b[j]

        vals[p] = s

    return vals

def warmup_numba():
    tri_verts = np.array([[[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]]], dtype=np.float64)
    centroids = tri_verts.mean(axis=1)
    tri_areas = np.array([0.5], dtype=np.float64)
    tri_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    _ = assemble_HG(centroids, tri_verts, tri_normals, tri_areas)

    grid = np.array([[0.25, 0.25, 0.25]], dtype=np.float64)
    u_b = np.array([0.0], dtype=np.float64)
    q_b = np.array([0.0], dtype=np.float64)
    _ = evaluate_interior(grid, tri_verts, tri_normals, tri_areas, u_b, q_b)


def main():
    warmup_numba()

    levels = [8, 16, 32]
    results = []

    grid_lin = np.linspace(0.2, 0.8, 5, dtype=np.float64)
    Xg, Yg, Zg = np.meshgrid(grid_lin, grid_lin, grid_lin, indexing="ij")
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

    for N in levels:
        t0 = perf_counter()

        nodes, elements, tri_verts, centroids, areas, normals, elem_bcs = build_cube_mesh(N)

        exact_u_cent = u_exact(centroids[:, 0], centroids[:, 1], centroids[:, 2])
        grad_cent = np.column_stack([
            centroids[:, 1] + centroids[:, 2],
            centroids[:, 0] + centroids[:, 2],
            centroids[:, 0] + centroids[:, 1],
        ])
        exact_q_cent = np.einsum("ij,ij->i", grad_cent, normals)

        dirichlet = (elem_bcs == 0)
        neumann = ~dirichlet

        u_known = np.zeros(elements.shape[0], dtype=np.float64)
        q_known = np.zeros(elements.shape[0], dtype=np.float64)

        u_known[dirichlet] = exact_u_cent[dirichlet]
        q_known[neumann] = exact_q_cent[neumann]

        H, G = assemble_HG(centroids, tri_verts, normals, areas)

        A = H.copy()
        A[:, dirichlet] = -G[:, dirichlet]
        b = G @ q_known - H @ u_known

        setup_time = perf_counter() - t0

        t1 = perf_counter()
        x = np.linalg.solve(A, b)
        solve_time = perf_counter() - t1

        u_b = u_known.copy()
        q_b = q_known.copy()
        u_b[neumann] = x[neumann]
        q_b[dirichlet] = x[dirichlet]

        t2 = perf_counter()
        u_num = evaluate_interior(grid_pts, tri_verts, normals, areas, u_b, q_b)
        eval_time = perf_counter() - t2

        u_ex = u_exact(grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2])
        rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)

        total_time = perf_counter() - t0
        ne = elements.shape[0]
        h = np.sqrt(1.0 / ne)

        results.append((N, ne, h, rel_l2, setup_time, solve_time, eval_time, total_time))

        del H, G, A, b, x

    hs = np.array([r[2] for r in results], dtype=np.float64)
    errs = np.array([r[3] for r in results], dtype=np.float64)
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]

    print(f"{'N':>4} {'Ne':>8} {'RelL2':>14} {'Setup(s)':>12} {'Solve(s)':>12} {'Eval(s)':>12} {'Total(s)':>12}")
    for N, ne, h, rel_l2, setup_time, solve_time, eval_time, total_time in results:
        print(f"{N:4d} {ne:8d} {rel_l2:14.6e} {setup_time:12.4f} {solve_time:12.4f} {eval_time:12.4f} {total_time:12.4f}")
    print(f"{slope:.6f}")


if __name__ == "__main__":
    main()