import numpy as np
import time
from numba import njit, prange
from numpy.polynomial.legendre import leggauss

PI4 = 4.0 * np.pi
EPS = 1e-14

# 7-point Dunavant rule on the reference triangle
TRI_LAM = np.array([
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [0.059715871789770, 0.470142064105115, 0.470142064105115],
    [0.470142064105115, 0.059715871789770, 0.470142064105115],
    [0.470142064105115, 0.470142064105115, 0.059715871789770],
    [0.797426985353087, 0.101286507323456, 0.101286507323456],
    [0.101286507323456, 0.797426985353087, 0.101286507323456],
    [0.101286507323456, 0.101286507323456, 0.797426985353087],
], dtype=np.float64)

TRI_W = np.array([
    0.225000000000000,
    0.132394152788506,
    0.132394152788506,
    0.132394152788506,
    0.125939180544827,
    0.125939180544827,
    0.125939180544827,
], dtype=np.float64) * 0.5

# Tensor-product Gauss-Legendre for Duffy transform
GL_X, GL_W = leggauss(8)
DX = 0.5 * (GL_X + 1.0)
DW = 0.5 * GL_W


def build_cube_mesh(N):
    node_map = {}
    nodes = []
    elements = []

    def get_node(ix, iy, iz):
        key = (ix, iy, iz)
        idx = node_map.get(key)
        if idx is None:
            idx = len(nodes)
            node_map[key] = idx
            nodes.append((ix / N, iy / N, iz / N))
        return idx

    def add_tri(a, b, c, expected_n):
        v0 = np.array(nodes[a], dtype=np.float64)
        v1 = np.array(nodes[b], dtype=np.float64)
        v2 = np.array(nodes[c], dtype=np.float64)
        cr = np.cross(v1 - v0, v2 - v0)
        if np.dot(cr, expected_n) < 0.0:
            elements.append((a, c, b))
        else:
            elements.append((a, b, c))

    face_specs = [
        ((-1.0, 0.0, 0.0), lambda i, j: (0, i, j),     lambda i, j: (0, i + 1, j),
         lambda i, j: (0, i + 1, j + 1), lambda i, j: (0, i, j + 1)),
        ((1.0, 0.0, 0.0),  lambda i, j: (N, i, j),     lambda i, j: (N, i + 1, j),
         lambda i, j: (N, i + 1, j + 1), lambda i, j: (N, i, j + 1)),
        ((0.0, -1.0, 0.0), lambda i, j: (i, 0, j),     lambda i, j: (i + 1, 0, j),
         lambda i, j: (i + 1, 0, j + 1), lambda i, j: (i, 0, j + 1)),
        ((0.0, 1.0, 0.0),  lambda i, j: (i, N, j),     lambda i, j: (i + 1, N, j),
         lambda i, j: (i + 1, N, j + 1), lambda i, j: (i, N, j + 1)),
        ((0.0, 0.0, -1.0), lambda i, j: (i, j, 0),     lambda i, j: (i + 1, j, 0),
         lambda i, j: (i + 1, j + 1, 0), lambda i, j: (i, j + 1, 0)),
        ((0.0, 0.0, 1.0),  lambda i, j: (i, j, N),     lambda i, j: (i + 1, j, N),
         lambda i, j: (i + 1, j + 1, N), lambda i, j: (i, j + 1, N)),
    ]

    for expected_n, p00f, p10f, p11f, p01f in face_specs:
        expected_n = np.array(expected_n, dtype=np.float64)
        for i in range(N):
            for j in range(N):
                p00 = get_node(*p00f(i, j))
                p10 = get_node(*p10f(i, j))
                p11 = get_node(*p11f(i, j))
                p01 = get_node(*p01f(i, j))
                add_tri(p00, p10, p11, expected_n)
                add_tri(p00, p11, p01, expected_n)

    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int32)

    normals = np.zeros((elements.shape[0], 3), dtype=np.float64)
    areas = np.zeros(elements.shape[0], dtype=np.float64)

    for e, tri in enumerate(elements):
        v0 = nodes[tri[0]]
        v1 = nodes[tri[1]]
        v2 = nodes[tri[2]]
        cr = np.cross(v1 - v0, v2 - v0)
        nrm = np.linalg.norm(cr)
        normals[e] = cr / nrm
        areas[e] = 0.5 * nrm

    return nodes, elements, normals, areas


def classify_bc(nodes):
    V = nodes.shape[0]
    bc_type = np.zeros(V, dtype=np.int32)  # 0: Dirichlet, 1: Neumann
    u_known = np.zeros(V, dtype=np.float64)
    q_known = np.zeros(V, dtype=np.float64)

    for i, (x, y, z) in enumerate(nodes):
        uex = x * y + y * z + z * x
        if abs(x - 0.0) < EPS or abs(y - 0.0) < EPS or abs(z - 0.0) < EPS:
            bc_type[i] = 0
            u_known[i] = uex
        else:
            bc_type[i] = 1
            if abs(x - 1.0) < EPS:
                q_known[i] = y + z
            elif abs(y - 1.0) < EPS:
                q_known[i] = x + z
            else:
                q_known[i] = x + y

    return bc_type, u_known, q_known


@njit(fastmath=True)
def _add_regular_contrib(x, v0, v1, v2, n, area, tri0, tri1, tri2, bc_type, u_known, q_known, row, rhs):
    h0 = 0.0
    h1 = 0.0
    h2 = 0.0
    g0 = 0.0
    g1 = 0.0
    g2 = 0.0

    for q in range(TRI_W.shape[0]):
        l0 = TRI_LAM[q, 0]
        l1 = TRI_LAM[q, 1]
        l2 = TRI_LAM[q, 2]

        yx = l0 * v0[0] + l1 * v1[0] + l2 * v2[0]
        yy = l0 * v0[1] + l1 * v1[1] + l2 * v2[1]
        yz = l0 * v0[2] + l1 * v1[2] + l2 * v2[2]

        rx = x[0] - yx
        ry = x[1] - yy
        rz = x[2] - yz

        r2 = rx * rx + ry * ry + rz * rz
        r = np.sqrt(r2)

        invr = 1.0 / r
        jac = 2.0 * area
        w = TRI_W[q] * jac

        gker = w * (invr / PI4)
        dot = rx * n[0] + ry * n[1] + rz * n[2]
        hker = w * (dot / (PI4 * r2 * r))

        g0 += gker * l0
        g1 += gker * l1
        g2 += gker * l2

        h0 += hker * l0
        h1 += hker * l1
        h2 += hker * l2

    j = tri0
    if bc_type[j] == 0:
        rhs[0] -= h0 * u_known[j]
        row[j] -= g0
    else:
        row[j] += h0
        rhs[0] += g0 * q_known[j]

    j = tri1
    if bc_type[j] == 0:
        rhs[0] -= h1 * u_known[j]
        row[j] -= g1
    else:
        row[j] += h1
        rhs[0] += g1 * q_known[j]

    j = tri2
    if bc_type[j] == 0:
        rhs[0] -= h2 * u_known[j]
        row[j] -= g2
    else:
        row[j] += h2
        rhs[0] += g2 * q_known[j]


@njit(fastmath=True)
def _add_singular_contrib(x, v0, v1, v2, n, area, tri0, tri1, tri2, sidx, bc_type, u_known, q_known, row, rhs):
    h0 = 0.0
    h1 = 0.0
    h2 = 0.0
    g0 = 0.0
    g1 = 0.0
    g2 = 0.0

    for a in range(DX.shape[0]):
        xi = DX[a]
        wx = DW[a]
        for b in range(DX.shape[0]):
            eta = DX[b]
            wy = DW[b]

            w = wx * wy * xi
            jac = 2.0 * area

            if sidx == 0:
                yx = v0[0] + xi * (v1[0] - v0[0]) + xi * eta * (v2[0] - v0[0])
                yy = v0[1] + xi * (v1[1] - v0[1]) + xi * eta * (v2[1] - v0[1])
                yz = v0[2] + xi * (v1[2] - v0[2]) + xi * eta * (v2[2] - v0[2])
                l0 = 1.0 - xi
                l1 = xi * (1.0 - eta)
                l2 = xi * eta
            elif sidx == 1:
                yx = v1[0] + xi * (v0[0] - v1[0]) + xi * eta * (v2[0] - v1[0])
                yy = v1[1] + xi * (v0[1] - v1[1]) + xi * eta * (v2[1] - v1[1])
                yz = v1[2] + xi * (v0[2] - v1[2]) + xi * eta * (v2[2] - v1[2])
                l1 = 1.0 - xi
                l0 = xi * (1.0 - eta)
                l2 = xi * eta
            else:
                yx = v2[0] + xi * (v0[0] - v2[0]) + xi * eta * (v1[0] - v2[0])
                yy = v2[1] + xi * (v0[1] - v2[1]) + xi * eta * (v1[1] - v2[1])
                yz = v2[2] + xi * (v0[2] - v2[2]) + xi * eta * (v1[2] - v2[2])
                l2 = 1.0 - xi
                l0 = xi * (1.0 - eta)
                l1 = xi * eta

            rx = x[0] - yx
            ry = x[1] - yy
            rz = x[2] - yz

            r2 = rx * rx + ry * ry + rz * rz
            r = np.sqrt(r2)

            invr = 1.0 / r
            gw = w * jac
            gker = gw * (invr / PI4)
            dot = rx * n[0] + ry * n[1] + rz * n[2]
            hker = gw * (dot / (PI4 * r2 * r))

            g0 += gker * l0
            g1 += gker * l1
            g2 += gker * l2

            h0 += hker * l0
            h1 += hker * l1
            h2 += hker * l2

    j = tri0
    if bc_type[j] == 0:
        rhs[0] -= h0 * u_known[j]
        row[j] -= g0
    else:
        row[j] += h0
        rhs[0] += g0 * q_known[j]

    j = tri1
    if bc_type[j] == 0:
        rhs[0] -= h1 * u_known[j]
        row[j] -= g1
    else:
        row[j] += h1
        rhs[0] += g1 * q_known[j]

    j = tri2
    if bc_type[j] == 0:
        rhs[0] -= h2 * u_known[j]
        row[j] -= g2
    else:
        row[j] += h2
        rhs[0] += g2 * q_known[j]


@njit(parallel=True, fastmath=True)
def assemble_system(nodes, elements, normals, areas, bc_type, u_known, q_known):
    V = nodes.shape[0]
    Ne = elements.shape[0]
    A = np.zeros((V, V), dtype=np.float64)
    b = np.zeros(V, dtype=np.float64)

    c_i = 0.5  # as requested

    for i in prange(V):
        x = nodes[i]
        row = np.zeros(V, dtype=np.float64)
        rhs = np.zeros(1, dtype=np.float64)

        for e in range(Ne):
            tri0 = elements[e, 0]
            tri1 = elements[e, 1]
            tri2 = elements[e, 2]

            v0 = nodes[tri0]
            v1 = nodes[tri1]
            v2 = nodes[tri2]

            if tri0 == i:
                _add_singular_contrib(x, v0, v1, v2, normals[e], areas[e],
                                      tri0, tri1, tri2, 0, bc_type, u_known, q_known, row, rhs)
            elif tri1 == i:
                _add_singular_contrib(x, v0, v1, v2, normals[e], areas[e],
                                      tri0, tri1, tri2, 1, bc_type, u_known, q_known, row, rhs)
            elif tri2 == i:
                _add_singular_contrib(x, v0, v1, v2, normals[e], areas[e],
                                      tri0, tri1, tri2, 2, bc_type, u_known, q_known, row, rhs)
            else:
                _add_regular_contrib(x, v0, v1, v2, normals[e], areas[e],
                                     tri0, tri1, tri2, bc_type, u_known, q_known, row, rhs)

        if bc_type[i] == 0:
            rhs[0] -= c_i * u_known[i]
        else:
            row[i] += c_i

        A[i, :] = row
        b[i] = rhs[0]

    return A, b


@njit(fastmath=True)
def eval_interior_point(x, nodes, elements, normals, areas, u_full, q_full):
    val = 0.0
    for e in range(elements.shape[0]):
        tri0 = elements[e, 0]
        tri1 = elements[e, 1]
        tri2 = elements[e, 2]

        v0 = nodes[tri0]
        v1 = nodes[tri1]
        v2 = nodes[tri2]
        n = normals[e]
        area = areas[e]

        for q in range(TRI_W.shape[0]):
            l0 = TRI_LAM[q, 0]
            l1 = TRI_LAM[q, 1]
            l2 = TRI_LAM[q, 2]

            yx = l0 * v0[0] + l1 * v1[0] + l2 * v2[0]
            yy = l0 * v0[1] + l1 * v1[1] + l2 * v2[1]
            yz = l0 * v0[2] + l1 * v1[2] + l2 * v2[2]

            rx = x[0] - yx
            ry = x[1] - yy
            rz = x[2] - yz

            r2 = rx * rx + ry * ry + rz * rz
            r = np.sqrt(r2)

            jac = 2.0 * area
            w = TRI_W[q] * jac

            gker = w * (1.0 / (PI4 * r))
            dot = rx * n[0] + ry * n[1] + rz * n[2]
            hker = w * (dot / (PI4 * r2 * r))

            u_loc = l0 * u_full[tri0] + l1 * u_full[tri1] + l2 * u_full[tri2]
            q_loc = l0 * q_full[tri0] + l1 * q_full[tri1] + l2 * q_full[tri2]

            val += gker * q_loc - hker * u_loc

    return val


@njit(parallel=True, fastmath=True)
def eval_interior(points, nodes, elements, normals, areas, u_full, q_full):
    m = points.shape[0]
    out = np.zeros(m, dtype=np.float64)
    for i in prange(m):
        out[i] = eval_interior_point(points[i], nodes, elements, normals, areas, u_full, q_full)
    return out


def recover_boundary_solution(nodes, bc_type, u_known, q_known, x_unknown):
    V = nodes.shape[0]
    u_full = np.zeros(V, dtype=np.float64)
    q_full = np.zeros(V, dtype=np.float64)

    for i in range(V):
        if bc_type[i] == 0:
            u_full[i] = u_known[i]
            q_full[i] = x_unknown[i]
        else:
            u_full[i] = x_unknown[i]
            q_full[i] = q_known[i]

    return u_full, q_full


def build_interior_grid(m=5):
    xs = np.linspace(0.2, 0.8, m)
    pts = np.array([(x, y, z) for x in xs for y in xs for z in xs], dtype=np.float64)
    return pts


def run_case(N):
    nodes, elements, normals, areas = build_cube_mesh(N)
    bc_type, u_known, q_known = classify_bc(nodes)

    t0 = time.perf_counter()
    A, b = assemble_system(nodes, elements, normals, areas, bc_type, u_known, q_known)
    setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    x = np.linalg.solve(A, b)
    solve = time.perf_counter() - t1

    u_full, q_full = recover_boundary_solution(nodes, bc_type, u_known, q_known, x)

    grid_pts = build_interior_grid(5)
    t2 = time.perf_counter()
    u_num = eval_interior(grid_pts, nodes, elements, normals, areas, u_full, q_full)
    eval_t = time.perf_counter() - t2

    u_ex = np.array([p[0] * p[1] + p[1] * p[2] + p[2] * p[0] for p in grid_pts], dtype=np.float64)
    rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)

    return {
        "N": N,
        "Ne": elements.shape[0],
        "V": nodes.shape[0],
        "rel_l2": rel_l2,
        "setup": setup,
        "solve": solve,
        "eval": eval_t,
        "total": setup + solve + eval_t,
    }


def main():
    print("N   | Ne     | V     | Rel L2 Error   | Setup (s) | Solve (s) | Eval (s) | Total (s) | Rate")
    print("-" * 96)

    prev_err = None
    prev_h = None

    for N in (8, 16, 32):
        res = run_case(N)
        h = np.sqrt(1.0 / res["Ne"])
        rate = ""
        if prev_err is not None:
            rate_val = np.log(prev_err / res["rel_l2"]) / np.log(prev_h / h)
            rate = f"{rate_val:5.2f}"
        print(f"{res['N']:>2d} | {res['Ne']:<6d} | {res['V']:<5d} | {res['rel_l2']:>13.6e} | {res['setup']:>8.3f} | {res['solve']:>8.3f} | {res['eval']:>8.3f} | {res['total']:>8.3f} | {rate:>4}")

        prev_err = res["rel_l2"]
        prev_h = h


if __name__ == "__main__":
    main()