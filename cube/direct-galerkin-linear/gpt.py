import math
import time
import numpy as np
from numba import njit, prange

PI4 = 4.0 * math.pi

TRI_P = np.array([
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
], dtype=np.float64)
TRI_W = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)
GX = np.array([0.1127016653792583, 0.5, 0.8872983346207417], dtype=np.float64)
GW = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0], dtype=np.float64)


# ------------------------------------------------------------
# Mesh
# ------------------------------------------------------------
def _add_node(node_map, nodes, x, y, z):
    key = (round(float(x), 6), round(float(y), 6), round(float(z), 6))
    if key in node_map:
        return node_map[key]
    idx = len(nodes)
    node_map[key] = idx
    nodes.append((float(x), float(y), float(z)))
    return idx


def generate_mesh(N):
    """Face-local linear triangular mesh on the 6 cube faces.

    Nodes are duplicated across faces. This avoids BC conflicts at edges/corners
    for the mixed Dirichlet/Neumann cube data.
    """
    geom_node_map = {}
    geom_nodes = []  # unique geometry tracker only

    nodes = []
    elems = []
    node_face = []  # 0=Dirichlet faces, 1=Neumann faces

    faces = [
        ("x", 0.0, np.array([-1.0, 0.0, 0.0]), (1, 2), 0),
        ("x", 1.0, np.array([1.0, 0.0, 0.0]), (1, 2), 1),
        ("y", 0.0, np.array([0.0, -1.0, 0.0]), (0, 2), 0),
        ("y", 1.0, np.array([0.0, 1.0, 0.0]), (0, 2), 1),
        ("z", 0.0, np.array([0.0, 0.0, -1.0]), (0, 1), 0),
        ("z", 1.0, np.array([0.0, 0.0, 1.0]), (0, 1), 1),
    ]

    axis_idx = {"x": 0, "y": 1, "z": 2}

    for fixed_axis, fixed_val, outward_normal, uv_axes, ftype in faces:
        grid = np.empty((N + 1, N + 1), dtype=np.int64)
        for i in range(N + 1):
            u = i / N
            for j in range(N + 1):
                v = j / N
                coord = [0.0, 0.0, 0.0]
                coord[axis_idx[fixed_axis]] = fixed_val
                coord[uv_axes[0]] = u
                coord[uv_axes[1]] = v

                # geometry tracker (unique coordinates)
                _add_node(geom_node_map, geom_nodes, coord[0], coord[1], coord[2])

                grid[i, j] = len(nodes)
                nodes.append(coord)
                node_face.append(ftype)

        for i in range(N):
            for j in range(N):
                n00 = int(grid[i, j])
                n10 = int(grid[i + 1, j])
                n01 = int(grid[i, j + 1])
                n11 = int(grid[i + 1, j + 1])

                tris = [(n00, n10, n11), (n00, n11, n01)]
                for a, b, c in tris:
                    v0 = np.array(nodes[a], dtype=np.float64)
                    v1 = np.array(nodes[b], dtype=np.float64)
                    v2 = np.array(nodes[c], dtype=np.float64)
                    cr = np.cross(v1 - v0, v2 - v0)
                    if np.dot(cr, outward_normal) < 0.0:
                        elems.append((a, c, b))
                    else:
                        elems.append((a, b, c))

    return (
        np.asarray(nodes, dtype=np.float64),
        np.asarray(elems, dtype=np.int64),
        np.asarray(node_face, dtype=np.int64),
    )


# ------------------------------------------------------------
# Geometry / BCs
# ------------------------------------------------------------
def compute_geometry(nodes, elems):
    v0 = nodes[elems[:, 0]]
    v1 = nodes[elems[:, 1]]
    v2 = nodes[elems[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cr, axis=1)
    areas = 0.5 * norms
    normals = cr / norms[:, None]
    return areas, normals


def build_node_elems(nodes, elems):
    conn = [[] for _ in range(nodes.shape[0])]
    for e, tri in enumerate(elems):
        conn[int(tri[0])].append(e)
        conn[int(tri[1])].append(e)
        conn[int(tri[2])].append(e)
    return [np.asarray(v, dtype=np.int64) for v in conn]


def assign_bcs(nodes, node_face):
    nn = nodes.shape[0]
    bc_type = np.empty(nn, dtype=np.int64)
    u_exact = np.empty(nn, dtype=np.float64)
    q_exact = np.empty(nn, dtype=np.float64)

    for i in range(nn):
        x, y, z = nodes[i]
        u_exact[i] = x * y + y * z + z * x
        if node_face[i] == 0:
            bc_type[i] = 0
            # Dirichlet faces: x=0, y=0, z=0
            if abs(x - 0.0) < 1e-12:
                q_exact[i] = -(y + z)
            elif abs(y - 0.0) < 1e-12:
                q_exact[i] = -(x + z)
            else:
                q_exact[i] = -(x + y)
        else:
            bc_type[i] = 1
            # Neumann faces: x=1, y=1, z=1
            if abs(x - 1.0) < 1e-12:
                q_exact[i] = y + z
            elif abs(y - 1.0) < 1e-12:
                q_exact[i] = x + z
            else:
                q_exact[i] = x + y

    return bc_type, u_exact, q_exact


# ------------------------------------------------------------
# Quadrature precompute
# ------------------------------------------------------------
def precompute_regular_quadrature(nodes, elems, areas):
    v0 = nodes[elems[:, 0]]
    v1 = nodes[elems[:, 1]]
    v2 = nodes[elems[:, 2]]
    ne = elems.shape[0]
    quad_xyz = np.empty((ne, 3, 3), dtype=np.float64)
    for q in range(3):
        l1, l2, l3 = TRI_P[q]
        quad_xyz[:, q, :] = l1 * v0 + l2 * v1 + l3 * v2
    quad_wJ = 2.0 * areas[:, None] * TRI_W[None, :]
    return quad_xyz, quad_wJ


# ------------------------------------------------------------
# Assembly
# ------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def assemble_system(nodes, elems, areas, normals, quad_xyz, quad_wJ, bc_type, u_exact, q_exact, tri_p, gx, gw):
    nn = nodes.shape[0]
    ne = elems.shape[0]

    A = np.zeros((nn, nn), dtype=np.float64)
    b = np.zeros(nn, dtype=np.float64)

    for i in prange(nn):
        xi = nodes[i, 0]
        yi = nodes[i, 1]
        zi = nodes[i, 2]
        row_rhs = 0.0

        # Smooth-face collocation coefficient. Face-local nodes avoid edge BC conflicts.
        c_i = 0.5
        if bc_type[i] == 1:
            A[i, i] += c_i
        else:
            row_rhs += c_i * u_exact[i]

        for e in range(ne):
            n0 = elems[e, 0]
            n1 = elems[e, 1]
            n2 = elems[e, 2]

            on_self = (i == n0) or (i == n1) or (i == n2)
            nx = normals[e, 0]
            ny = normals[e, 1]
            nz = normals[e, 2]

            # For a collocation point on the same planar triangle, q* vanishes exactly.
            # Only the G-kernel needs the Duffy transform.
            if on_self:
                if i == n0:
                    ax0 = nodes[n0, 0]
                    ax1 = nodes[n0, 1]
                    ax2 = nodes[n0, 2]
                    bx0 = nodes[n1, 0]
                    bx1 = nodes[n1, 1]
                    bx2 = nodes[n1, 2]
                    cx0 = nodes[n2, 0]
                    cx1 = nodes[n2, 1]
                    cx2 = nodes[n2, 2]
                    cA = n0
                    cB = n1
                    cC = n2
                elif i == n1:
                    ax0 = nodes[n1, 0]
                    ax1 = nodes[n1, 1]
                    ax2 = nodes[n1, 2]
                    bx0 = nodes[n2, 0]
                    bx1 = nodes[n2, 1]
                    bx2 = nodes[n2, 2]
                    cx0 = nodes[n0, 0]
                    cx1 = nodes[n0, 1]
                    cx2 = nodes[n0, 2]
                    cA = n1
                    cB = n2
                    cC = n0
                else:
                    ax0 = nodes[n2, 0]
                    ax1 = nodes[n2, 1]
                    ax2 = nodes[n2, 2]
                    bx0 = nodes[n0, 0]
                    bx1 = nodes[n0, 1]
                    bx2 = nodes[n0, 2]
                    cx0 = nodes[n1, 0]
                    cx1 = nodes[n1, 1]
                    cx2 = nodes[n1, 2]
                    cA = n2
                    cB = n0
                    cC = n1

                bx0m = bx0 - ax0
                bx1m = bx1 - ax1
                bx2m = bx2 - ax2
                cx0m = cx0 - ax0
                cx1m = cx1 - ax1
                cx2m = cx2 - ax2

                detJ = 2.0 * areas[e]

                # Duffy map on the full triangle from the target vertex A.
                for iu in range(3):
                    u = gx[iu]
                    wu = gw[iu]
                    lamA = 1.0 - u
                    for iv in range(3):
                        v = gx[iv]
                        wv = gw[iv]
                        t0 = (1.0 - v) * bx0m + v * cx0m
                        t1 = (1.0 - v) * bx1m + v * cx1m
                        t2 = (1.0 - v) * bx2m + v * cx2m
                        y0 = ax0 + u * t0
                        y1 = ax1 + u * t1
                        y2 = ax2 + u * t2

                        # Jacobian contributes u and the triangle area map contributes detJ.
                        w = wu * wv * detJ * u

                        rx = xi - y0
                        ry = yi - y1
                        rz = zi - y2
                        r2 = rx * rx + ry * ry + rz * rz
                        if r2 < 1e-28:
                            continue
                        r = math.sqrt(r2)
                        gker = 1.0 / (PI4 * r)

                        lB = u * (1.0 - v)
                        lC = u * v

                        gA = w * lamA * gker
                        gB = w * lB * gker
                        gC = w * lC * gker

                        # q* integral is zero for the coincident planar triangle.
                        if bc_type[cA] == 1:
                            A[i, cA] += -gA
                        else:
                            row_rhs += gA * q_exact[cA]

                        if bc_type[cB] == 1:
                            A[i, cB] += -gB
                        else:
                            row_rhs += gB * q_exact[cB]

                        if bc_type[cC] == 1:
                            A[i, cC] += -gC
                        else:
                            row_rhs += gC * q_exact[cC]

            else:
                for q in range(3):
                    y0 = quad_xyz[e, q, 0]
                    y1 = quad_xyz[e, q, 1]
                    y2 = quad_xyz[e, q, 2]
                    w = quad_wJ[e, q]

                    rx = xi - y0
                    ry = yi - y1
                    rz = zi - y2
                    r2 = rx * rx + ry * ry + rz * rz
                    if r2 < 1e-28:
                        continue
                    r = math.sqrt(r2)
                    invr = 1.0 / r
                    invr3 = invr / r2
                    dotn = rx * nx + ry * ny + rz * nz
                    gker = invr / PI4
                    qstar = dotn * invr3 / PI4

                    b0 = tri_p[q, 0]
                    b1 = tri_p[q, 1]
                    b2 = tri_p[q, 2]

                    g0 = w * b0 * gker
                    g1 = w * b1 * gker
                    g2 = w * b2 * gker
                    h0 = w * b0 * qstar
                    h1 = w * b1 * qstar
                    h2 = w * b2 * qstar

                    # Known contributions to RHS.
                    if bc_type[n0] == 1:
                        row_rhs += g0 * q_exact[n0]
                    else:
                        row_rhs -= h0 * u_exact[n0]

                    if bc_type[n1] == 1:
                        row_rhs += g1 * q_exact[n1]
                    else:
                        row_rhs -= h1 * u_exact[n1]

                    if bc_type[n2] == 1:
                        row_rhs += g2 * q_exact[n2]
                    else:
                        row_rhs -= h2 * u_exact[n2]

                    # Unknown contributions to A.
                    if bc_type[n0] == 1:
                        A[i, n0] += h0
                    else:
                        A[i, n0] += -g0

                    if bc_type[n1] == 1:
                        A[i, n1] += h1
                    else:
                        A[i, n1] += -g1

                    if bc_type[n2] == 1:
                        A[i, n2] += h2
                    else:
                        A[i, n2] += -g2

        b[i] = row_rhs

    return A, b


@njit(parallel=True, fastmath=True)
def evaluate_interior(points, elems, normals, quad_xyz, quad_wJ, u_b, q_b, tri_p):
    npnt = points.shape[0]
    ne = elems.shape[0]
    out = np.zeros(npnt, dtype=np.float64)

    for p in prange(npnt):
        xp = points[p, 0]
        yp = points[p, 1]
        zp = points[p, 2]
        val = 0.0

        for e in range(ne):
            n0 = elems[e, 0]
            n1 = elems[e, 1]
            n2 = elems[e, 2]
            nx = normals[e, 0]
            ny = normals[e, 1]
            nz = normals[e, 2]

            for q in range(3):
                y0 = quad_xyz[e, q, 0]
                y1 = quad_xyz[e, q, 1]
                y2 = quad_xyz[e, q, 2]
                w = quad_wJ[e, q]

                rx = xp - y0
                ry = yp - y1
                rz = zp - y2
                r2 = rx * rx + ry * ry + rz * rz
                r = math.sqrt(r2)
                invr = 1.0 / r
                invr3 = invr / r2
                dotn = rx * nx + ry * ny + rz * nz
                gker = invr / PI4
                qstar = dotn * invr3 / PI4

                l0 = tri_p[q, 0]
                l1 = tri_p[q, 1]
                l2 = tri_p[q, 2]
                ub = l0 * u_b[n0] + l1 * u_b[n1] + l2 * u_b[n2]
                qb = l0 * q_b[n0] + l1 * q_b[n1] + l2 * q_b[n2]
                val += w * (gker * qb - qstar * ub)

        out[p] = val

    return out


def recover_UQ(x, bc_type, u_exact, q_exact):
    u = np.where(bc_type == 1, x, u_exact)
    q = np.where(bc_type == 0, x, q_exact)
    return u, q


def exact_solution(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return x * y + y * z + z * x


def run_case(N):
    t0 = time.perf_counter()
    nodes, elems, node_face = generate_mesh(N)
    areas, normals = compute_geometry(nodes, elems)
    _ = build_node_elems(nodes, elems)
    bc_type, u_exact_b, q_exact_b = assign_bcs(nodes, node_face)
    quad_xyz, quad_wJ = precompute_regular_quadrature(nodes, elems, areas)
    A, b = assemble_system(nodes, elems, areas, normals, quad_xyz, quad_wJ, bc_type, u_exact_b, q_exact_b, TRI_P, GX, GW)
    setup_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    x = np.linalg.solve(A, b)
    solve_time = time.perf_counter() - t1

    u_b, q_b = recover_UQ(x, bc_type, u_exact_b, q_exact_b)

    t2 = time.perf_counter()
    grid = np.linspace(0.2, 0.8, 5)
    pts = np.array([[a, b_, c] for a in grid for b_ in grid for c in grid], dtype=np.float64)
    u_num = evaluate_interior(pts, elems, normals, quad_xyz, quad_wJ, u_b, q_b, TRI_P)
    u_ex = exact_solution(pts)
    rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)
    eval_time = time.perf_counter() - t2
    total_time = setup_time + solve_time + eval_time

    return {
        "N": N,
        "Ne": int(elems.shape[0]),
        "err": float(rel_l2),
        "setup": setup_time,
        "solve": solve_time,
        "eval": eval_time,
        "total": total_time,
    }


def main():
    results = []
    for N in [8, 16, 32]:
        results.append(run_case(N))

    h1 = 1.0 / results[0]["N"]
    h3 = 1.0 / results[2]["N"]
    slope = math.log(results[2]["err"] / results[0]["err"]) / math.log(h3 / h1)

    print("N    | Ne      | Rel L2 Error   | Setup (s) | Solve (s) | Eval (s) | Total (s)")
    for r in results:
        print(f"{r['N']:<4d} | {r['Ne']:<7d} | {r['err']:>13.6e} | {r['setup']:8.4f} | {r['solve']:8.4f} | {r['eval']:8.4f} | {r['total']:8.4f}")
    print("Convergence Analysis:")
    print(f"Computed Slope: {slope:.4f}")
    print("Expected Slope: ~2.0000 (O(h^2) for linear elements)")


if __name__ == "__main__":
    main()
