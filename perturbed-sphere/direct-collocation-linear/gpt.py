import math
import time
import numpy as np
from numba import njit, prange


# ============================================================
# 3D DIRECT MIXED BEM - COLLOCATION - CONTINUOUS LINEAR
# Bumpy Potato Domain via Cubed-Sphere-like face projection
# ============================================================
# Structural fix:
#   - Build the mixed BEM system in reduced form directly.
#   - Unknown vector x = [u on Neumann nodes, q on Dirichlet nodes]
#   - Use the consistent interior BIE:
#         c u + H u = G q
#   - Use a harmonic manufactured solution:
#         u(x,y,z) = x*y + z
#     so that ∇²u = 0 exactly.
# ============================================================

PI = math.pi


# -----------------------------
# Exact solution / BCs
# -----------------------------

def u_exact(x, y, z):
    return x * y + z


def grad_exact(x, y, z):
    return y, x, 1.0


# -----------------------------
# Mesh generation
# -----------------------------

def _round_key(x, y, z, nd=6):
    return (round(float(x), nd), round(float(y), nd), round(float(z), nd))


def generate_continuous_bumpy_mesh(N):
    """
    Generate a continuous surface mesh on a bumpy potato-like closed surface.

    Construction:
    - Start from 6 cube faces on [-1,1]^3.
    - Subdivide each face into N x N squares -> 2 triangles per square.
    - Project each face-node radially from the cube to the unit sphere.
    - Apply radius perturbation r(theta, phi).
    - Merge duplicate vertices using dictionary tracker rounded to 6 decimals.
    """
    node_dict = {}
    nodes = []
    elems = []

    def get_node_id(x, y, z):
        key = _round_key(x, y, z)
        idx = node_dict.get(key)
        if idx is None:
            idx = len(nodes)
            node_dict[key] = idx
            nodes.append([x, y, z])
        return idx

    s = np.linspace(-1.0, 1.0, N + 1)
    t = np.linspace(-1.0, 1.0, N + 1)

    # (fixed_axis, fixed_value, varying_axis_1, varying_axis_2)
    face_specs = [
        (0, 1.0, 1, 2),
        (0, -1.0, 2, 1),
        (1, 1.0, 2, 0),
        (1, -1.0, 0, 2),
        (2, 1.0, 0, 1),
        (2, -1.0, 1, 0),
    ]

    for fixed_axis, fixed_val, a1, a2 in face_specs:
        local_ids = np.empty((N + 1, N + 1), dtype=np.int64)

        for i in range(N + 1):
            for j in range(N + 1):
                coord = [0.0, 0.0, 0.0]
                coord[fixed_axis] = fixed_val
                coord[a1] = s[i]
                coord[a2] = t[j]
                X, Y, Z = coord

                # Radial projection from cube to unit sphere
                Rcube = math.sqrt(X * X + Y * Y + Z * Z)
                Xs = X / Rcube
                Ys = Y / Rcube
                Zs = Z / Rcube

                theta = math.acos(max(-1.0, min(1.0, Zs)))
                phi = math.atan2(Ys, Xs)

                # Bumpy potato radius
                r = 1.5 + 0.3 * math.sin(4.0 * theta) * math.cos(5.0 * phi)

                xf = r * math.sin(theta) * math.cos(phi)
                yf = r * math.sin(theta) * math.sin(phi)
                zf = r * math.cos(theta)
                local_ids[i, j] = get_node_id(xf, yf, zf)

        for i in range(N):
            for j in range(N):
                n00 = local_ids[i, j]
                n10 = local_ids[i + 1, j]
                n01 = local_ids[i, j + 1]
                n11 = local_ids[i + 1, j + 1]

                elems.append([n00, n10, n11])
                elems.append([n00, n11, n01])

    nodes = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(elems, dtype=np.int64)
    return nodes, elems


# -----------------------------
# Geometry
# -----------------------------

@njit(parallel=True, fastmath=True)
def compute_geometry_numba(nodes, elems):
    ne = elems.shape[0]
    nn = nodes.shape[0]
    areas = np.zeros(ne, dtype=np.float64)
    enormals = np.zeros((ne, 3), dtype=np.float64)
    nodal_normals = np.zeros((nn, 3), dtype=np.float64)
    nodal_area = np.zeros(nn, dtype=np.float64)

    for e in prange(ne):
        i0 = elems[e, 0]
        i1 = elems[e, 1]
        i2 = elems[e, 2]

        x0 = nodes[i0, 0]
        y0 = nodes[i0, 1]
        z0 = nodes[i0, 2]
        x1 = nodes[i1, 0]
        y1 = nodes[i1, 1]
        z1 = nodes[i1, 2]
        x2 = nodes[i2, 0]
        y2 = nodes[i2, 1]
        z2 = nodes[i2, 2]

        ux = x1 - x0
        uy = y1 - y0
        uz = z1 - z0
        vx = x2 - x0
        vy = y2 - y0
        vz = z2 - z0

        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx

        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm > 1e-30:
            nx /= norm
            ny /= norm
            nz /= norm
        else:
            nx = 0.0
            ny = 0.0
            nz = 0.0

        # Ensure outward orientation using centroid radial check
        cx = (x0 + x1 + x2) / 3.0
        cy = (y0 + y1 + y2) / 3.0
        cz = (z0 + z1 + z2) / 3.0
        if nx * cx + ny * cy + nz * cz < 0.0:
            nx = -nx
            ny = -ny
            nz = -nz

        area = 0.5 * norm
        areas[e] = area
        enormals[e, 0] = nx
        enormals[e, 1] = ny
        enormals[e, 2] = nz

    for e in range(ne):
        i0 = elems[e, 0]
        i1 = elems[e, 1]
        i2 = elems[e, 2]
        a = areas[e]
        nx = enormals[e, 0]
        ny = enormals[e, 1]
        nz = enormals[e, 2]

        nodal_normals[i0, 0] += a * nx
        nodal_normals[i0, 1] += a * ny
        nodal_normals[i0, 2] += a * nz
        nodal_area[i0] += a

        nodal_normals[i1, 0] += a * nx
        nodal_normals[i1, 1] += a * ny
        nodal_normals[i1, 2] += a * nz
        nodal_area[i1] += a

        nodal_normals[i2, 0] += a * nx
        nodal_normals[i2, 1] += a * ny
        nodal_normals[i2, 2] += a * nz
        nodal_area[i2] += a

    for i in prange(nn):
        a = nodal_area[i]
        if a > 1e-30:
            nodal_normals[i, 0] /= a
            nodal_normals[i, 1] /= a
            nodal_normals[i, 2] /= a
            nrm = math.sqrt(
                nodal_normals[i, 0] * nodal_normals[i, 0]
                + nodal_normals[i, 1] * nodal_normals[i, 1]
                + nodal_normals[i, 2] * nodal_normals[i, 2]
            )
            if nrm > 1e-30:
                nodal_normals[i, 0] /= nrm
                nodal_normals[i, 1] /= nrm
                nodal_normals[i, 2] /= nrm
        else:
            x = nodes[i, 0]
            y = nodes[i, 1]
            z = nodes[i, 2]
            nrm = math.sqrt(x * x + y * y + z * z)
            if nrm > 1e-30:
                nodal_normals[i, 0] = x / nrm
                nodal_normals[i, 1] = y / nrm
                nodal_normals[i, 2] = z / nrm

    return areas, enormals, nodal_normals


def compute_geometry(nodes, elems):
    return compute_geometry_numba(nodes, elems)


# -----------------------------
# Boundary conditions
# -----------------------------

def assign_mixed_bcs(nodes, nodal_normals):
    nn = nodes.shape[0]
    bc_type = np.zeros(nn, dtype=np.int64)  # 0 Dirichlet, 1 Neumann
    bc_val = np.zeros(nn, dtype=np.float64)
    u_known = np.zeros(nn, dtype=np.float64)
    q_known = np.zeros(nn, dtype=np.float64)

    for i in range(nn):
        x, y, z = nodes[i]
        u = u_exact(float(x), float(y), float(z))
        du_dx, du_dy, du_dz = grad_exact(float(x), float(y), float(z))
        nx, ny, nz = nodal_normals[i]
        q = du_dx * nx + du_dy * ny + du_dz * nz

        if x > 0.0:
            bc_type[i] = 0
            bc_val[i] = u
            u_known[i] = u
        else:
            bc_type[i] = 1
            bc_val[i] = q
            q_known[i] = q

    return bc_type, bc_val, u_known, q_known


# -----------------------------
# Quadrature rules
# -----------------------------

def gauss_legendre_1d(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x.astype(np.float64), w.astype(np.float64)


def dunavant7_rule():
    w = np.array([
        0.2250000000000000,
        0.1323941527885062,
        0.1323941527885062,
        0.1323941527885062,
        0.1259391805448272,
        0.1259391805448272,
        0.1259391805448272,
    ], dtype=np.float64)

    l1 = np.array([
        1.0 / 3.0,
        0.0597158717897700,
        0.4701420641051150,
        0.4701420641051150,
        0.7974269853530872,
        0.1012865073234563,
        0.1012865073234563,
    ], dtype=np.float64)

    l2 = np.array([
        1.0 / 3.0,
        0.4701420641051150,
        0.0597158717897700,
        0.4701420641051150,
        0.1012865073234563,
        0.7974269853530872,
        0.1012865073234563,
    ], dtype=np.float64)

    l3 = np.array([
        1.0 / 3.0,
        0.4701420641051150,
        0.4701420641051150,
        0.0597158717897700,
        0.1012865073234563,
        0.1012865073234563,
        0.7974269853530872,
    ], dtype=np.float64)

    return l1, l2, l3, w


# -----------------------------
# Kernels
# -----------------------------

@njit(fastmath=True)
def _dist(xi, yi, zi, x, y, z):
    dx = xi - x
    dy = yi - y
    dz = zi - z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


@njit(fastmath=True)
def _kernel_G(xi, yi, zi, x, y, z):
    r = _dist(xi, yi, zi, x, y, z)
    if r < 1e-15:
        return 0.0
    return 1.0 / (4.0 * PI * r)


@njit(fastmath=True)
def _kernel_H(xi, yi, zi, nx, ny, nz, x, y, z):
    # Positive kernel corresponding to q* = (X-Y)·n / (4π r^3)
    dx = xi - x
    dy = yi - y
    dz = zi - z
    r2 = dx * dx + dy * dy + dz * dz
    if r2 < 1e-30:
        return 0.0
    r = math.sqrt(r2)
    return (dx * nx + dy * ny + dz * nz) / (4.0 * PI * r2 * r)


# -----------------------------
# Mixed system assembly
# -----------------------------

@njit(parallel=True, fastmath=True)
def assemble_mixed_system(
    nodes,
    elems,
    areas,
    enormals,
    bc_type,
    u_known,
    q_known,
    u_pos,
    q_pos,
    dun_l1,
    dun_l2,
    dun_l3,
    dun_w,
    gq12_x,
    gq12_w,
):
    nn = nodes.shape[0]
    ne = elems.shape[0]
    A = np.zeros((nn, nn), dtype=np.float64)
    b = np.zeros(nn, dtype=np.float64)

    nq7 = dun_w.shape[0]
    nq12 = gq12_x.shape[0]
    c_jump = 0.5

    for i in prange(nn):
        xi = nodes[i, 0]
        yi = nodes[i, 1]
        zi = nodes[i, 2]

        # c_i u_i term
        if bc_type[i] == 1:
            A[i, u_pos[i]] += c_jump
        else:
            b[i] -= c_jump * u_known[i]

        for e in range(ne):
            i0 = elems[e, 0]
            i1 = elems[e, 1]
            i2 = elems[e, 2]

            x0 = nodes[i0, 0]
            y0 = nodes[i0, 1]
            z0 = nodes[i0, 2]
            x1 = nodes[i1, 0]
            y1 = nodes[i1, 1]
            z1 = nodes[i1, 2]
            x2 = nodes[i2, 0]
            y2 = nodes[i2, 1]
            z2 = nodes[i2, 2]

            exn = enormals[e, 0]
            eyn = enormals[e, 1]
            ezn = enormals[e, 2]
            area = areas[e]

            d0 = (xi - x0) * (xi - x0) + (yi - y0) * (yi - y0) + (zi - z0) * (zi - z0)
            d1 = (xi - x1) * (xi - x1) + (yi - y1) * (yi - y1) + (zi - z1) * (zi - z1)
            d2 = (xi - x2) * (xi - x2) + (yi - y2) * (yi - y2) + (zi - z2) * (zi - z2)

            cx = (x0 + x1 + x2) / 3.0
            cy = (y0 + y1 + y2) / 3.0
            cz = (z0 + z1 + z2) / 3.0
            dc2 = (xi - cx) * (xi - cx) + (yi - cy) * (yi - cy) + (zi - cz) * (zi - cz)

            near_vertex = (d0 < 1e-26) or (d1 < 1e-26) or (d2 < 1e-26)
            near_singular = dc2 < 20.0 * area

            g0 = 0.0
            g1 = 0.0
            g2 = 0.0
            h0 = 0.0
            h1 = 0.0
            h2 = 0.0

            if near_vertex:
                singular_local = -1
                if i == i0:
                    singular_local = 0
                elif i == i1:
                    singular_local = 1
                elif i == i2:
                    singular_local = 2

                if singular_local == 0:
                    ax = x0
                    ay = y0
                    az = z0
                    bx = x1
                    by = y1
                    bz = z1
                    cx2 = x2
                    cy2 = y2
                    cz2 = z2
                elif singular_local == 1:
                    ax = x1
                    ay = y1
                    az = z1
                    bx = x2
                    by = y2
                    bz = z2
                    cx2 = x0
                    cy2 = y0
                    cz2 = z0
                else:
                    ax = x2
                    ay = y2
                    az = z2
                    bx = x0
                    by = y0
                    bz = z0
                    cx2 = x1
                    cy2 = y1
                    cz2 = z1

                if singular_local != -1:
                    # Duffy on a square -> triangle, singular vertex at target node
                    for iu in range(nq12):
                        u = 0.5 * (gq12_x[iu] + 1.0)
                        wu = 0.5 * gq12_w[iu]
                        for iv in range(nq12):
                            v = 0.5 * (gq12_x[iv] + 1.0)
                            wv = 0.5 * gq12_w[iv]

                            lam1 = 1.0 - u
                            lam2 = u * (1.0 - v)
                            lam3 = u * v

                            px = lam1 * ax + lam2 * bx + lam3 * cx2
                            py = lam1 * ay + lam2 * by + lam3 * cy2
                            pz = lam1 * az + lam2 * bz + lam3 * cz2
                            jac = 2.0 * area * u

                            Gv = _kernel_G(xi, yi, zi, px, py, pz)
                            Hv = _kernel_H(xi, yi, zi, exn, eyn, ezn, px, py, pz)

                            if singular_local == 0:
                                phi0 = lam1
                                phi1 = lam2
                                phi2 = lam3
                            elif singular_local == 1:
                                phi0 = lam3
                                phi1 = lam1
                                phi2 = lam2
                            else:
                                phi0 = lam2
                                phi1 = lam3
                                phi2 = lam1

                            wgt = wu * wv * jac
                            g0 += wgt * phi0 * Gv
                            g1 += wgt * phi1 * Gv
                            g2 += wgt * phi2 * Gv
                            h0 += wgt * phi0 * Hv
                            h1 += wgt * phi1 * Hv
                            h2 += wgt * phi2 * Hv
            elif near_singular:
                # Denser tensor-product quadrature for near-singular panels
                for iu in range(nq12):
                    u = 0.5 * (gq12_x[iu] + 1.0)
                    wu = 0.5 * gq12_w[iu]
                    for iv in range(nq12):
                        v = 0.5 * (gq12_x[iv] + 1.0)
                        wv = 0.5 * gq12_w[iv]
                        lam1 = 1.0 - u
                        lam2 = u * (1.0 - v)
                        lam3 = u * v
                        px = lam1 * x0 + lam2 * x1 + lam3 * x2
                        py = lam1 * y0 + lam2 * y1 + lam3 * y2
                        pz = lam1 * z0 + lam2 * z1 + lam3 * z2
                        jac = 2.0 * area * u
                        Gv = _kernel_G(xi, yi, zi, px, py, pz)
                        Hv = _kernel_H(xi, yi, zi, exn, eyn, ezn, px, py, pz)
                        wgt = wu * wv * jac
                        g0 += wgt * lam1 * Gv
                        g1 += wgt * lam2 * Gv
                        g2 += wgt * lam3 * Gv
                        h0 += wgt * lam1 * Hv
                        h1 += wgt * lam2 * Hv
                        h2 += wgt * lam3 * Hv
            else:
                # Regular 7-point Dunavant quadrature
                for q in range(nq7):
                    l1 = dun_l1[q]
                    l2 = dun_l2[q]
                    l3 = dun_l3[q]
                    px = l1 * x0 + l2 * x1 + l3 * x2
                    py = l1 * y0 + l2 * y1 + l3 * y2
                    pz = l1 * z0 + l2 * z1 + l3 * z2
                    Gv = _kernel_G(xi, yi, zi, px, py, pz)
                    Hv = _kernel_H(xi, yi, zi, exn, eyn, ezn, px, py, pz)
                    wgt = dun_w[q] * 2.0 * area
                    g0 += wgt * l1 * Gv
                    g1 += wgt * l2 * Gv
                    g2 += wgt * l3 * Gv
                    h0 += wgt * l1 * Hv
                    h1 += wgt * l2 * Hv
                    h2 += wgt * l3 * Hv

            # Reduced mixed system assembly:
            # c u + H u = G q
            # Unknowns: u on Neumann nodes, q on Dirichlet nodes
            # RHS:  -c*u_known - H*u_known + G*q_known
            # Contribution from local node i0
            if bc_type[i0] == 1:
                A[i, u_pos[i0]] += h0
                b[i] += g0 * q_known[i0]
            else:
                A[i, q_pos[i0]] += -g0
                b[i] += -h0 * u_known[i0]

            # Contribution from local node i1
            if bc_type[i1] == 1:
                A[i, u_pos[i1]] += h1
                b[i] += g1 * q_known[i1]
            else:
                A[i, q_pos[i1]] += -g1
                b[i] += -h1 * u_known[i1]

            # Contribution from local node i2
            if bc_type[i2] == 1:
                A[i, u_pos[i2]] += h2
                b[i] += g2 * q_known[i2]
            else:
                A[i, q_pos[i2]] += -g2
                b[i] += -h2 * u_known[i2]

    return A, b


# -----------------------------
# Solve system
# -----------------------------

def solve_boundary_system(A, b, bc_type, bc_val):
    nn = A.shape[0]
    dir_idx = np.where(bc_type == 0)[0]
    neu_idx = np.where(bc_type == 1)[0]

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, b, rcond=None)[0]

    u = np.zeros(nn, dtype=np.float64)
    q = np.zeros(nn, dtype=np.float64)

    # Unknown vector layout: [u_neu, q_dir]
    u[dir_idx] = bc_val[dir_idx]
    q[neu_idx] = bc_val[neu_idx]

    u[neu_idx] = x[:neu_idx.size]
    q[dir_idx] = x[neu_idx.size:]

    return u, q


# -----------------------------
# Interior evaluation
# -----------------------------

@njit(fastmath=True)
def eval_point(xi, yi, zi, nodes, elems, u, q, areas, l1, l2, l3, w):
    ne = elems.shape[0]
    val = 0.0

    for e in range(ne):
        i0 = elems[e, 0]
        i1 = elems[e, 1]
        i2 = elems[e, 2]

        x0 = nodes[i0, 0]
        y0 = nodes[i0, 1]
        z0 = nodes[i0, 2]
        x1 = nodes[i1, 0]
        y1 = nodes[i1, 1]
        z1 = nodes[i1, 2]
        x2 = nodes[i2, 0]
        y2 = nodes[i2, 1]
        z2 = nodes[i2, 2]

        ux = x1 - x0
        uy = y1 - y0
        uz = z1 - z0
        vx = x2 - x0
        vy = y2 - y0
        vz = z2 - z0
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx

        nnrm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nnrm > 1e-30:
            nx /= nnrm
            ny /= nnrm
            nz /= nnrm

        area = areas[e]

        for k in range(7):
            px = l1[k] * x0 + l2[k] * x1 + l3[k] * x2
            py = l1[k] * y0 + l2[k] * y1 + l3[k] * y2
            pz = l1[k] * z0 + l2[k] * z1 + l3[k] * z2

            r = _dist(xi, yi, zi, px, py, pz)
            if r < 1e-15:
                continue

            Gv = 1.0 / (4.0 * PI * r)
            dx = xi - px
            dy = yi - py
            dz = zi - pz
            Hv = (dx * nx + dy * ny + dz * nz) / (4.0 * PI * r * r * r)

            ue = u[i0] * l1[k] + u[i1] * l2[k] + u[i2] * l3[k]
            qe = q[i0] * l1[k] + q[i1] * l2[k] + q[i2] * l3[k]

            # Interior representation: u = G q - H u
            val += w[k] * 2.0 * area * (qe * Gv - ue * Hv)

    return val


def evaluate_interior(nodes, elems, u, q, areas):
    pts = np.linspace(-0.5, 0.5, 5)
    l1, l2, l3, w = dunavant7_rule()

    vals = []
    exact = []
    for x in pts:
        for y in pts:
            for z in pts:
                vals.append(
                    eval_point(
                        float(x), float(y), float(z),
                        nodes, elems, u, q, areas, l1, l2, l3, w
                    )
                )
                exact.append(u_exact(float(x), float(y), float(z)))

    vals = np.asarray(vals, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    return np.linalg.norm(vals - exact) / np.linalg.norm(exact)


# -----------------------------
# Driver
# -----------------------------

def run_case(N):
    t0 = time.perf_counter()

    nodes, elems = generate_continuous_bumpy_mesh(N)
    areas, enormals, nodal_normals = compute_geometry(nodes, elems)
    bc_type, bc_val, u_known, q_known = assign_mixed_bcs(nodes, nodal_normals)

    nn = nodes.shape[0]
    u_pos = np.full(nn, -1, dtype=np.int64)
    q_pos = np.full(nn, -1, dtype=np.int64)

    nu = 0
    nq = 0
    for i in range(nn):
        if bc_type[i] == 1:
            u_pos[i] = nu
            nu += 1
        else:
            q_pos[i] = nq
            nq += 1

    dun_l1, dun_l2, dun_l3, dun_w = dunavant7_rule()
    gq12_x, gq12_w = gauss_legendre_1d(12)

    A, b = assemble_mixed_system(
        nodes, elems, areas, enormals,
        bc_type, u_known, q_known, u_pos, q_pos,
        dun_l1, dun_l2, dun_l3, dun_w,
        gq12_x, gq12_w
    )
    t1 = time.perf_counter()

    u, q = solve_boundary_system(A, b, bc_type, bc_val)
    t2 = time.perf_counter()

    rel_l2 = evaluate_interior(nodes, elems, u, q, areas)
    t3 = time.perf_counter()

    return {
        "N": N,
        "Ne": elems.shape[0],
        "N_nodes": nodes.shape[0],
        "Rel_L2": rel_l2,
        "Setup": t1 - t0,
        "Solve": t2 - t1,
        "Eval": t3 - t2,
        "Total": t3 - t0,
    }


def main():
    Ns = [8, 16, 32]

    # Warm-up JIT
    _nodes, _elems = generate_continuous_bumpy_mesh(2)
    _areas, _enormals, _numn = compute_geometry(_nodes, _elems)
    _bc_type, _bc_val, _u_known, _q_known = assign_mixed_bcs(_nodes, _numn)

    _nn = _nodes.shape[0]
    _u_pos = np.full(_nn, -1, dtype=np.int64)
    _q_pos = np.full(_nn, -1, dtype=np.int64)
    _nu = 0
    _nq = 0
    for i in range(_nn):
        if _bc_type[i] == 1:
            _u_pos[i] = _nu
            _nu += 1
        else:
            _q_pos[i] = _nq
            _nq += 1

    dun_l1, dun_l2, dun_l3, dun_w = dunavant7_rule()
    gq12_x, gq12_w = gauss_legendre_1d(12)

    _A, _b = assemble_mixed_system(
        _nodes, _elems, _areas, _enormals,
        _bc_type, _u_known, _q_known, _u_pos, _q_pos,
        dun_l1, dun_l2, dun_l3, dun_w,
        gq12_x, gq12_w
    )
    _u, _q = solve_boundary_system(_A, _b, _bc_type, _bc_val)
    _ = evaluate_interior(_nodes, _elems, _u, _q, _areas)

    results = []
    for N in Ns:
        results.append(run_case(N))

    e1 = results[0]["Rel_L2"]
    e2 = results[2]["Rel_L2"]
    h1 = 1.0 / 8.0
    h2 = 1.0 / 32.0
    slope = (math.log(e2) - math.log(e1)) / (math.log(h2) - math.log(h1))

    print("N    | Ne      | N_nodes | Rel L2 Error   | Setup (s) | Solve (s) | Eval (s) | Total (s)")
    for r in results:
        print(
            f"{r['N']:<4d} | {r['Ne']:<7d} | {r['N_nodes']:<7d} | "
            f"{r['Rel_L2']:<13.6e} | {r['Setup']:<9.4f} | {r['Solve']:<9.4f} | "
            f"{r['Eval']:<8.4f} | {r['Total']:<8.4f}"
        )
    print("Convergence Analysis:")
    print(f"Computed Slope: {slope:.4f}")
    print("Expected Slope: ~2.0000 (O(h^2) for linear elements)")


if __name__ == "__main__":
    main()