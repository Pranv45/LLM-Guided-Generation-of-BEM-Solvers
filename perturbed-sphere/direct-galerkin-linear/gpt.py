import math
import time
import numpy as np
from numba import njit, prange


PI = math.pi
FOUR_PI = 4.0 * PI


def exact_u(x, y, z):
    return np.sinh(x) * np.sin(y) + np.cosh(y) * np.cos(z)


def exact_grad(x, y, z):
    du_dx = np.cosh(x) * np.sin(y)
    du_dy = np.sinh(x) * np.cos(y) + np.sinh(y) * np.cos(z)
    du_dz = -np.cosh(y) * np.sin(z)
    return du_dx, du_dy, du_dz


def generate_continuous_bumpy_mesh(N):
    node_map = {}
    nodes = []
    elems = []

    def add_node(x, y, z):
        key = (round(float(x), 6), round(float(y), 6), round(float(z), 6))
        idx = node_map.get(key, -1)
        if idx < 0:
            idx = len(nodes)
            node_map[key] = idx
            nodes.append([x, y, z])
        return idx

    lin = np.linspace(-1.0, 1.0, N + 1)

    face_builders = [
        lambda a, b: (1.0, a, b),
        lambda a, b: (-1.0, a, b),
        lambda a, b: (a, 1.0, b),
        lambda a, b: (a, -1.0, b),
        lambda a, b: (a, b, 1.0),
        lambda a, b: (a, b, -1.0),
    ]

    for fb in face_builders:
        local = np.empty((N + 1, N + 1), dtype=np.int64)
        for i, a in enumerate(lin):
            for j, b in enumerate(lin):
                X, Y, Z = fb(a, b)
                R = math.sqrt(X * X + Y * Y + Z * Z)
                xs, ys, zs = X / R, Y / R, Z / R
                theta = math.acos(max(-1.0, min(1.0, zs)))
                phi = math.atan2(ys, xs)
                r = 1.5 + 0.3 * math.sin(4.0 * theta) * math.cos(5.0 * phi)
                xf = r * math.sin(theta) * math.cos(phi)
                yf = r * math.sin(theta) * math.sin(phi)
                zf = r * math.cos(theta)
                local[i, j] = add_node(xf, yf, zf)

        for i in range(N):
            for j in range(N):
                n00 = local[i, j]
                n10 = local[i + 1, j]
                n01 = local[i, j + 1]
                n11 = local[i + 1, j + 1]
                elems.append([n00, n10, n11])
                elems.append([n00, n11, n01])

    nodes = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(elems, dtype=np.int64)

    for e in range(elems.shape[0]):
        i0, i1, i2 = elems[e]
        x0, y0, z0 = nodes[i0]
        x1, y1, z1 = nodes[i1]
        x2, y2, z2 = nodes[i2]
        ux, uy, uz = x1 - x0, y1 - y0, z1 - z0
        vx, vy, vz = x2 - x0, y2 - y0, z2 - z0
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        cx = (x0 + x1 + x2) / 3.0
        cy = (y0 + y1 + y2) / 3.0
        cz = (z0 + z1 + z2) / 3.0
        if nx * cx + ny * cy + nz * cz < 0.0:
            elems[e, 1], elems[e, 2] = elems[e, 2], elems[e, 1]

    return nodes, elems


def compute_geometry(nodes, elems):
    ne = elems.shape[0]
    nn = nodes.shape[0]
    areas = np.empty(ne, dtype=np.float64)
    normals = np.empty((ne, 3), dtype=np.float64)
    centroids = np.empty((ne, 3), dtype=np.float64)
    nodal_normals = np.zeros((nn, 3), dtype=np.float64)

    for e in range(ne):
        i0, i1, i2 = elems[e]
        x0, y0, z0 = nodes[i0]
        x1, y1, z1 = nodes[i1]
        x2, y2, z2 = nodes[i2]

        ux, uy, uz = x1 - x0, y1 - y0, z1 - z0
        vx, vy, vz = x2 - x0, y2 - y0, z2 - z0
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm == 0.0:
            raise ValueError("Degenerate triangle encountered.")
        area = 0.5 * norm
        nx /= norm
        ny /= norm
        nz /= norm

        cx = (x0 + x1 + x2) / 3.0
        cy = (y0 + y1 + y2) / 3.0
        cz = (z0 + z1 + z2) / 3.0
        if nx * cx + ny * cy + nz * cz < 0.0:
            nx, ny, nz = -nx, -ny, -nz

        areas[e] = area
        normals[e, 0] = nx
        normals[e, 1] = ny
        normals[e, 2] = nz
        centroids[e, 0] = cx
        centroids[e, 1] = cy
        centroids[e, 2] = cz

        wnx = area * nx
        wny = area * ny
        wnz = area * nz
        nodal_normals[i0, 0] += wnx
        nodal_normals[i0, 1] += wny
        nodal_normals[i0, 2] += wnz
        nodal_normals[i1, 0] += wnx
        nodal_normals[i1, 1] += wny
        nodal_normals[i1, 2] += wnz
        nodal_normals[i2, 0] += wnx
        nodal_normals[i2, 1] += wny
        nodal_normals[i2, 2] += wnz

    for i in range(nn):
        x, y, z = nodes[i]
        nx, ny, nz = nodal_normals[i]
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm > 0.0:
            nx /= norm
            ny /= norm
            nz /= norm
        if x * nx + y * ny + z * nz < 0.0:
            nx, ny, nz = -nx, -ny, -nz
        nodal_normals[i, 0] = nx
        nodal_normals[i, 1] = ny
        nodal_normals[i, 2] = nz

    return areas, normals, centroids, nodal_normals


def assign_mixed_bcs(nodes, nodal_normals):
    x = nodes[:, 0]
    u_ex = exact_u(nodes[:, 0], nodes[:, 1], nodes[:, 2])
    du_dx, du_dy, du_dz = exact_grad(nodes[:, 0], nodes[:, 1], nodes[:, 2])
    q_ex = du_dx * nodal_normals[:, 0] + du_dy * nodal_normals[:, 1] + du_dz * nodal_normals[:, 2]
    is_dirichlet = x > 0.0
    bc_type = np.where(is_dirichlet, 0, 1).astype(np.int8)
    bc_val = np.where(is_dirichlet, u_ex, q_ex)
    return bc_type, bc_val, u_ex, q_ex


Q3_BARY = np.array(
    [
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    ],
    dtype=np.float64,
)
Q3_W = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

Q7_BARY = np.array(
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
Q7_W = np.array(
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

GL4_X, GL4_W = np.polynomial.legendre.leggauss(4)
GL4_X = 0.5 * (GL4_X + 1.0)
GL4_W = 0.5 * GL4_W


@njit(inline="always", fastmath=True)
def _bary_point(ax, ay, az, bx, by, bz, cx, cy, cz, l0, l1, l2):
    return (
        l0 * ax + l1 * bx + l2 * cx,
        l0 * ay + l1 * by + l2 * cy,
        l0 * az + l1 * bz + l2 * cz,
    )


@njit(inline="always", fastmath=True)
def _norm3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


@njit(inline="always", fastmath=True)
def _kernel_u(rx, ry, rz):
    r2 = rx * rx + ry * ry + rz * rz
    if r2 < 1e-18:
        r2 = 1e-18
    return 1.0 / (FOUR_PI * math.sqrt(r2))


@njit(inline="always", fastmath=True)
def _kernel_q(rx, ry, rz, nx, ny, nz):
    r2 = rx * rx + ry * ry + rz * rz
    if r2 < 1e-18:
        r2 = 1e-18
    r = math.sqrt(r2)
    dot = rx * nx + ry * ny + rz * nz
    return dot / (FOUR_PI * r2 * r)


@njit(inline="always", fastmath=True)
def _add9(M, i0, i1, i2, j0, j1, j2, coef, a0, a1, a2, b0, b1, b2):
    M[i0, j0] += coef * a0 * b0
    M[i0, j1] += coef * a0 * b1
    M[i0, j2] += coef * a0 * b2
    M[i1, j0] += coef * a1 * b0
    M[i1, j1] += coef * a1 * b1
    M[i1, j2] += coef * a1 * b2
    M[i2, j0] += coef * a2 * b0
    M[i2, j1] += coef * a2 * b1
    M[i2, j2] += coef * a2 * b2


def assemble_mass_matrix(nodes, elems, areas):
    nn = nodes.shape[0]
    M = np.zeros((nn, nn), dtype=np.float64)
    for e in range(elems.shape[0]):
        i0, i1, i2 = elems[e]
        a = areas[e] / 12.0
        M[i0, i0] += 2.0 * a
        M[i0, i1] += a
        M[i0, i2] += a
        M[i1, i0] += a
        M[i1, i1] += 2.0 * a
        M[i1, i2] += a
        M[i2, i0] += a
        M[i2, i1] += a
        M[i2, i2] += 2.0 * a
    return M


@njit(cache=True, fastmath=True)
def assemble_galerkin_operators(nodes, elems, areas, centroids, normals, q3_bary, q3_w, q7_bary, q7_w, gl_x, gl_w):
    nn = nodes.shape[0]
    ne = elems.shape[0]
    H = np.zeros((nn, nn), dtype=np.float64)
    G = np.zeros((nn, nn), dtype=np.float64)

    for t in range(ne):
        ti0 = elems[t, 0]
        ti1 = elems[t, 1]
        ti2 = elems[t, 2]

        tax = nodes[ti0, 0]
        tay = nodes[ti0, 1]
        taz = nodes[ti0, 2]
        tbx = nodes[ti1, 0]
        tby = nodes[ti1, 1]
        tbz = nodes[ti1, 2]
        tcx = nodes[ti2, 0]
        tcy = nodes[ti2, 1]
        tcz = nodes[ti2, 2]

        area_t = areas[t]
        ctx = centroids[t, 0]
        cty = centroids[t, 1]
        ctz = centroids[t, 2]

        for s in range(ne):
            sj0 = elems[s, 0]
            sj1 = elems[s, 1]
            sj2 = elems[s, 2]

            sax = nodes[sj0, 0]
            say = nodes[sj0, 1]
            saz = nodes[sj0, 2]
            sbx = nodes[sj1, 0]
            sby = nodes[sj1, 1]
            sbz = nodes[sj1, 2]
            scx = nodes[sj2, 0]
            scy = nodes[sj2, 1]
            scz = nodes[sj2, 2]

            area_s = areas[s]
            nx = normals[s, 0]
            ny = normals[s, 1]
            nz = normals[s, 2]

            dx = ctx - centroids[s, 0]
            dy = cty - centroids[s, 1]
            dz = ctz - centroids[s, 2]
            dist2 = dx * dx + dy * dy + dz * dz

            if t == s:
                for it in range(q7_bary.shape[0]):
                    lt0 = q7_bary[it, 0]
                    lt1 = q7_bary[it, 1]
                    lt2 = q7_bary[it, 2]
                    xt, yt, zt = _bary_point(tax, tay, taz, tbx, tby, tbz, tcx, tcy, tcz, lt0, lt1, lt2)
                    wt = area_t * q7_w[it]

                    j0x = sax - xt
                    j0y = say - yt
                    j0z = saz - zt
                    j0ux = sbx - sax
                    j0uy = sby - say
                    j0uz = sbz - saz
                    j0 = _norm3(
                        j0y * j0uz - j0z * j0uy,
                        j0z * j0ux - j0x * j0uz,
                        j0x * j0uy - j0y * j0ux,
                    )

                    j1x = sbx - xt
                    j1y = sby - yt
                    j1z = sbz - zt
                    j1ux = scx - sbx
                    j1uy = scy - sby
                    j1uz = scz - sbz
                    j1 = _norm3(
                        j1y * j1uz - j1z * j1uy,
                        j1z * j1ux - j1x * j1uz,
                        j1x * j1uy - j1y * j1ux,
                    )

                    j2x = scx - xt
                    j2y = scy - yt
                    j2z = scz - zt
                    j2ux = sax - scx
                    j2uy = say - scy
                    j2uz = saz - scz
                    j2 = _norm3(
                        j2y * j2uz - j2z * j2uy,
                        j2z * j2ux - j2x * j2uz,
                        j2x * j2uy - j2y * j2ux,
                    )

                    for iu in range(gl_x.shape[0]):
                        u = gl_x[iu]
                        wu = gl_w[iu]
                        omu = 1.0 - u
                        for iv in range(gl_x.shape[0]):
                            v = gl_x[iv]
                            wv = gl_w[iv]

                            # subtriangle [x, V0, V1]
                            yx = xt + u * (sax - xt) + u * v * (sbx - sax)
                            yy = yt + u * (say - yt) + u * v * (sby - say)
                            yz = zt + u * (saz - zt) + u * v * (sbz - saz)
                            rx = xt - yx
                            ry = yt - yy
                            rz = zt - yz
                            ustar = _kernel_u(rx, ry, rz)
                            jac = u * j0
                            coefG = wt * wu * wv * jac * ustar
                            ns0 = omu * lt0 + u * (1.0 - v)
                            ns1 = omu * lt1 + u * v
                            ns2 = omu * lt2
                            _add9(G, ti0, ti1, ti2, sj0, sj1, sj2, coefG, lt0, lt1, lt2, ns0, ns1, ns2)

                            # subtriangle [x, V1, V2]
                            yx = xt + u * (sbx - xt) + u * v * (scx - sbx)
                            yy = yt + u * (sby - yt) + u * v * (scy - sby)
                            yz = zt + u * (sbz - zt) + u * v * (scz - sbz)
                            rx = xt - yx
                            ry = yt - yy
                            rz = zt - yz
                            ustar = _kernel_u(rx, ry, rz)
                            jac = u * j1
                            coefG = wt * wu * wv * jac * ustar
                            ns0 = omu * lt0
                            ns1 = omu * lt1 + u * (1.0 - v)
                            ns2 = omu * lt2 + u * v
                            _add9(G, ti0, ti1, ti2, sj0, sj1, sj2, coefG, lt0, lt1, lt2, ns0, ns1, ns2)

                            # subtriangle [x, V2, V0]
                            yx = xt + u * (scx - xt) + u * v * (sax - scx)
                            yy = yt + u * (scy - yt) + u * v * (say - scy)
                            yz = zt + u * (scz - zt) + u * v * (saz - scz)
                            rx = xt - yx
                            ry = yt - yy
                            rz = zt - yz
                            ustar = _kernel_u(rx, ry, rz)
                            jac = u * j2
                            coefG = wt * wu * wv * jac * ustar
                            ns0 = omu * lt0 + u * v
                            ns1 = omu * lt1
                            ns2 = omu * lt2 + u * (1.0 - v)
                            _add9(G, ti0, ti1, ti2, sj0, sj1, sj2, coefG, lt0, lt1, lt2, ns0, ns1, ns2)
                continue

            if dist2 < 20.0 * (area_t + area_s):
                qb = q7_bary
                qw = q7_w
            else:
                qb = q3_bary
                qw = q3_w

            for it in range(qb.shape[0]):
                lt0 = qb[it, 0]
                lt1 = qb[it, 1]
                lt2 = qb[it, 2]
                xt, yt, zt = _bary_point(tax, tay, taz, tbx, tby, tbz, tcx, tcy, tcz, lt0, lt1, lt2)
                wt = area_t * qw[it]

                for isrc in range(qb.shape[0]):
                    ls0 = qb[isrc, 0]
                    ls1 = qb[isrc, 1]
                    ls2 = qb[isrc, 2]
                    ysx, ysy, ysz = _bary_point(sax, say, saz, sbx, sby, sbz, scx, scy, scz, ls0, ls1, ls2)
                    rx = xt - ysx
                    ry = yt - ysy
                    rz = zt - ysz
                    ustar = _kernel_u(rx, ry, rz)
                    qstar = _kernel_q(rx, ry, rz, nx, ny, nz)
                    coef = wt * area_s * qw[isrc]
                    _add9(H, ti0, ti1, ti2, sj0, sj1, sj2, coef * qstar, lt0, lt1, lt2, ls0, ls1, ls2)
                    _add9(G, ti0, ti1, ti2, sj0, sj1, sj2, coef * ustar, lt0, lt1, lt2, ls0, ls1, ls2)

    return H, G


@njit(parallel=True, cache=True, fastmath=True)
def evaluate_interior(points, nodes, elems, areas, normals, u_nodes, q_nodes, q7_bary, q7_w):
    npnts = points.shape[0]
    vals = np.zeros(npnts, dtype=np.float64)

    for p in prange(npnts):
        xp = points[p, 0]
        yp = points[p, 1]
        zp = points[p, 2]
        acc = 0.0

        for e in range(elems.shape[0]):
            i0 = elems[e, 0]
            i1 = elems[e, 1]
            i2 = elems[e, 2]

            ax = nodes[i0, 0]
            ay = nodes[i0, 1]
            az = nodes[i0, 2]
            bx = nodes[i1, 0]
            by = nodes[i1, 1]
            bz = nodes[i1, 2]
            cx = nodes[i2, 0]
            cy = nodes[i2, 1]
            cz = nodes[i2, 2]

            nx = normals[e, 0]
            ny = normals[e, 1]
            nz = normals[e, 2]
            area = areas[e]

            for k in range(q7_bary.shape[0]):
                l0 = q7_bary[k, 0]
                l1 = q7_bary[k, 1]
                l2 = q7_bary[k, 2]
                sx = l0 * ax + l1 * bx + l2 * cx
                sy = l0 * ay + l1 * by + l2 * cy
                sz = l0 * az + l1 * bz + l2 * cz
                rx = xp - sx
                ry = yp - sy
                rz = zp - sz
                ustar = _kernel_u(rx, ry, rz)
                qstar = _kernel_q(rx, ry, rz, nx, ny, nz)
                uy = l0 * u_nodes[i0] + l1 * u_nodes[i1] + l2 * u_nodes[i2]
                qy = l0 * q_nodes[i0] + l1 * q_nodes[i1] + l2 * q_nodes[i2]
                acc += area * q7_w[k] * (qy * ustar - uy * qstar)

        vals[p] = acc

    return vals


def solve_mixed_system(H, G, M, bc_type, u_exact_nodes, q_exact_nodes):
    nn = H.shape[0]
    K = 0.5 * M + H

    is_dir = (bc_type == 0)
    u_known = np.where(is_dir, u_exact_nodes, 0.0)
    q_known = np.where(is_dir, 0.0, q_exact_nodes)

    rhs = -K @ u_known + G @ q_known

    A = np.empty((nn, nn), dtype=np.float64)
    for j in range(nn):
        if is_dir[j]:
            A[:, j] = -G[:, j]
        else:
            A[:, j] = K[:, j]

    sol = np.linalg.solve(A, rhs)

    u_num = u_known.copy()
    q_num = q_known.copy()

    for j in range(nn):
        if is_dir[j]:
            q_num[j] = sol[j]
        else:
            u_num[j] = sol[j]

    return u_num, q_num


def run_case(N):
    t0 = time.perf_counter()
    nodes, elems = generate_continuous_bumpy_mesh(N)
    areas, normals, centroids, nodal_normals = compute_geometry(nodes, elems)
    bc_type, bc_val, u_ex_nodes, q_ex_nodes = assign_mixed_bcs(nodes, nodal_normals)

    M = assemble_mass_matrix(nodes, elems, areas)
    H, G = assemble_galerkin_operators(nodes, elems, areas, centroids, normals, Q3_BARY, Q3_W, Q7_BARY, Q7_W, GL4_X, GL4_W)
    t_setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    u_num, q_num = solve_mixed_system(H, G, M, bc_type, u_ex_nodes, q_ex_nodes)
    t_solve = time.perf_counter() - t1

    grid = np.linspace(-0.7, 0.7, 10)
    pts = np.array([[x, y, z] for x in grid for y in grid for z in grid], dtype=np.float64)

    t2 = time.perf_counter()
    u_int = evaluate_interior(pts, nodes, elems, areas, normals, u_num, q_num, Q7_BARY, Q7_W)
    t_eval = time.perf_counter() - t2

    u_exact_int = exact_u(pts[:, 0], pts[:, 1], pts[:, 2])
    rel_l2 = np.linalg.norm(u_int - u_exact_int) / np.linalg.norm(u_exact_int)

    total = time.perf_counter() - t0
    return nodes.shape[0], elems.shape[0], rel_l2, t_setup, t_solve, t_eval, total


def main():
    Ns = [8, 16, 32]
    results = []
    for N in Ns:
        nn, ne, err, setup, solve, evalt, total = run_case(N)
        results.append((N, ne, nn, err, setup, solve, evalt, total))

    print("N    | Ne      | N_nodes | Rel L2 Error   | Setup (s) | Solve (s) | Eval (s) | Total (s)")
    print()
    for N, ne, nn, err, setup, solve, evalt, total in results:
        print(f"{N:<4d} | {ne:<7d} | {nn:<7d} | {err:0.6e}   | {setup:0.4f}    | {solve:0.4f}    | {evalt:0.4f}   | {total:0.4f}")

    h4 = 1.0 / results[0][0]
    h16 = 1.0 / results[2][0]
    e4 = results[0][3]
    e16 = results[2][3]
    slope = math.log(e4 / e16) / math.log(h4 / h16)

    print()
    print(f"Computed Slope: {slope:0.4f}")
    print("Expected Slope: ~2.0000 (O(h^2) for linear elements)")


if __name__ == "__main__":
    main()