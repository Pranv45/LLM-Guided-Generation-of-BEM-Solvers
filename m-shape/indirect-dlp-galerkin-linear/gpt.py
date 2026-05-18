import gc
import time
import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import gmres

np.set_printoptions(precision=6, suppress=True)

VERTICES = np.array(
    [
        (-1.5,  1.0),
        (-0.75, 1.0),
        ( 0.0,   0.4),
        ( 0.75,  1.0),
        ( 1.5,   1.0),
        ( 1.5,   0.0),
        ( 1.5,  -1.0),
        ( 0.75, -1.0),
        ( 0.75,  0.3),
        ( 0.0,  -0.3),
        (-0.75,  0.3),
        (-0.75, -1.0),
        (-1.5,  -1.0),
        (-1.5,   0.0),
    ],
    dtype=np.float64,
)

N_VALUES = [400, 800, 1600, 3200, 6400]
NGRID = 200
NQ = 8
PI2 = 2.0 * np.pi


def u_exact_xy(x, y):
    return x**3 - 3.0 * x * y**2


def polygon_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def distribute_segments(lengths, total_segments):
    nv = len(lengths)
    if total_segments < nv:
        raise ValueError("total_segments must be >= number of polygon vertices")

    remaining = total_segments - nv
    if remaining == 0:
        return np.ones(nv, dtype=np.int64)

    frac = lengths / lengths.sum()
    extra = remaining * frac
    add = np.floor(extra).astype(np.int64)
    counts = np.ones(nv, dtype=np.int64) + add
    rem = remaining - int(add.sum())
    if rem > 0:
        order = np.argsort(-(extra - add))
        counts[order[:rem]] += 1
    return counts


def build_mesh(vertices, total_elements):
    verts = np.array(vertices, dtype=np.float64, copy=True)
    if polygon_area(verts) < 0.0:
        verts = verts[::-1].copy()

    nv = verts.shape[0]
    edge_lengths = np.array(
        [np.linalg.norm(verts[(k + 1) % nv] - verts[k]) for k in range(nv)],
        dtype=np.float64,
    )
    counts = distribute_segments(edge_lengths, total_elements)

    nodes = [verts[i].copy() for i in range(nv)]
    elements = []
    for k in range(nv):
        m = int(counts[k])
        p0 = verts[k]
        p1 = verts[(k + 1) % nv]

        seq = [k]
        for r in range(1, m):
            t = 0.5 * (1.0 - np.cos(np.pi * r / m))
            pt = (1.0 - t) * p0 + t * p1
            nodes.append(pt.copy())
            seq.append(len(nodes) - 1)
        seq.append((k + 1) % nv)

        for a, b in zip(seq[:-1], seq[1:]):
            elements.append((a, b))

    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)

    elem_start = nodes[elements[:, 0]]
    elem_end = nodes[elements[:, 1]]
    elem_vec = elem_end - elem_start
    elem_len = np.sqrt(np.sum(elem_vec**2, axis=1))
    elem_normals = np.column_stack((elem_vec[:, 1], -elem_vec[:, 0])) / elem_len[:, None]
    elem_c = np.sum(elem_normals * elem_start, axis=1)

    return nodes, elements, elem_start, elem_vec, elem_len, elem_normals, elem_c, verts


def points_in_polygon(points, vertices):
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=np.bool_)

    x0 = vertices[-1, 0]
    y0 = vertices[-1, 1]
    for k in range(vertices.shape[0]):
        x1 = vertices[k, 0]
        y1 = vertices[k, 1]
        cond = ((y0 > y) != (y1 > y)) & (
            x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-300) + x0
        )
        inside ^= cond
        x0, y0 = x1, y1
    return inside


def min_dist_to_boundary(points, vertices):
    dmin = np.full(points.shape[0], np.inf, dtype=np.float64)
    for k in range(vertices.shape[0]):
        a = vertices[k]
        b = vertices[(k + 1) % vertices.shape[0]]
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        ap = points - a
        t = np.clip((ap @ ab) / ab2, 0.0, 1.0)
        proj = a + t[:, None] * ab[None, :]
        d = np.sqrt(np.sum((points - proj) ** 2, axis=1))
        dmin = np.minimum(dmin, d)
    return dmin


def gauss_01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def assemble_mass_and_rhs(nodes, elements, qx, qw):
    n = nodes.shape[0]
    A = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)

    for e in range(elements.shape[0]):
        a = int(elements[e, 0])
        b = int(elements[e, 1])
        p0 = nodes[a]
        p1 = nodes[b]
        vec = p1 - p0
        L = float(np.sqrt(np.dot(vec, vec)))

        A[a, a] += -L / 6.0
        A[a, b] += -L / 12.0
        A[b, a] += -L / 12.0
        A[b, b] += -L / 6.0

        for k in range(qx.shape[0]):
            s = float(qx[k])
            w = float(qw[k])
            x = p0 + s * vec
            g = u_exact_xy(x[0], x[1])
            phi0 = 1.0 - s
            phi1 = s
            rhs[a] += w * L * phi0 * g
            rhs[b] += w * L * phi1 * g

    return A, rhs


@njit(fastmath=True)
def same_line(ex, ey, elem_normals, elem_c):
    n1x = elem_normals[ex, 0]
    n1y = elem_normals[ex, 1]
    n2x = elem_normals[ey, 0]
    n2y = elem_normals[ey, 1]
    c1 = elem_c[ex]
    c2 = elem_c[ey]
    tol = 1e-13

    if abs(n1x - n2x) < tol and abs(n1y - n2y) < tol and abs(c1 - c2) < tol:
        return True
    if abs(n1x + n2x) < tol and abs(n1y + n2y) < tol and abs(c1 + c2) < tol:
        return True
    return False


@njit(fastmath=True)
def shape_from_shared(local_pos, shared_local_pos, s):
    if local_pos == shared_local_pos:
        return 1.0 - s
    return s


@njit(fastmath=True)
def pair_disjoint_scalar(ex, ey, lx, ly, nodes, elements, elem_start, elem_vec, elem_len, elem_normals, qx, qw):
    p0x = elem_start[ex, 0]
    p0y = elem_start[ex, 1]
    vx = elem_vec[ex, 0]
    vy = elem_vec[ex, 1]
    q0x = elem_start[ey, 0]
    q0y = elem_start[ey, 1]
    ux = elem_vec[ey, 0]
    uy = elem_vec[ey, 1]
    nyx = elem_normals[ey, 0]
    nyy = elem_normals[ey, 1]
    Lx = elem_len[ex]
    Ly = elem_len[ey]

    acc = 0.0
    nq = qx.shape[0]
    for iu in range(nq):
        sx = qx[iu]
        wx = qw[iu]
        phi_x = 1.0 - sx if lx == 0 else sx
        px = p0x + sx * vx
        py = p0y + sx * vy
        for iv in range(nq):
            sy = qx[iv]
            wy = qw[iv]
            phi_y = 1.0 - sy if ly == 0 else sy
            qxv = q0x + sy * ux
            qyv = q0y + sy * uy
            dx = px - qxv
            dy = py - qyv
            r2 = dx * dx + dy * dy
            kern = (dx * nyx + dy * nyy) / (PI2 * r2)
            acc += wx * wy * phi_x * phi_y * kern * Lx * Ly
    return acc


@njit(fastmath=True)
def pair_duffy_scalar(ex, ey, lx, ly, nodes, elements, elem_normals, qx, qw):
    a0 = elements[ex, 0]
    a1 = elements[ex, 1]
    b0 = elements[ey, 0]
    b1 = elements[ey, 1]

    shared = -1
    if a0 == b0 or a0 == b1:
        shared = a0
    elif a1 == b0 or a1 == b1:
        shared = a1

    if shared < 0:
        return 0.0

    shared_x = nodes[shared, 0]
    shared_y = nodes[shared, 1]

    if a0 == shared:
        ex_other = a1
        shared_loc_x = 0
    else:
        ex_other = a0
        shared_loc_x = 1

    if b0 == shared:
        ey_other = b1
        shared_loc_y = 0
    else:
        ey_other = b0
        shared_loc_y = 1

    vx = nodes[ex_other, 0] - shared_x
    vy = nodes[ex_other, 1] - shared_y
    ux = nodes[ey_other, 0] - shared_x
    uy = nodes[ey_other, 1] - shared_y

    nyx = elem_normals[ey, 0]
    nyy = elem_normals[ey, 1]
    Lx = np.sqrt(vx * vx + vy * vy)
    Ly = np.sqrt(ux * ux + uy * uy)

    acc = 0.0
    nq = qx.shape[0]
    for iu in range(nq):
        u = qx[iu]
        wu = qw[iu]
        for iv in range(nq):
            v = qx[iv]
            wv = qw[iv]

            sx = u
            sy = u * v
            x1 = shared_x + sx * vx
            y1 = shared_y + sx * vy
            qx1 = shared_x + sy * ux
            qy1 = shared_y + sy * uy
            phi_x = shape_from_shared(lx, shared_loc_x, sx)
            phi_y = shape_from_shared(ly, shared_loc_y, sy)
            dx = x1 - qx1
            dy = y1 - qy1
            r2 = dx * dx + dy * dy
            kern = (dx * nyx + dy * nyy) / (PI2 * r2)
            acc += wu * wv * phi_x * phi_y * kern * u * Lx * Ly

            sx = u * v
            sy = u
            x1 = shared_x + sx * vx
            y1 = shared_y + sx * vy
            qx1 = shared_x + sy * ux
            qy1 = shared_y + sy * uy
            phi_x = shape_from_shared(lx, shared_loc_x, sx)
            phi_y = shape_from_shared(ly, shared_loc_y, sy)
            dx = x1 - qx1
            dy = y1 - qy1
            r2 = dx * dx + dy * dy
            kern = (dx * nyx + dy * nyy) / (PI2 * r2)
            acc += wu * wv * phi_x * phi_y * kern * u * Lx * Ly

    return acc


@njit(parallel=True, fastmath=True)
def add_double_layer_matrix(A, nodes, elements, elem_start, elem_vec, elem_len, elem_normals, elem_c, node_support_elems, node_support_lpos, qx, qw):
    n = nodes.shape[0]
    for i in prange(n):
        ei0 = node_support_elems[i, 0]
        ei1 = node_support_elems[i, 1]
        li0 = node_support_lpos[i, 0]
        li1 = node_support_lpos[i, 1]
        for j in range(n):
            ej0 = node_support_elems[j, 0]
            ej1 = node_support_elems[j, 1]
            lj0 = node_support_lpos[j, 0]
            lj1 = node_support_lpos[j, 1]

            val = 0.0

            ex = ei0
            ey = ej0
            if not same_line(ex, ey, elem_normals, elem_c):
                a0 = elements[ex, 0]
                a1 = elements[ex, 1]
                b0 = elements[ey, 0]
                b1 = elements[ey, 1]
                shared = (a0 == b0) or (a0 == b1) or (a1 == b0) or (a1 == b1)
                if shared:
                    val += pair_duffy_scalar(ex, ey, li0, lj0, nodes, elements, elem_normals, qx, qw)
                else:
                    val += pair_disjoint_scalar(ex, ey, li0, lj0, nodes, elements, elem_start, elem_vec, elem_len, elem_normals, qx, qw)

            ex = ei0
            ey = ej1
            if not same_line(ex, ey, elem_normals, elem_c):
                a0 = elements[ex, 0]
                a1 = elements[ex, 1]
                b0 = elements[ey, 0]
                b1 = elements[ey, 1]
                shared = (a0 == b0) or (a0 == b1) or (a1 == b0) or (a1 == b1)
                if shared:
                    val += pair_duffy_scalar(ex, ey, li0, lj1, nodes, elements, elem_normals, qx, qw)
                else:
                    val += pair_disjoint_scalar(ex, ey, li0, lj1, nodes, elements, elem_start, elem_vec, elem_len, elem_normals, qx, qw)

            ex = ei1
            ey = ej0
            if not same_line(ex, ey, elem_normals, elem_c):
                a0 = elements[ex, 0]
                a1 = elements[ex, 1]
                b0 = elements[ey, 0]
                b1 = elements[ey, 1]
                shared = (a0 == b0) or (a0 == b1) or (a1 == b0) or (a1 == b1)
                if shared:
                    val += pair_duffy_scalar(ex, ey, li1, lj0, nodes, elements, elem_normals, qx, qw)
                else:
                    val += pair_disjoint_scalar(ex, ey, li1, lj0, nodes, elements, elem_start, elem_vec, elem_len, elem_normals, qx, qw)

            ex = ei1
            ey = ej1
            if not same_line(ex, ey, elem_normals, elem_c):
                a0 = elements[ex, 0]
                a1 = elements[ex, 1]
                b0 = elements[ey, 0]
                b1 = elements[ey, 1]
                shared = (a0 == b0) or (a0 == b1) or (a1 == b0) or (a1 == b1)
                if shared:
                    val += pair_duffy_scalar(ex, ey, li1, lj1, nodes, elements, elem_normals, qx, qw)
                else:
                    val += pair_disjoint_scalar(ex, ey, li1, lj1, nodes, elements, elem_start, elem_vec, elem_len, elem_normals, qx, qw)

            A[i, j] += val


@njit(parallel=True, fastmath=True)
def evaluate_potential(points, nodes, elements, elem_normals, mu, qx, qw):
    m = points.shape[0]
    ne = elements.shape[0]
    nq = qx.shape[0]
    out = np.zeros(m, dtype=np.float64)

    for p in prange(m):
        xp = points[p, 0]
        yp = points[p, 1]
        acc = 0.0

        for e in range(ne):
            a = elements[e, 0]
            b = elements[e, 1]
            p0x = nodes[a, 0]
            p0y = nodes[a, 1]
            vx = nodes[b, 0] - p0x
            vy = nodes[b, 1] - p0y
            nyx = elem_normals[e, 0]
            nyy = elem_normals[e, 1]
            mu0 = mu[a]
            mu1 = mu[b]
            L = np.sqrt(vx * vx + vy * vy)

            for k in range(nq):
                s = qx[k]
                w = qw[k]
                yx = p0x + s * vx
                yy = p0y + s * vy
                dx = xp - yx
                dy = yp - yy
                r2 = dx * dx + dy * dy
                kern = (dx * nyx + dy * nyy) / (PI2 * r2)
                mus = (1.0 - s) * mu0 + s * mu1
                acc += w * L * kern * mus

        out[p] = acc

    return out


def main():
    qx, qw = gauss_01(NQ)

    xs = np.linspace(-1.5, 1.5, NGRID)
    ys = np.linspace(-1.0, 1.0, NGRID)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])

    interior_mask = points_in_polygon(all_pts, VERTICES)
    interior_pts = all_pts[interior_mask]

    perimeter = np.sum(np.linalg.norm(np.roll(VERTICES, -1, axis=0) - VERTICES, axis=1))
    N_min = min(N_VALUES)
    h_coarse = perimeter / N_min
    delta = 2.0 * h_coarse

    dist = min_dist_to_boundary(interior_pts, VERTICES)
    grid_pts = interior_pts[dist > delta]
    uex_grid = u_exact_xy(grid_pts[:, 0], grid_pts[:, 1])

    print(f"{'N':>6} {'Unknowns':>10} {'GMRES':>8} {'RelL2':>14} {'Setup(s)':>10} {'Solve(s)':>10} {'Eval(s)':>10} {'Total(s)':>10}")

    errors = []

    for N in N_VALUES:
        t0 = time.perf_counter()

        nodes, elements, elem_start, elem_vec, elem_len, elem_normals, elem_c, verts = build_mesh(VERTICES, N)

        n_nodes = nodes.shape[0]
        node_support_elems = -np.ones((n_nodes, 2), dtype=np.int64)
        node_support_lpos = -np.ones((n_nodes, 2), dtype=np.int64)
        fill = np.zeros(n_nodes, dtype=np.int64)
        for e in range(elements.shape[0]):
            a = int(elements[e, 0])
            b = int(elements[e, 1])

            s = fill[a]
            node_support_elems[a, s] = e
            node_support_lpos[a, s] = 0
            fill[a] += 1

            s = fill[b]
            node_support_elems[b, s] = e
            node_support_lpos[b, s] = 1
            fill[b] += 1

        A, rhs = assemble_mass_and_rhs(nodes, elements, qx, qw)
        add_double_layer_matrix(A, nodes, elements, elem_start, elem_vec, elem_len, elem_normals, elem_c, node_support_elems, node_support_lpos, qx, qw)

        setup_time = time.perf_counter() - t0

        iters = [0]

        def cb(_):
            iters[0] += 1

        t1 = time.perf_counter()
        mu, info = gmres(
            A,
            rhs,
            rtol=1e-10,
            atol=1e-10,
            restart=n_nodes,
            maxiter=1,
            callback=cb,
            callback_type="pr_norm",
        )
        solve_time = time.perf_counter() - t1

        t2 = time.perf_counter()
        u_num = evaluate_potential(grid_pts, nodes, elements, elem_normals, mu, qx, qw)
        eval_time = time.perf_counter() - t2

        rel_err = np.linalg.norm(u_num - uex_grid) / np.linalg.norm(uex_grid)
        total_time = setup_time + solve_time + eval_time
        errors.append(rel_err)

        print(f"{N:6d} {n_nodes:10d} {iters[0]:8d} {rel_err:14.6e} {setup_time:10.4f} {solve_time:10.4f} {eval_time:10.4f} {total_time:10.4f}")

        del nodes, elements, elem_start, elem_vec, elem_len, elem_normals, elem_c, node_support_elems, node_support_lpos, A, rhs, mu, u_num
        gc.collect()

    log_h = np.log(1.0 / np.array(N_VALUES, dtype=np.float64))
    log_err = np.log(np.array(errors, dtype=np.float64))
    slope, _ = np.polyfit(log_h, log_err, 1)
    print(f"{'Estimated convergence order':>28} = {slope:.6f}")


if __name__ == "__main__":
    main()