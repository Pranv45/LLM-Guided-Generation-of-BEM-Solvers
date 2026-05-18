import time
import numpy as np
from numba import njit, prange

PI = np.pi


def u_exact(x, y):
    return x**3 - 3.0 * x * y**2


def q_exact(x, y, nx, ny):
    ux = 3.0 * x**2 - 3.0 * y**2
    uy = -6.0 * x * y
    return ux * nx + uy * ny


def camber_and_slope(x, m, p):
    yc = np.zeros_like(x, dtype=np.float64)
    dyc = np.zeros_like(x, dtype=np.float64)

    left = x < p
    right = ~left

    if p > 0.0:
        yc[left] = m / (p * p) * (2.0 * p * x[left] - x[left] ** 2)
        dyc[left] = 2.0 * m / (p * p) * (p - x[left])

    if p < 1.0:
        yc[right] = m / ((1.0 - p) ** 2) * (
            (1.0 - 2.0 * p) + 2.0 * p * x[right] - x[right] ** 2
        )
        dyc[right] = 2.0 * m / ((1.0 - p) ** 2) * (p - x[right])

    return yc, dyc


def thickness(x, t):
    return 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )


def generate_naca4318_nodes(N):
    N = int(N)
    if N < 4:
        raise ValueError("N must be at least 4")

    m, p, t = 0.04, 0.3, 0.18

    n_upper = N // 2 + 1
    n_lower = N - n_upper + 1

    s_upper = np.linspace(0.0, 1.0, n_upper)
    s_lower = np.linspace(0.0, 1.0, n_lower)

    x_upper = 0.5 * (1.0 + np.cos(np.pi * s_upper))  # TE -> LE
    x_lower = 0.5 * (1.0 - np.cos(np.pi * s_lower))  # LE -> TE

    yc_u, dyc_u = camber_and_slope(x_upper, m, p)
    yc_l, dyc_l = camber_and_slope(x_lower, m, p)

    yt_u = thickness(x_upper, t)
    yt_l = thickness(x_lower, t)

    th_u = np.arctan(dyc_u)
    th_l = np.arctan(dyc_l)

    upper = np.column_stack(
        [x_upper - yt_u * np.sin(th_u), yc_u + yt_u * np.cos(th_u)]
    )
    lower = np.column_stack(
        [x_lower + yt_l * np.sin(th_l), yc_l - yt_l * np.cos(th_l)]
    )

    nodes = np.vstack([upper, lower[1:]])  # double node at TE

    x = nodes[:, 0]
    y = nodes[:, 1]
    area2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area2 < 0.0:
        nodes = nodes[::-1].copy()

    return nodes.astype(np.float64)


def build_geometry(nodes):
    N = nodes.shape[0]
    elements = np.column_stack([np.arange(N - 1), np.arange(1, N)]).astype(np.int64)

    a = nodes[elements[:, 0]]
    b = nodes[elements[:, 1]]

    dx = b[:, 0] - a[:, 0]
    dy = b[:, 1] - a[:, 1]
    lengths = np.sqrt(dx * dx + dy * dy)
    if np.any(lengths <= 0.0):
        raise ValueError("Zero-length element encountered.")

    half = 0.5 * lengths
    tx = dx / lengths
    ty = dy / lengths
    normals = np.column_stack([dy / lengths, -dx / lengths])

    node_normals = np.zeros_like(nodes)
    node_normals[0] = normals[0]
    node_normals[-1] = normals[-1]
    for i in range(1, N - 1):
        v = normals[i - 1] + normals[i]
        nv = np.linalg.norm(v)
        if nv < 1e-14:
            v = normals[i]
            nv = np.linalg.norm(v)
        node_normals[i] = v / nv

    node_bcs = (nodes[:, 0] < 0.8).astype(np.int8)  # 1 = Neumann, 0 = Dirichlet

    return elements, lengths, half, tx, ty, normals, node_normals, node_bcs


def gauss_rule_unit(order=8):
    xi, wi = np.polynomial.legendre.leggauss(order)
    xi = 0.5 * (xi + 1.0)
    wi = 0.5 * wi
    return xi.astype(np.float64), wi.astype(np.float64)


def build_support_maps(N):
    support_elems = -np.ones((N, 2), dtype=np.int64)
    support_locs = -np.ones((N, 2), dtype=np.int64)

    support_elems[0, 0] = 0
    support_locs[0, 0] = 0

    support_elems[-1, 0] = N - 2
    support_locs[-1, 0] = 1

    for i in range(1, N - 1):
        support_elems[i, 0] = i - 1
        support_locs[i, 0] = 1
        support_elems[i, 1] = i
        support_locs[i, 1] = 0

    return support_elems, support_locs


@njit(inline="always")
def _basis_from_s(s):
    return 1.0 - s, s


@njit(inline="always")
def _basis_from_a(a, singular_is_first):
    if singular_is_first:
        return 1.0 - a, a
    else:
        return a, 1.0 - a


@njit(inline="always")
def _pair_block_same(L):
    Hloc = np.zeros((2, 2), dtype=np.float64)
    Gloc = np.zeros((2, 2), dtype=np.float64)

    c = -(L * L) / (2.0 * np.pi)
    logL = np.log(L)

    # Analytical double integrals for log|s-t| with linear shape functions
    I11 = -25.0 / 72.0
    I12 = -19.0 / 72.0
    I22 = -45.0 / 72.0

    Gloc[0, 0] = c * (0.25 * logL + I11)
    Gloc[0, 1] = c * (0.25 * logL + I12)
    Gloc[1, 0] = c * (0.25 * logL + I12)
    Gloc[1, 1] = c * (0.25 * logL + I22)

    return Hloc, Gloc


@njit(inline="always")
def _pair_block_disjoint(x0, x1, y0, y1, nx_y, ny_y, Lx, Ly, xi, wi):
    Hloc = np.zeros((2, 2), dtype=np.float64)
    Gloc = np.zeros((2, 2), dtype=np.float64)

    nq = xi.shape[0]
    for a in range(nq):
        s = xi[a]
        wx = wi[a]
        phi_x0, phi_x1 = _basis_from_s(s)

        px = x0[0] * phi_x0 + x1[0] * phi_x1
        py = x0[1] * phi_x0 + x1[1] * phi_x1

        for b in range(nq):
            t = xi[b]
            wy = wi[b]
            phi_y0, phi_y1 = _basis_from_s(t)

            qx = y0[0] * phi_y0 + y1[0] * phi_y1
            qy = y0[1] * phi_y0 + y1[1] * phi_y1

            rx = px - qx
            ry = py - qy
            r2 = rx * rx + ry * ry
            r = np.sqrt(r2)

            ustar = -(1.0 / (2.0 * np.pi)) * np.log(r)
            qstar = (1.0 / (2.0 * np.pi)) * ((rx * nx_y + ry * ny_y) / r2)

            w = Lx * Ly * wx * wy

            Gloc[0, 0] += w * ustar * phi_x0 * phi_y0
            Gloc[0, 1] += w * ustar * phi_x0 * phi_y1
            Gloc[1, 0] += w * ustar * phi_x1 * phi_y0
            Gloc[1, 1] += w * ustar * phi_x1 * phi_y1

            Hloc[0, 0] += w * qstar * phi_x0 * phi_y0
            Hloc[0, 1] += w * qstar * phi_x0 * phi_y1
            Hloc[1, 0] += w * qstar * phi_x1 * phi_y0
            Hloc[1, 1] += w * qstar * phi_x1 * phi_y1

    return Hloc, Gloc


@njit(inline="always")
def _pair_block_adjacent(vs, vx, vy, sing_x_first, sing_y_first, nx_y, ny_y, Lx, Ly, xi, wi):
    Hloc = np.zeros((2, 2), dtype=np.float64)
    Gloc = np.zeros((2, 2), dtype=np.float64)

    nq = xi.shape[0]

    for region in range(2):
        for a in range(nq):
            r = xi[a]
            wr = wi[a]
            for b in range(nq):
                v = xi[b]
                wv = wi[b]

                if region == 0:
                    aa = r
                    bb = r * v
                else:
                    aa = r * v
                    bb = r

                phi_x0, phi_x1 = _basis_from_a(aa, sing_x_first)
                phi_y0, phi_y1 = _basis_from_a(bb, sing_y_first)

                px = vs[0] + aa * vx[0]
                py = vs[1] + aa * vx[1]
                qx = vs[0] + bb * vy[0]
                qy = vs[1] + bb * vy[1]

                rx = px - qx
                ry = py - qy
                r2 = rx * rx + ry * ry
                rnorm = np.sqrt(r2)

                ustar = -(1.0 / (2.0 * np.pi)) * np.log(rnorm)
                qstar = (1.0 / (2.0 * np.pi)) * ((rx * nx_y + ry * ny_y) / r2)

                w = Lx * Ly * wr * wv * r

                Gloc[0, 0] += w * ustar * phi_x0 * phi_y0
                Gloc[0, 1] += w * ustar * phi_x0 * phi_y1
                Gloc[1, 0] += w * ustar * phi_x1 * phi_y0
                Gloc[1, 1] += w * ustar * phi_x1 * phi_y1

                Hloc[0, 0] += w * qstar * phi_x0 * phi_y0
                Hloc[0, 1] += w * qstar * phi_x0 * phi_y1
                Hloc[1, 0] += w * qstar * phi_x1 * phi_y0
                Hloc[1, 1] += w * qstar * phi_x1 * phi_y1

    return Hloc, Gloc


@njit(inline="always")
def _pair_block(nodes, elements, lengths, ex, ey, normals, xi, wi):
    Hloc = np.zeros((2, 2), dtype=np.float64)
    Gloc = np.zeros((2, 2), dtype=np.float64)

    if ex == ey:
        Hloc, Gloc = _pair_block_same(lengths[ex])
        return Hloc, Gloc

    ax = elements[ex, 0]
    bx = elements[ex, 1]
    ay = elements[ey, 0]
    by = elements[ey, 1]

    x0 = nodes[ax]
    x1 = nodes[bx]
    y0 = nodes[ay]
    y1 = nodes[by]

    if abs(ex - ey) == 1:
        vs = nodes[bx] if ex < ey else nodes[ax]

        if ex < ey:
            vx = nodes[ax] - vs
            vy = nodes[by] - vs
            sing_x_first = False  # singular at element ex local node 1
            sing_y_first = True   # singular at element ey local node 0
        else:
            vx = nodes[bx] - vs
            vy = nodes[ay] - vs
            sing_x_first = True   # singular at element ex local node 0
            sing_y_first = False  # singular at element ey local node 1

        return _pair_block_adjacent(
            vs,
            vx,
            vy,
            sing_x_first,
            sing_y_first,
            normals[ey, 0],
            normals[ey, 1],
            lengths[ex],
            lengths[ey],
            xi,
            wi,
        )

    return _pair_block_disjoint(
        x0,
        x1,
        y0,
        y1,
        normals[ey, 0],
        normals[ey, 1],
        lengths[ex],
        lengths[ey],
        xi,
        wi,
    )


@njit(parallel=True)
def assemble_HG_galerkin(nodes, elements, lengths, normals, support_elems, support_locs, xi, wi):
    N = nodes.shape[0]
    H = np.zeros((N, N), dtype=np.float64)
    G = np.zeros((N, N), dtype=np.float64)

    for i in prange(N):
        for j in range(N):
            gval = 0.0
            hval = 0.0

            for pi in range(2):
                ex = support_elems[i, pi]
                if ex < 0:
                    continue
                lix = support_locs[i, pi]

                for pj in range(2):
                    ey = support_elems[j, pj]
                    if ey < 0:
                        continue
                    ljy = support_locs[j, pj]

                    Hloc, Gloc = _pair_block(nodes, elements, lengths, ex, ey, normals, xi, wi)

                    gval += Gloc[lix, ljy]
                    if i != j:
                        hval += Hloc[lix, ljy]

            G[i, j] = gval
            if i != j:
                H[i, j] = hval

        s = 0.0
        for j in range(N):
            if j != i:
                s += H[i, j]
        H[i, i] = -s

    return H, G


@njit(parallel=True)
def evaluate_interior(points, nodes, elements, lengths, normals, u_nodes, q_nodes, xi, wi):
    npnt = points.shape[0]
    Ne = elements.shape[0]
    nq = xi.shape[0]
    out = np.zeros(npnt, dtype=np.float64)

    for p in prange(npnt):
        x = points[p, 0]
        y = points[p, 1]
        val = 0.0

        for e in range(Ne):
            a = elements[e, 0]
            b = elements[e, 1]

            ax = nodes[a, 0]
            ay = nodes[a, 1]
            bx = nodes[b, 0]
            by = nodes[b, 1]

            ua = u_nodes[a]
            ub = u_nodes[b]
            qa = q_nodes[a]
            qb = q_nodes[b]

            integ = 0.0
            for k in range(nq):
                s = xi[k]
                w = wi[k]

                phi1 = 1.0 - s
                phi2 = s

                px = ax * phi1 + bx * phi2
                py = ay * phi1 + by * phi2

                rx = x - px
                ry = y - py
                r2 = rx * rx + ry * ry
                r = np.sqrt(r2)

                ustar = -(1.0 / (2.0 * np.pi)) * np.log(r)
                qstar = (1.0 / (2.0 * np.pi)) * ((rx * normals[e, 0] + ry * normals[e, 1]) / r2)

                uval = ua * phi1 + ub * phi2
                qval = qa * phi1 + qb * phi2

                integ += w * (ustar * qval - qstar * uval)

            val += lengths[e] * integ

        out[p] = val

    return out


@njit(parallel=True)
def point_in_polygon(points, poly):
    n = poly.shape[0]
    m = points.shape[0]
    inside = np.zeros(m, dtype=np.bool_)

    for i in prange(m):
        x = points[i, 0]
        y = points[i, 1]
        c = False

        xj = poly[n - 1, 0]
        yj = poly[n - 1, 1]

        for k in range(n):
            xi = poly[k, 0]
            yi = poly[k, 1]

            if ((yi > y) != (yj > y)):
                xint = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
                if x < xint:
                    c = not c

            xj = xi
            yj = yi

        inside[i] = c

    return inside


@njit(parallel=True)
def min_dist_to_segments(points, ax, ay, bx, by):
    m = points.shape[0]
    ne = ax.shape[0]
    out = np.empty(m, dtype=np.float64)

    for i in prange(m):
        px = points[i, 0]
        py = points[i, 1]
        dmin = 1e300

        for e in range(ne):
            x1 = ax[e]
            y1 = ay[e]
            x2 = bx[e]
            y2 = by[e]

            vx = x2 - x1
            vy = y2 - y1
            wx = px - x1
            wy = py - y1

            c1 = wx * vx + wy * vy
            if c1 <= 0.0:
                dx = px - x1
                dy = py - y1
                d = np.sqrt(dx * dx + dy * dy)
            else:
                c2 = vx * vx + vy * vy
                if c1 >= c2:
                    dx = px - x2
                    dy = py - y2
                    d = np.sqrt(dx * dx + dy * dy)
                else:
                    b = c1 / c2
                    projx = x1 + b * vx
                    projy = y1 + b * vy
                    dx = px - projx
                    dy = py - projy
                    d = np.sqrt(dx * dx + dy * dy)

            if d < dmin:
                dmin = d

        out[i] = dmin

    return out


def warm_up_numba():
    nodes = np.array(
        [[1.0, 0.0], [0.5, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float64
    )
    elements = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    lengths = np.array([0.5, 0.5, 1.0], dtype=np.float64)
    normals = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    support_elems = np.array([[0, -1], [0, 1], [1, 2], [2, -1]], dtype=np.int64)
    support_locs = np.array([[0, -1], [1, 0], [1, 0], [1, -1]], dtype=np.int64)
    xi, wi = gauss_rule_unit(4)

    _ = assemble_HG_galerkin(nodes, elements, lengths, normals, support_elems, support_locs, xi, wi)

    pts = np.array([[0.2, 0.1]], dtype=np.float64)
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    _ = point_in_polygon(pts, poly)

    ax = np.array([0.0, 1.0], dtype=np.float64)
    ay = np.array([0.0, 0.0], dtype=np.float64)
    bx = np.array([1.0, 1.0], dtype=np.float64)
    by = np.array([0.0, 1.0], dtype=np.float64)
    _ = min_dist_to_segments(pts, ax, ay, bx, by)

    u_nodes = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    q_nodes = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    _ = evaluate_interior(pts, nodes, elements, lengths, normals, u_nodes, q_nodes, xi, wi)


def run_case(N, xi, wi, grid_n=200, N_values_ref=(100, 200, 400, 800)):
    t0 = time.perf_counter()

    nodes = generate_naca4318_nodes(N)
    elements, lengths, half, tx, ty, normals, node_normals, node_bcs = build_geometry(nodes)
    support_elems, support_locs = build_support_maps(N)

    u_known = u_exact(nodes[:, 0], nodes[:, 1])
    q_known = q_exact(nodes[:, 0], nodes[:, 1], node_normals[:, 0], node_normals[:, 1])

    H, G = assemble_HG_galerkin(nodes, elements, lengths, normals, support_elems, support_locs, xi, wi)

    t1 = time.perf_counter()

    A = np.empty_like(H)
    b = np.zeros(N, dtype=np.float64)

    for j in range(N):
        if node_bcs[j] == 0:
            A[:, j] = -G[:, j]
            b -= H[:, j] * u_known[j]
        else:
            A[:, j] = H[:, j]
            b += G[:, j] * q_known[j]

    xsol = np.linalg.solve(A, b)

    t2 = time.perf_counter()

    u_nodes = np.where(node_bcs == 0, u_known, xsol)
    q_nodes = np.where(node_bcs == 1, q_known, xsol)

    xs = np.linspace(-0.1, 1.1, grid_n)
    ys = np.linspace(-0.2, 0.2, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])

    interior = point_in_polygon(all_pts, nodes)
    interior_pts = all_pts[interior]

    N_min = min(N_values_ref)
    perimeter = np.sum(lengths)
    h_coarse = perimeter / N_min
    delta = 2.0 * h_coarse

    ax = nodes[elements[:, 0], 0]
    ay = nodes[elements[:, 0], 1]
    bx = nodes[elements[:, 1], 0]
    by = nodes[elements[:, 1], 1]

    dist = min_dist_to_segments(interior_pts, ax, ay, bx, by)
    grid_pts = interior_pts[dist > delta]

    u_num = evaluate_interior(
        grid_pts, nodes, elements, lengths, normals, u_nodes, q_nodes, xi, wi
    )
    u_ex = u_exact(grid_pts[:, 0], grid_pts[:, 1])

    rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)

    t3 = time.perf_counter()

    return {
        "N": N,
        "unknowns": N,
        "rel_l2": rel_l2,
        "assembly": t1 - t0,
        "solve": t2 - t1,
        "evaluation": t3 - t2,
        "total": t3 - t0,
    }


def main():
    warm_up_numba()

    N_values = [400, 800, 1600, 3200, 6400]
    xi, wi = gauss_rule_unit(8)

    results = []
    for N in N_values:
        results.append(run_case(N, xi, wi, grid_n=200, N_values_ref=N_values))

    hs = 1.0 / np.array(N_values, dtype=np.float64)
    errs = np.array([r["rel_l2"] for r in results], dtype=np.float64)
    slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)

    print("N      Unknowns   Rel_L2_Error        Setup_Assem(s)   Solve(s)   Eval(s)   Total(s)")
    for r in results:
        print(
            f"{r['N']:<6d} {r['unknowns']:<10d} {r['rel_l2']:>14.6e}   "
            f"{r['assembly']:>13.6f}   {r['solve']:>8.6f}   "
            f"{r['evaluation']:>8.6f}   {r['total']:>8.6f}"
        )

    print(f"Estimated convergence order = {slope:.6f}")


if __name__ == "__main__":
    main()