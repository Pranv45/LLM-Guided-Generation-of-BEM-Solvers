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


def naca_4digit_boundary(m, p, t, N_points):
    N_points = int(N_points)
    if N_points < 4:
        raise ValueError("N_points must be at least 4")

    n_upper = N_points // 2
    n_lower = N_points - n_upper

    s_upper = np.linspace(0.0, 1.0, n_upper + 1)
    s_lower = np.linspace(0.0, 1.0, n_lower + 1)

    x_upper = 0.5 * (1.0 + np.cos(np.pi * s_upper))   # 1 -> 0
    x_lower = 0.5 * (1.0 - np.cos(np.pi * s_lower))   # 0 -> 1

    def camber(x):
        yc = np.empty_like(x)
        dyc = np.empty_like(x)

        mask = x < p
        if p > 0.0:
            yc[mask] = m / (p * p) * (2.0 * p * x[mask] - x[mask] ** 2)
            dyc[mask] = 2.0 * m / (p * p) * (p - x[mask])
        else:
            yc[mask] = 0.0
            dyc[mask] = 0.0

        mask2 = ~mask
        if p < 1.0:
            yc[mask2] = m / ((1.0 - p) ** 2) * (
                (1.0 - 2.0 * p) + 2.0 * p * x[mask2] - x[mask2] ** 2
            )
            dyc[mask2] = 2.0 * m / ((1.0 - p) ** 2) * (p - x[mask2])
        else:
            yc[mask2] = 0.0
            dyc[mask2] = 0.0

        return yc, dyc

    def thickness(x):
        return 5.0 * t * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1036 * x**4
        )

    yu, dyu = camber(x_upper)
    yl, dyl = camber(x_lower)
    tu = thickness(x_upper)
    tl = thickness(x_lower)

    th_u = np.arctan(dyu)
    th_l = np.arctan(dyl)

    upper = np.column_stack([x_upper - tu * np.sin(th_u), yu + tu * np.cos(th_u)])
    lower = np.column_stack([x_lower + tl * np.sin(th_l), yl - tl * np.cos(th_l)])

    # TE upper -> LE -> TE lower, with exact TE closure
    verts = np.vstack([upper, lower[1:]])

    # Enforce CCW orientation
    unique = verts[:-1]
    x0 = unique[:, 0]
    y0 = unique[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    area2 = np.sum(x0 * y1 - x1 * y0)
    if area2 < 0.0:
        unique = unique[::-1].copy()
        verts = np.vstack([unique, unique[:1]])

    return verts


def build_geometry(vertices):
    p0 = vertices[:-1].copy()
    p1 = vertices[1:].copy()

    ax = p0[:, 0]
    ay = p0[:, 1]
    bx = p1[:, 0]
    by = p1[:, 1]

    dx = bx - ax
    dy = by - ay
    L = np.sqrt(dx * dx + dy * dy)
    if np.any(L <= 0.0):
        raise ValueError("Zero-length element encountered.")

    cx = 0.5 * (ax + bx)
    cy = 0.5 * (ay + by)

    tx = dx / L
    ty = dy / L

    nx = dy / L
    ny = -dx / L

    half = 0.5 * L

    return ax, ay, bx, by, cx, cy, L, half, tx, ty, nx, ny


def classify_bcs(cx):
    return (cx < 0.8).astype(np.int8)  # 1 = Neumann, 0 = Dirichlet


def gauss_rule(order=8):
    xi, wi = np.polynomial.legendre.leggauss(order)
    return xi.astype(np.float64), wi.astype(np.float64)


@njit(parallel=True)
def assemble_HG(cx, cy, ax, ay, bx, by, half, tx, ty, nx, ny, L, xi, wi):
    ne = cx.shape[0]
    H = np.zeros((ne, ne), dtype=np.float64)
    G = np.zeros((ne, ne), dtype=np.float64)

    for i in prange(ne):
        H[i, i] = 0.5
        G[i, i] = (L[i] / (2.0 * np.pi)) * (1.0 - np.log(L[i] / 2.0))

        xcol = cx[i]
        ycol = cy[i]

        for j in range(ne):
            if j == i:
                continue

            h = 0.0
            g = 0.0

            for k in range(xi.shape[0]):
                s = xi[k]
                w = wi[k]

                px = cx[j] + half[j] * s * tx[j]
                py = cy[j] + half[j] * s * ty[j]

                rx = xcol - px
                ry = ycol - py
                r2 = rx * rx + ry * ry

                log_r = 0.5 * np.log(r2)
                ustar = -(1.0 / (2.0 * np.pi)) * log_r
                qstar = (1.0 / (2.0 * np.pi)) * ((rx * nx[j] + ry * ny[j]) / r2)

                g += w * ustar
                h += w * qstar

            G[i, j] = half[j] * g
            H[i, j] = half[j] * h

    return H, G


@njit(parallel=True)
def evaluate_interior(points, cx, cy, half, tx, ty, nx, ny, L, u_bnd, q_bnd, xi, wi):
    npnt = points.shape[0]
    ne = cx.shape[0]
    out = np.zeros(npnt, dtype=np.float64)

    for p in prange(npnt):
        x = points[p, 0]
        y = points[p, 1]
        val = 0.0

        for j in range(ne):
            int_u = 0.0
            int_q = 0.0

            for k in range(xi.shape[0]):
                s = xi[k]
                w = wi[k]

                px = cx[j] + half[j] * s * tx[j]
                py = cy[j] + half[j] * s * ty[j]

                rx = x - px
                ry = y - py
                r2 = rx * rx + ry * ry

                log_r = 0.5 * np.log(r2)
                ustar = -(1.0 / (2.0 * np.pi)) * log_r
                qstar = (1.0 / (2.0 * np.pi)) * ((rx * nx[j] + ry * ny[j]) / r2)

                int_u += w * ustar * q_bnd[j]
                int_q += w * qstar * u_bnd[j]

            val += half[j] * (int_u - int_q)

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

            cond = ((yi > y) != (yj > y))
            if cond:
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

        for j in range(ne):
            vx = bx[j] - ax[j]
            vy = by[j] - ay[j]
            wx = px - ax[j]
            wy = py - ay[j]

            c1 = wx * vx + wy * vy
            if c1 <= 0.0:
                dx = px - ax[j]
                dy = py - ay[j]
                d = np.sqrt(dx * dx + dy * dy)
            else:
                c2 = vx * vx + vy * vy
                if c1 >= c2:
                    dx = px - bx[j]
                    dy = py - by[j]
                    d = np.sqrt(dx * dx + dy * dy)
                else:
                    b = c1 / c2
                    projx = ax[j] + b * vx
                    projy = ay[j] + b * vy
                    dx = px - projx
                    dy = py - projy
                    d = np.sqrt(dx * dx + dy * dy)

            if d < dmin:
                dmin = d

        out[i] = dmin

    return out


def warm_up_numba():
    ax = np.array([0.0, 1.0], dtype=np.float64)
    ay = np.array([0.0, 0.0], dtype=np.float64)
    bx = np.array([1.0, 1.0], dtype=np.float64)
    by = np.array([0.0, 1.0], dtype=np.float64)
    cx = 0.5 * (ax + bx)
    cy = 0.5 * (ay + by)
    dx = bx - ax
    dy = by - ay
    L = np.sqrt(dx * dx + dy * dy)
    half = 0.5 * L
    tx = dx / L
    ty = dy / L
    nx = dy / L
    ny = -dx / L
    xi, wi = gauss_rule(4)

    _ = assemble_HG(cx, cy, ax, ay, bx, by, half, tx, ty, nx, ny, L, xi, wi)

    pts = np.array([[0.25, 0.25]], dtype=np.float64)
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    _ = point_in_polygon(pts, poly)
    _ = min_dist_to_segments(pts, ax, ay, bx, by)

    u_bnd = np.array([1.0, 1.0], dtype=np.float64)
    q_bnd = np.array([0.0, 0.0], dtype=np.float64)
    _ = evaluate_interior(pts, cx, cy, half, tx, ty, nx, ny, L, u_bnd, q_bnd, xi, wi)


def run_case(N_points, xi, wi, grid_n=200, N_values_ref=(100, 200, 400, 800)):
    t0 = time.perf_counter()

    vertices = naca_4digit_boundary(0.04, 0.3, 0.18, N_points)
    ax, ay, bx, by, cx, cy, L, half, tx, ty, nx, ny = build_geometry(vertices)
    bcs_type = classify_bcs(cx)

    u_known = u_exact(cx, cy)
    q_known = q_exact(cx, cy, nx, ny)

    H, G = assemble_HG(cx, cy, ax, ay, bx, by, half, tx, ty, nx, ny, L, xi, wi)

    A = np.empty_like(H)
    b = np.zeros(N_points, dtype=np.float64)

    for j in range(N_points):
        if bcs_type[j] == 0:
            A[:, j] = -G[:, j]
            b -= H[:, j] * u_known[j]
        else:
            A[:, j] = H[:, j]
            b += G[:, j] * q_known[j]

    t1 = time.perf_counter()

    xsol = np.linalg.solve(A, b)

    t2 = time.perf_counter()

    u_bnd = np.where(bcs_type == 0, u_known, xsol)
    q_bnd = np.where(bcs_type == 1, q_known, xsol)

    xs = np.linspace(-0.1, 1.1, grid_n)
    ys = np.linspace(-0.2, 0.2, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])

    poly = vertices[:-1].copy()
    interior = point_in_polygon(all_pts, poly)
    interior_pts = all_pts[interior]

    N_min = min(N_values_ref)
    perimeter = np.sum(L)
    h_coarse = perimeter / N_min
    delta = 2.0 * h_coarse

    dist = min_dist_to_segments(interior_pts, ax, ay, bx, by)
    grid_pts = interior_pts[dist > delta]

    if grid_pts.shape[0] == 0:
        raise RuntimeError("No interior evaluation points remain after boundary exclusion.")

    u_num = evaluate_interior(grid_pts, cx, cy, half, tx, ty, nx, ny, L, u_bnd, q_bnd, xi, wi)
    u_ex = u_exact(grid_pts[:, 0], grid_pts[:, 1])

    rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)

    t3 = time.perf_counter()

    return {
        "N": N_points,
        "unknowns": N_points,
        "rel_l2": rel_l2,
        "setup_assembly": t1 - t0,
        "solve": t2 - t1,
        "eval": t3 - t2,
        "total": t3 - t0,
    }


def main():
    warm_up_numba()

    N_values = [400, 800, 1600, 3200, 6400]
    xi, wi = gauss_rule(8)

    results = []
    for N in N_values:
        results.append(run_case(N, xi, wi, grid_n=200, N_values_ref=N_values))

    hs = 1.0 / np.array(N_values, dtype=np.float64)
    errs = np.array([r["rel_l2"] for r in results], dtype=np.float64)
    slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)

    print("N      Unknowns    Rel_L2_Error        Setup+Assemble(s)   Solve(s)      Eval(s)       Total(s)")
    for r in results:
        print(
            f'{r["N"]:<6d} {r["unknowns"]:<10d} {r["rel_l2"]:>14.6e}   '
            f'{r["setup_assembly"]:>14.6f}   {r["solve"]:>9.6f}   '
            f'{r["eval"]:>9.6f}   {r["total"]:>9.6f}'
        )

    print(f"Estimated convergence order = {slope:.6f}")


if __name__ == "__main__":
    main()
