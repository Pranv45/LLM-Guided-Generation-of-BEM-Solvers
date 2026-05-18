import time
import math
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# ------------------------------------------------------------
# Geometry / exact solution
# ------------------------------------------------------------

VERTICES = np.array(
    [
        (-1.5,  1.0),
        (-0.75, 1.0),
        ( 0.0,  0.4),
        ( 0.75, 1.0),
        ( 1.5,  1.0),
        ( 1.5,  0.0),
        ( 1.5, -1.0),
        ( 0.75, -1.0),
        ( 0.75,  0.3),
        ( 0.0, -0.3),
        (-0.75, 0.3),
        (-0.75,-1.0),
        (-1.5,-1.0),
        (-1.5, 0.0),
    ],
    dtype=np.float64,
)

N_VALUES = [400, 800, 1600, 3200, 6400]
NGRID = 200
GRID_XMIN = -1.5
GRID_XMAX = 1.5
GRID_YMIN = -1.0
GRID_YMAX = 1.0
QUAD_ORDER = 16  # >= 8 as requested

TWOPI = 2.0 * math.pi


def u_exact(x, y):
    return x * x * x - 3.0 * x * y * y


def allocate_elements_per_edge(vertices: np.ndarray, total_elements: int) -> np.ndarray:
    m = vertices.shape[0]
    nxt = np.roll(vertices, -1, axis=0)
    edge_lengths = np.linalg.norm(nxt - vertices, axis=1)
    length_sum = edge_lengths.sum()

    raw = total_elements * edge_lengths / length_sum
    counts = np.floor(raw).astype(np.int64)
    counts[counts < 1] = 1

    diff = int(total_elements - counts.sum())
    frac = raw - np.floor(raw)

    if diff > 0:
        order = np.argsort(-frac)
        idx = 0
        while diff > 0:
            counts[order[idx % m]] += 1
            diff -= 1
            idx += 1
    elif diff < 0:
        order = np.argsort(frac)
        idx = 0
        safety = 0
        while diff < 0 and safety < 100000:
            j = order[idx % m]
            if counts[j] > 1:
                counts[j] -= 1
                diff += 1
            idx += 1
            safety += 1

    return counts


def build_geometry(vertices: np.ndarray, total_elements: int, quad_order: int):
    edge_counts = allocate_elements_per_edge(vertices, total_elements)
    m = vertices.shape[0]
    nxt = np.roll(vertices, -1, axis=0)

    xi, wi = np.polynomial.legendre.leggauss(quad_order)

    elements = []
    midpoints = []
    lengths = []
    normals = []
    qx = []
    qy = []
    qw = []

    for e in range(m):
        p0 = vertices[e]
        p1 = nxt[e]
        d = p1 - p0
        edge_len = float(np.linalg.norm(d))
        normal = np.array([d[1], -d[0]], dtype=np.float64) / edge_len  # CCW => outward

        ne = int(edge_counts[e])
        t = np.arange(ne + 1, dtype=np.float64) / ne
        s = 0.5 * (1.0 - np.cos(math.pi * t))  # cosine clustering on each edge
        nodes = p0[None, :] + s[:, None] * d[None, :]

        for j in range(ne):
            a = nodes[j]
            b = nodes[j + 1]
            mid = 0.5 * (a + b)
            L = float(np.linalg.norm(b - a))

            elements.append(np.stack([a, b], axis=0))
            midpoints.append(mid)
            lengths.append(L)
            normals.append(normal)

            y = mid[None, :] + 0.5 * xi[:, None] * (b - a)[None, :]
            w = 0.5 * L * wi
            qx.append(y[:, 0].copy())
            qy.append(y[:, 1].copy())
            qw.append(w.copy())

    return (
        np.asarray(elements, dtype=np.float64),
        np.asarray(midpoints, dtype=np.float64),
        np.asarray(lengths, dtype=np.float64),
        np.asarray(normals, dtype=np.float64),
        np.asarray(qx, dtype=np.float64),
        np.asarray(qy, dtype=np.float64),
        np.asarray(qw, dtype=np.float64),
        edge_counts,
    )


@njit(fastmath=True)
def point_in_polygon(x, y, px, py):
    inside = False
    n = px.shape[0]
    j = n - 1
    for i in range(n):
        xi = px[i]
        yi = py[i]
        xj = px[j]
        yj = py[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


@njit(fastmath=True)
def points_in_polygon(all_pts, vertices):
    npts = all_pts.shape[0]
    px = vertices[:, 0]
    py = vertices[:, 1]
    out = np.empty(npts, dtype=np.bool_)
    for i in range(npts):
        out[i] = point_in_polygon(all_pts[i, 0], all_pts[i, 1], px, py)
    return out


@njit(fastmath=True)
def point_segment_dist(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby + 1e-300
    t = (apx * abx + apy * aby) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    qx = ax + t * abx
    qy = ay + t * aby
    dx = px - qx
    dy = py - qy
    return math.sqrt(dx * dx + dy * dy)


@njit(fastmath=True)
def min_dist_to_boundary(points, vertices):
    npts = points.shape[0]
    nv = vertices.shape[0]
    dists = np.empty(npts, dtype=np.float64)
    for i in range(npts):
        px = points[i, 0]
        py = points[i, 1]
        dmin = 1e300
        for k in range(nv):
            ax = vertices[k, 0]
            ay = vertices[k, 1]
            bx = vertices[(k + 1) % nv, 0]
            by = vertices[(k + 1) % nv, 1]
            d = point_segment_dist(px, py, ax, ay, bx, by)
            if d < dmin:
                dmin = d
        dists[i] = dmin
    return dists


@njit(parallel=True, fastmath=True)
def assemble_operator(mu, colloc_x, colloc_y, qx, qy, qw, nx, ny):
    ne = mu.shape[0]
    nq = qx.shape[1]

    out = np.empty(ne, dtype=np.float64)

    mu_mean = 0.0
    for j in range(ne):
        mu_mean += mu[j]
    mu_mean /= ne

    for i in prange(ne):
        xi = colloc_x[i]
        yi = colloc_y[i]

        acc = 0.5 * (mu[i] - mu_mean)  # interior trace for the convention used here

        for j in range(ne):
            if j == i:
                continue

            muj = mu[j] - mu_mean
            nxj = nx[j]
            nyj = ny[j]

            s = 0.0
            for k in range(nq):
                rx = xi - qx[j, k]
                ry = yi - qy[j, k]
                r2 = rx * rx + ry * ry
                s += qw[j, k] * ((rx * nxj + ry * nyj) / r2)

            acc += muj * s / TWOPI

        out[i] = acc

    out_mean = 0.0
    for i in range(ne):
        out_mean += out[i]
    out_mean /= ne

    for i in prange(ne):
        out[i] -= out_mean

    return out


@njit(parallel=True, fastmath=True)
def evaluate_interior(px, py, mu, qx, qy, qw, nx, ny):
    npnt = px.shape[0]
    ne = mu.shape[0]
    nq = qx.shape[1]

    out = np.empty(npnt, dtype=np.float64)

    mu_mean = 0.0
    for j in range(ne):
        mu_mean += mu[j]
    mu_mean /= ne

    for p in prange(npnt):
        x = px[p]
        y = py[p]
        acc = 0.0

        for j in range(ne):
            muj = mu[j] - mu_mean
            nxj = nx[j]
            nyj = ny[j]
            s = 0.0
            for k in range(nq):
                rx = x - qx[j, k]
                ry = y - qy[j, k]
                r2 = rx * rx + ry * ry
                s += qw[j, k] * ((rx * nxj + ry * nyj) / r2)
            acc += muj * s / TWOPI

        out[p] = acc

    return out


def build_evaluation_grid(vertices: np.ndarray, ngrid: int):
    xs = np.linspace(GRID_XMIN, GRID_XMAX, ngrid)
    ys = np.linspace(GRID_YMIN, GRID_YMAX, ngrid)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])

    mask_in = points_in_polygon(all_pts, vertices)
    interior = all_pts[mask_in].copy()

    N_min = min(N_VALUES)
    nv = len(vertices)
    perim = sum(np.linalg.norm(vertices[(k + 1) % nv] - vertices[k]) for k in range(nv))
    h_coarse = perim / N_min
    delta = 2.0 * h_coarse

    dist = min_dist_to_boundary(interior, vertices)
    mask_far = dist > delta
    grid_pts = interior[mask_far].copy()

    return grid_pts[:, 0], grid_pts[:, 1], delta


def run_case(total_elements: int, eval_x, eval_y):
    t0 = time.perf_counter()

    elements, midpoints, lengths, normals, qx, qy, qw, edge_counts = build_geometry(
        VERTICES, total_elements, QUAD_ORDER
    )

    rhs = u_exact(midpoints[:, 0], midpoints[:, 1])
    rhs = rhs - rhs.mean()

    ne = midpoints.shape[0]
    colloc_x = midpoints[:, 0].copy()
    colloc_y = midpoints[:, 1].copy()
    nx = normals[:, 0].copy()
    ny = normals[:, 1].copy()

    # warm-up JIT on actual shapes
    _ = assemble_operator(
        np.zeros(ne, dtype=np.float64), colloc_x, colloc_y, qx, qy, qw, nx, ny
    )
    _ = evaluate_interior(
        eval_x[:1], eval_y[:1], np.zeros(ne, dtype=np.float64), qx, qy, qw, nx, ny
    )

    t_setup = time.perf_counter() - t0

    def matvec(v):
        return assemble_operator(v, colloc_x, colloc_y, qx, qy, qw, nx, ny)

    A = LinearOperator((ne, ne), matvec=matvec, dtype=np.float64)

    iter_counter = {"count": 0}

    def cb(_):
        iter_counter["count"] += 1

    t1 = time.perf_counter()
    mu, info = gmres(
        A,
        rhs,
        restart=ne,
        rtol=1e-10,
        atol=1e-10,
        callback=cb,
        callback_type="pr_norm",
    )
    solve_time = time.perf_counter() - t1

    mu = mu - mu.mean()

    t2 = time.perf_counter()
    u_num = evaluate_interior(eval_x, eval_y, mu, qx, qy, qw, nx, ny)
    u_ex = u_exact(eval_x, eval_y)
    rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)
    eval_time = time.perf_counter() - t2

    total_time = time.perf_counter() - t0

    return {
        "N": total_elements,
        "unknowns": ne,
        "gmres_iters": iter_counter["count"],
        "rel_l2": rel_l2,
        "setup": t_setup,
        "solve": solve_time,
        "eval": eval_time,
        "total": total_time,
        "info": info,
        "u_num": u_num,
        "u_ex": u_ex,
    }


def prewarm_numba():
    px = np.array([0.0], dtype=np.float64)
    py = np.array([0.0], dtype=np.float64)

    mu = np.zeros(2, dtype=np.float64)
    colloc_x = np.zeros(2, dtype=np.float64)
    colloc_y = np.zeros(2, dtype=np.float64)
    qx = np.zeros((2, 2), dtype=np.float64)
    qy = np.zeros((2, 2), dtype=np.float64)
    qw = np.ones((2, 2), dtype=np.float64)
    nx = np.zeros(2, dtype=np.float64)
    ny = np.zeros(2, dtype=np.float64)

    _ = point_in_polygon(
        0.1,
        0.1,
        np.array([0.0, 1.0, 1.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    _ = assemble_operator(mu, colloc_x, colloc_y, qx, qy, qw, nx, ny)
    _ = evaluate_interior(px, py, mu, qx, qy, qw, nx, ny)
    _ = min_dist_to_boundary(np.array([[0.0, 0.0]], dtype=np.float64), VERTICES)


def plot_error_with_boundary(x, y, u_num, u_ex, vertices):
    error = np.abs(u_num - u_ex)
    tri = Triangulation(x, y)

    plt.figure(figsize=(7, 6))
    tcf = plt.tricontourf(tri, error, levels=50)
    plt.colorbar(tcf, label="Absolute Error")

    verts_closed = np.vstack([vertices, vertices[0]])
    plt.plot(verts_closed[:, 0], verts_closed[:, 1], "k-", lw=2)

    plt.title("Absolute Error on Filtered Interior Grid")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def main():
    prewarm_numba()

    eval_x, eval_y, delta = build_evaluation_grid(VERTICES, NGRID)

    results = []
    for N in N_VALUES:
        results.append(run_case(N, eval_x, eval_y))

    print(f"Filtered evaluation grid: ngrid={NGRID}, delta={delta:.6e}")
    print("N | unknowns | gmres_iters | rel_L2 | setup_s | solve_s | eval_s | total_s | info")
    for r in results:
        print(
            f"{r['N']} | {r['unknowns']} | {r['gmres_iters']} | "
            f"{r['rel_l2']:.6e} | {r['setup']:.4f} | {r['solve']:.4f} | "
            f"{r['eval']:.4f} | {r['total']:.4f} | {r['info']}"
        )

    best = results[-1]
    plot_error_with_boundary(eval_x, eval_y, best["u_num"], best["u_ex"], VERTICES)


if __name__ == "__main__":
    main()