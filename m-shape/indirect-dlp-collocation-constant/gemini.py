import numpy as np
import scipy.sparse.linalg as spla
from numba import njit, prange
import time
import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# Exact Solution
# -------------------------------------------------------------------------
@njit(fastmath=True)
def u_exact(x, y):
    return x**3 - 3.0 * x * y**2

# -------------------------------------------------------------------------
# Numba Kernels for Grid Filtering
# -------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def points_in_polygon(pts, poly):
    n_pts = len(pts)
    n_verts = len(poly)
    inside = np.zeros(n_pts, dtype=np.bool_)

    for i in prange(n_pts):
        x, y = pts[i, 0], pts[i, 1]
        c = False
        p1x, p1y = poly[0, 0], poly[0, 1]
        for j in range(1, n_verts + 1):
            p2x, p2y = poly[j % n_verts, 0], poly[j % n_verts, 1]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                c = not c
            p1x, p1y = p2x, p2y
        inside[i] = c
    return inside

@njit(parallel=True, fastmath=True)
def min_dist_to_boundary(pts, poly):
    n_pts = len(pts)
    n_verts = len(poly)
    dists = np.zeros(n_pts)

    for i in prange(n_pts):
        px, py = pts[i, 0], pts[i, 1]
        min_d2 = 1e30

        for j in range(n_verts):
            p1x, p1y = poly[j, 0], poly[j, 1]
            p2x, p2y = poly[(j + 1) % n_verts, 0], poly[(j + 1) % n_verts, 1]

            vx, vy = p2x - p1x, p2y - p1y
            wx, wy = px - p1x, py - p1y

            c1 = wx*vx + wy*vy
            if c1 <= 0.0:
                d2 = wx*wx + wy*wy
            else:
                c2 = vx*vx + vy*vy
                if c2 <= c1:
                    dx, dy = px - p2x, py - p2y
                    d2 = dx*dx + dy*dy
                else:
                    b = c1 / c2
                    cx, cy = p1x + b*vx, p1y + b*vy
                    dx, dy = px - cx, py - cy
                    d2 = dx*dx + dy*dy

            if d2 < min_d2:
                min_d2 = d2

        dists[i] = np.sqrt(min_d2)
    return dists

# -------------------------------------------------------------------------
# Numba Kernels for BEM Integrals
# -------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def matvec(mu, midpoints, quad_pts, quad_wts, normals):
    Ne = len(mu)
    Nq = quad_wts.shape[1]
    out = np.zeros(Ne)

    for i in prange(Ne):
        xi = midpoints[i, 0]
        yi = midpoints[i, 1]
        integral = 0.0

        for j in range(Ne):
            if i == j:
                continue
            nx = normals[j, 0]
            ny = normals[j, 1]
            muj = mu[j]
            for q in range(Nq):
                qx = quad_pts[j, q, 0]
                qy = quad_pts[j, q, 1]
                qw = quad_wts[j, q]

                rx = xi - qx
                ry = yi - qy
                r2 = rx*rx + ry*ry

                rdotn = rx*nx + ry*ny
                integral += qw * (rdotn / r2) * muj

        out[i] = -0.5 * mu[i] + (1.0 / (2.0 * np.pi)) * integral
    return out

@njit(parallel=True, fastmath=True)
def eval_interior(eval_pts, mu, quad_pts, quad_wts, normals):
    N_eval = len(eval_pts)
    Ne = len(mu)
    Nq = quad_wts.shape[1]
    out = np.zeros(N_eval)

    for i in prange(N_eval):
        xi = eval_pts[i, 0]
        yi = eval_pts[i, 1]
        integral = 0.0

        for j in range(Ne):
            nx = normals[j, 0]
            ny = normals[j, 1]
            muj = mu[j]
            for q in range(Nq):
                qx = quad_pts[j, q, 0]
                qy = quad_pts[j, q, 1]
                qw = quad_wts[j, q]

                rx = xi - qx
                ry = yi - qy
                r2 = rx*rx + ry*ry

                rdotn = rx*nx + ry*ny
                integral += qw * (rdotn / r2) * muj

        out[i] = (1.0 / (2.0 * np.pi)) * integral
    return out

# -------------------------------------------------------------------------
# GMRES Helper Classes
# -------------------------------------------------------------------------
class BEMOperator(spla.LinearOperator):
    def __init__(self, Ne, midpoints, quad_pts, quad_wts, normals):
        super().__init__(shape=(Ne, Ne), dtype=np.float64)
        self.midpoints = midpoints
        self.quad_pts = quad_pts
        self.quad_wts = quad_wts
        self.normals = normals

    def _matvec(self, mu):
        return matvec(mu, self.midpoints, self.quad_pts, self.quad_wts, self.normals)

class GMRESCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, pr_norm):
        self.niter += 1

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
def main():
    VERTICES = np.array([
        [-1.5,  1.0], [-0.75, 1.0], [ 0.0,  0.4], [ 0.75, 1.0],
        [ 1.5,  1.0], [ 1.5,  0.0], [ 1.5, -1.0], [ 0.75, -1.0],
        [ 0.75,  0.3], [ 0.0, -0.3], [-0.75,  0.3], [-0.75,-1.0],
        [-1.5, -1.0], [-1.5,  0.0]
    ])

    # Reverse to enforce CCW ordering -> Outward normals
    VERTICES = VERTICES[::-1]

    # Pre-compile Numba functions
    dummy_pts = np.zeros((1, 2))
    _ = points_in_polygon(dummy_pts, VERTICES)
    _ = min_dist_to_boundary(dummy_pts, VERTICES)
    _ = matvec(np.zeros(2), np.zeros((2,2)), np.zeros((2,1,2)), np.zeros((2,1)), np.zeros((2,2)))
    _ = eval_interior(dummy_pts, np.zeros(2), np.zeros((2,1,2)), np.zeros((2,1)), np.zeros((2,2)))

    # ── Interior grid (fixed) ──────────────────────────────
    ngrid = 200
    xs = np.linspace(-1.5, 1.5, ngrid)
    ys = np.linspace(-1.0, 1.0, ngrid)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])

    # Step 1: keep only interior points
    mask_in    = points_in_polygon(all_pts, VERTICES)
    interior   = all_pts[mask_in].copy()

    # ── Refinement study ───────────────────────────────────
    N_values = [400, 800, 1600, 3200, 6400]

    # Step 2: exclude points closer than delta to any boundary edge.
    N_min      = min(N_values)
    nv_        = len(VERTICES)
    perim      = sum(np.linalg.norm(VERTICES[(k+1) % nv_] - VERTICES[k]) for k in range(nv_))
    h_coarse   = perim / N_min
    delta      = 2.0 * h_coarse

    dist       = min_dist_to_boundary(interior, VERTICES)
    mask_far   = dist > delta
    grid_pts   = interior[mask_far].copy()

    u_ex_interior = np.array([u_exact(p[0], p[1]) for p in grid_pts])
    norm_u_ex = np.linalg.norm(u_ex_interior)

    print(f"Total points inside domain: {len(interior)}")
    print(f"Points excluded by buffer (delta={delta:.4f}): {np.sum(~mask_far)}")
    print(f"Final evaluation grid size: {len(grid_pts)}")
    print("\n" + "-" * 88)
    print(f"{'N':<5} | {'Unknowns':<8} | {'Iters':<5} | {'Rel L2 Error':<12} | {'Setup (s)':<9} | {'Solve (s)':<9} | {'Eval (s)':<9} | {'Total (s)':<9}")
    print("-" * 88)

    q_pts_1d, q_wts_1d = np.polynomial.legendre.leggauss(10)

    for N in N_values:
        t_start = time.time()

        edge_lengths = np.zeros(nv_)
        for i in range(nv_):
            p1 = VERTICES[i]
            p2 = VERTICES[(i + 1) % nv_]
            edge_lengths[i] = np.linalg.norm(p2 - p1)

        frac = N * edge_lengths / perim
        counts = np.floor(frac).astype(int)
        counts = np.maximum(counts, 1)
        diff = N - np.sum(counts)
        if diff > 0:
            idx = np.argsort(frac - counts)[::-1]
            for i in range(diff): counts[idx[i]] += 1
        elif diff < 0:
            idx = np.argsort(frac - counts)
            removed = 0
            for i in idx:
                if counts[i] > 1:
                    counts[i] -= 1
                    removed += 1
                    if removed == -diff: break

        midpoints = np.zeros((N, 2))
        normals = np.zeros((N, 2))
        quad_pts = np.zeros((N, len(q_pts_1d), 2))
        quad_wts = np.zeros((N, len(q_pts_1d)))

        idx = 0
        for i in range(nv_):
            p1 = VERTICES[i]
            p2 = VERTICES[(i + 1) % nv_]
            Ne_edge = counts[i]

            t = np.linspace(0, 1, Ne_edge + 1)
            s = (1.0 - np.cos(np.pi * t)) / 2.0

            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            nrm = np.array([edge_vec[1], -edge_vec[0]]) / edge_len

            for j in range(Ne_edge):
                ep1 = p1 + s[j] * edge_vec
                ep2 = p1 + s[j+1] * edge_vec

                midpoints[idx] = (ep1 + ep2) / 2.0
                normals[idx] = nrm
                L_elem = np.linalg.norm(ep2 - ep1)

                for q in range(len(q_pts_1d)):
                    xi = q_pts_1d[q]
                    quad_pts[idx, q] = 0.5 * (1 - xi) * ep1 + 0.5 * (1 + xi) * ep2
                    quad_wts[idx, q] = 0.5 * L_elem * q_wts_1d[q]

                idx += 1

        t_setup = time.time() - t_start

        # Linear Solve
        t_solve_start = time.time()
        rhs = np.array([u_exact(m[0], m[1]) for m in midpoints])

        operator = BEMOperator(N, midpoints, quad_pts, quad_wts, normals)
        counter = GMRESCounter()
        mu, info = spla.gmres(
            operator, rhs,
            rtol=1e-10, atol=1e-10,
            restart=N,
            callback=counter,
            callback_type='pr_norm'
        )
        t_solve = time.time() - t_solve_start

        # Interior Evaluation on filtered grid
        t_eval_start = time.time()
        u_num_interior = eval_interior(grid_pts, mu, quad_pts, quad_wts, normals)
        t_eval = time.time() - t_eval_start

        # Error Computation
        err_l2 = np.linalg.norm(u_num_interior - u_ex_interior) / norm_u_ex
        t_total = time.time() - t_start

        print(f"{N:<5} | {N:<8} | {counter.niter:<5} | {err_l2:<12.5e} | {t_setup:<9.4f} | {t_solve:<9.4f} | {t_eval:<9.4f} | {t_total:<9.4f}")

if __name__ == '__main__':
    main()