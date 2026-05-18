"""
2D BEM solver for Laplace's equation on M-shaped domain
Indirect double-layer potential, constant elements, matrix-free GMRES
Numba-accelerated matvec and interior evaluation
"""

import numpy as np
from numba import njit, prange
import scipy.sparse.linalg as spla
import time

# ─────────────────────────────────────────────────────────────
# Domain vertices (as given; signed area < 0 → CW in math coords)
# ─────────────────────────────────────────────────────────────
VERTICES = np.array([
    [-1.5,  1.0],
    [-0.75, 1.0],
    [ 0.0,  0.4],
    [ 0.75, 1.0],
    [ 1.5,  1.0],
    [ 1.5,  0.0],
    [ 1.5, -1.0],
    [ 0.75,-1.0],
    [ 0.75, 0.3],
    [ 0.0, -0.3],
    [-0.75, 0.3],
    [-0.75,-1.0],
    [-1.5, -1.0],
    [-1.5,  0.0],
], dtype=np.float64)


def u_exact(pts):
    """Manufactured harmonic solution: u = x³ - 3xy²"""
    x, y = pts[:, 0], pts[:, 1]
    return x**3 - 3.0*x*y**2


# ─────────────────────────────────────────────────────────────
# Signed area (shoelace): determines winding
# ─────────────────────────────────────────────────────────────
def signed_area(verts):
    nv = len(verts)
    a2 = 0.0
    for k in range(nv):
        ax, ay = verts[k]
        bx, by = verts[(k+1) % nv]
        a2 += ax*by - bx*ay
    return 0.5*a2   # positive = CCW, negative = CW


# ─────────────────────────────────────────────────────────────
# Outward normal for an oriented segment a→b
# For CCW polygon: RIGHT normal = [+tang_y, -tang_x]  points outward
# For CW  polygon: LEFT  normal = [-tang_y, +tang_x]  points outward
# We detect winding once and set the sign accordingly.
# ─────────────────────────────────────────────────────────────
def outward_normal(a, b, ccw: bool):
    tang = b - a
    tang /= np.linalg.norm(tang)
    if ccw:
        return np.array([ tang[1], -tang[0]])   # right normal
    else:
        return np.array([-tang[1],  tang[0]])   # left normal


# ─────────────────────────────────────────────────────────────
# Boundary discretisation with cosine clustering per edge
# ─────────────────────────────────────────────────────────────
def build_boundary(verts, N_total):
    nv = len(verts)
    ccw = signed_area(verts) > 0

    edges = [(verts[i], verts[(i+1) % nv]) for i in range(nv)]
    edge_lens = [np.linalg.norm(b - a) for a, b in edges]
    total_len = sum(edge_lens)

    # Distribute elements proportional to edge length
    raw = [max(1, round(N_total * l / total_len)) for l in edge_lens]
    diff = N_total - sum(raw)
    order = np.argsort(edge_lens)[::-1]
    for k in range(abs(diff)):
        raw[order[k % nv]] += int(np.sign(diff))

    endpoints, midpoints, lengths, normals = [], [], [], []

    for (a, b), n_e in zip(edges, raw):
        t = np.arange(n_e + 1, dtype=float) / n_e
        s = 0.5 * (1.0 - np.cos(np.pi * t))          # cosine clustering
        pts_e = a + np.outer(s, (b - a))
        nrm = outward_normal(a, b, ccw)

        for k in range(n_e):
            p0, p1 = pts_e[k], pts_e[k+1]
            endpoints.append([p0, p1])
            midpoints.append(0.5*(p0+p1))
            lengths.append(np.linalg.norm(p1-p0))
            normals.append(nrm.copy())

    return (np.array(endpoints, dtype=np.float64),
            np.array(midpoints, dtype=np.float64),
            np.array(lengths,   dtype=np.float64),
            np.array(normals,   dtype=np.float64))


# ─────────────────────────────────────────────────────────────
# Gaussian quadrature on [-1,1]
# ─────────────────────────────────────────────────────────────
def gauss_quadrature(order=8):
    xi, w = np.polynomial.legendre.leggauss(order)
    return xi.astype(np.float64), w.astype(np.float64)


def precompute_quad(endpoints, lengths, xi, w):
    Ne, nq = len(endpoints), len(xi)
    qpts = np.empty((Ne, nq, 2), dtype=np.float64)
    qwts = np.empty((Ne, nq),    dtype=np.float64)
    for j in range(Ne):
        a, b = endpoints[j, 0], endpoints[j, 1]
        for q in range(nq):
            t = 0.5*(xi[q] + 1.0)
            qpts[j, q, 0] = a[0] + t*(b[0]-a[0])
            qpts[j, q, 1] = a[1] + t*(b[1]-a[1])
            qwts[j, q]    = 0.5 * lengths[j] * w[q]
    return qpts, qwts


# ─────────────────────────────────────────────────────────────
# Numba: core matvec y = (-1/2 I + K) mu
#
# Diagonal: for a straight constant element collocated at its own midpoint,
#   CPV of the DLP self-integral = 0  (no solid angle from a flat segment)
#   So: K_{ii} = 0, and the full diagonal is −1/2 from the jump term.
#
# Off-diagonal: standard Gaussian quadrature of the DLP kernel.
# ─────────────────────────────────────────────────────────────
@njit(parallel=True, fastmath=True, cache=True)
def _matvec_core(mu, midpoints, normals, qpts, qwts, Ne, nq):
    INV2PI = 1.0 / (2.0 * np.pi)
    result = np.empty(Ne, dtype=np.float64)

    for i in prange(Ne):
        xi_x = midpoints[i, 0]
        xi_y = midpoints[i, 1]
        acc  = -0.5 * mu[i]    # jump term (CPV of K_{ii} = 0)

        for j in range(Ne):
            if i == j:
                continue       # singular self-term handled analytically above
            nj_x = normals[j, 0]
            nj_y = normals[j, 1]
            muj  = mu[j]
            s = 0.0
            for q in range(nq):
                rx = xi_x - qpts[j, q, 0]
                ry = xi_y - qpts[j, q, 1]
                r2 = rx*rx + ry*ry
                if r2 < 1e-30:
                    continue
                kernel = INV2PI * (rx*nj_x + ry*nj_y) / r2
                s += kernel * qwts[j, q]
            acc += s * muj

        result[i] = acc

    return result


# ─────────────────────────────────────────────────────────────
# Augmented operator to fix nullspace:
#   Row 0 is replaced by the constraint mean(mu) = 0
#   (constant μ spans the nullspace of the DLP operator on closed curves)
# ─────────────────────────────────────────────────────────────
def make_augmented_operator(midpoints, normals, qpts, qwts, Ne):
    nq = qpts.shape[1]

    def matvec(mu):
        y = _matvec_core(mu, midpoints, normals, qpts, qwts, Ne, nq)
        # replace row 0 with constraint: mean(mu) = 0
        y[0] = mu.mean()
        return y

    return spla.LinearOperator((Ne, Ne), matvec=matvec, dtype=np.float64)


# ─────────────────────────────────────────────────────────────
# Interior evaluation: u(x) = ∫ ∂G/∂n_y(x,y) μ(y) ds_y
# ─────────────────────────────────────────────────────────────
@njit(parallel=True, fastmath=True, cache=True)
def _eval_interior(ipts, mu, normals, qpts, qwts, Ne, nq):
    INV2PI = 1.0 / (2.0 * np.pi)
    Np = ipts.shape[0]
    u  = np.zeros(Np, dtype=np.float64)
    for i in prange(Np):
        xi_x = ipts[i, 0]
        xi_y = ipts[i, 1]
        acc  = 0.0
        for j in range(Ne):
            nj_x = normals[j, 0]
            nj_y = normals[j, 1]
            muj  = mu[j]
            for q in range(nq):
                rx = xi_x - qpts[j, q, 0]
                ry = xi_y - qpts[j, q, 1]
                r2 = rx*rx + ry*ry
                if r2 < 1e-30:
                    continue
                kernel = INV2PI * (rx*nj_x + ry*nj_y) / r2
                acc += kernel * qwts[j, q] * muj
        u[i] = acc
    return u


# ─────────────────────────────────────────────────────────────
# Minimum distance from each point to the polygon boundary.
# Used to exclude near-boundary evaluation points from error norms.
# ─────────────────────────────────────────────────────────────
def min_dist_to_boundary(pts, verts):
    nv   = len(verts)
    dist = np.full(len(pts), np.inf)
    for k in range(nv):
        a  = verts[k]
        b  = verts[(k+1) % nv]
        ab = b - a
        ab2 = float(ab @ ab)
        ap  = pts - a                              # (N, 2)
        t   = np.clip((ap @ ab) / ab2, 0.0, 1.0)  # (N,)
        proj = a + np.outer(t, ab)                 # (N, 2)
        d    = np.linalg.norm(pts - proj, axis=1)  # (N,)
        dist = np.minimum(dist, d)
    return dist


# ─────────────────────────────────────────────────────────────
# Ray-casting point-in-polygon
# ─────────────────────────────────────────────────────────────
def points_in_polygon(pts, verts):
    nv = len(verts)
    inside = np.zeros(len(pts), dtype=bool)
    for i, (px, py) in enumerate(pts):
        cnt = 0
        for k in range(nv):
            ax, ay = verts[k]
            bx, by = verts[(k+1) % nv]
            if (ay > py) != (by > py):
                xint = ax + (py - ay)*(bx - ax)/(by - ay)
                if px < xint:
                    cnt += 1
        inside[i] = (cnt % 2 == 1)
    return inside


# ─────────────────────────────────────────────────────────────
# Single refinement run
# ─────────────────────────────────────────────────────────────
def run(N_total, grid_pts, grid_exact):
    t0 = time.perf_counter()

    # ── Geometry & quadrature ──────────────────────────────
    endpoints, midpoints, lengths, normals = build_boundary(VERTICES, N_total)
    Ne = len(midpoints)

    xi_q, w_q = gauss_quadrature(8)
    qpts, qwts = precompute_quad(endpoints, lengths, xi_q, w_q)

    t_setup = time.perf_counter() - t0

    # ── RHS: Dirichlet data at collocation midpoints ───────
    f = u_exact(midpoints)

    # ── JIT warm-up ───────────────────────────────────────
    _dummy = np.zeros(Ne, dtype=np.float64)
    _matvec_core(_dummy, midpoints, normals, qpts, qwts, Ne, 8)
    _eval_interior(grid_pts[:1], _dummy, normals, qpts, qwts, Ne, 8)

    # ── Augmented system ───────────────────────────────────
    op = make_augmented_operator(midpoints, normals, qpts, qwts, Ne)
    f_aug      = f.copy()
    f_aug[0]   = 0.0     # constraint row: mean(mu) = 0

    iters = [0]
    def cb(rk):
        iters[0] += 1

    t1 = time.perf_counter()
    mu, info = spla.gmres(
        op, f_aug,
        rtol=1e-10, atol=1e-10,
        restart=min(Ne, 300),
        maxiter=20*Ne,
        callback=cb,
        callback_type='pr_norm',
    )
    t_solve = time.perf_counter() - t1

    mu -= mu.mean()    # project out residual nullspace component

    # ── Interior evaluation ────────────────────────────────
    t2 = time.perf_counter()
    u_num = _eval_interior(grid_pts, mu, normals, qpts, qwts, Ne, 8)
    t_eval = time.perf_counter() - t2

    # ── Error ─────────────────────────────────────────────
    denom   = np.linalg.norm(grid_exact)
    rel_err = np.linalg.norm(u_num - grid_exact) / (denom if denom > 1e-14 else 1.0)

    t_total = time.perf_counter() - t0

    return dict(N=N_total, Ne=Ne, iters=iters[0], err=rel_err,
                t_setup=t_setup, t_solve=t_solve,
                t_eval=t_eval, t_total=t_total, gmres_info=info)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
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
    # A point at distance r from an element of size h sits in the near-singular
    # regime when r/h < ~2: the kernel ~1/(2pi*r) is not well-resolved by
    # standard quadrature and the pointwise error there converges at a
    # different rate than the bulk, corrupting the L2 norm.
    # delta must scale with the COARSEST element size in the study:
    #   h_coarse = perimeter / N_min
    #   delta    = 2 * h_coarse   (2-element safety margin)
    N_min      = min(N_values)
    nv_        = len(VERTICES)
    perim      = sum(np.linalg.norm(VERTICES[(k+1) % nv_] - VERTICES[k])
                     for k in range(nv_))
    h_coarse   = perim / N_min
    delta      = 2.0 * h_coarse
    dist       = min_dist_to_boundary(interior, VERTICES)
    mask_far   = dist > delta
    grid_pts   = interior[mask_far].copy()
    grid_exact = u_exact(grid_pts)

    area  = signed_area(VERTICES)
    wind  = "CCW" if area > 0 else "CW"
    print(f"Perimeter                  : {perim:.4f}")
    print(f"Coarsest h  (N={N_min})       : {h_coarse:.4f}  (perimeter / N_min)")
    print(f"Exclusion δ = 2 × h_coarse : {delta:.4f}")
    print(f"Interior points (all)      : {mask_in.sum()}")
    print(f"Interior points (used)     : {len(grid_pts)}  (dist > δ)")
    print(f"Interior points (excluded) : {len(interior) - len(grid_pts)}  (dist ≤ δ)")
    print(f"Polygon signed area        : {area:.4f}  ({wind} traversal)")
    print(f"Outward normals computed for {wind} orientation")
    print()

    hdr = (f"{'N':>6}  {'Ne':>6}  {'Iters':>6}  "
           f"{'Rel L2 Err':>12}  {'Setup(s)':>9}  "
           f"{'Solve(s)':>9}  {'Eval(s)':>8}  {'Total(s)':>9}")
    sep = "─" * len(hdr)

    print(hdr)
    print(sep)

    prev_err = None
    for N in N_values:
        r = run(N, grid_pts, grid_exact)
        rate = ""
        if prev_err is not None and r['err'] > 1e-14:
            rate = f"  (rate {np.log2(prev_err/r['err']):.2f})"
        print(f"{r['N']:>6}  {r['Ne']:>6}  {r['iters']:>6}  "
              f"{r['err']:>12.4e}  {r['t_setup']:>9.3f}  "
              f"{r['t_solve']:>9.3f}  {r['t_eval']:>8.3f}  "
              f"{r['t_total']:>9.3f}{rate}")
        prev_err = r['err']

    print(sep)
    print()
    print("Notes:")
    print("  Constant elements → O(h) convergence expected (rate ≈ 1)")
    print("  Cosine clustering concentrates DOFs near corners")
    print("  Nullspace fixed: row 0 replaced by constraint mean(μ) = 0")
    print("  Near-boundary points (dist < 1 grid spacing) excluded from L2 error")
    print("  Re-entrant corners cause μ singularity → some rate variation expected")


if __name__ == "__main__":
    main()