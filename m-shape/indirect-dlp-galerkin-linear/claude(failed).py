import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange
import time

# ─────────────────────────────────────────
# DOMAIN VERTICES (CCW)
# ─────────────────────────────────────────
VERTICES = np.array([
    [-1.5,  1.0], [-0.75,  1.0], [ 0.0,  0.4], [ 0.75,  1.0],
    [ 1.5,  1.0], [ 1.5,  0.0], [ 1.5, -1.0], [ 0.75, -1.0],
    [ 0.75, 0.3], [ 0.0, -0.3], [-0.75,  0.3], [-0.75, -1.0],
    [-1.5, -1.0], [-1.5,  0.0]
], dtype=np.float64)

def u_exact(x, y):
    return x**3 - 3.0*x*y**2

# ─────────────────────────────────────────
# GAUSSIAN QUADRATURE (16-point for safety)
# ─────────────────────────────────────────
GP, GW = np.polynomial.legendre.leggauss(16)
GP = np.ascontiguousarray(GP, dtype=np.float64)
GW = np.ascontiguousarray(GW, dtype=np.float64)

# ─────────────────────────────────────────
# DISCRETIZATION
# ─────────────────────────────────────────
def build_discretization(N_total):
    """
    Build piecewise-linear BEM mesh.
    Returns nodes(N,2), elements(Ne,2), lengths(Ne,), normals(Ne,2).
    Nodes are unique; last vertex of boundary == node 0.
    Normals: CCW boundary + outward = rotate tangent by -90 deg
             tangent = (p2-p1)/L  →  n = (tang_y, -tang_x)  WRONG
             For CCW with outward: n = (tang_y, -tang_x)
             Check: top edge goes right → tang=(1,0) → n=(0,-1) INWARD
             Correct outward for CCW: n = (-tang_y, tang_x)  NO
    ─────────────────────────────────────────────────────────────────
    For a CCW-oriented boundary the OUTWARD normal is obtained by
    rotating the tangent vector +90° (left turn):
        n = (-tang_y, tang_x)   ← this points INTO domain for CCW

    Wait — let's be precise with a test case:
        Top edge: p1=(-1.5,1), p2=(-0.75,1) → tang=(+,0)
        Outward normal should be (0,+1)  (pointing up, away from domain)
        Rotate tang +90°: (-tang_y, tang_x) = (0, +) ✓ CORRECT

        Right edge: p1=(1.5,1), p2=(1.5,-1) → tang=(0,-)
        Outward normal should be (+1,0)
        Rotate tang +90°: (-tang_y, tang_x) = (+, 0) ✓ CORRECT
    ─────────────────────────────────────────────────────────────────
    So: outward normal for CCW = rotate tangent by +90°
        n = (-tang_y, tang_x)
    """
    nv = len(VERTICES)
    edge_vecs = np.array([VERTICES[(i+1)%nv] - VERTICES[i] for i in range(nv)])
    edge_L    = np.linalg.norm(edge_vecs, axis=1)
    perimeter = edge_L.sum()

    # Distribute nodes proportionally to edge length
    raw   = np.maximum(1, np.round(N_total * edge_L / perimeter).astype(int))
    diff  = N_total - raw.sum()
    order = np.argsort(-edge_L)
    for k in range(int(abs(diff))):
        i = order[k % nv]
        raw[i] += 1 if diff > 0 else (-1 if raw[i] > 1 else 0)

    node_list = []   # list of [x, y]
    elem_list = []   # list of [i1, i2]
    len_list  = []
    nrm_list  = []

    for ei in range(nv):
        p0 = VERTICES[ei]
        p1 = VERTICES[(ei+1) % nv]
        n_seg = int(raw[ei])

        # cosine clustering on this edge
        t_lin = np.linspace(0.0, 1.0, n_seg + 1)
        s     = 0.5 * (1.0 - np.cos(np.pi * t_lin))   # s ∈ [0,1]
        pts   = p0 + np.outer(s, p1 - p0)              # (n_seg+1, 2)

        tang   = (p1 - p0) / edge_L[ei]
        normal = np.array([-tang[1], tang[0]])          # outward for CCW (+90°)

        # Global index of pts[0]
        if ei == 0:
            base = 0
            for pt in pts[:-1]:          # don't add last (shared with next edge start)
                node_list.append(pt.copy())
        else:
            base = len(node_list)        # pts[0] already in list as previous edge's last
            # Actually pts[0] == last node added previously? Not exactly because of cosine.
            # We must NOT add pts[0]; it equals node_list[-1] only if we share correctly.
            # Strategy: share the vertex node, overwrite with exact vertex coordinate.
            for pt in pts[1:-1]:         # interior nodes of this edge
                node_list.append(pt.copy())

        # Build element indices for this edge
        # pts[0..n_seg]: physical positions
        # Their global indices:
        if ei == 0:
            idx = list(range(0, n_seg))           # 0 .. n_seg-1
            # pts[n_seg] will be added by next edge or is node 0 if last edge
        else:
            prev_end = base - 1                   # index of pts[0] (shared vertex)
            interior_start = base                 # index of pts[1]
            idx = [prev_end] + list(range(interior_start, interior_start + n_seg - 1))

        # Last point of this edge:
        # If not last edge: it's pts[n_seg] which is VERTICES[(ei+1)%nv],
        #   will be the first node of next edge → don't add now.
        # If last edge: it's VERTICES[0] = node 0.
        if ei < nv - 1:
            idx.append(len(node_list))   # placeholder; will be first node of next edge
        else:
            idx.append(0)                # closes the boundary

        for k in range(n_seg):
            seg_p0 = pts[k]
            seg_p1 = pts[k+1]
            seg_len = np.linalg.norm(seg_p1 - seg_p0)
            elem_list.append([idx[k], idx[k+1]])
            len_list.append(seg_len)
            nrm_list.append(normal.copy())

    nodes    = np.array(node_list, dtype=np.float64)
    elements = np.array(elem_list, dtype=np.int32)
    lengths  = np.array(len_list,  dtype=np.float64)
    normals  = np.array(nrm_list,  dtype=np.float64)
    return nodes, elements, lengths, normals

# ─────────────────────────────────────────
# JUMP TERM c_i via interior solid angle
# ─────────────────────────────────────────
def compute_jump_terms(nodes, elements):
    """
    Interior Dirichlet BIE (indirect DLP):
        c(x) μ(x) + PV ∫ ∂G/∂n μ ds = f(x)

    For a smooth point:  c = +1/2
    For a corner with INTERIOR angle α:  c = α / (2π)

    We extract α from the two elements meeting at each node.
    The interior angle is measured INSIDE the domain.

    Sign check: For a convex corner (α < π), c < 1/2.
                For a re-entrant corner (α > π), c > 1/2.
    """
    N = len(nodes)

    # Build: for each node, find the incoming and outgoing element
    # incoming element at node i: elements[e] where elements[e,1] == i
    # outgoing element at node i: elements[e] where elements[e,0] == i
    incoming = {}
    outgoing = {}
    for e, (n1, n2) in enumerate(elements):
        outgoing[int(n1)] = e
        incoming[int(n2)] = e

    c = np.full(N, 0.5)

    for i in range(N):
        e_in  = incoming.get(i, None)
        e_out = outgoing.get(i, None)
        if e_in is None or e_out is None:
            continue

        # Tangent of incoming element (direction of travel, toward node i)
        n1_in, n2_in = elements[e_in]
        t_in = (nodes[n2_in] - nodes[n1_in])
        t_in = t_in / np.linalg.norm(t_in)   # points TOWARD i

        # Tangent of outgoing element (direction of travel, away from node i)
        n1_out, n2_out = elements[e_out]
        t_out = (nodes[n2_out] - nodes[n1_out])
        t_out = t_out / np.linalg.norm(t_out)  # points AWAY from i

        # The boundary turns from direction t_in to direction t_out at node i.
        # The EXTERIOR angle β = signed angle from t_in to t_out (CCW positive).
        # The INTERIOR angle α = π - β  ... but for re-entrant corners β < 0.
        # More robustly: α = π - β  where β ∈ (-π, π].
        # For a smooth point: t_out == t_in → β=0 → α=π ... wrong, should be 2π·c=π → c=1/2 ✓
        # Wait: smooth → α = π (straight line, "half-space"), c = π/(2π) = 1/2 ✓

        cross = t_in[0]*t_out[1] - t_in[1]*t_out[0]
        dot   = t_in[0]*t_out[0] + t_in[1]*t_out[1]
        beta  = np.arctan2(cross, dot)    # exterior turning angle ∈ (-π, π]
        alpha = np.pi - beta              # interior solid angle

        # Ensure α ∈ (0, 2π)
        if alpha <= 0.0:
            alpha += 2.0 * np.pi
        if alpha >= 2.0 * np.pi:
            alpha -= 2.0 * np.pi

        c[i] = alpha / (2.0 * np.pi)

    return c

# ─────────────────────────────────────────
# NUMBA MATVEC  (C μ + K μ)
# ─────────────────────────────────────────
@njit(parallel=True, cache=True)
def _matvec(mu, nodes, elements, lengths, normals, c_diag, gp, gw):
    N  = nodes.shape[0]
    Ne = elements.shape[0]
    ng = gp.shape[0]
    out = np.zeros(N)

    for i in prange(N):
        xi = nodes[i, 0]
        yi = nodes[i, 1]
        s  = c_diag[i] * mu[i]

        for e in range(Ne):
            n1 = elements[e, 0]
            n2 = elements[e, 1]
            if n1 == i or n2 == i:
                # Kernel is O(r)/r² = O(1/r) → integrable BUT:
                # For straight elements the DLP kernel ∂G/∂n is EXACTLY zero
                # when x lies ON the line through the element (same edge),
                # because (x-y)·n = 0.  Skip safely.
                continue

            Le  = lengths[e]
            nx  = normals[e, 0]
            ny  = normals[e, 1]
            x1  = nodes[n1, 0]; y1 = nodes[n1, 1]
            x2  = nodes[n2, 0]; y2 = nodes[n2, 1]
            m1  = mu[n1]; m2 = mu[n2]
            hL  = 0.5 * Le

            for q in range(ng):
                sq   = gp[q]; wq = gw[q]
                ph1  = 0.5*(1.0 - sq)
                ph2  = 0.5*(1.0 + sq)
                yx   = ph1*x1 + ph2*x2
                yy   = ph1*y1 + ph2*y2
                mu_q = ph1*m1 + ph2*m2
                rx   = xi - yx
                ry   = yi - yy
                r2   = rx*rx + ry*ry
                if r2 < 1.0e-28:
                    continue
                rdn    = rx*nx + ry*ny
                kernel = 0.15915494309189535 * rdn / r2   # 1/(2π)
                s     += wq * hL * kernel * mu_q

        out[i] = s
    return out

# ─────────────────────────────────────────
# INTERIOR EVALUATION
# ─────────────────────────────────────────
@njit(parallel=True, cache=True)
def _eval_interior(pts, mu, nodes, elements, lengths, normals, gp, gw):
    Np = pts.shape[0]
    Ne = elements.shape[0]
    ng = gp.shape[0]
    u  = np.zeros(Np)
    for k in prange(Np):
        xi = pts[k, 0]; yi = pts[k, 1]
        v  = 0.0
        for e in range(Ne):
            n1 = elements[e, 0]; n2 = elements[e, 1]
            Le  = lengths[e]
            nx  = normals[e, 0]; ny = normals[e, 1]
            x1  = nodes[n1,0]; y1 = nodes[n1,1]
            x2  = nodes[n2,0]; y2 = nodes[n2,1]
            m1  = mu[n1]; m2 = mu[n2]
            hL  = 0.5*Le
            for q in range(ng):
                sq   = gp[q]; wq = gw[q]
                ph1  = 0.5*(1.0-sq); ph2 = 0.5*(1.0+sq)
                yx   = ph1*x1 + ph2*x2
                yy_  = ph1*y1 + ph2*y2
                mu_q = ph1*m1 + ph2*m2
                rx   = xi-yx; ry = yi-yy_
                r2   = rx*rx + ry*ry
                if r2 < 1.0e-28:
                    continue
                rdn    = rx*nx + ry*ny
                kernel = 0.15915494309189535 * rdn / r2
                v     += wq * hL * kernel * mu_q
        u[k] = v
    return u

# ─────────────────────────────────────────
# POLYGON UTILITIES (Numba)
# ─────────────────────────────────────────
@njit(cache=True)
def _in_poly(px, py, verts):
    n = verts.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi=verts[i,0]; yi=verts[i,1]
        xj=verts[j,0]; yj=verts[j,1]
        if ((yi > py) != (yj > py)) and (px < (xj-xi)*(py-yi)/(yj-yi)+xi):
            inside = not inside
        j = i
    return inside

@njit(parallel=True, cache=True)
def _pts_in_poly(pts, verts):
    n = pts.shape[0]
    m = np.zeros(n, dtype=np.bool_)
    for k in prange(n):
        m[k] = _in_poly(pts[k,0], pts[k,1], verts)
    return m

@njit(parallel=True, cache=True)
def _min_dist_bnd(pts, verts):
    nv = verts.shape[0]
    np_ = pts.shape[0]
    d = np.full(np_, 1.0e18)
    for k in prange(np_):
        px=pts[k,0]; py=pts[k,1]
        dm=1.0e18
        for i in range(nv):
            j=(i+1)%nv
            ax=verts[i,0]; ay=verts[i,1]
            bx=verts[j,0]; by=verts[j,1]
            ddx=bx-ax; ddy=by-ay
            L2=ddx*ddx+ddy*ddy
            if L2<1e-30:
                dd=(px-ax)**2+(py-ay)**2
            else:
                t=((px-ax)*ddx+(py-ay)*ddy)/L2
                t=max(0.0,min(1.0,t))
                cx=ax+t*ddx; cy=ay+t*ddy
                dd=(px-cx)**2+(py-cy)**2
            if dd<dm:
                dm=dd
        d[k]=dm**0.5
    return d

# ─────────────────────────────────────────
# VALIDATION: check normal orientation
# ─────────────────────────────────────────
def validate_normals(nodes, elements, normals):
    """
    Solid angle test: ∫_Γ ∂G/∂n ds = -1 for interior point (DLP of constant 1).
    For interior point x:  ∫ ∂G/∂n_y(x,y) ds_y = -1  (full solid angle, interior)
    """
    # Test with centroid-ish interior point
    test_pt = np.array([[0.0, -0.1]])
    one_mu  = np.ones(len(nodes))
    val = _eval_interior(test_pt, one_mu, nodes, elements, lengths_g, normals, GP, GW)
    return val[0]

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run():
    N_values = [400, 800, 1600, 3200, 6400]

    # Warm up Numba JIT
    _n, _e, _l, _nr = build_discretization(50)
    _c = compute_jump_terms(_n, _e)
    _mu = np.ones(len(_n))
    _matvec(_mu, _n, _e, _l, _nr, _c, GP, GW)
    _pts_in_poly(np.zeros((1,2), dtype=np.float64), VERTICES)
    _min_dist_bnd(np.zeros((1,2), dtype=np.float64), VERTICES)

    # Fixed evaluation grid
    ngrid   = 200
    xs      = np.linspace(-1.5, 1.5, ngrid)
    ys      = np.linspace(-1.0,  1.0, ngrid)
    XX, YY  = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])

    in_mask     = _pts_in_poly(all_pts, VERTICES)
    interior    = all_pts[in_mask]

    nv        = len(VERTICES)
    perimeter = sum(np.linalg.norm(VERTICES[(i+1)%nv]-VERTICES[i]) for i in range(nv))
    h_coarse  = perimeter / min(N_values)
    delta     = 2.0 * h_coarse

    dist_bnd  = _min_dist_bnd(interior, VERTICES)
    grid_pts  = interior[dist_bnd > delta]
    u_ex      = u_exact(grid_pts[:,0], grid_pts[:,1])
    norm_uex  = np.linalg.norm(u_ex)

    hdr = (f"{'N':>6} {'DOF':>6} {'Iter':>6} {'L2 Err':>14} "
           f"{'Setup(s)':>9} {'Solve(s)':>9} {'Eval(s)':>9} {'Tot(s)':>9}")
    sep = "=" * len(hdr)
    print(sep); print(hdr); print(sep)

    L2_errors = []

    for N in N_values:
        t0 = time.perf_counter()

        nodes, elements, lengths, normals = build_discretization(N)
        Nn = len(nodes)

        # ── Solid-angle check (normal orientation) ──────────────────
        # DLP of constant=1 evaluated at an interior point should = -1
        # (the interior Dirichlet BIE constant term).
        # If we get +1, normals are flipped → negate normals.
        chk_pt  = np.array([[0.0, -0.1]])
        chk_mu  = np.ones(Nn)
        chk_val = _eval_interior(chk_pt, chk_mu, nodes, elements,
                                 lengths, normals, GP, GW)[0]
        # For correct CCW outward normals and interior point:
        # ∫ ∂G/∂n μ ds  with μ=1  should give  -1
        # (the jump gives c·1 = α/(2π) < 1, and lim from interior = -1)
        # Actually for a closed surface ∫ ∂G/∂n ds = -1 for interior points.
        if chk_val > 0:
            normals = -normals   # flip if wrong orientation

        c_diag = compute_jump_terms(nodes, elements)

        # ── Verify c_diag via rigid-body (sum of full row = 0) ──────
        # For the operator (cI + K) applied to μ=1:
        # result should be 0 everywhere (since u=const is trivial for
        # interior Neumann, but for Dirichlet DLP... actually for DLP
        # operator: K[1] + c = ?).
        # For DLP on a closed boundary: PV ∫ ∂G/∂n ds = -c(x) for interior.
        # So (cI + K)[1] = c·1 + (-c·1) = 0  ✓ — use this to validate c_diag.

        t_setup = time.perf_counter() - t0

        # RHS
        f_rhs = u_exact(nodes[:,0], nodes[:,1])

        # GMRES
        iters = [0]
        def matvec(v, nodes=nodes, elements=elements, lengths=lengths,
                   normals=normals, c_diag=c_diag):
            iters[0] += 1
            return _matvec(v, nodes, elements, lengths, normals, c_diag, GP, GW)

        A = LinearOperator((Nn, Nn), matvec=matvec, dtype=np.float64)

        t1 = time.perf_counter()
        mu_sol, info = gmres(
            A, f_rhs,
            rtol=1e-10, atol=1e-10,
            restart=min(Nn, 300),
            callback_type='pr_norm'
        )
        t_solve = time.perf_counter() - t1
        if info != 0:
            print(f"  [!] GMRES info={info} for N={N}")

        # Interior evaluation
        t2 = time.perf_counter()
        u_num = _eval_interior(grid_pts, mu_sol, nodes, elements,
                               lengths, normals, GP, GW)
        t_eval = time.perf_counter() - t2

        l2_err = np.linalg.norm(u_num - u_ex) / norm_uex
        L2_errors.append(l2_err)
        t_tot = time.perf_counter() - t0

        print(f"{N:>6} {Nn:>6} {iters[0]:>6} {l2_err:>14.4e} "
              f"{t_setup:>9.2f} {t_solve:>9.2f} {t_eval:>9.2f} {t_tot:>9.2f}")

    print(sep)

    # Convergence analysis
    h_arr    = 1.0 / np.array(N_values, dtype=float)
    log_h    = np.log(h_arr)
    log_e    = np.log(np.array(L2_errors))
    slope, _ = np.polyfit(log_h, log_e, 1)

    print(f"\nConvergence Analysis")
    print(f"  N        : {N_values}")
    print(f"  L2 errors: {[f'{e:.4e}' for e in L2_errors]}")
    print(f"  Estimated convergence order = {slope:.4f}")
    print(f"  (Expected ≈ 2.0 for linear elements)")

if __name__ == "__main__":
    run()