import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange
import time

# ─────────────────────────────────────────────
# DOMAIN
# ─────────────────────────────────────────────
VERTICES = np.array([
    [-1.5,  1.0], [-0.75,  1.0], [ 0.0,  0.4], [ 0.75,  1.0],
    [ 1.5,  1.0], [ 1.5,  0.0], [ 1.5, -1.0], [ 0.75, -1.0],
    [ 0.75, 0.3], [ 0.0, -0.3], [-0.75,  0.3], [-0.75, -1.0],
    [-1.5, -1.0], [-1.5,  0.0]
], dtype=np.float64)

def u_exact(x, y):
    return x**3 - 3*x*y**2

# ─────────────────────────────────────────────
# GAUSSIAN QUADRATURE (order 8)
# ─────────────────────────────────────────────
def gauss_quadrature(n=8):
    return np.polynomial.legendre.leggauss(n)

# ─────────────────────────────────────────────
# BOUNDARY DISCRETIZATION
# ─────────────────────────────────────────────
def discretize_boundary(vertices, N_total):
    nv = len(vertices)
    edges = [(vertices[i], vertices[(i+1) % nv]) for i in range(nv)]
    edge_lengths = [np.linalg.norm(e[1]-e[0]) for e in edges]
    perimeter = sum(edge_lengths)

    nodes_per_edge = []
    assigned = 0
    for i, L in enumerate(edge_lengths):
        if i < nv - 1:
            n = max(1, round(N_total * L / perimeter))
        else:
            n = N_total - assigned
        nodes_per_edge.append(n)
        assigned += n

    all_nodes = []
    elements = []
    lengths = []
    normals = []

    node_idx = 0
    for i, (p0, p1) in enumerate(edges):
        n = nodes_per_edge[i]
        t_vals = np.linspace(0, 1, n+1)
        s_vals = (1 - np.cos(np.pi * t_vals)) / 2
        pts = p0[np.newaxis, :] + s_vals[:, np.newaxis] * (p1 - p0)[np.newaxis, :]

        edge_vec = p1 - p0
        L = np.linalg.norm(edge_vec)
        tang = edge_vec / L
        nrm = np.array([tang[1], -tang[0]])  # outward normal (CCW → right-hand rule)

        start = node_idx
        if i == 0:
            for pt in pts[:-1]:
                all_nodes.append(pt)
                node_idx += 1
        else:
            for pt in pts[1:-1]:
                all_nodes.append(pt)
                node_idx += 1

        for k in range(n):
            n1 = start + k if i == 0 else start + k - 1 + (1 if k > 0 else 0)
            pass

    # Simpler flat approach:
    all_nodes = []
    elements = []
    lengths = []
    normals = []

    global_nodes = []
    edge_node_indices = []

    for i, (p0, p1) in enumerate(edges):
        n = nodes_per_edge[i]
        t_vals = np.linspace(0, 1, n+1)
        s_vals = (1 - np.cos(np.pi * t_vals)) / 2
        pts = [p0 + sv * (p1 - p0) for sv in s_vals]
        edge_node_indices.append([])
        for j, pt in enumerate(pts):
            if i == 0 or j > 0:
                global_nodes.append(pt)
            edge_node_indices[-1].append(len(global_nodes)-1 if (i > 0 and j == 0) else len(global_nodes)-1)

    # Rebuild properly
    global_nodes = []
    edge_node_idx = []

    for i, (p0, p1) in enumerate(edges):
        n = nodes_per_edge[i]
        t_vals = np.linspace(0, 1, n+1)
        s_vals = (1 - np.cos(np.pi * t_vals)) / 2
        pts = p0[np.newaxis, :] + s_vals[:, np.newaxis] * (p1 - p0)[np.newaxis, :]
        idxs = []
        for j in range(n+1):
            if i == 0:
                global_nodes.append(pts[j])
                idxs.append(len(global_nodes)-1)
            else:
                if j == 0:
                    idxs.append(edge_node_idx[-1][-1])
                else:
                    global_nodes.append(pts[j])
                    idxs.append(len(global_nodes)-1)
        edge_node_idx.append(idxs)

    # last node of last edge = first node of first edge
    last_edge_last = edge_node_idx[-1][-1]
    first_node = edge_node_idx[0][0]
    # merge: replace last_edge_last references with first_node
    # Actually just pop the last node and remap
    global_nodes_arr = np.array(global_nodes[:last_edge_last] + global_nodes[last_edge_last+1:] if last_edge_last != first_node else global_nodes[:last_edge_last], dtype=np.float64)

    # Clean restart with definitive implementation
    return _discretize(vertices, nodes_per_edge)

def _discretize(vertices, nodes_per_edge):
    nv = len(vertices)
    edges = [(vertices[i], vertices[(i+1) % nv]) for i in range(nv)]

    node_list = []
    elem_list = []
    len_list = []
    nrm_list = []

    for i, (p0, p1) in enumerate(edges):
        n = nodes_per_edge[i]
        t_vals = np.linspace(0, 1, n+1)
        s_vals = (1 - np.cos(np.pi * t_vals)) / 2
        pts = p0 + np.outer(s_vals, p1 - p0)

        edge_vec = p1 - p0
        Le = np.linalg.norm(edge_vec)
        tang = edge_vec / Le
        nrm = np.array([tang[1], -tang[0]])

        start_global = len(node_list)

        if i == 0:
            for pt in pts:
                node_list.append(pt.copy())
        else:
            for pt in pts[1:]:
                node_list.append(pt.copy())

        # element node indices on this edge
        # pts[0..n] → global indices
        if i == 0:
            idx = list(range(start_global, start_global + n + 1))
        else:
            idx = [start_global - 1] + list(range(start_global, start_global + n))

        for k in range(n):
            seg_len = np.linalg.norm(pts[k+1] - pts[k])
            elem_list.append([idx[k], idx[k+1]])
            len_list.append(seg_len)
            nrm_list.append(nrm.copy())

    # Close: last node of boundary = first node
    # The last element connects last node added to node 0
    # Fix: replace last node index in elems with 0
    total_nodes_raw = len(node_list)
    # Last node should be same as node 0
    # Find elements referencing the last raw node index
    last_raw = total_nodes_raw - 1
    for e in elem_list:
        if e[1] == last_raw:
            e[1] = 0
        if e[0] == last_raw:
            e[0] = 0

    # Remove duplicate last node
    node_arr = np.array(node_list[:-1], dtype=np.float64)
    # Update indices >= last_raw: no indices should be last_raw now
    # (already replaced above), but indices > last_raw don't exist.

    elem_arr = np.array(elem_list, dtype=np.int32)
    len_arr  = np.array(len_list,  dtype=np.float64)
    nrm_arr  = np.array(nrm_list,  dtype=np.float64)

    return node_arr, elem_arr, len_arr, nrm_arr

def build_discretization(N_total):
    nv = len(VERTICES)
    edges = [(VERTICES[i], VERTICES[(i+1) % nv]) for i in range(nv)]
    edge_lengths = np.array([np.linalg.norm(e[1]-e[0]) for e in edges])
    perimeter = edge_lengths.sum()

    raw = np.round(N_total * edge_lengths / perimeter).astype(int)
    raw = np.maximum(raw, 1)
    diff = N_total - raw.sum()
    # distribute remainder
    order = np.argsort(-edge_lengths)
    for k in range(abs(diff)):
        if diff > 0:
            raw[order[k % nv]] += 1
        else:
            if raw[order[k % nv]] > 1:
                raw[order[k % nv]] -= 1

    return _discretize(VERTICES, raw)

# ─────────────────────────────────────────────
# JUMP TERM (Equipotential / Solid-angle trick)
# ─────────────────────────────────────────────
def compute_jump_terms(nodes, elements, lengths, normals):
    """
    c_i = -1/(2π) * interior_angle_i  (for CCW, outward normal convention)
    But the standard result for the interior Dirichlet BIE with DLP:
      (c_i + K) μ = f
    where c_i = interior solid angle / (2π).
    For interior problem, jump from exterior: c(x) = θ_int/(2π).
    We use the Rigid Body / Equipotential trick:
      sum over all elements of K_ij = -c_i  (row sum = -c_i)
    But we compute c_i directly from interior angle.
    """
    N = len(nodes)
    nv = len(VERTICES)
    # Map each node to its adjacent edges to get interior angle
    # For a smooth boundary: c_i = 1/2
    # For a corner with interior angle θ_int: c_i = θ_int / (2π)

    # Build node→element adjacency
    node_elems = [[] for _ in range(N)]
    for e_idx, (n1, n2) in enumerate(elements):
        node_elems[n1].append(e_idx)
        node_elems[n2].append(e_idx)

    c = np.zeros(N)
    for i in range(N):
        adj = node_elems[i]
        if len(adj) < 2:
            c[i] = 0.5
            continue
        # Get the two tangent vectors of adjacent elements at node i
        # For each adjacent element, find tangent pointing AWAY from node i
        tangents = []
        for e_idx in adj:
            n1, n2 = elements[e_idx]
            Le = lengths[e_idx]
            tang = (nodes[n2] - nodes[n1]) / Le
            if n2 == i:
                tang = -tang  # point away from i
            tangents.append(tang)
        if len(tangents) < 2:
            c[i] = 0.5
            continue
        t1, t2 = tangents[0], tangents[1]
        # Interior angle: angle between inward tangents
        # t1 points along boundary AWAY from node on one side
        # t2 points along boundary AWAY from node on other side
        # The interior angle is the angle you turn through staying inside
        cross = t1[0]*t2[1] - t1[1]*t2[0]
        dot   = t1[0]*t2[0] + t1[1]*t2[1]
        angle_between = np.arctan2(cross, dot)  # angle from t1 to t2
        # Interior angle = π - angle_between for convex corners
        # More carefully: use the exterior angle
        # For CCW boundary, interior angle at corner:
        # The two edges meeting at node i:
        # incoming edge tangent (toward i): -t_incoming_away
        # outgoing edge tangent (away from i): t_outgoing_away
        # We need to identify which is incoming and which is outgoing
        # Incoming: element where node i == n2
        # Outgoing: element where node i == n1
        in_tang = None
        out_tang = None
        for e_idx in adj:
            n1e, n2e = elements[e_idx]
            Le = lengths[e_idx]
            tang_fwd = (nodes[n2e] - nodes[n1e]) / Le
            if n2e == i:
                in_tang = tang_fwd   # incoming direction (points TO i)
            elif n1e == i:
                out_tang = tang_fwd  # outgoing direction (points FROM i)
        if in_tang is None or out_tang is None:
            c[i] = 0.5
            continue
        # Interior angle: angle between (-in_tang) and (out_tang) measured CCW inside domain
        v1 = -in_tang   # direction along boundary arriving at i, reversed
        v2 = out_tang   # direction along boundary leaving i
        cross2 = v1[0]*v2[1] - v1[1]*v2[0]
        dot2   = v1[0]*v2[0] + v1[1]*v2[1]
        theta = np.arctan2(cross2, dot2)
        if theta <= 0:
            theta += 2*np.pi
        # theta is now the interior angle
        c[i] = theta / (2*np.pi)
    return c

# ─────────────────────────────────────────────
# NUMBA MATVEC KERNEL
# ─────────────────────────────────────────────
@njit(parallel=True, cache=True)
def _matvec_kernel(mu, nodes, elements, lengths, normals, c_diag, gp, gw):
    N  = nodes.shape[0]
    Ne = elements.shape[0]
    ng = gp.shape[0]
    result = np.zeros(N)

    for i in prange(N):
        xi = nodes[i, 0]
        yi = nodes[i, 1]
        val = c_diag[i] * mu[i]

        for e in range(Ne):
            n1 = elements[e, 0]
            n2 = elements[e, 1]
            # skip singular: collocation node is endpoint of element
            if n1 == i or n2 == i:
                continue

            Le   = lengths[e]
            nx   = normals[e, 0]
            ny   = normals[e, 1]
            x1   = nodes[n1, 0]; y1 = nodes[n1, 1]
            x2   = nodes[n2, 0]; y2 = nodes[n2, 1]
            mu1  = mu[n1]; mu2 = mu[n2]

            half_L = 0.5 * Le
            cx_e   = 0.5 * (x1 + x2)
            cy_e   = 0.5 * (y1 + y2)

            for q in range(ng):
                s  = gp[q]             # ∈ [-1,1]
                w  = gw[q]
                phi1 = 0.5 * (1.0 - s)
                phi2 = 0.5 * (1.0 + s)
                # physical point
                yx = phi1 * x1 + phi2 * x2
                yy = phi1 * y1 + phi2 * y2
                mu_q = phi1 * mu1 + phi2 * mu2

                rx = xi - yx
                ry = yi - yy
                r2 = rx*rx + ry*ry
                if r2 < 1e-30:
                    continue
                rdn = rx*nx + ry*ny
                kernel = (1.0 / (2.0 * np.pi)) * rdn / r2
                val += w * half_L * kernel * mu_q

        result[i] = val
    return result

# ─────────────────────────────────────────────
# POLYGON UTILITIES
# ─────────────────────────────────────────────
@njit(cache=True)
def _ray_cast(px, py, verts):
    n = verts.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi = verts[i, 0]; yi = verts[i, 1]
        xj = verts[j, 0]; yj = verts[j, 1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi)*(py - yi)/(yj - yi) + xi):
            inside = not inside
        j = i
    return inside

@njit(parallel=True, cache=True)
def points_in_polygon_numba(pts, verts):
    n = pts.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    for k in prange(n):
        mask[k] = _ray_cast(pts[k, 0], pts[k, 1], verts)
    return mask

@njit(parallel=True, cache=True)
def min_dist_to_boundary_numba(pts, verts):
    nv = verts.shape[0]
    np_ = pts.shape[0]
    dist = np.full(np_, 1e18)
    for k in prange(np_):
        px = pts[k, 0]; py = pts[k, 1]
        d_min = 1e18
        for i in range(nv):
            j = (i + 1) % nv
            ax = verts[i, 0]; ay = verts[i, 1]
            bx = verts[j, 0]; by = verts[j, 1]
            dx = bx - ax; dy = by - ay
            L2 = dx*dx + dy*dy
            if L2 < 1e-30:
                d = (px-ax)**2 + (py-ay)**2
            else:
                t = ((px-ax)*dx + (py-ay)*dy) / L2
                t = max(0.0, min(1.0, t))
                cx = ax + t*dx; cy = ay + t*dy
                d = (px-cx)**2 + (py-cy)**2
            if d < d_min:
                d_min = d
        dist[k] = np.sqrt(d_min)
    return dist

# ─────────────────────────────────────────────
# INTERIOR EVALUATION
# ─────────────────────────────────────────────
@njit(parallel=True, cache=True)
def _eval_interior(pts, mu, nodes, elements, lengths, normals, gp, gw):
    Np = pts.shape[0]
    Ne = elements.shape[0]
    ng = gp.shape[0]
    u_num = np.zeros(Np)

    for k in prange(Np):
        xi = pts[k, 0]; yi = pts[k, 1]
        val = 0.0
        for e in range(Ne):
            n1 = elements[e, 0]; n2 = elements[e, 1]
            Le   = lengths[e]
            nx   = normals[e, 0]; ny = normals[e, 1]
            x1   = nodes[n1, 0]; y1 = nodes[n1, 1]
            x2   = nodes[n2, 0]; y2 = nodes[n2, 1]
            mu1  = mu[n1]; mu2 = mu[n2]
            half_L = 0.5 * Le

            for q in range(ng):
                s    = gp[q]; w = gw[q]
                phi1 = 0.5*(1.0-s); phi2 = 0.5*(1.0+s)
                yx   = phi1*x1 + phi2*x2
                yy_  = phi1*y1 + phi2*y2
                mu_q = phi1*mu1 + phi2*mu2
                rx   = xi - yx; ry = yi - yy_
                r2   = rx*rx + ry*ry
                if r2 < 1e-30:
                    continue
                rdn  = rx*nx + ry*ny
                kernel = (1.0/(2.0*np.pi)) * rdn / r2
                val   += w * half_L * kernel * mu_q
        u_num[k] = val
    return u_num

# ─────────────────────────────────────────────
# MAIN REFINEMENT STUDY
# ─────────────────────────────────────────────
def run_study():
    gp, gw = gauss_quadrature(8)
    gp = np.asarray(gp, dtype=np.float64)
    gw = np.asarray(gw, dtype=np.float64)

    N_values = [400, 800, 1600, 3200, 6400]
    L2_errors = []

    # Pre-warm Numba
    _dummy_pts = np.array([[0.0, 0.0]])
    _dummy_v   = VERTICES
    points_in_polygon_numba(_dummy_pts, _dummy_v)
    min_dist_to_boundary_numba(_dummy_pts, _dummy_v)

    # Build evaluation grid (fixed)
    ngrid = 200
    xs = np.linspace(-1.5, 1.5, ngrid)
    ys = np.linspace(-1.0,  1.0, ngrid)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])

    interior_mask = points_in_polygon_numba(all_pts, VERTICES)
    interior_pts  = all_pts[interior_mask]

    nv        = len(VERTICES)
    edges_v   = [(VERTICES[i], VERTICES[(i+1) % nv]) for i in range(nv)]
    perimeter = sum(np.linalg.norm(e[1]-e[0]) for e in edges_v)
    N_min     = min(N_values)
    h_coarse  = perimeter / N_min
    delta     = 2 * h_coarse

    dist_to_bnd = min_dist_to_boundary_numba(interior_pts, VERTICES)
    grid_pts    = interior_pts[dist_to_bnd > delta]
    u_ex_grid   = u_exact(grid_pts[:, 0], grid_pts[:, 1])

    header = (
        f"{'N':>6} | {'Unknowns':>8} | {'GMRES its':>9} | "
        f"{'L2 error':>12} | {'Setup(s)':>8} | {'Solve(s)':>8} | "
        f"{'Eval(s)':>8} | {'Total(s)':>8}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for N in N_values:
        t0 = time.perf_counter()

        nodes, elements, lengths, normals = build_discretization(N)
        Nn = len(nodes)
        c_diag = compute_jump_terms(nodes, elements, lengths, normals)

        # Warm up matvec JIT on first call
        _test_mu = np.ones(Nn)
        _matvec_kernel(_test_mu, nodes, elements, lengths, normals, c_diag, gp, gw)

        t_setup = time.perf_counter() - t0

        # RHS
        f_rhs = u_exact(nodes[:, 0], nodes[:, 1])

        # Linear operator
        iters = [0]
        def matvec(mu_vec):
            iters[0] += 1
            return _matvec_kernel(mu_vec, nodes, elements, lengths, normals, c_diag, gp, gw)

        A = LinearOperator((Nn, Nn), matvec=matvec, dtype=np.float64)

        t1 = time.perf_counter()
        mu_sol, info = gmres(
            A, f_rhs,
            rtol=1e-10, atol=1e-10,
            restart=min(Nn, 200),
            callback_type='pr_norm'
        )
        t_solve = time.perf_counter() - t1

        if info != 0:
            print(f"  GMRES did not converge (info={info})")

        # Interior evaluation
        t2 = time.perf_counter()
        u_num = _eval_interior(grid_pts, mu_sol, nodes, elements, lengths, normals, gp, gw)
        t_eval = time.perf_counter() - t2

        err_vec = u_num - u_ex_grid
        l2_err = np.linalg.norm(err_vec) / np.linalg.norm(u_ex_grid)
        L2_errors.append(l2_err)

        t_total = time.perf_counter() - t0

        print(
            f"{N:>6} | {Nn:>8} | {iters[0]:>9} | "
            f"{l2_err:>12.4e} | {t_setup:>8.2f} | {t_solve:>8.2f} | "
            f"{t_eval:>8.2f} | {t_total:>8.2f}"
        )

    print("=" * len(header))

    # Convergence analysis
    h_vals   = 1.0 / np.array(N_values, dtype=np.float64)
    log_h    = np.log(h_vals)
    log_err  = np.log(np.array(L2_errors))
    slope, _ = np.polyfit(log_h, log_err, 1)

    print(f"\nConvergence Analysis")
    print(f"  h values : {h_vals}")
    print(f"  L2 errors: {np.array(L2_errors)}")
    print(f"  Estimated convergence order = {slope:.4f}")
    print(f"  (Expected ~2.0 for linear elements)")

if __name__ == "__main__":
    run_study()