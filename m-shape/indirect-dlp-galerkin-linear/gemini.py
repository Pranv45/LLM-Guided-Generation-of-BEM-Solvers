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
            j_mod = j if j < n_verts else 0 # Avoid % in loop
            p2x, p2y = poly[j_mod, 0], poly[j_mod, 1]
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
            j_next = j + 1
            if j_next == n_verts: j_next = 0

            p1x, p1y = poly[j, 0], poly[j, 1]
            p2x, p2y = poly[j_next, 0], poly[j_next, 1]

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
# Numba Kernels for BEM Integrals (Galerkin, Linear Elements)
# -------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def assemble_system(nodes, elements, lengths, normals, q_pts_1d, q_wts_1d):
    N = len(nodes)
    Ne = len(elements)
    Nq = len(q_pts_1d)

    A = np.zeros((N, N))
    b = np.zeros(N)
    m_vec = np.zeros(N)

    pts01 = (q_pts_1d + 1.0) / 2.0
    wts01 = q_wts_1d / 2.0

    # 1. Pre-compute neighbor mappings (Avoids O(N^2) search overhead)
    node2elems = np.full((N, 2), -1, dtype=np.int32)
    node2phi = np.full((N, 2), -1, dtype=np.int32)
    counts = np.zeros(N, dtype=np.int32)

    for e in range(Ne):
        n1 = elements[e, 0]
        n2 = elements[e, 1]
        if counts[n1] < 2:
            node2elems[n1, counts[n1]] = e
            node2phi[n1, counts[n1]] = 1
            counts[n1] += 1
        if counts[n2] < 2:
            node2elems[n2, counts[n2]] = e
            node2phi[n2, counts[n2]] = 2
            counts[n2] += 1

    # Thread-safe assembly
    for i in prange(N):
        e_list_0 = node2elems[i, 0]
        e_list_1 = node2elems[i, 1]
        pidx_0 = node2phi[i, 0]
        pidx_1 = node2phi[i, 1]

        # Outer integral over the two attached elements
        for k in range(2):
            e_x = e_list_0 if k == 0 else e_list_1
            pidx = pidx_0 if k == 0 else pidx_1

            Lx = lengths[e_x]
            nx1 = elements[e_x, 0]
            nx2 = elements[e_x, 1]

            # SCALARIZATION: No array creation
            px1_0 = nodes[nx1, 0]; px1_1 = nodes[nx1, 1]
            px2_0 = nodes[nx2, 0]; px2_1 = nodes[nx2, 1]

            m_vec[i] += 0.5 * Lx

            if pidx == 1:
                A[i, nx1] += -0.5 * (Lx / 3.0)
                A[i, nx2] += -0.5 * (Lx / 6.0)
            else:
                A[i, nx1] += -0.5 * (Lx / 6.0)
                A[i, nx2] += -0.5 * (Lx / 3.0)

            for qx in range(Nq):
                sx = q_pts_1d[qx]
                wx = q_wts_1d[qx] * 0.5 * Lx
                phix1 = 0.5 * (1.0 - sx)
                phix2 = 0.5 * (1.0 + sx)
                phix_i = phix1 if pidx == 1 else phix2

                xc_0 = phix1 * px1_0 + phix2 * px2_0
                xc_1 = phix1 * px1_1 + phix2 * px2_1
                f_val = u_exact(xc_0, xc_1)
                b[i] += phix_i * f_val * wx

            # Double-Layer Integral Assembly
            for e_y in range(Ne):
                if e_x == e_y:
                    continue

                ny1 = elements[e_y, 0]
                ny2 = elements[e_y, 1]
                Ly = lengths[e_y]
                nrm_y_0 = normals[e_y, 0]; nrm_y_1 = normals[e_y, 1]
                py1_0 = nodes[ny1, 0]; py1_1 = nodes[ny1, 1]
                py2_0 = nodes[ny2, 0]; py2_1 = nodes[ny2, 1]

                is_adj = False
                shared_x_at_1 = False; shared_y_at_1 = False

                if nx2 == ny1:
                    is_adj = True; shared_x_at_1 = True; shared_y_at_1 = False
                elif nx1 == ny2:
                    is_adj = True; shared_x_at_1 = False; shared_y_at_1 = True
                elif nx1 == ny1:
                    is_adj = True; shared_x_at_1 = False; shared_y_at_1 = False
                elif nx2 == ny2:
                    is_adj = True; shared_x_at_1 = True; shared_y_at_1 = True

                if is_adj:
                    # DUFFY TRANSFORMATION
                    for qu in range(Nq):
                        u = pts01[qu]
                        wu = wts01[qu]
                        for qv in range(Nq):
                            v = pts01[qv]
                            wv = wts01[qv]

                            for tri in range(2):
                                if tri == 0:
                                    s_val, t_val = u, u*v
                                else:
                                    s_val, t_val = u*v, u

                                weight = wu * wv * u * Lx * Ly

                                sx_ref = (1.0 - 2.0*s_val) if shared_x_at_1 else (-1.0 + 2.0*s_val)
                                sy_ref = (1.0 - 2.0*t_val) if shared_y_at_1 else (-1.0 + 2.0*t_val)

                                phi_x1 = 0.5 * (1.0 - sx_ref); phi_x2 = 0.5 * (1.0 + sx_ref)
                                phi_y1 = 0.5 * (1.0 - sy_ref); phi_y2 = 0.5 * (1.0 + sy_ref)
                                phi_xi = phi_x1 if pidx == 1 else phi_x2

                                xc_0 = phi_x1 * px1_0 + phi_x2 * px2_0
                                xc_1 = phi_x1 * px1_1 + phi_x2 * px2_1
                                yc_0 = phi_y1 * py1_0 + phi_y2 * py2_0
                                yc_1 = phi_y1 * py1_1 + phi_y2 * py2_1

                                rx = xc_0 - yc_0
                                ry = xc_1 - yc_1
                                r2 = rx*rx + ry*ry

                                if r2 > 1e-14:
                                    rdotn = rx*nrm_y_0 + ry*nrm_y_1
                                    kernel = (1.0 / (2.0 * np.pi)) * (rdotn / r2)
                                    A[i, ny1] += phi_xi * kernel * phi_y1 * weight
                                    A[i, ny2] += phi_xi * kernel * phi_y2 * weight
                else:
                    # STANDARD TENSOR-PRODUCT GAUSS INTEGRATION
                    for qx in range(Nq):
                        sx = q_pts_1d[qx]
                        wx = q_wts_1d[qx] * 0.5 * Lx
                        phi_x1 = 0.5 * (1.0 - sx)
                        phi_x2 = 0.5 * (1.0 + sx)
                        phi_xi = phi_x1 if pidx == 1 else phi_x2
                        xc_0 = phi_x1 * px1_0 + phi_x2 * px2_0
                        xc_1 = phi_x1 * px1_1 + phi_x2 * px2_1

                        for qy in range(Nq):
                            sy = q_pts_1d[qy]
                            wy = q_wts_1d[qy] * 0.5 * Ly
                            phi_y1 = 0.5 * (1.0 - sy)
                            phi_y2 = 0.5 * (1.0 + sy)
                            yc_0 = phi_y1 * py1_0 + phi_y2 * py2_0
                            yc_1 = phi_y1 * py1_1 + phi_y2 * py2_1

                            rx = xc_0 - yc_0
                            ry = xc_1 - yc_1
                            r2 = rx*rx + ry*ry

                            rdotn = rx*nrm_y_0 + ry*nrm_y_1
                            kernel = (1.0 / (2.0 * np.pi)) * (rdotn / r2)
                            weight = wx * wy

                            A[i, ny1] += phi_xi * kernel * phi_y1 * weight
                            A[i, ny2] += phi_xi * kernel * phi_y2 * weight

        # Equipotential Trick
        row_sum_off = 0.0
        for j in range(N):
            if j != i:
                row_sum_off += A[i, j]
        A[i, i] = -m_vec[i] - row_sum_off

    return A, b

@njit(parallel=True, fastmath=True)
def eval_interior_full(eval_pts, mu, nodes, elements, quad_pts, quad_wts, normals, q_pts_1d):
    N_eval = len(eval_pts)
    Ne = len(elements)
    Nq = quad_wts.shape[1]
    out = np.zeros(N_eval)

    for i in prange(N_eval):
        xi = eval_pts[i, 0]
        yi = eval_pts[i, 1]
        integral = 0.0

        for e in range(Ne):
            n1 = elements[e, 0]
            n2 = elements[e, 1]

            nx = normals[e, 0]
            ny = normals[e, 1]
            mu1 = mu[n1]
            mu2 = mu[n2]

            for q in range(Nq):
                qx = quad_pts[e, q, 0]
                qy = quad_pts[e, q, 1]
                qw = quad_wts[e, q]
                s = q_pts_1d[q]

                phi1 = 0.5 * (1.0 - s)
                phi2 = 0.5 * (1.0 + s)
                mu_q = phi1 * mu1 + phi2 * mu2

                rx = xi - qx
                ry = yi - qy
                r2 = rx*rx + ry*ry

                rdotn = rx*nx + ry*ny
                integral += qw * (rdotn / r2) * mu_q

        out[i] = (1.0 / (2.0 * np.pi)) * integral
    return out

# -------------------------------------------------------------------------
# GMRES Helper Class
# -------------------------------------------------------------------------
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

    VERTICES = VERTICES[::-1]

    # Pre-compile Numba
    dummy_pts = np.zeros((2, 2))
    dummy_el = np.zeros((2, 2), dtype=np.int32)
    _ = points_in_polygon(dummy_pts, VERTICES)
    _ = min_dist_to_boundary(dummy_pts, VERTICES)
    _ = assemble_system(dummy_pts, dummy_el, np.zeros(2), np.zeros((2,2)), np.zeros(1), np.zeros(1))
    _ = eval_interior_full(dummy_pts, np.zeros(2), dummy_pts, dummy_el, np.zeros((2,1,2)), np.zeros((2,1)), np.zeros((2,2)), np.zeros(1))

    ngrid = 200
    xs = np.linspace(-1.5, 1.5, ngrid)
    ys = np.linspace(-1.0, 1.0, ngrid)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])

    mask_in    = points_in_polygon(all_pts, VERTICES)
    interior   = all_pts[mask_in].copy()

    N_values = [400, 800, 1600, 3200, 6400]

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

    print(f"{'N':<5} | {'Unknowns':<8} | {'Iters':<5} | {'Rel L2 Error':<12} | {'Setup (s)':<9} | {'Solve (s)':<9} | {'Eval (s)':<9} | {'Total (s)':<9}")
    print("-" * 88)

    q_pts_1d, q_wts_1d = np.polynomial.legendre.leggauss(10)
    errors = []

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

        nodes_list = []
        for i in range(nv_):
            p1 = VERTICES[i]
            p2 = VERTICES[(i + 1) % nv_]
            Ne_edge = counts[i]
            t = np.linspace(0, 1, Ne_edge + 1)
            s = (1.0 - np.cos(np.pi * t)) / 2.0
            edge_vec = p2 - p1
            for j in range(Ne_edge):
                nodes_list.append(p1 + s[j] * edge_vec)

        nodes = np.array(nodes_list)
        actual_N = len(nodes)

        elements = np.zeros((actual_N, 2), dtype=np.int32)
        normals = np.zeros((actual_N, 2))
        lengths = np.zeros(actual_N)
        quad_pts = np.zeros((actual_N, len(q_pts_1d), 2))
        quad_wts = np.zeros((actual_N, len(q_pts_1d)))

        for i in range(actual_N):
            n1 = i
            n2 = (i + 1) % actual_N
            elements[i, 0] = n1
            elements[i, 1] = n2

            p1 = nodes[n1]
            p2 = nodes[n2]

            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            lengths[i] = edge_len
            normals[i] = np.array([edge_vec[1], -edge_vec[0]]) / edge_len

            for q in range(len(q_pts_1d)):
                xi = q_pts_1d[q]
                quad_pts[i, q] = 0.5 * (1 - xi) * p1 + 0.5 * (1 + xi) * p2
                quad_wts[i, q] = 0.5 * edge_len * q_wts_1d[q]

        A, b = assemble_system(nodes, elements, lengths, normals, q_pts_1d, q_wts_1d)
        t_setup = time.time() - t_start

        t_solve_start = time.time()
        counter = GMRESCounter()

        # NOTE: At very large dense scales (N >= 3200), scipy.linalg.solve (LAPACK)
        # may actually beat GMRES due to BLAS3 vectorization.
        mu, info = spla.gmres(
            A, b,
            rtol=1e-10, atol=1e-10,
            restart=actual_N,
            callback=counter,
            callback_type='pr_norm'
        )
        t_solve = time.time() - t_solve_start

        t_eval_start = time.time()
        u_num_interior = eval_interior_full(grid_pts, mu, nodes, elements, quad_pts, quad_wts, normals, q_pts_1d)
        t_eval = time.time() - t_eval_start

        err_l2 = np.linalg.norm(u_num_interior - u_ex_interior) / norm_u_ex
        errors.append(err_l2)
        t_total = time.time() - t_start

        print(f"{actual_N:<5} | {actual_N:<8} | {counter.niter:<5} | {err_l2:<12.5e} | {t_setup:<9.4f} | {t_solve:<9.4f} | {t_eval:<9.4f} | {t_total:<9.4f}")

    print("-" * 88)

    h_vals = 1.0 / np.array(N_values)
    log_h = np.log(h_vals)
    log_err = np.log(errors)
    slope, _ = np.polyfit(log_h, log_err, 1)

    print(f"Estimated convergence order = {slope:.4f}")

if __name__ == '__main__':
    main()