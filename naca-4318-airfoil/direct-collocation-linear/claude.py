"""
2D BEM Solver — Direct Collocation, Mixed BC, Linear Elements
Domain: NACA 4318 airfoil interior
Manufactured solution: u(x,y) = x^3 - 3xy^2
"""

import numpy as np
import time
import sys
from numba import njit, prange

# ============================================================
# 1. NACA 4-digit geometry
# ============================================================

def naca4_points(m, p, t, N_elem):
    n_half = N_elem // 2
    beta = np.linspace(0, np.pi, n_half + 1)
    xc = 0.5 * (1 - np.cos(beta))

    yc  = np.where(xc < p,
                   m/p**2*(2*p*xc - xc**2),
                   m/(1-p)**2*((1-2*p) + 2*p*xc - xc**2))
    dyc = np.where(xc < p,
                   2*m/p**2*(p - xc),
                   2*m/(1-p)**2*(p - xc))
    yt  = 5*t*(0.2969*np.sqrt(np.maximum(xc,0)) - 0.1260*xc
               - 0.3516*xc**2 + 0.2843*xc**3 - 0.1015*xc**4)
    theta = np.arctan(dyc)

    xu = xc - yt*np.sin(theta);  yu = yc + yt*np.cos(theta)
    xl = xc + yt*np.sin(theta);  yl = yc - yt*np.cos(theta)

    upper_x = xu[::-1];  upper_y = yu[::-1]
    lower_x = xl[1:];    lower_y = yl[1:]

    xs = np.concatenate([upper_x, lower_x])
    ys = np.concatenate([upper_y, lower_y])
    xs[0] = 1.0;  ys[0] = 0.0
    xs[-1] = 1.0; ys[-1] = 0.0

    nodes = np.column_stack([xs, ys])
    elements = np.column_stack([
        np.arange(N_elem), np.arange(1, N_elem+1)
    ]).astype(np.int32)

    dx = nodes[elements[:,1],0] - nodes[elements[:,0],0]
    dy = nodes[elements[:,1],1] - nodes[elements[:,0],1]
    lengths = np.sqrt(dx**2 + dy**2)
    normals = np.column_stack([dy/lengths, -dx/lengths])

    return nodes, elements, lengths, normals


def assign_bcs(nodes, split_x=0.8):
    return (nodes[:,0] >= split_x).astype(np.int32)


def u_exact(x, y):
    return x**3 - 3*x*y**2

def q_exact(x, y, nx, ny):
    return (3*x**2 - 3*y**2)*nx + (-6*x*y)*ny


# ============================================================
# 2. Gaussian quadrature on [0,1]
# ============================================================

def gauss_ref(n):
    pts, wts = np.polynomial.legendre.leggauss(n)
    return 0.5*(pts+1), 0.5*wts


# ============================================================
# 3. Analytical singular integrals for G
# ============================================================

def G_singular_analytical(xi_col, L):
    ng = 64
    gp, gw = gauss_ref(ng)

    def integrate_phi_log(a, b, s_lo, s_hi, xi_sing):
        results = 0.0
        intervals = []
        if s_lo < xi_sing < s_hi:
            intervals = [(s_lo, xi_sing), (xi_sing, s_hi)]
        else:
            intervals = [(s_lo, s_hi)]
        for (lo, hi) in intervals:
            xi = lo + (hi-lo)*gp
            val = (a + b*xi) * np.log(L*np.abs(xi - xi_sing + 1e-300)) * (hi-lo)
            results += np.dot(gw, val)
        return results

    eps = 1e-14
    if xi_col <= eps:
        g1 = L * (-1/(2*np.pi)) * (np.log(L)*0.5 + (-3/4))
        g2 = L * (-1/(2*np.pi)) * (np.log(L)*0.5 + (-1/4))
    elif xi_col >= 1-eps:
        g1 = L * (-1/(2*np.pi)) * (np.log(L)*0.5 + (-1/4))
        g2 = L * (-1/(2*np.pi)) * (np.log(L)*0.5 + (-3/4))
    else:
        g1 = L * (-1/(2*np.pi)) * integrate_phi_log(1, -1, 0, 1, xi_col)
        g2 = L * (-1/(2*np.pi)) * integrate_phi_log(0,  1, 0, 1, xi_col)

    return g1, g2


# ============================================================
# 4. Numba kernels
# ============================================================

@njit(parallel=True, cache=True)
def _assemble_HG_nb(nodes_x, nodes_y, elem0, elem1,
                     lengths, normals_x, normals_y,
                     G_sing0, G_sing1,          # (N_nodes,) precomputed singular G contributions
                     sing_na, sing_nb,           # (N_nodes,) element endpoints for singular elems
                     gp, gw):
    """
    Full collocation assembly parallelised over rows (collocation nodes).
    Singular rows are handled via precomputed G_sing values.
    Returns H (N_nodes, N_nodes) and G (N_nodes, N_nodes).
    """
    N_nodes = nodes_x.shape[0]
    N_elem  = elem0.shape[0]
    ng      = gp.shape[0]
    INV2PI  = 1.0 / (6.283185307179586)

    H = np.zeros((N_nodes, N_nodes))
    G = np.zeros((N_nodes, N_nodes))

    for i in prange(N_nodes):
        xi_x = nodes_x[i]; xi_y = nodes_y[i]

        for e in range(N_elem):
            na = elem0[e]; nb = elem1[e]
            Le  = lengths[e]
            nyx = normals_x[e]; nyy = normals_y[e]
            ax  = nodes_x[na]; ay = nodes_y[na]
            bx  = nodes_x[nb]; by = nodes_y[nb]

            on_elem = (na == i or nb == i)

            if on_elem:
                # Singular: H=0, G from precomputed analytical values
                if na == i:
                    G[i, na] += G_sing0[e]
                    G[i, nb] += G_sing1[e]
                else:
                    # nb == i  → local_xi=1, g1↔g2 swapped
                    G[i, na] += G_sing1[e]
                    G[i, nb] += G_sing0[e]
                # H contribution is 0
            else:
                # Regular Gauss
                sumG0 = 0.0; sumG1 = 0.0
                sumH0 = 0.0; sumH1 = 0.0
                for k in range(ng):
                    xi  = gp[k]; w = gw[k]
                    yx  = ax + xi*(bx-ax)
                    yy  = ay + xi*(by-ay)
                    rx  = xi_x - yx; ry = xi_y - yy
                    r2  = rx*rx + ry*ry
                    if r2 < 1e-300: r2 = 1e-300
                    kG  = -INV2PI * 0.5 * np.log(r2) * Le
                    kH  =  INV2PI * (rx*nyx + ry*nyy) / r2 * Le
                    phi1 = 1.0 - xi; phi2 = xi
                    sumG0 += w * kG * phi1
                    sumG1 += w * kG * phi2
                    sumH0 += w * kH * phi1
                    sumH1 += w * kH * phi2
                G[i, na] += sumG0; G[i, nb] += sumG1
                H[i, na] += sumH0; H[i, nb] += sumH1

    # Equipotential trick
    for i in range(N_nodes):
        s = 0.0
        for j in range(N_nodes):
            if j != i:
                s += H[i, j]
        H[i, i] = -s

    return H, G


@njit(parallel=True, cache=True)
def _eval_interior_nb(pts_x, pts_y,
                       qx_e, qy_e,
                       u_e, q_e,
                       ny_x, ny_y, Le,
                       gw):
    N_pts  = pts_x.shape[0]
    N_elem = qx_e.shape[0]
    ng     = gw.shape[0]
    INV2PI = 1.0 / (6.283185307179586)
    u_int  = np.zeros(N_pts)

    for k in prange(N_pts):
        xp = pts_x[k]; yp = pts_y[k]
        val = 0.0
        for e in range(N_elem):
            Le_e = Le[e]
            for j in range(ng):
                rx = xp - qx_e[e, j];  ry = yp - qy_e[e, j]
                r2 = rx*rx + ry*ry
                if r2 < 1e-300: r2 = 1e-300
                kG = -INV2PI * 0.5 * np.log(r2)
                kH =  INV2PI * (rx*ny_x[e] + ry*ny_y[e]) / r2
                val += gw[j] * (kG*q_e[e,j] - kH*u_e[e,j]) * Le_e
        u_int[k] = val

    return u_int


# ============================================================
# 5. Matrix assembly (Python wrapper)
# ============================================================

def assemble_HG(nodes, elements, lengths, normals, ng=10):
    N_nodes = len(nodes)
    N_elem  = len(elements)

    gp, gw = gauss_ref(ng)
    gp = np.ascontiguousarray(gp)
    gw = np.ascontiguousarray(gw)

    # Precompute singular G contributions for each element (xi=0 end)
    G_sing0 = np.zeros(N_elem)   # contribution to node_a (local xi=0)
    G_sing1 = np.zeros(N_elem)   # contribution to node_b (local xi=1)
    for e in range(N_elem):
        g1, g2 = G_singular_analytical(0.0, lengths[e])
        G_sing0[e] = g1   # phi1 end (na, local xi=0)
        G_sing1[e] = g2   # phi2 end (nb, local xi=1)

    nodes_x  = np.ascontiguousarray(nodes[:,0])
    nodes_y  = np.ascontiguousarray(nodes[:,1])
    elem0    = np.ascontiguousarray(elements[:,0])
    elem1    = np.ascontiguousarray(elements[:,1])
    norms_x  = np.ascontiguousarray(normals[:,0])
    norms_y  = np.ascontiguousarray(normals[:,1])

    H, G = _assemble_HG_nb(
        nodes_x, nodes_y, elem0, elem1,
        np.ascontiguousarray(lengths), norms_x, norms_y,
        G_sing0, G_sing1, elem0, elem1,
        gp, gw)

    return H, G


# ============================================================
# 6. Solve mixed BVP
# ============================================================

def solve_bem(nodes, elements, lengths, normals, H, G, split_x=0.8):
    N_nodes = len(nodes)
    node_bc = assign_bcs(nodes, split_x)

    node_n = np.zeros((N_nodes, 2))
    node_c = np.zeros(N_nodes)
    for e in range(len(elements)):
        for loc in range(2):
            n = elements[e,loc]
            node_n[n] += normals[e]
            node_c[n] += 1
    for n in range(N_nodes):
        if node_c[n] > 0:
            nl = np.linalg.norm(node_n[n]/node_c[n])
            if nl > 0: node_n[n] = (node_n[n]/node_c[n]) / nl

    u_known = np.zeros(N_nodes)
    q_known = np.zeros(N_nodes)
    for i in range(N_nodes):
        xi, yi = nodes[i]
        if node_bc[i] == 1:
            u_known[i] = u_exact(xi, yi)
        else:
            q_known[i] = q_exact(xi, yi, node_n[i,0], node_n[i,1])

    D_idx = np.where(node_bc==1)[0]
    N_idx = np.where(node_bc==0)[0]

    A = np.zeros((N_nodes, N_nodes))
    b = np.zeros(N_nodes)

    for k, i in enumerate(N_idx):   A[:, k]            =  H[:, i]
    for k, i in enumerate(D_idx):   A[:, len(N_idx)+k] = -G[:, i]

    for i in N_idx:  b += G[:,i]*q_known[i]
    for i in D_idx:  b -= H[:,i]*u_known[i]

    x = np.linalg.solve(A, b)

    u_vec = u_known.copy();  q_vec = q_known.copy()
    for k, i in enumerate(N_idx):  u_vec[i] = x[k]
    for k, i in enumerate(D_idx):  q_vec[i] = x[len(N_idx)+k]

    return u_vec, q_vec


# ============================================================
# 7. Interior evaluation
# ============================================================

def eval_interior(pts, nodes, elements, lengths, normals, u_vec, q_vec, ng=10):
    gp, gw = gauss_ref(ng)
    gp = np.ascontiguousarray(gp)
    gw = np.ascontiguousarray(gw)

    n0_all = elements[:,0];  n1_all = elements[:,1]
    Ex0 = nodes[n0_all];     Ex1 = nodes[n1_all]
    U0  = u_vec[n0_all];     U1  = u_vec[n1_all]
    Q0  = q_vec[n0_all];     Q1  = q_vec[n1_all]

    qx_e = Ex0[:,0:1] + gp[None,:]*(Ex1[:,0:1]-Ex0[:,0:1])
    qy_e = Ex0[:,1:2] + gp[None,:]*(Ex1[:,1:2]-Ex0[:,1:2])
    u_e  = (1-gp[None,:])*U0[:,None] + gp[None,:]*U1[:,None]
    q_e  = (1-gp[None,:])*Q0[:,None] + gp[None,:]*Q1[:,None]

    return _eval_interior_nb(
        np.ascontiguousarray(pts[:,0]),
        np.ascontiguousarray(pts[:,1]),
        np.ascontiguousarray(qx_e),
        np.ascontiguousarray(qy_e),
        np.ascontiguousarray(u_e),
        np.ascontiguousarray(q_e),
        np.ascontiguousarray(normals[:,0]),
        np.ascontiguousarray(normals[:,1]),
        np.ascontiguousarray(lengths),
        gw)


# ============================================================
# 8. Polygon utilities
# ============================================================

def points_in_polygon(pts, poly):
    n = len(poly)
    px = pts[:,0]; py = pts[:,1]
    inside = np.zeros(len(pts), dtype=bool)
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[(i+1)%n]
        mask = ((yi>py) != (yj>py))
        x_int = (xj-xi)*(py-yi)/((yj-yi)+1e-300)+xi
        inside ^= (mask & (px < x_int))
    return inside


def min_dist_to_boundary(pts, poly):
    n = len(poly)
    dists = np.full(len(pts), np.inf)
    for i in range(n):
        A = np.array(poly[i], dtype=float)
        B = np.array(poly[(i+1)%n], dtype=float)
        AB = B-A; L2 = np.dot(AB,AB)
        if L2 < 1e-20:
            d = np.linalg.norm(pts-A, axis=1)
        else:
            t = np.clip(((pts-A)@AB)/L2, 0, 1)
            d = np.linalg.norm(pts-(A+t[:,None]*AB), axis=1)
        dists = np.minimum(dists, d)
    return dists


# ============================================================
# 9. Refinement study
# ============================================================

def run_study():
    N_values  = [400, 800, 1600, 3200, 6400]
    m, p, t_val = 0.04, 0.3, 0.18
    split_x   = 0.8

    ngrid = 200
    xs = np.linspace(-0.1, 1.1, ngrid)
    ys = np.linspace(-0.2, 0.2, ngrid)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])

    nodes_ref, _, lengths_ref, _ = naca4_points(m, p, t_val, 100)
    poly_ref  = nodes_ref[:-1]
    perim_ref = np.sum(lengths_ref)
    h_coarse  = perim_ref / min(N_values)
    delta     = 2 * h_coarse

    interior_mask = points_in_polygon(all_pts, poly_ref)
    interior_pts  = all_pts[interior_mask]
    dist          = min_dist_to_boundary(interior_pts, poly_ref)
    grid_pts      = interior_pts[dist > delta]
    u_ex_grid     = u_exact(grid_pts[:,0], grid_pts[:,1])

    # Numba warmup
    print("Warming up Numba JIT ...", flush=True)
    _n, _e, _l, _no = naca4_points(m, p, t_val, 10)
    assemble_HG(_n, _e, _l, _no, ng=4)
    _gp, _gw = gauss_ref(4); _Ne=10; _ng=4
    _qx=np.zeros((_Ne,_ng)); _qy=np.zeros((_Ne,_ng))
    _ue=np.zeros((_Ne,_ng)); _qe=np.zeros((_Ne,_ng))
    _eval_interior_nb(np.zeros(2), np.zeros(2), _qx, _qy, _ue, _qe,
                      np.zeros(_Ne), np.zeros(_Ne), np.ones(_Ne), _gw)

    L2_errors = []

    print("=" * 75)
    print(f"{'BEM SOLVER | NACA 4318 | Collocation | Mixed BC | Linear Elements':^75}")
    print("=" * 75)
    print(f"  Manufactured solution : u(x,y) = x^3 - 3xy^2")
    print(f"  Neumann (x < 0.8)     : q = (3x^2-3y^2)*nx - 6xy*ny")
    print(f"  Dirichlet (x >= 0.8)  : u = x^3 - 3xy^2")
    print(f"  Trailing edge         : double nodes (node 0 and node N)")
    print(f"  Interior eval points  : {len(grid_pts)}")
    print("=" * 75)
    print()
    print(f"  {'N':>5}  {'Nodes':>6}  {'L2 Error':>12}  {'Assembly(s)':>12}  "
          f"{'Solve(s)':>9}  {'Eval(s)':>8}  {'Total(s)':>9}")
    print("  " + "-" * 70)

    for N_elem in N_values:
        t0 = time.time()

        nodes, elements, lengths, normals = naca4_points(m, p, t_val, N_elem)
        N_nodes = len(nodes)

        t_asm = time.time()
        H, G = assemble_HG(nodes, elements, lengths, normals)
        t_asm = time.time() - t_asm

        t_slv = time.time()
        u_vec, q_vec = solve_bem(nodes, elements, lengths, normals, H, G, split_x)
        t_slv = time.time() - t_slv

        t_ev = time.time()
        u_num = eval_interior(grid_pts, nodes, elements, lengths, normals, u_vec, q_vec)
        t_ev = time.time() - t_ev

        L2 = np.linalg.norm(u_num - u_ex_grid) / (np.linalg.norm(u_ex_grid) + 1e-300)
        L2_errors.append(L2)
        t_tot = time.time() - t0

        print(f"  {N_elem:>5}  {N_nodes:>6}  {L2:>12.4e}  {t_asm:>12.2f}  "
              f"{t_slv:>9.3f}  {t_ev:>8.2f}  {t_tot:>9.2f}")
        sys.stdout.flush()

    print("  " + "-" * 70)
    print()

    h_arr    = 1.0 / np.array(N_values, dtype=float)
    log_h    = np.log(h_arr)
    log_err  = np.log(np.array(L2_errors))
    slope, _ = np.polyfit(log_h, log_err, 1)

    print("=" * 75)
    print("  CONVERGENCE ANALYSIS")
    print("=" * 75)
    print(f"  {'N':>5}  {'h = 1/N':>10}  {'L2 Error':>12}  {'log(h)':>10}  {'log(err)':>10}")
    print("  " + "-" * 55)
    for i, N in enumerate(N_values):
        print(f"  {N:>5}  {h_arr[i]:>10.6f}  {L2_errors[i]:>12.4e}  "
              f"{log_h[i]:>10.4f}  {log_err[i]:>10.4f}")
    print()
    print(f"  Estimated convergence order (least-squares slope): {slope:.3f}")
    print(f"  Expected (linear collocation BEM):                 ~2.0")
    print("=" * 75)


if __name__ == "__main__":
    run_study()