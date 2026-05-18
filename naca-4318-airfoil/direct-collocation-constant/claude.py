"""
2D BEM Solver — Direct Collocation, Constant Elements, Mixed BC
Domain: NACA 4318 airfoil interior
Manufactured solution: u(x,y) = x^3 - 3xy^2
"""

import numpy as np
import time
import sys
from numba import njit, prange

# ============================================================
# 1. NACA 4-digit geometry (constant elements)
# ============================================================

def naca4_elements(m, p, t, N_elem):
    n_half = N_elem // 2
    beta = np.linspace(0, np.pi, n_half + 1)
    xc = 0.5 * (1 - np.cos(beta))

    yc  = np.where(xc < p,
                   m/p**2*(2*p*xc - xc**2),
                   m/(1-p)**2*((1-2*p) + 2*p*xc - xc**2))
    dyc = np.where(xc < p,
                   2*m/p**2*(p - xc),
                   2*m/(1-p)**2*(p - xc))
    yt  = 5*t*(0.2969*np.sqrt(np.maximum(xc, 0)) - 0.1260*xc
               - 0.3516*xc**2 + 0.2843*xc**3 - 0.1015*xc**4)
    theta = np.arctan(dyc)

    xu = xc - yt*np.sin(theta);  yu = yc + yt*np.cos(theta)
    xl = xc + yt*np.sin(theta);  yl = yc - yt*np.cos(theta)

    upper_x = xu[::-1];  upper_y = yu[::-1]
    lower_x = xl[1:];    lower_y = yl[1:]

    xs = np.concatenate([upper_x, lower_x])
    ys = np.concatenate([upper_y, lower_y])
    xs[0] = 1.0; ys[0] = 0.0
    xs[-1] = 1.0; ys[-1] = 0.0

    verts_x = xs[:N_elem]; verts_y = ys[:N_elem]
    idx0 = np.arange(N_elem); idx1 = (np.arange(N_elem) + 1) % N_elem

    x0 = verts_x[idx0]; y0 = verts_y[idx0]
    x1 = verts_x[idx1]; y1 = verts_y[idx1]

    midpoints = np.column_stack([0.5*(x0+x1), 0.5*(y0+y1)])
    dx = x1-x0; dy = y1-y0
    lengths = np.sqrt(dx**2 + dy**2)
    normals = np.column_stack([dy/lengths, -dx/lengths])
    endpoints = np.stack([np.column_stack([x0,y0]),
                          np.column_stack([x1,y1])], axis=1)

    return endpoints, midpoints, lengths, normals


def assign_bcs(midpoints, split_x=0.8):
    return (midpoints[:,0] >= split_x).astype(np.int32)


def u_exact(x, y):
    return x**3 - 3*x*y**2

def q_exact(x, y, nx, ny):
    return (3*x**2 - 3*y**2)*nx + (-6*x*y)*ny


# ============================================================
# 2. Gaussian quadrature on [-1,1]
# ============================================================

def gauss_std(n):
    return np.polynomial.legendre.leggauss(n)


# ============================================================
# 3. Numba kernels
# ============================================================

@njit(parallel=True, cache=True)
def _assemble_HG_nb(mid_x, mid_y, cx, cy, hx, hy,
                     lengths, normals_x, normals_y,
                     G_diag, gp, gw):
    """
    Full constant-element assembly parallelised over rows.
    G_diag: precomputed analytical self-term for each element.
    """
    Ne     = mid_x.shape[0]
    ng     = gp.shape[0]
    INV2PI = 1.0 / (6.283185307179586)

    H = np.zeros((Ne, Ne))
    G = np.zeros((Ne, Ne))

    for i in prange(Ne):
        xi_x = mid_x[i]; xi_y = mid_y[i]
        for j in range(Ne):
            if i == j:
                H[i, i] = 0.5
                G[i, i] = G_diag[i]
            else:
                Lj  = lengths[j]
                nxj = normals_x[j]; nyj = normals_y[j]
                cxj = cx[j]; cyj = cy[j]
                hxj = hx[j]; hyj = hy[j]
                half_L = 0.5 * Lj
                sumG = 0.0; sumH = 0.0
                for k in range(ng):
                    yx  = cxj + hxj*gp[k]
                    yy  = cyj + hyj*gp[k]
                    rx  = xi_x - yx; ry = xi_y - yy
                    r2  = rx*rx + ry*ry
                    if r2 < 1e-300: r2 = 1e-300
                    sumG += gw[k] * (-INV2PI * 0.5 * np.log(r2) * half_L)
                    sumH += gw[k] * ( INV2PI * (rx*nxj + ry*nyj) / r2 * half_L)
                G[i, j] = sumG
                H[i, j] = sumH

    return H, G


@njit(parallel=True, cache=True)
def _eval_interior_nb(pts_x, pts_y,
                       qx_e, qy_e,
                       u_e, q_e,
                       ny_x, ny_y, half_L,
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
            hL  = half_L[e]
            ue  = u_e[e]; qe = q_e[e]
            nyx = ny_x[e]; nyy = ny_y[e]
            for j in range(ng):
                rx  = xp - qx_e[e, j];  ry = yp - qy_e[e, j]
                r2  = rx*rx + ry*ry
                if r2 < 1e-300: r2 = 1e-300
                kG  = -INV2PI * 0.5 * np.log(r2)
                kH  =  INV2PI * (rx*nyx + ry*nyy) / r2
                val += gw[j] * (kG*qe - kH*ue) * hL
        u_int[k] = val

    return u_int


# ============================================================
# 4. Assembly wrapper
# ============================================================

def assemble_HG(midpoints, endpoints, lengths, normals, ng=10):
    Ne = len(midpoints)
    gp, gw = gauss_std(ng)
    gp = np.ascontiguousarray(gp); gw = np.ascontiguousarray(gw)

    Amx = endpoints[:,0,0]; Amy = endpoints[:,0,1]
    Bmx = endpoints[:,1,0]; Bmy = endpoints[:,1,1]
    cx  = np.ascontiguousarray(0.5*(Amx+Bmx))
    cy  = np.ascontiguousarray(0.5*(Amy+Bmy))
    hx  = np.ascontiguousarray(0.5*(Bmx-Amx))
    hy  = np.ascontiguousarray(0.5*(Bmy-Amy))

    # Precompute analytical diagonal G
    INV2PI = 1.0 / (2.0*np.pi)
    G_diag = (lengths * INV2PI) * (1.0 - np.log(lengths / 2.0))

    H, G = _assemble_HG_nb(
        np.ascontiguousarray(midpoints[:,0]),
        np.ascontiguousarray(midpoints[:,1]),
        cx, cy, hx, hy,
        np.ascontiguousarray(lengths),
        np.ascontiguousarray(normals[:,0]),
        np.ascontiguousarray(normals[:,1]),
        np.ascontiguousarray(G_diag),
        gp, gw)

    return H, G


# ============================================================
# 5. Solve
# ============================================================

def solve_bem(midpoints, lengths, normals, H, G, split_x=0.8):
    Ne = len(midpoints)
    bc = assign_bcs(midpoints, split_x)

    u_known = np.zeros(Ne); q_known = np.zeros(Ne)
    for i in range(Ne):
        xi, yi = midpoints[i]; nx, ny = normals[i]
        if bc[i] == 1:
            u_known[i] = u_exact(xi, yi)
        else:
            q_known[i] = q_exact(xi, yi, nx, ny)

    D_idx = np.where(bc==1)[0]
    N_idx = np.where(bc==0)[0]

    A = np.zeros((Ne, Ne)); b = np.zeros(Ne)
    for j in N_idx:
        A[:, j] =  H[:, j]
        b       += G[:, j] * q_known[j]
    for j in D_idx:
        A[:, j] = -G[:, j]
        b       -= H[:, j] * u_known[j]

    x = np.linalg.solve(A, b)

    u_vec = u_known.copy(); q_vec = q_known.copy()
    for j in N_idx: u_vec[j] = x[j]
    for j in D_idx: q_vec[j] = x[j]

    return u_vec, q_vec


# ============================================================
# 6. Interior evaluation
# ============================================================

def eval_interior(pts, midpoints, endpoints, lengths, normals, u_vec, q_vec, ng=10):
    gp, gw = gauss_std(ng)
    gp = np.ascontiguousarray(gp); gw = np.ascontiguousarray(gw)

    Amx = endpoints[:,0,0]; Amy = endpoints[:,0,1]
    Bmx = endpoints[:,1,0]; Bmy = endpoints[:,1,1]
    cx  = 0.5*(Amx+Bmx); cy = 0.5*(Amy+Bmy)
    hx  = 0.5*(Bmx-Amx); hy = 0.5*(Bmy-Amy)

    Ne = len(midpoints)
    qx_e = (cx[:,None] + hx[:,None]*gp[None,:]).astype(np.float64)
    qy_e = (cy[:,None] + hy[:,None]*gp[None,:]).astype(np.float64)

    return _eval_interior_nb(
        np.ascontiguousarray(pts[:,0]),
        np.ascontiguousarray(pts[:,1]),
        np.ascontiguousarray(qx_e),
        np.ascontiguousarray(qy_e),
        np.ascontiguousarray(u_vec),
        np.ascontiguousarray(q_vec),
        np.ascontiguousarray(normals[:,0]),
        np.ascontiguousarray(normals[:,1]),
        np.ascontiguousarray(lengths * 0.5),
        gw)


# ============================================================
# 7. Polygon utilities
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
# 8. Refinement study
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

    endpoints_ref, midpoints_ref, lengths_ref, normals_ref = naca4_elements(m, p, t_val, 100)
    poly_ref  = endpoints_ref[:,0,:]
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
    _ep, _mp, _le, _no = naca4_elements(m, p, t_val, 10)
    assemble_HG(_mp, _ep, _le, _no, ng=4)
    _gp, _gw = gauss_std(4); _Ne=10; _ng=4
    _qx=np.zeros((_Ne,_ng)); _qy=np.zeros((_Ne,_ng))
    _eval_interior_nb(np.zeros(2), np.zeros(2), _qx, _qy,
                      np.zeros(_Ne), np.zeros(_Ne),
                      np.zeros(_Ne), np.zeros(_Ne),
                      np.ones(_Ne), _gw)

    L2_errors = []

    print("=" * 75)
    print(f"{'BEM SOLVER | NACA 4318 | Collocation | Constant Elements | Mixed BC':^75}")
    print("=" * 75)
    print(f"  Manufactured solution : u(x,y) = x^3 - 3xy^2")
    print(f"  Neumann (x < 0.8)     : q = (3x^2-3y^2)*nx - 6xy*ny")
    print(f"  Dirichlet (x >= 0.8)  : u = x^3 - 3xy^2")
    print(f"  Collocation           : element midpoints  (c_i = 1/2)")
    print(f"  Interior eval points  : {len(grid_pts)}")
    print("=" * 75)
    print()
    print(f"  {'N':>5}  {'Unknowns':>8}  {'L2 Error':>12}  {'Assembly(s)':>12}  "
          f"{'Solve(s)':>9}  {'Eval(s)':>8}  {'Total(s)':>9}")
    print("  " + "-" * 72)

    for N_elem in N_values:
        t0 = time.time()

        endpoints, midpoints, lengths, normals = naca4_elements(m, p, t_val, N_elem)
        Ne = N_elem

        t_asm = time.time()
        H, G = assemble_HG(midpoints, endpoints, lengths, normals, ng=10)
        t_asm = time.time() - t_asm

        t_slv = time.time()
        u_vec, q_vec = solve_bem(midpoints, lengths, normals, H, G, split_x)
        t_slv = time.time() - t_slv

        t_ev = time.time()
        u_num = eval_interior(grid_pts, midpoints, endpoints, lengths, normals,
                              u_vec, q_vec, ng=10)
        t_ev = time.time() - t_ev

        L2 = np.linalg.norm(u_num - u_ex_grid) / (np.linalg.norm(u_ex_grid) + 1e-300)
        L2_errors.append(L2)
        t_tot = time.time() - t0

        print(f"  {N_elem:>5}  {Ne:>8}  {L2:>12.4e}  {t_asm:>12.3f}  "
              f"{t_slv:>9.3f}  {t_ev:>8.2f}  {t_tot:>9.2f}")
        sys.stdout.flush()

    print("  " + "-" * 72)
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
    print(f"  Expected (constant element BEM):                   ~1.0")
    print("=" * 75)


if __name__ == "__main__":
    run_study()