# ================================================================
# 2D BEM Solver — Laplace on Annulus (1 < r < 2)
# Direct collocation | LINEAR elements | Matrix-free GMRES
# Numba-accelerated assembly, matvec, interior evaluation
#
# Exact solution: u(r,θ) = (r³ + r⁻³) cos(3θ)
# Inner BC (r=1): Dirichlet  u = 2 cos(3θ)
# Outer BC (r=2): Neumann    ∂u/∂n = (189/16) cos(3θ)
#
# THE BUG that caused your results (errors >0.1, rate ~0.5):
#   In the Numba _build_HG, for self-element rows (is_a or is_b),
#   the numerical G accumulation was SKIPPED, then (anal - num) was added.
#   Net result: anal - num  instead of  num + (anal - num) = anal.
#
# FIX: always accumulate G numerically for ALL rows (including self),
#   then for self rows add the correction (anal - num) on top.
#   This gives: num + (anal - num) = anal. Correct.
# ================================================================

import numpy as np
from scipy.special import roots_legendre
from scipy.sparse.linalg import LinearOperator, gmres
import numba as nb
import time

# ----------------------------------------------------------------
# Exact solution
# ----------------------------------------------------------------

def u_exact(r, theta):
    return (r**3 + r**-3) * np.cos(3 * theta)

def dudr_exact(r, theta):
    return (3 * r**2 - 3 * r**-4) * np.cos(3 * theta)

# ----------------------------------------------------------------
# Quadrature
# ----------------------------------------------------------------

def gauss_quad(order):
    xi, wi = roots_legendre(order)
    return xi.astype(np.float64), wi.astype(np.float64)

# ----------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------

def build_geometry(N):
    """
    Linear elements on inner (r=1) and outer (r=2) circles.

    Nodes:
      0 .. N-1    : inner circle, angles 0, dθ, 2dθ, ...
      N .. 2N-1   : outer circle, angles 0, dθ, 2dθ, ...

    Elements:
      e in [0, N):    inner, nodes [e, (e+1)%N]
      e in [N, 2N):   outer, nodes [N+k, N+(k+1)%N]

    Normal convention (outward from domain):
      Inner: −r̂   Outer: +r̂
    Normal evaluated exactly at each quadrature point.
    """
    dth = 2 * np.pi / N
    ti  = np.arange(N) * dth
    to  = np.arange(N) * dth

    nodes = np.vstack([
        np.stack([np.cos(ti), np.sin(ti)],   axis=1),
        np.stack([2*np.cos(to), 2*np.sin(to)], axis=1)
    ])

    conn = np.zeros((2*N, 2), dtype=np.int32)
    for k in range(N):
        conn[k]   = [k,   (k+1) % N]
        conn[N+k] = [N+k, N + (k+1) % N]

    bc         = np.array([0]*N + [1]*N, dtype=np.int32)
    r_node     = np.concatenate([np.ones(N), 2.0*np.ones(N)])
    theta_node = np.concatenate([ti, to])

    N_elems = 2*N
    el_th_a = np.zeros(N_elems)
    el_dth  = np.zeros(N_elems)
    el_r    = np.zeros(N_elems)

    for e in range(N_elems):
        ya = nodes[conn[e, 0]]; yb = nodes[conn[e, 1]]
        r_avg = (np.linalg.norm(ya) + np.linalg.norm(yb)) / 2.0
        th_a  = np.arctan2(ya[1], ya[0])
        th_b  = np.arctan2(yb[1], yb[0])
        dth_e = th_b - th_a
        if dth_e >  np.pi: dth_e -= 2*np.pi
        if dth_e < -np.pi: dth_e += 2*np.pi
        el_th_a[e] = th_a
        el_dth[e]  = dth_e
        el_r[e]    = r_avg

    return nodes, conn, bc, r_node, theta_node, el_th_a, el_dth, el_r

# ----------------------------------------------------------------
# Numba: build H and G matrices
# ----------------------------------------------------------------

@nb.njit(cache=True, parallel=True, fastmath=True)
def _build_HG(nodes, conn, el_th_a, el_dth, el_r, xi_g, wi_g, H, Gm):
    """
    Assemble H and G for linear BEM.

    Arc parametrisation: y(ξ) = r_e*(cos(θ_a + dθ/2*(ξ+1)), sin(...))
    jac = r_e * |dθ| / 2,   φ₁ = (1-ξ)/2,   φ₂ = (1+ξ)/2
    n(y) = exact outward unit normal at y_q

    G self-term analytical correction:
      G[a,a] = G[b,b] = -La/(4π)·(log La − 3/2)
      G[a,b] = G[b,a] = -La/(4π)·(log La − 1/2)
      La = r_e * |dθ|   (arc length)

    CRITICAL: Accumulate G numerically for ALL rows first,
    then add (anal − num) correction for self rows.
    This gives: num + (anal − num) = anal.  ← correct
    (Old bug: skipped numerical in self branch → got anal − num ← WRONG)
    """
    N_nodes = nodes.shape[0]
    N_elems = conn.shape[0]
    nq      = xi_g.shape[0]
    INV2PI  = 1.0 / (2.0 * np.pi)

    for i in nb.prange(N_nodes):
        cx = nodes[i, 0];  cy = nodes[i, 1]

        for e in range(N_elems):
            a_idx = conn[e, 0];  b_idx = conn[e, 1]
            th_a  = el_th_a[e];  dth_e = el_dth[e];  r_e = el_r[e]
            jac   = r_e * abs(dth_e) / 2.0
            sign  = -1.0 if r_e < 1.5 else 1.0
            is_a  = (i == a_idx)
            is_b  = (i == b_idx)

            H_a = 0.0;  H_b = 0.0
            G_a = 0.0;  G_b = 0.0
            # For self correction: track the numerical self-element integrals
            G_self_phi1 = 0.0;  G_self_phi2 = 0.0

            for k in range(nq):
                tq   = th_a + dth_e * 0.5 * (xi_g[k] + 1.0)
                yx   = r_e * np.cos(tq)
                yy_  = r_e * np.sin(tq)
                nx_q = sign * np.cos(tq)
                ny_q = sign * np.sin(tq)
                rx   = cx - yx;  ry = cy - yy_
                d2   = rx*rx + ry*ry
                if d2 < 1e-60:
                    continue
                phi1 = 0.5 * (1.0 - xi_g[k])
                phi2 = 0.5 * (1.0 + xi_g[k])
                w    = wi_g[k] * jac
                rdotn = rx * nx_q + ry * ny_q
                H_kern = INV2PI * rdotn / d2 * w
                G_kern = -INV2PI * 0.5 * np.log(d2) * w
                H_a += H_kern * phi1;  H_b += H_kern * phi2
                # Accumulate G numerically for ALL rows (including self)
                G_a += G_kern * phi1;  G_b += G_kern * phi2
                # Track self-element integrals for correction
                if is_a or is_b:
                    G_self_phi1 += G_kern * phi1
                    G_self_phi2 += G_kern * phi2

            H[i, a_idx] += H_a;  H[i, b_idx] += H_b
            Gm[i, a_idx] += G_a;  Gm[i, b_idx] += G_b

            # G self-correction: add (anal − num) on top of already-added num
            if is_a or is_b:
                La     = r_e * abs(dth_e)
                log_La = np.log(La)
                G_aa_anal = -La / (4.0*np.pi) * (log_La - 1.5)
                G_ab_anal = -La / (4.0*np.pi) * (log_La - 0.5)
                if is_a:
                    Gm[i, a_idx] += G_aa_anal - G_self_phi1
                    Gm[i, b_idx] += G_ab_anal - G_self_phi2
                else:   # is_b: G[b,b]=G_aa, G[b,a]=G_ab (same arc length, by symmetry)
                    Gm[i, a_idx] += G_ab_anal - G_self_phi1
                    Gm[i, b_idx] += G_aa_anal - G_self_phi2

# ----------------------------------------------------------------
# Numba matvec
# ----------------------------------------------------------------

@nb.njit(cache=True, parallel=True, fastmath=True)
def _matvec_nb(x, A_mat, out):
    N = A_mat.shape[0]
    for i in nb.prange(N):
        s = 0.0
        for j in range(N):
            s += A_mat[i, j] * x[j]
        out[i] = s

# ----------------------------------------------------------------
# Numba interior evaluation
# ----------------------------------------------------------------

@nb.njit(cache=True, parallel=True, fastmath=True)
def _eval_interior(pts_x, pts_y, conn, el_th_a, el_dth, el_r,
                   u_all, q_all, xi_g, wi_g, u_int):
    """
    u(x) = Σ_e ∫_e [ G(x,y) q(y) − ∂G/∂n_y(x,y) u(y) ] ds_y
    with linear interpolation of u and q on each element.
    Uses 16-pt Gauss to avoid near-singular inaccuracies for
    interior points close to the boundary.
    """
    M      = pts_x.shape[0]
    N_elems = conn.shape[0]
    nq     = xi_g.shape[0]
    INV2PI = 1.0 / (2.0 * np.pi)

    for ip in nb.prange(M):
        cx  = pts_x[ip];  cy = pts_y[ip]
        val = 0.0
        for e in range(N_elems):
            a_idx = conn[e, 0];  b_idx = conn[e, 1]
            th_a  = el_th_a[e];  dth_e = el_dth[e];  r_e = el_r[e]
            jac   = r_e * abs(dth_e) / 2.0
            sign  = -1.0 if r_e < 1.5 else 1.0
            u_a   = u_all[a_idx];  u_b = u_all[b_idx]
            q_a   = q_all[a_idx];  q_b = q_all[b_idx]
            for k in range(nq):
                tq   = th_a + dth_e * 0.5 * (xi_g[k] + 1.0)
                yx   = r_e * np.cos(tq)
                yy_  = r_e * np.sin(tq)
                nx_q = sign * np.cos(tq)
                ny_q = sign * np.sin(tq)
                rx   = cx - yx;  ry = cy - yy_
                d2   = rx*rx + ry*ry
                if d2 < 1e-60:
                    continue
                phi1 = 0.5 * (1.0 - xi_g[k])
                phi2 = 0.5 * (1.0 + xi_g[k])
                w    = wi_g[k] * jac
                G_k  = -INV2PI * 0.5 * np.log(d2)
                H_k  =  INV2PI * (rx * nx_q + ry * ny_q) / d2
                val += G_k * (q_a*phi1 + q_b*phi2) * w
                val -= H_k * (u_a*phi1 + u_b*phi2) * w
        u_int[ip] = val

# ----------------------------------------------------------------
# Main BEM solver
# ----------------------------------------------------------------

def solve_bem(N, xi_g, wi_g, xi_int, wi_int, verbose=True):
    Ntot = 2 * N
    t0   = time.perf_counter()

    nodes, conn, bc, r_node, theta_node, el_th_a, el_dth, el_r = \
        build_geometry(N)

    # Known boundary data
    u_bc = np.zeros(Ntot);  q_bc = np.zeros(Ntot)
    u_bc[bc == 0] = u_exact(r_node[bc == 0], theta_node[bc == 0])
    q_bc[bc == 1] = dudr_exact(r_node[bc == 1], theta_node[bc == 1])

    # Assemble H and G
    H  = np.zeros((Ntot, Ntot))
    Gm = np.zeros((Ntot, Ntot))
    _build_HG(nodes, conn, el_th_a, el_dth, el_r, xi_g, wi_g, H, Gm)
    for i in range(Ntot):
        H[i, i] += 0.5   # free term c = 1/2

    t_setup = time.perf_counter() - t0

    # BIE: H @ u = Gm @ q
    # Unknowns x = [q_inner (N), u_outer (N)]
    A   = np.zeros((Ntot, Ntot))
    rhs = np.zeros(Ntot)
    for j in range(Ntot):
        if bc[j] == 0:           # Dirichlet: q unknown, u known
            A[:, j]  = -Gm[:, j]
            rhs     -=  H[:, j] * u_bc[j]
        else:                    # Neumann: u unknown, q known
            A[:, j]  =  H[:, j]
            rhs     +=  Gm[:, j] * q_bc[j]

    iters   = [0]
    out_buf = np.zeros(Ntot)

    def matvec(x):
        iters[0] += 1
        _matvec_nb(x, A, out_buf)
        return out_buf.copy()

    Aop = LinearOperator((Ntot, Ntot), matvec=matvec, dtype=np.float64)

    t1 = time.perf_counter()
    x_sol, info = gmres(Aop, rhs, rtol=1e-10, atol=1e-12,
                        restart=200, maxiter=2000)
    t_solve = time.perf_counter() - t1

    if verbose:
        print(f"    GMRES: info={info}, matvecs={iters[0]}")

    u_all = u_bc.copy();  q_all = q_bc.copy()
    q_all[bc == 0] = x_sol[bc == 0]
    u_all[bc == 1] = x_sol[bc == 1]

    # Interior evaluation (16-pt Gauss avoids near-singular errors)
    t2 = time.perf_counter()
    ngrid = 60
    xg    = np.linspace(-1.9, 1.9, ngrid)
    XX, YY = np.meshgrid(xg, xg)
    rr     = np.sqrt(XX**2 + YY**2)
    mask   = (rr > 1.0) & (rr < 2.0)
    pts_x  = XX[mask].astype(np.float64)
    pts_y  = YY[mask].astype(np.float64)
    u_int  = np.zeros(len(pts_x))

    _eval_interior(pts_x, pts_y, conn, el_th_a, el_dth, el_r,
                   u_all, q_all, xi_int, wi_int, u_int)

    t_eval = time.perf_counter() - t2

    u_ref  = u_exact(np.sqrt(pts_x**2 + pts_y**2), np.arctan2(pts_y, pts_x))
    rel_l2 = np.linalg.norm(u_int - u_ref) / np.linalg.norm(u_ref)
    t_tot  = time.perf_counter() - t0

    return dict(N=N, unknowns=Ntot, iters=iters[0], rel_l2=rel_l2,
                t_setup=t_setup, t_solve=t_solve, t_eval=t_eval, t_total=t_tot)

# ----------------------------------------------------------------
# Run
# ----------------------------------------------------------------

xi_g,  wi_g   = gauss_quad(10)   # assembly quadrature
xi_int, wi_int = gauss_quad(16)  # interior eval (16-pt avoids near-singular issues at 10-pt)

print("Warming up Numba JIT...")
solve_bem(8, xi_g, wi_g, xi_int, wi_int, verbose=False)
print("Warmup done.\n")

N_values = [160, 320, 640, 1280, 2560]
results  = []

hdr = (f"{'N':>6} {'Unknowns':>10} {'GMRES Mvecs':>12} {'Rel L2 Err':>14} "
       f"{'Conv Rate':>10} {'Setup(s)':>10} {'Solve(s)':>10} {'Eval(s)':>10} {'Total(s)':>10}")
print(hdr)
print("-" * len(hdr))

for N in N_values:
    print(f"  Running N={N} ...")
    res = solve_bem(N, xi_g, wi_g, xi_int, wi_int, verbose=True)
    results.append(res)

    if len(results) > 1:
        prev = results[-2]
        rate = (np.log(prev['rel_l2'] / res['rel_l2'])
                / np.log(res['N'] / prev['N']))
        rate_str = f"{rate:>10.2f}"
    else:
        rate_str = f"{'—':>10}"

    print(f"{res['N']:>6} {res['unknowns']:>10} {res['iters']:>12} "
          f"{res['rel_l2']:>14.6e} {rate_str} {res['t_setup']:>10.3f} "
          f"{res['t_solve']:>10.3f} {res['t_eval']:>10.3f} {res['t_total']:>10.3f}")
    print()

Ns   = np.array([r['N']      for r in results], dtype=float)
errs = np.array([r['rel_l2'] for r in results])
coeffs = np.polyfit(np.log2(Ns), np.log2(errs), 1)
print(f"Overall convergence order (least-squares fit): {-coeffs[0]:.3f}")
print(f"  (theoretical: ~2.0 for linear elements on smooth domains)")