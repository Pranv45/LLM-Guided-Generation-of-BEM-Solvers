import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange

# ---------------------------------------------------------------------------
# Quadrature rules  (identical to reference)
#   pts_gl / wts_gl : 16-pt Gauss–Legendre on [−1,1]
#   pts_01 / wts_01 : same nodes mapped to [0,1]  (used inside log-sing loop)
#   pts_log/ wts_log: 8-pt Gauss–log-weight rule on [0,1] for ∫log(t)f(t)dt
# ---------------------------------------------------------------------------
@njit(cache=True)
def get_gauss_points():
    pts_gl = np.array([-0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030,
                       -0.6178762444026438, -0.4580167776572274, -0.2816035507792589, -0.0950125098376374,
                        0.0950125098376374,  0.2816035507792589,  0.4580167776572274,  0.6178762444026438,
                        0.7554044083550030,  0.8656312023878318,  0.9445750230732326,  0.9894009349916499])
    wts_gl = np.array([0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
                       0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
                       0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
                       0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541])

    pts_01 = 0.5 * pts_gl + 0.5
    wts_01 = 0.5 * wts_gl

    pts_log = np.array([0.0133202436, 0.0797504274, 0.1978710287, 0.3541539561,
                        0.5294585752, 0.7018145299, 0.8493793204, 0.9533264500])
    wts_log = np.array([0.1644166048, 0.2375256098, 0.2268419844, 0.1757540790,
                        0.1129240291, 0.0578722107, 0.0209790737, 0.0036864071])

    return pts_gl, wts_gl, pts_01, wts_01, pts_log, wts_log

# ---------------------------------------------------------------------------
# Exact solution and its normal derivative for the rectangle problem
#
#   u(x,y)  = sinh(πx)/sinh(2π) · cos(πy)
#   ∂u/∂x   = π cosh(πx)/sinh(2π) · cos(πy)
#   ∂u/∂y   = −π sinh(πx)/sinh(2π) · sin(πy)
#   q = ∂u/∂n = ∇u · n
# ---------------------------------------------------------------------------
@njit(cache=True)
def exact_u(x, y):
    return np.sinh(np.pi * x) / np.sinh(2.0 * np.pi) * np.cos(np.pi * y)

@njit(cache=True)
def exact_q(x, y, nx, ny):
    S2pi = np.sinh(2.0 * np.pi)
    dudx = np.pi * np.cosh(np.pi * x) / S2pi * np.cos(np.pi * y)
    dudy = -np.pi * np.sinh(np.pi * x) / S2pi * np.sin(np.pi * y)
    return dudx * nx + dudy * ny

# ---------------------------------------------------------------------------
# Galerkin matrix assembly  (identical logic to reference)
#
# Assembles H_mat and G_mat, the full N_tot × N_tot Galerkin matrices, by
# integrating over every pair of elements (et, es).
#
# Self-interaction (et == es):
#   H block : 0.5 × mass matrix   (regularised jump in H)
#   G block : analytical log-singular integration split into
#             a constant part + a Gauss × log-weight inner quadrature
#
# Off-diagonal (et ≠ es):
#   Regular 16 × 16 tensor-product Gauss rule for both G and H blocks.
#
# Shape functions: N1(ξ) = (1−ξ)/2,  N2(ξ) = (1+ξ)/2  on [−1,1]
# Jacobians: Lt/2 for the test element, Ls/2 for the source element.
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def compute_galerkin_matrices(nodes_x, nodes_y, elems, nx, ny, lengths,
                               pts_gl, wts_gl, pts_01, wts_01, pts_log, wts_log):
    N_tot = len(nodes_x)
    N_el  = len(elems)
    H_mat = np.zeros((N_tot, N_tot))
    G_mat = np.zeros((N_tot, N_tot))

    for et in prange(N_el):
        it1 = elems[et, 0]
        it2 = elems[et, 1]
        Lt  = lengths[et]

        for es in range(N_el):
            js1 = elems[es, 0]
            js2 = elems[es, 1]
            Ls  = lengths[es]

            h11, h12, h21, h22 = 0.0, 0.0, 0.0, 0.0
            g11, g12, g21, g22 = 0.0, 0.0, 0.0, 0.0

            if et == es:
                # ---- Self-interaction ----
                # H: regularised free-term → 0.5 × mass matrix
                mass11 = Lt / 3.0
                mass12 = Lt / 6.0
                mass22 = Lt / 3.0
                h11 += 0.5 * mass11
                h12 += 0.5 * mass12
                h21 += 0.5 * mass12
                h22 += 0.5 * mass22

                # G: constant part = −(Lt²/8π) ln(Lt/2) × [[1,1],[1,1]]
                const_part = -(Lt * Lt / (8.0 * np.pi)) * np.log(Lt / 2.0)
                g11 += const_part
                g12 += const_part
                g21 += const_part
                g22 += const_part

                # G: log-singular remainder via 16-pt outer × (log+regular) inner rules
                for kt in range(len(pts_gl)):
                    xt = pts_gl[kt]
                    wt = wts_gl[kt]
                    Nt1 = 0.5 * (1.0 - xt)
                    Nt2 = 0.5 * (1.0 + xt)

                    I_log_1 = 0.0
                    I_log_2 = 0.0

                    # Left sub-interval: source runs from xt down to −1
                    len_left = xt + 1.0
                    if len_left > 1e-12:
                        # Regular part: ∫ ln(len_left) × ... ds  (pts_01)
                        for ks in range(len(pts_01)):
                            s  = pts_01[ks]
                            ws = wts_01[ks]
                            xs = xt - len_left * s
                            Ns1 = 0.5 * (1.0 - xs)
                            Ns2 = 0.5 * (1.0 + xs)
                            term = len_left * np.log(len_left) * ws
                            I_log_1 += term * Ns1
                            I_log_2 += term * Ns2
                        # Log-weight part: ∫ ln(s) × ... ds  (pts_log)
                        for ks in range(len(pts_log)):
                            s  = pts_log[ks]
                            ws = wts_log[ks]
                            xs = xt - len_left * s
                            Ns1 = 0.5 * (1.0 - xs)
                            Ns2 = 0.5 * (1.0 + xs)
                            term = -len_left * ws
                            I_log_1 += term * Ns1
                            I_log_2 += term * Ns2

                    # Right sub-interval: source runs from xt up to +1
                    len_right = 1.0 - xt
                    if len_right > 1e-12:
                        for ks in range(len(pts_01)):
                            s  = pts_01[ks]
                            ws = wts_01[ks]
                            xs = xt + len_right * s
                            Ns1 = 0.5 * (1.0 - xs)
                            Ns2 = 0.5 * (1.0 + xs)
                            term = len_right * np.log(len_right) * ws
                            I_log_1 += term * Ns1
                            I_log_2 += term * Ns2
                        for ks in range(len(pts_log)):
                            s  = pts_log[ks]
                            ws = wts_log[ks]
                            xs = xt + len_right * s
                            Ns1 = 0.5 * (1.0 - xs)
                            Ns2 = 0.5 * (1.0 + xs)
                            term = -len_right * ws
                            I_log_1 += term * Ns1
                            I_log_2 += term * Ns2

                    # Outer weight: −(Lt²/8π) × wt  (Lt²/4 from Jacobians, 1/(2π) from kernel)
                    factor = -(Lt * Lt) / (8.0 * np.pi) * wt
                    g11 += Nt1 * I_log_1 * factor
                    g12 += Nt1 * I_log_2 * factor
                    g21 += Nt2 * I_log_1 * factor
                    g22 += Nt2 * I_log_2 * factor

            else:
                # ---- Off-diagonal: 16 × 16 tensor-product Gauss quadrature ----
                for kt in range(len(pts_gl)):
                    xt = pts_gl[kt]
                    wt = wts_gl[kt]
                    Nt1 = 0.5 * (1.0 - xt)
                    Nt2 = 0.5 * (1.0 + xt)

                    pt_x = Nt1 * nodes_x[it1] + Nt2 * nodes_x[it2]
                    pt_y = Nt1 * nodes_y[it1] + Nt2 * nodes_y[it2]

                    for ks in range(len(pts_gl)):
                        xs = pts_gl[ks]
                        ws = wts_gl[ks]
                        Ns1 = 0.5 * (1.0 - xs)
                        Ns2 = 0.5 * (1.0 + xs)

                        ps_x = Ns1 * nodes_x[js1] + Ns2 * nodes_x[js2]
                        ps_y = Ns1 * nodes_y[js1] + Ns2 * nodes_y[js2]

                        rx = pt_x - ps_x
                        ry = pt_y - ps_y
                        r2 = rx * rx + ry * ry

                        if r2 > 1e-14:
                            G_val  = -1.0 / (4.0 * np.pi) * np.log(r2)
                            dGdn   = (rx * nx[es] + ry * ny[es]) / (2.0 * np.pi * r2)
                            w_tot  = wt * ws * 0.25 * Lt * Ls

                            h11 += Nt1 * Ns1 * dGdn * w_tot
                            h12 += Nt1 * Ns2 * dGdn * w_tot
                            h21 += Nt2 * Ns1 * dGdn * w_tot
                            h22 += Nt2 * Ns2 * dGdn * w_tot

                            g11 += Nt1 * Ns1 * G_val * w_tot
                            g12 += Nt1 * Ns2 * G_val * w_tot
                            g21 += Nt2 * Ns1 * G_val * w_tot
                            g22 += Nt2 * Ns2 * G_val * w_tot

            # Scatter into global matrices
            H_mat[it1, js1] += h11
            H_mat[it1, js2] += h12
            H_mat[it2, js1] += h21
            H_mat[it2, js2] += h22

            G_mat[it1, js1] += g11
            G_mat[it1, js2] += g12
            G_mat[it2, js1] += g21
            G_mat[it2, js2] += g22

    return H_mat, G_mat

# ---------------------------------------------------------------------------
# Interior evaluation (same as reference)
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def eval_interior(ix_pts, iy_pts, u_all, q_all, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts):
    M     = len(ix_pts)
    u_int = np.zeros(M)
    for i in prange(M):
        val = 0.0
        for e in range(len(elems)):
            j1 = elems[e, 0]
            j2 = elems[e, 1]
            L  = lengths[e]
            for k in range(len(pts)):
                xi = pts[k]
                w  = wts[k]
                N1 = 0.5 * (1.0 - xi)
                N2 = 0.5 * (1.0 + xi)
                yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]

                rx = ix_pts[i] - yx
                ry = iy_pts[i] - yy
                r2 = rx * rx + ry * ry

                G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                dGdn  = (rx * nx[e] + ry * ny[e]) / (2.0 * np.pi * r2)

                q_y = N1 * q_all[j1] + N2 * q_all[j2]
                u_y = N1 * u_all[j1] + N2 * u_all[j2]

                val += (G_val * q_y - dGdn * u_y) * w * 0.5 * L
        u_int[i] = val
    return u_int

class IterationCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, pr_norm):
        self.niter += 1

# ---------------------------------------------------------------------------
# Main solver
#
# Rectangle Ω = [0,2]×[0,1]
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  Node / element ordering (CCW traversal, outward normal = (dy,−dx)/L)   │
# │                                                                          │
# │  Sides & #elements proportional to side length  (N per unit length):    │
# │    Bottom (y=0, x: 0→2) : 2N elements,  nodes 0 … 2N                   │
# │    Right  (x=2, y: 0→1) :  N elements,  nodes 2N … 3N                  │
# │    Top    (y=1, x: 2→0) : 2N elements,  nodes 3N … 5N                  │
# │    Left   (x=0, y: 1→0) :  N elements,  nodes 5N … 6N (=0 wrap-around) │
# │                                                                          │
# │  Total nodes N_tot = 6N  (shared corners), total elements N_el = 6N     │
# │                                                                          │
# │  BC assignment:                                                          │
# │    Dirichlet (u known, q unknown): Right & Left sides                    │
# │      column indices: 2N … 3N  (right)  and  5N … 6N=0 (left)           │
# │    Neumann   (q known, u unknown): Bottom & Top sides                   │
# │      column indices: 0 … 2N   (bottom) and  3N … 5N  (top)             │
# │                                                                          │
# │  A[:,j] = −G[:,j]  if j is Dirichlet (q unknown)                        │
# │  A[:,j] =  H[:,j]  if j is Neumann   (u unknown)                        │
# └──────────────────────────────────────────────────────────────────────────┘
# ---------------------------------------------------------------------------
def solve_bem(N):
    t_start = time.time()

    N_el  = 6 * N               # elements  (= nodes for linear elements, closed polygon)
    N_tot = 6 * N               # nodes

    nodes_x = np.zeros(N_tot)
    nodes_y = np.zeros(N_tot)
    elems   = np.zeros((N_el, 2), dtype=np.int32)

    # ---- Nodes ----
    # Bottom: 0 … 2N-1  (start node of each bottom element, last is shared with right)
    # We store every node once; node k is the start of element k.
    # Node layout:
    #   0 … 2N  : bottom (x=k/N, y=0 for k=0..2N)  — note 2N+1 nodes but shared
    # We build them segment by segment so corners are naturally shared.

    # Bottom nodes (indices 0 … 2N-1): x = k/(N), y = 0, k=0..2N-1
    for k in range(2 * N):
        nodes_x[k] = k / N
        nodes_y[k] = 0.0

    # Right nodes (indices 2N … 3N-1): x=2, y = k/(N), k=0..N-1
    # Note: node 2N coincides with (2,0) — shared corner between bottom and right
    for k in range(N):
        nodes_x[2 * N + k] = 2.0
        nodes_y[2 * N + k] = k / N

    # Top nodes (indices 3N … 5N-1): x = 2 − k/N, y=1, k=0..2N-1
    # Node 3N = (2,1) — shared corner between right and top
    for k in range(2 * N):
        nodes_x[3 * N + k] = 2.0 - k / N
        nodes_y[3 * N + k] = 1.0

    # Left nodes (indices 5N … 6N-1): x=0, y = 1 − k/N, k=0..N-1
    # Node 5N = (0,1) — shared corner between top and left
    # Node 5N+N-1 = (0, 1/N) — last left node; next (closing) node is 0 = (0,0)
    for k in range(N):
        nodes_x[5 * N + k] = 0.0
        nodes_y[5 * N + k] = 1.0 - k / N
    # Corner (0,0) = node 0; the closing element left→bottom shares that node.

    # ---- Elements ----
    # Bottom: element k connects node k to node (k+1 mod 6N) for k=0..2N-1
    for k in range(2 * N):
        elems[k, 0] = k
        elems[k, 1] = k + 1 if k < 2 * N - 1 else 2 * N   # last bottom node → first right node

    # Right: element 2N+k connects node 2N+k to 2N+k+1 for k=0..N-1
    for k in range(N):
        elems[2 * N + k, 0] = 2 * N + k
        elems[2 * N + k, 1] = 2 * N + k + 1 if k < N - 1 else 3 * N   # → first top node

    # Top: element 3N+k connects node 3N+k to 3N+k+1 for k=0..2N-1
    for k in range(2 * N):
        elems[3 * N + k, 0] = 3 * N + k
        elems[3 * N + k, 1] = 3 * N + k + 1 if k < 2 * N - 1 else 5 * N   # → first left node

    # Left: element 5N+k connects node 5N+k to next for k=0..N-1
    for k in range(N):
        elems[5 * N + k, 0] = 5 * N + k
        elems[5 * N + k, 1] = 5 * N + k + 1 if k < N - 1 else 0   # close the loop → node 0

    # ---- Lengths and outward normals ----
    # CCW traversal → outward normal = (dy, −dx) / L
    nx = np.zeros(N_el)
    ny = np.zeros(N_el)
    lengths = np.zeros(N_el)
    for e in range(N_el):
        n1 = elems[e, 0]
        n2 = elems[e, 1]
        dx = nodes_x[n2] - nodes_x[n1]
        dy = nodes_y[n2] - nodes_y[n1]
        L  = np.hypot(dx, dy)
        lengths[e] = L
        nx[e] =  dy / L
        ny[e] = -dx / L

    # ---- BC classification ----
    # Dirichlet nodes: Right side (2N .. 3N) and Left side (5N .. 6N-1)
    # Neumann nodes:   Bottom (0 .. 2N-1) and Top (3N .. 5N-1)
    # Corner nodes belong to two sides; we assign them as Dirichlet (they lie
    # on both a Dirichlet and a Neumann side, and the Dirichlet value is given).
    dirichlet_set = set()
    for k in range(N + 1):             # right: nodes 2N .. 3N
        dirichlet_set.add(2 * N + k)
    for k in range(N):                 # left:  nodes 5N .. 6N-1
        dirichlet_set.add(5 * N + k)
    # Corner (0,0) = node 0 is shared bottom/left → Dirichlet (u=0)
    dirichlet_set.add(0)
    # Corner (2,0) = node 2N already in right set.
    # Corner (2,1) = node 3N already in right set.
    # Corner (0,1) = node 5N already in left set.

    # ---- Known boundary data ----
    u_known = np.array([exact_u(nodes_x[i], nodes_y[i]) for i in range(N_tot)])
    q_known = np.zeros(N_tot)   # q = 0 everywhere on Neumann sides (∂u/∂y = 0)
    # For Neumann nodes we don't need exact q (it's prescribed as 0).
    # For Dirichlet nodes we don't use q_known in the assembly.

    # ---- Galerkin matrix assembly ----
    pts_gl, wts_gl, pts_01, wts_01, pts_log, wts_log = get_gauss_points()

    H, G = compute_galerkin_matrices(nodes_x, nodes_y, elems, nx, ny, lengths,
                                     pts_gl, wts_gl, pts_01, wts_01, pts_log, wts_log)

    # ---- Build system A x = b ----
    # Unknown: v[j] = q_j  if j is Dirichlet;  v[j] = u_j  if j is Neumann
    A = np.zeros((N_tot, N_tot))
    b = np.zeros(N_tot)

    for j in range(N_tot):
        if j in dirichlet_set:           # Dirichlet: q unknown → A col = −G col
            A[:, j] = -G[:, j]
            b      -= H[:, j] * u_known[j]    # move known H·u to RHS
        else:                             # Neumann: u unknown → A col = +H col
            A[:, j] = H[:, j]
            b      += G[:, j] * q_known[j]    # move known G·q to RHS (q=0 here)

    t_setup = time.time() - t_start

    # ---- GMRES solve ----
    t_solve_start = time.time()
    def matvec(v):
        return A @ v
    LO      = LinearOperator((N_tot, N_tot), matvec=matvec)
    counter = IterationCounter()
    sol, info = gmres(LO, b, callback=counter, rtol=1e-12, atol=1e-12,
                      maxiter=N_tot, callback_type='pr_norm')
    t_solve = time.time() - t_solve_start

    # ---- Reconstruct full boundary solution ----
    u_all = np.zeros(N_tot)
    q_all = np.zeros(N_tot)
    for i in range(N_tot):
        if i in dirichlet_set:
            u_all[i] = u_known[i]
            q_all[i] = sol[i]
        else:
            u_all[i] = sol[i]
            q_all[i] = q_known[i]

    # ---- Interior evaluation ----
    t_eval_start = time.time()
    ngrid_x = 80; ngrid_y = 40
    _gx = np.linspace(0.01, 1.99, ngrid_x)
    _gy = np.linspace(0.01, 0.99, ngrid_y)
    gx, gy = np.meshgrid(_gx, _gy)
    gx = gx.flatten(); gy = gy.flatten()
    ix_pts = gx; iy_pts = gy

    u_interior  = eval_interior(ix_pts, iy_pts, u_all, q_all,
                                nodes_x, nodes_y, elems, nx, ny, lengths,
                                pts_gl, wts_gl)
    u_exact_int = np.array([exact_u(px, py) for px, py in zip(ix_pts, iy_pts)])
    rel_l2      = np.linalg.norm(u_interior - u_exact_int) / np.linalg.norm(u_exact_int)
    t_eval = time.time() - t_eval_start

    return {
        "N":        N,
        "Unknowns": N_tot,
        "GMRES":    counter.niter,
        "Error":    rel_l2,
        "Setup":    t_setup,
        "Solve":    t_solve,
        "Eval":     t_eval,
        "Total":    t_setup + t_solve + t_eval,
    }

# ---------------------------------------------------------------------------
# Refinement study
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Warm-up: compile Numba kernels (excluded from timing)
    _ = solve_bem(10)

    N_values = [20, 40, 80, 160, 320]
    results  = []

    print(f"{'N':<4} {'Unknowns':<10} {'GMRES':<7} {'Rel L2 Error':<13} {'Conv Rate':<10} "
          f"{'Setup':<9} {'Solve':<9} {'Eval':<9} {'Total':<9}")
    print("-" * 90)

    for n in N_values:
        res = solve_bem(n)
        results.append(res)

    for i, res in enumerate(results):
        if i == 0:
            rate_str = "      ---"
        else:
            prev = results[i - 1]
            rate = np.log(prev['Error'] / res['Error']) / np.log(res['N'] / prev['N'])
            rate_str = f"{rate:9.2f}"
        print(f"{res['N']:<4d} {res['Unknowns']:<10d} {res['GMRES']:<7d} {res['Error']:13.6e} "
              f"{rate_str:10s} {res['Setup']:<9.3f} {res['Solve']:<9.3f} "
              f"{res['Eval']:<9.3f} {res['Total']:<9.3f}")

    if len(results) > 1:
        log_h   = np.log([1.0 / res['N'] for res in results])
        log_err = np.log([res['Error']    for res in results])
        A_fit   = np.vstack([log_h, np.ones(len(log_h))]).T
        m, c    = np.linalg.lstsq(A_fit, log_err, rcond=None)[0]
        print("-" * 90)
        print(f"Final Convergence Order: {m:.2f}")