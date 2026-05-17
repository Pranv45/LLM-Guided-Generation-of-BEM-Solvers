import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange

@njit(cache=True)
def get_gauss_points():
    pts = np.array([-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
                     0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363])
    wts = np.array([ 0.1012285362903763,  0.2223810344533745,  0.3137066458778873,  0.3626837833783620,
                     0.3626837833783620,  0.3137066458778873,  0.2223810344533745,  0.1012285362903763])
    return pts, wts

@njit(cache=True)
def exact_u(x, y):
    # u(x,y) = (sinh(pi*x) / sinh(2*pi)) * cos(pi*y)
    return (np.sinh(np.pi * x) / np.sinh(2.0 * np.pi)) * np.cos(np.pi * y)

@njit(cache=True)
def exact_q(x, y, nnx, nny):
    # exact normal derivative q = grad(u) . n
    ux = np.pi * np.cosh(np.pi * x) / np.sinh(2.0 * np.pi) * np.cos(np.pi * y)
    uy = -np.pi * np.sinh(np.pi * x) / np.sinh(2.0 * np.pi) * np.sin(np.pi * y)
    return ux * nnx + uy * nny

@njit(parallel=True, fastmath=True, cache=True)
def bem_matvec(v, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts, is_dirichlet, c_coeff):
    N_tot = len(nodes_x)
    out = np.zeros(N_tot)

    for i in prange(N_tot):
        row_val = 0.0

        # If the node is Neumann, v[i] holds unknown u_i.
        # It must be multiplied by the free term c(x) and remain on the LHS.
        if is_dirichlet[i] == 0:
            row_val -= c_coeff[i] * v[i]

        for e in range(N_tot):
            j1 = elems[e, 0]
            j2 = elems[e, 1]
            L = lengths[e]

            if i == j1:
                G1 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.75 * L)
                G2 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.25 * L)
                H1 = 0.0
                H2 = 0.0
            elif i == j2:
                G1 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.25 * L)
                G2 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.75 * L)
                H1 = 0.0
                H2 = 0.0
            else:
                G1 = 0.0; G2 = 0.0; H1 = 0.0; H2 = 0.0
                for k in range(len(pts)):
                    xi = pts[k]
                    w = wts[k]
                    N1 = 0.5 * (1.0 - xi)
                    N2 = 0.5 * (1.0 + xi)
                    yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                    yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]

                    rx = nodes_x[i] - yx
                    ry = nodes_y[i] - yy
                    r2 = rx*rx + ry*ry

                    G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                    H_val = 1.0 / (2.0 * np.pi * r2) * (rx * nx[e] + ry * ny[e])

                    wL2 = w * L * 0.5
                    G1 += G_val * N1 * wL2
                    G2 += G_val * N2 * wL2
                    H1 += H_val * N1 * wL2
                    H2 += H_val * N2 * wL2

            if is_dirichlet[j1] == 1:
                row_val += G1 * v[j1]
            else:
                row_val -= H1 * v[j1]

            if is_dirichlet[j2] == 1:
                row_val += G2 * v[j2]
            else:
                row_val -= H2 * v[j2]

        out[i] = row_val
    return out

@njit(parallel=True, fastmath=True, cache=True)
def build_rhs(u_known, q_known, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts, is_dirichlet, c_coeff):
    N_tot = len(nodes_x)
    rhs = np.zeros(N_tot)

    for i in prange(N_tot):
        row_val = 0.0

        # If the node is Dirichlet, u_i is known, so the free term moves to the RHS.
        if is_dirichlet[i] == 1:
            row_val += c_coeff[i] * u_known[i]

        for e in range(N_tot):
            j1 = elems[e, 0]
            j2 = elems[e, 1]
            L = lengths[e]

            if i == j1:
                G1 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.75 * L)
                G2 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.25 * L)
                H1 = 0.0
                H2 = 0.0
            elif i == j2:
                G1 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.25 * L)
                G2 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.75 * L)
                H1 = 0.0
                H2 = 0.0
            else:
                G1 = 0.0; G2 = 0.0; H1 = 0.0; H2 = 0.0
                for k in range(len(pts)):
                    xi = pts[k]
                    w = wts[k]
                    N1 = 0.5 * (1.0 - xi)
                    N2 = 0.5 * (1.0 + xi)
                    yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                    yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]

                    rx = nodes_x[i] - yx
                    ry = nodes_y[i] - yy
                    r2 = rx*rx + ry*ry

                    G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                    H_val = 1.0 / (2.0 * np.pi * r2) * (rx * nx[e] + ry * ny[e])

                    wL2 = w * L * 0.5
                    G1 += G_val * N1 * wL2
                    G2 += G_val * N2 * wL2
                    H1 += H_val * N1 * wL2
                    H2 += H_val * N2 * wL2

            if is_dirichlet[j1] == 1:
                row_val += H1 * u_known[j1]
            else:
                row_val -= G1 * q_known[j1]

            if is_dirichlet[j2] == 1:
                row_val += H2 * u_known[j2]
            else:
                row_val -= G2 * q_known[j2]

        rhs[i] = row_val
    return rhs

@njit(parallel=True, fastmath=True, cache=True)
def eval_interior(ix_pts, iy_pts, u_all, q_all, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts):
    M = len(ix_pts)
    N_tot = len(nodes_x)
    u_int = np.zeros(M)

    for i in prange(M):
        val = 0.0
        for e in range(N_tot):
            j1 = elems[e, 0]
            j2 = elems[e, 1]
            L = lengths[e]

            G1 = 0.0; G2 = 0.0; H1 = 0.0; H2 = 0.0
            for k in range(len(pts)):
                xi = pts[k]
                w = wts[k]
                N1 = 0.5 * (1.0 - xi)
                N2 = 0.5 * (1.0 + xi)
                yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]

                rx = ix_pts[i] - yx
                ry = iy_pts[i] - yy
                r2 = rx*rx + ry*ry

                G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                H_val = 1.0 / (2.0 * np.pi * r2) * (rx * nx[e] + ry * ny[e])

                wL2 = w * L * 0.5
                G1 += G_val * N1 * wL2
                G2 += G_val * N2 * wL2
                H1 += H_val * N1 * wL2
                H2 += H_val * N2 * wL2

            val += (G1 * q_all[j1] + G2 * q_all[j2]) - (H1 * u_all[j1] + H2 * u_all[j2])
        u_int[i] = val
    return u_int

class IterationCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

def solve_bem(N):
    t_start = time.time()

    # Domain [0,2] x [0,1]. Number of continuous linear panels per unit length = N
    # Total nodes = 2*N (bottom) + N (right) + 2*N (top) + N (left) = 6*N
    N_tot = 6 * N
    nodes_x = np.zeros(N_tot)
    nodes_y = np.zeros(N_tot)
    is_dirichlet = np.zeros(N_tot, dtype=np.int32)
    c_coeff = np.full(N_tot, 0.5)

    idx = 0

    # Bottom edge (y=0, x from 0 to 2)
    for i in range(2 * N):
        nodes_x[idx] = i * (1.0 / N)
        nodes_y[idx] = 0.0
        # Node 0 is the corner (0,0), Left BC wins (Dirichlet)
        is_dirichlet[idx] = 1 if i == 0 else 0
        c_coeff[idx] = 0.25 if i == 0 else 0.5
        idx += 1

    # Right edge (x=2, y from 0 to 1) -> All Dirichlet
    for i in range(N):
        nodes_x[idx] = 2.0
        nodes_y[idx] = i * (1.0 / N)
        is_dirichlet[idx] = 1
        c_coeff[idx] = 0.25 if i == 0 else 0.5
        idx += 1

    # Top edge (y=1, x from 2 to 0)
    for i in range(2 * N):
        nodes_x[idx] = 2.0 - i * (1.0 / N)
        nodes_y[idx] = 1.0
        # Node 3N is corner (2,1), Right BC wins (Dirichlet)
        is_dirichlet[idx] = 1 if i == 0 else 0
        c_coeff[idx] = 0.25 if i == 0 else 0.5
        idx += 1

    # Left edge (x=0, y from 1 to 0) -> All Dirichlet
    for i in range(N):
        nodes_x[idx] = 0.0
        nodes_y[idx] = 1.0 - i * (1.0 / N)
        is_dirichlet[idx] = 1
        c_coeff[idx] = 0.25 if i == 0 else 0.5
        idx += 1

    # Build Elements Connectivity
    elems = np.zeros((N_tot, 2), dtype=np.int32)
    nx = np.zeros(N_tot)
    ny = np.zeros(N_tot)
    lengths = np.zeros(N_tot)

    for e in range(N_tot):
        j1 = e
        j2 = (e + 1) % N_tot
        elems[e, 0] = j1
        elems[e, 1] = j2

        dx = nodes_x[j2] - nodes_x[j1]
        dy = nodes_y[j2] - nodes_y[j1]
        L = np.sqrt(dx**2 + dy**2)
        lengths[e] = L

        # Outward pointing normal vector for a CCW parameterized domain
        nx[e] = dy / L
        ny[e] = -dx / L

    # Calculate Known Vectors
    u_known = np.zeros(N_tot)
    q_known = np.zeros(N_tot)

    for i in range(N_tot):
        u_known[i] = exact_u(nodes_x[i], nodes_y[i])

        # Approximate normal vector at nodes as the average of adjoining elements
        e_prev = (i - 1) % N_tot
        e_curr = i
        nnx = nx[e_prev] + nx[e_curr]
        nny = ny[e_prev] + ny[e_curr]
        norm = np.sqrt(nnx**2 + nny**2)
        nnx /= norm
        nny /= norm

        q_known[i] = exact_q(nodes_x[i], nodes_y[i], nnx, nny)

    pts, wts = get_gauss_points()

    # Assemble right-hand side
    rhs = build_rhs(u_known, q_known, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts, is_dirichlet, c_coeff)
    t_setup = time.time() - t_start

    # Matrix-free setup
    t_solve_start = time.time()
    def matvec(v):
        return bem_matvec(v, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts, is_dirichlet, c_coeff)

    LO = LinearOperator((N_tot, N_tot), matvec=matvec)
    counter = IterationCounter()
    sol, info = gmres(LO, rhs, callback=counter, rtol=1e-12, maxiter=N_tot, callback_type='legacy')
    t_solve = time.time() - t_solve_start

    # Recover the full fields u and q everywhere
    u_all = np.zeros(N_tot)
    q_all = np.zeros(N_tot)
    for i in range(N_tot):
        if is_dirichlet[i] == 1:
            u_all[i] = u_known[i]
            q_all[i] = sol[i]
        else:
            u_all[i] = sol[i]
            q_all[i] = q_known[i]

    # Evaluate at internal points
    t_eval_start = time.time()

    _grid_x = np.linspace(0.01, 1.99, 80)
    _grid_y = np.linspace(0.01, 0.99, 40)
    gx, gy = np.meshgrid(_grid_x, _grid_y)
    ix_pts = gx.flatten()
    iy_pts = gy.flatten()

    u_interior = eval_interior(ix_pts, iy_pts, u_all, q_all, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts)
    u_exact_int = np.array([exact_u(px, py) for px, py in zip(ix_pts, iy_pts)])
    rel_l2 = np.linalg.norm(u_interior - u_exact_int) / np.linalg.norm(u_exact_int)

    t_eval = time.time() - t_eval_start

    return {
        "N": N,
        "tot_N": N_tot,
        "iters": counter.niter,
        "error": rel_l2,
        "t_setup": t_setup,
        "t_solve": t_solve,
        "t_eval": t_eval,
        "t_total": t_setup + t_solve + t_eval
    }

if __name__ == "__main__":
    # Warm-up compile pass for accurate timing benchmarks
    _ = solve_bem(4)

    N_values = [20, 40, 80, 160, 320]
    results = []

    print(f"{'N':>5} | {'DOFs':>6} | {'Asm(s)':>8} | {'Solve(s)':>8} | {'Eval(s)':>8} | {'Total(s)':>8} | {'RelL2':>12} | {'Rate':>8}")
    print("-" * 92)

    for n in N_values:
        res = solve_bem(n)
        results.append(res)

    for i, res in enumerate(results):
        if i == 0:
            rate_str = "     ---"
        else:
            prev = results[i-1]
            rate = np.log(prev['error'] / res['error']) / np.log(res['tot_N'] / prev['tot_N'])
            rate_str = f"{rate:8.2f}"

        print(f"{res['N']:5d} | {res['tot_N']:6d} | {res['t_setup']:8.3f} | {res['t_solve']:8.3f} | {res['t_eval']:8.3f} | {res['t_total']:8.3f} | {res['error']:12.6e} | {rate_str}")

    if len(results) > 1:
        overall_rate = np.log(results[0]['error'] / results[-1]['error']) / np.log(results[-1]['tot_N'] / results[0]['tot_N'])
        print("-" * 92)
        print(f"Overall Convergence Order (vs DOFs): {overall_rate:.2f}")