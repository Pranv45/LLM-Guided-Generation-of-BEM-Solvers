import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange

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

@njit(cache=True)
def exact_u(x, y):
    return (np.sinh(np.pi * x) / np.sinh(2.0 * np.pi)) * np.cos(np.pi * y)

@njit(cache=True)
def exact_q(x, y, nx, ny):
    ux = np.pi * (np.cosh(np.pi * x) / np.sinh(2.0 * np.pi)) * np.cos(np.pi * y)
    uy = -np.pi * (np.sinh(np.pi * x) / np.sinh(2.0 * np.pi)) * np.sin(np.pi * y)
    return ux * nx + uy * ny

@njit(parallel=True, fastmath=True, cache=True)
def compute_galerkin_matrices(nodes_x, nodes_y, elems, nx, ny, lengths, pts_gl, wts_gl, pts_01, wts_01, pts_log, wts_log):
    N_tot = len(nodes_x)
    N_el = len(elems)
    H_mat = np.zeros((N_tot, N_tot))
    G_mat = np.zeros((N_tot, N_tot))

    for et in prange(N_el):
        it1 = elems[et, 0]
        it2 = elems[et, 1]
        Lt = lengths[et]

        for es in range(N_el):
            js1 = elems[es, 0]
            js2 = elems[es, 1]
            Ls = lengths[es]

            h11, h12, h21, h22 = 0.0, 0.0, 0.0, 0.0
            g11, g12, g21, g22 = 0.0, 0.0, 0.0, 0.0

            if et == es:
                mass11 = Lt / 3.0
                mass12 = Lt / 6.0
                mass22 = Lt / 3.0
                h11 += 0.5 * mass11
                h12 += 0.5 * mass12
                h21 += 0.5 * mass12
                h22 += 0.5 * mass22

                const_part = -(Lt * Lt / (8.0 * np.pi)) * np.log(Lt / 2.0)
                g11 += const_part
                g12 += const_part
                g21 += const_part
                g22 += const_part

                for kt in range(len(pts_gl)):
                    xt = pts_gl[kt]
                    wt = wts_gl[kt]
                    Nt1 = 0.5 * (1.0 - xt)
                    Nt2 = 0.5 * (1.0 + xt)

                    I_log_1 = 0.0
                    I_log_2 = 0.0

                    len_left = xt + 1.0
                    if len_left > 1e-12:
                        for ks in range(len(pts_01)):
                            s = pts_01[ks]
                            ws = wts_01[ks]
                            xs = xt - len_left * s
                            Ns1 = 0.5 * (1.0 - xs)
                            Ns2 = 0.5 * (1.0 + xs)
                            term = len_left * np.log(len_left) * ws
                            I_log_1 += term * Ns1
                            I_log_2 += term * Ns2
                        for ks in range(len(pts_log)):
                            s = pts_log[ks]
                            ws = wts_log[ks]
                            xs = xt - len_left * s
                            Ns1 = 0.5 * (1.0 - xs)
                            Ns2 = 0.5 * (1.0 + xs)
                            term = -len_left * ws
                            I_log_1 += term * Ns1
                            I_log_2 += term * Ns2

                    len_right = 1.0 - xt
                    if len_right > 1e-12:
                        for ks in range(len(pts_01)):
                            s = pts_01[ks]
                            ws = wts_01[ks]
                            xs = xt + len_right * s
                            Ns1 = 0.5 * (1.0 - xs)
                            Ns2 = 0.5 * (1.0 + xs)
                            term = len_right * np.log(len_right) * ws
                            I_log_1 += term * Ns1
                            I_log_2 += term * Ns2
                        for ks in range(len(pts_log)):
                            s = pts_log[ks]
                            ws = wts_log[ks]
                            xs = xt + len_right * s
                            Ns1 = 0.5 * (1.0 - xs)
                            Ns2 = 0.5 * (1.0 + xs)
                            term = -len_right * ws
                            I_log_1 += term * Ns1
                            I_log_2 += term * Ns2

                    factor = -(Lt * Lt) / (8.0 * np.pi) * wt
                    g11 += Nt1 * I_log_1 * factor
                    g12 += Nt1 * I_log_2 * factor
                    g21 += Nt2 * I_log_1 * factor
                    g22 += Nt2 * I_log_2 * factor
            else:
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
                        r2 = rx*rx + ry*ry

                        if r2 > 1e-14:
                            G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                            dGdn = (rx * nx[es] + ry * ny[es]) / (2.0 * np.pi * r2)

                            w_tot = wt * ws * 0.25 * Lt * Ls

                            h11 += Nt1 * Ns1 * dGdn * w_tot
                            h12 += Nt1 * Ns2 * dGdn * w_tot
                            h21 += Nt2 * Ns1 * dGdn * w_tot
                            h22 += Nt2 * Ns2 * dGdn * w_tot

                            g11 += Nt1 * Ns1 * G_val * w_tot
                            g12 += Nt1 * Ns2 * G_val * w_tot
                            g21 += Nt2 * Ns1 * G_val * w_tot
                            g22 += Nt2 * Ns2 * G_val * w_tot

            H_mat[it1, js1] += h11
            H_mat[it1, js2] += h12
            H_mat[it2, js1] += h21
            H_mat[it2, js2] += h22

            G_mat[it1, js1] += g11
            G_mat[it1, js2] += g12
            G_mat[it2, js1] += g21
            G_mat[it2, js2] += g22

    return H_mat, G_mat

@njit(parallel=True, fastmath=True, cache=True)
def eval_interior(ix_pts, iy_pts, u_all, q_all, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts):
    M = len(ix_pts)
    N_tot = len(nodes_x)
    u_int = np.zeros(M)
    for i in prange(M):
        val = 0.0
        for e in range(len(elems)):
            j1 = elems[e, 0]
            j2 = elems[e, 1]
            L = lengths[e]
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
                dGdn = (rx * nx[e] + ry * ny[e]) / (2.0 * np.pi * r2)

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

def solve_bem(N):
    t_start = time.time()

    # Domain [0,2] x [0,1]. N panels on short edges, 2N on long edges. Total = 6N panels
    N_tot = 6 * N
    nodes_x = np.zeros(N_tot)
    nodes_y = np.zeros(N_tot)
    elems = np.zeros((N_tot, 2), dtype=np.int32)
    nx = np.zeros(N_tot)
    ny = np.zeros(N_tot)
    lengths = np.zeros(N_tot)
    is_dirichlet = np.zeros(N_tot, dtype=np.int32)

    idx = 0
    # Bottom edge (y=0), x from 0 to 2
    for i in range(2 * N):
        nodes_x[idx] = i * (1.0 / N)
        nodes_y[idx] = 0.0
        is_dirichlet[idx] = 1 if i == 0 else 0  # Corner (0,0) is Dirichlet
        idx += 1

    # Right edge (x=2), y from 0 to 1
    for i in range(N):
        nodes_x[idx] = 2.0
        nodes_y[idx] = i * (1.0 / N)
        is_dirichlet[idx] = 1                   # Whole right edge is Dirichlet
        idx += 1

    # Top edge (y=1), x from 2 to 0
    for i in range(2 * N):
        nodes_x[idx] = 2.0 - i * (1.0 / N)
        nodes_y[idx] = 1.0
        is_dirichlet[idx] = 1 if i == 0 else 0  # Corner (2,1) is Dirichlet
        idx += 1

    # Left edge (x=0), y from 1 to 0
    for i in range(N):
        nodes_x[idx] = 0.0
        nodes_y[idx] = 1.0 - i * (1.0 / N)
        is_dirichlet[idx] = 1                   # Whole left edge is Dirichlet
        idx += 1

    # Connectivity and Elements Definition
    for e in range(N_tot):
        n1 = e
        n2 = (e + 1) % N_tot
        elems[e, 0] = n1
        elems[e, 1] = n2

        dx = nodes_x[n2] - nodes_x[n1]
        dy = nodes_y[n2] - nodes_y[n1]
        L = np.hypot(dx, dy)
        lengths[e] = L
        nx[e] = dy / L
        ny[e] = -dx / L

    pts_gl, wts_gl, pts_01, wts_01, pts_log, wts_log = get_gauss_points()

    H, G = compute_galerkin_matrices(nodes_x, nodes_y, elems, nx, ny, lengths, pts_gl, wts_gl, pts_01, wts_01, pts_log, wts_log)

    A = np.zeros((N_tot, N_tot))
    b = np.zeros(N_tot)

    u_known = np.zeros(N_tot)
    q_known = np.zeros(N_tot)

    for i in range(N_tot):
        u_known[i] = exact_u(nodes_x[i], nodes_y[i])

        # Approximate boundary normal at the node for proper analytical q mapping
        prev_e = (i - 1 + N_tot) % N_tot
        next_e = i
        nnx = nx[prev_e] + nx[next_e]
        nny = ny[prev_e] + ny[next_e]
        norm = np.hypot(nnx, nny)

        if norm > 1e-12:
            node_nx, node_ny = nnx / norm, nny / norm
        else:
            node_nx, node_ny = nx[next_e], ny[next_e]

        q_known[i] = exact_q(nodes_x[i], nodes_y[i], node_nx, node_ny)

    # Matrix Assembly
    for j in range(N_tot):
        if is_dirichlet[j] == 1:
            A[:, j] = -G[:, j]
            b -= H[:, j] * u_known[j]
        else:
            A[:, j] = H[:, j]
            b += G[:, j] * q_known[j]

    t_setup = time.time() - t_start

    t_solve_start = time.time()
    def matvec(v):
        return A @ v

    LO = LinearOperator((N_tot, N_tot), matvec=matvec)
    counter = IterationCounter()
    sol, info = gmres(LO, b, callback=counter, rtol=1e-12, atol=1e-12, maxiter=N_tot, callback_type='pr_norm')
    t_solve = time.time() - t_solve_start

    # Assemble back continuous fields for evaluation
    u_all = np.zeros(N_tot)
    q_all = np.zeros(N_tot)
    for i in range(N_tot):
        if is_dirichlet[i] == 1:
            u_all[i] = u_known[i]
            q_all[i] = sol[i]
        else:
            u_all[i] = sol[i]
            q_all[i] = q_known[i]

    # Interior evaluation
    t_eval_start = time.time()
    ngrid_x = 60
    ngrid_y = 30
    _grid_x = np.linspace(0.01, 1.99, ngrid_x)
    _grid_y = np.linspace(0.01, 0.99, ngrid_y)
    gx, gy = np.meshgrid(_grid_x, _grid_y)
    ix_pts = gx.flatten()
    iy_pts = gy.flatten()

    u_interior = eval_interior(ix_pts, iy_pts, u_all, q_all, nodes_x, nodes_y, elems, nx, ny, lengths, pts_gl, wts_gl)
    u_exact_int = np.array([exact_u(px, py) for px, py in zip(ix_pts, iy_pts)])
    rel_l2 = np.linalg.norm(u_interior - u_exact_int) / np.linalg.norm(u_exact_int)
    t_eval = time.time() - t_eval_start

    return {
        "N": N,
        "DOFs": N_tot,
        "GMRES": counter.niter,
        "RelL2": rel_l2,
        "Asm(s)": t_setup,
        "Solve(s)": t_solve,
        "Eval(s)": t_eval,
        "Total(s)": t_setup + t_solve + t_eval
    }

if __name__ == "__main__":
    # Numba JIT warmup sequence
    _ = solve_bem(4)

    N_values = [20, 40, 80, 160, 320]
    results = []

    print(f"{'N':<4} {'DOFs':<8} {'GMRES':<7} {'RelL2':<13} {'Conv Rate':<10} {'Asm(s)':<9} {'Solve(s)':<9} {'Eval(s)':<9} {'Total(s)':<9}")
    print("-" * 92)

    for n in N_values:
        res = solve_bem(n)
        results.append(res)

    for i, res in enumerate(results):
        if i == 0:
            rate_str = "      ---"
        else:
            prev = results[i-1]
            rate = np.log(prev['RelL2'] / res['RelL2']) / np.log(res['DOFs'] / prev['DOFs'])
            rate_str = f"{rate:9.2f}"

        print(f"{res['N']:<4d} {res['DOFs']:<8d} {res['GMRES']:<7d} {res['RelL2']:13.6e} {rate_str:10s} {res['Asm(s)']:<9.3f} {res['Solve(s)']:<9.3f} {res['Eval(s)']:<9.3f} {res['Total(s)']:<9.3f}")

    if len(results) > 1:
        log_h = np.log([1.0 / res['DOFs'] for res in results])
        log_err = np.log([res['RelL2'] for res in results])
        A = np.vstack([log_h, np.ones(len(log_h))]).T
        m, c = np.linalg.lstsq(A, log_err, rcond=None)[0]
        print("-" * 92)
        print(f"Final Convergence Order: {m:.2f}")