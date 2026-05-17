#
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
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (r**3 + r**(-3)) * np.cos(3 * theta)

@njit(cache=True)
def exact_q(x, y, nx, ny):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    dudr = (3 * r**2 - 3 * r**(-4)) * np.cos(3 * theta)
    dudt = -3 * (r**3 + r**(-3)) * np.sin(3 * theta)
    ux = dudr * np.cos(theta) - (1.0/r) * dudt * np.sin(theta)
    uy = dudr * np.sin(theta) + (1.0/r) * dudt * np.cos(theta)
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

    N_tot = 2 * N
    nodes_x = np.zeros(N_tot)
    nodes_y = np.zeros(N_tot)
    elems = np.zeros((N_tot, 2), dtype=np.int32)
    nx = np.zeros(N_tot)
    ny = np.zeros(N_tot)
    lengths = np.zeros(N_tot)

    for i in range(N):
        theta = -2.0 * np.pi * i / N
        nodes_x[i] = 1.0 * np.cos(theta)
        nodes_y[i] = 1.0 * np.sin(theta)
        elems[i, 0] = i
        elems[i, 1] = (i + 1) % N

    for i in range(N):
        theta = 2.0 * np.pi * i / N
        nodes_x[N+i] = 2.0 * np.cos(theta)
        nodes_y[N+i] = 2.0 * np.sin(theta)
        elems[N+i, 0] = N + i
        elems[N+i, 1] = N + ((i + 1) % N)

    for e in range(N_tot):
        n1 = elems[e, 0]
        n2 = elems[e, 1]
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
        if i < N:
            node_nx, node_ny = -nodes_x[i], -nodes_y[i]
        else:
            node_nx, node_ny = nodes_x[i]/2.0, nodes_y[i]/2.0
        q_known[i] = exact_q(nodes_x[i], nodes_y[i], node_nx, node_ny)

    for j in range(N_tot):
        if j < N:
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

    u_all = np.zeros(N_tot)
    q_all = np.zeros(N_tot)
    for i in range(N_tot):
        if i < N:
            u_all[i] = u_known[i]
            q_all[i] = sol[i]
        else:
            u_all[i] = sol[i]
            q_all[i] = q_known[i]

    t_eval_start = time.time()
    ngrid = 60
    _grid = np.linspace(-1.9, 1.9, ngrid)
    gx, gy = np.meshgrid(_grid, _grid)
    gx = gx.flatten()
    gy = gy.flatten()

    r_val = np.sqrt(gx**2 + gy**2)
    mask = (r_val > 1.0) & (r_val < 2.0)
    ix_pts = gx[mask]
    iy_pts = gy[mask]

    u_interior = eval_interior(ix_pts, iy_pts, u_all, q_all, nodes_x, nodes_y, elems, nx, ny, lengths, pts_gl, wts_gl)
    u_exact_int = np.array([exact_u(px, py) for px, py in zip(ix_pts, iy_pts)])
    rel_l2 = np.linalg.norm(u_interior - u_exact_int) / np.linalg.norm(u_exact_int)
    t_eval = time.time() - t_eval_start

    return {
        "N": N,
        "Unknowns": N_tot,
        "GMRES": counter.niter,
        "Error": rel_l2,
        "Setup": t_setup,
        "Solve": t_solve,
        "Eval": t_eval,
        "Total": t_setup + t_solve + t_eval
    }

if __name__ == "__main__":
    _ = solve_bem(10)

    N_values = [160, 320, 640, 1280, 2560, 5120]
    results = []

    print(f"{'N':<4} {'Unknowns':<10} {'GMRES':<7} {'Rel L2 Error':<13} {'Conv Rate':<10} {'Setup':<9} {'Solve':<9} {'Eval':<9} {'Total':<9}")
    print("-" * 90)

    for n in N_values:
        res = solve_bem(n)
        results.append(res)

    for i, res in enumerate(results):
        if i == 0:
            rate_str = "      ---"
        else:
            prev = results[i-1]
            rate = np.log(prev['Error'] / res['Error']) / np.log(res['N'] / prev['N'])
            rate_str = f"{rate:9.2f}"

        print(f"{res['N']:<4d} {res['Unknowns']:<10d} {res['GMRES']:<7d} {res['Error']:13.6e} {rate_str:10s} {res['Setup']:<9.3f} {res['Solve']:<9.3f} {res['Eval']:<9.3f} {res['Total']:<9.3f}")

    if len(results) > 1:
        log_h = np.log([1.0 / res['N'] for res in results])
        log_err = np.log([res['Error'] for res in results])
        A = np.vstack([log_h, np.ones(len(log_h))]).T
        m, c = np.linalg.lstsq(A, log_err, rcond=None)[0]
        print("-" * 90)
        print(f"Final Convergence Order: {m:.2f}")