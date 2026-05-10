import numpy as np
import time
from scipy.sparse.linalg import gmres
from numba import njit, prange

def bem_setup(M, n_freq):
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    nodes = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    f_vals = np.cos(n_freq * theta)
    idx1 = np.arange(M)
    idx2 = (idx1 + 1) % M
    p1, p2 = nodes[idx1], nodes[idx2]
    L = np.linalg.norm(p2 - p1, axis=1)
    mid = (p1 + p2) / 2.0
    normals = mid / np.linalg.norm(mid, axis=1)[:, np.newaxis]
    q_pts, q_w = np.polynomial.legendre.leggauss(8)
    phi1, phi2 = (1 - q_pts) / 2.0, (1 + q_pts) / 2.0
    return nodes, f_vals, idx1, idx2, L, normals, q_pts, q_w, phi1, phi2

@njit(parallel=True, fastmath=True)
def assemble_matrix_numba(M, nodes, idx1, idx2, L, normals, q_w, phi1, phi2):
    Mass = np.zeros((M, M))
    K_mat = np.zeros((M, M))
    Q = len(q_w)

    # 1. Mass Matrix Assembly
    for i in range(M):
        n1, n2 = idx1[i], idx2[i]
        L3, L6 = L[i] / 3.0, L[i] / 6.0
        Mass[n1, n1] += L3
        Mass[n2, n2] += L3
        Mass[n1, n2] += L6
        Mass[n2, n1] += L6

    # 2. Vectorized Kernel Assembly (Galerkin Double Integration)
    for target_e in prange(M):
        t1, t2 = idx1[target_e], idx2[target_e]
        L_t = L[target_e]

        for source_k in range(M):
            s1, s2 = idx1[source_k], idx2[source_k]
            L_s = L[source_k]
            nk = normals[source_k]

            v11 = v12 = v21 = v22 = 0.0

            for qt in range(Q):
                wt = q_w[qt] * L_t / 2.0
                pt1, pt2 = phi1[qt], phi2[qt]
                ytx = nodes[t1, 0] * pt1 + nodes[t2, 0] * pt2
                yty = nodes[t1, 1] * pt1 + nodes[t2, 1] * pt2

                for qs in range(Q):
                    ws = q_w[qs] * L_s / 2.0
                    ps1, ps2 = phi1[qs], phi2[qs]
                    ysx = nodes[s1, 0] * ps1 + nodes[s2, 0] * ps2
                    ysy = nodes[s1, 1] * ps1 + nodes[s2, 1] * ps2

                    if target_e == source_k:
                        kernel = 0.0 # Analytically zero for linear segments on circle
                    else:
                        dx = ytx - ysx
                        dy = yty - ysy
                        dist_sq = dx*dx + dy*dy
                        dot_prod = dx*nk[0] + dy*nk[1]
                        kernel = -1.0 / (2.0 * np.pi) * (dot_prod / dist_sq)

                    term = kernel * wt * ws
                    v11 += term * pt1 * ps1
                    v12 += term * pt1 * ps2
                    v21 += term * pt2 * ps1
                    v22 += term * pt2 * ps2

            K_mat[t1, s1] += v11
            K_mat[t1, s2] += v12
            K_mat[t2, s1] += v21
            K_mat[t2, s2] += v22

    return 0.5 * Mass + K_mat

@njit(fastmath=True)
def assemble_rhs_numba(nodes, f_vals, idx1, idx2, L, q_w, phi1, phi2):
    M = len(nodes)
    b = np.zeros(M)
    Q = len(q_w)
    for k in range(M):
        n1, n2 = idx1[k], idx2[k]
        jw = L[k] / 2.0
        for q in range(Q):
            w = q_w[q] * jw
            p1, p2 = phi1[q], phi2[q]
            fq = f_vals[n1] * p1 + f_vals[n2] * p2
            term = fq * w
            b[n1] += term * p1
            b[n2] += term * p2
    return b

def solve_system(A, nodes, f_vals, idx1, idx2, L, q_w, phi1, phi2):
    b = assemble_rhs_numba(nodes, f_vals, idx1, idx2, L, q_w, phi1, phi2)
    it_count = [0]
    def cb(rk): it_count[0] += 1
    mu, info = gmres(A, b, callback=cb, atol=1e-10, callback_type='legacy')
    return mu, it_count[0]

@njit(parallel=True, fastmath=True)
def bem_evaluate_numba(mu, nodes, idx1, idx2, L, normals, q_w, phi1, phi2, eval_pts):
    N_eval = eval_pts.shape[0]
    M = len(L)
    Q = len(q_w)
    u_eval = np.zeros(N_eval)

    for i in prange(N_eval):
        ex, ey = eval_pts[i, 0], eval_pts[i, 1]
        val = 0.0
        for k in range(M):
            s1, s2 = idx1[k], idx2[k]
            nk = normals[k]
            jw = L[k] / 2.0

            for q in range(Q):
                ps1, ps2 = phi1[q], phi2[q]
                w = q_w[q] * jw

                yqx = nodes[s1, 0] * ps1 + nodes[s2, 0] * ps2
                yqy = nodes[s1, 1] * ps1 + nodes[s2, 1] * ps2
                mu_q = mu[s1] * ps1 + mu[s2] * ps2

                dx = ex - yqx
                dy = ey - yqy
                dist_sq = dx*dx + dy*dy
                dot_prod = dx*nk[0] + dy*nk[1]

                kernel = -1.0 / (2.0 * np.pi) * (dot_prod / dist_sq)
                val += kernel * w * mu_q

        u_eval[i] = val
    return u_eval

def run_bem(M, n_freq=3):
    t0 = time.perf_counter()
    nodes, f_vals, idx1, idx2, L, normals, q_pts, q_w, phi1, phi2 = bem_setup(M, n_freq)
    A = assemble_matrix_numba(M, nodes, idx1, idx2, L, normals, q_w, phi1, phi2)
    t_setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    mu, iters = solve_system(A, nodes, f_vals, idx1, idx2, L, q_w, phi1, phi2)
    t_solve = time.perf_counter() - t1

    # Grid Evaluation
    Nx = Ny = 120
    xx, yy = np.linspace(-0.9, 0.9, Nx), np.linspace(-0.9, 0.9, Ny)
    X, Y = np.meshgrid(xx, yy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    mask = (X**2 + Y**2).ravel() < 0.9**2
    eval_pts = pts[mask]

    t2 = time.perf_counter()
    u_num = bem_evaluate_numba(mu, nodes, idx1, idx2, L, normals, q_w, phi1, phi2, eval_pts)
    t_eval = time.perf_counter() - t2

    r, th = np.sqrt(np.sum(eval_pts**2, axis=1)), np.arctan2(eval_pts[:, 1], eval_pts[:, 0])
    u_ex = (r**n_freq) * np.cos(n_freq * th)
    rel_l2 = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)
    linf = np.max(np.abs(u_num - u_ex))

    return {"M": M, "iters": iters, "setup": t_setup, "solve": t_solve, "eval": t_eval, "l2": rel_l2, "linf": linf}

if __name__ == "__main__":
    # Warm-up Numba compilers so the time isn't added to the first M run
    _ = run_bem(10)

    M_list = [200, 400, 800, 1600]
    print(f"{'M':>6} | {'Iters':>5} | {'Setup(s)':>8} | {'Solve(s)':>8} | {'Eval(s)':>8} | {'L2 Error':>10}")
    print("-" * 65)
    for M in M_list:
        res = run_bem(M)
        print(f"{res['M']:6d} | {res['iters']:5d} | {res['setup']:8.3f} | {res['solve']:8.3f} | {res['eval']:8.3f} | {res['l2']:10.4e}")