import numpy as np
import time
from numba import njit, prange

def generate_naca4318(N_elems):
    m = 0.04
    p = 0.3
    t = 0.18
    N_half = N_elems // 2

    beta = np.linspace(0, np.pi, N_half + 1)
    x = 0.5 * (1 - np.cos(beta))

    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    yt[-1] = 0.0

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] < p:
            yc[i] = (m / p**2) * (2 * p * x[i] - x[i]**2)
            dyc_dx[i] = (2 * m / p**2) * (p - x[i])
        else:
            yc[i] = (m / (1 - p)**2) * (1 - 2 * p + 2 * p * x[i] - x[i]**2)
            dyc_dx[i] = (2 * m / (1 - p)**2) * (p - x[i])

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    X = np.concatenate((xu[::-1][:-1], xl))
    Y = np.concatenate((yu[::-1][:-1], yl))

    nodes = np.column_stack((X, Y))

    elements = np.zeros((N_elems, 2), dtype=np.int32)
    for i in range(N_elems):
        elements[i, 0] = i
        elements[i, 1] = i + 1

    return nodes, elements

@njit
def evaluate_kernels(xi, eta, ex, ey, nodes, elements, n_ey):
    n0x = elements[ex, 0]; n1x = elements[ex, 1]
    n0y = elements[ey, 0]; n1y = elements[ey, 1]
    x_val_0 = nodes[n0x, 0] * (1 - xi) + nodes[n1x, 0] * xi
    x_val_1 = nodes[n0x, 1] * (1 - xi) + nodes[n1x, 1] * xi
    y_val_0 = nodes[n0y, 0] * (1 - eta) + nodes[n1y, 0] * eta
    y_val_1 = nodes[n0y, 1] * (1 - eta) + nodes[n1y, 1] * eta

    dx = x_val_0 - y_val_0
    dy = x_val_1 - y_val_1
    r2 = dx*dx + dy*dy

    u_star = -0.5 / np.pi * np.log(np.sqrt(r2))
    q_star = 0.5 / np.pi * (dx * n_ey[0] + dy * n_ey[1]) / r2
    return u_star, q_star

@njit
def compute_pair(ex, ey, local_i, nodes, elements, L, n_arr, u2, v2, w2):
    H0 = 0.0; H1 = 0.0; G0 = 0.0; G1 = 0.0
    Lx = L[ex]; Ly = L[ey]
    n_ey = n_arr[ey]

    if ex == ey:
        I_mat = np.array([[-7.0/16.0, -5.0/16.0], [-5.0/16.0, -7.0/16.0]])
        val0 = -Lx * Ly / (2 * np.pi) * (0.25 * np.log(Lx) + I_mat[local_i, 0])
        val1 = -Lx * Ly / (2 * np.pi) * (0.25 * np.log(Lx) + I_mat[local_i, 1])
        return 0.0, 0.0, val0, val1

    n0x = elements[ex, 0]; n1x = elements[ex, 1]
    n0y = elements[ey, 0]; n1y = elements[ey, 1]

    vx = -1; vy = -1
    if n0x == n0y: vx = 0; vy = 0
    elif n0x == n1y: vx = 0; vy = 1
    elif n1x == n0y: vx = 1; vy = 0
    elif n1x == n1y: vx = 1; vy = 1

    if vx != -1:
        for k in range(64):
            u = u2[k]; v = v2[k]; w = w2[k]; J = u
            s = u; t = u * v
            xi = vx + (1.0 if vx == 0 else -1.0) * s
            eta = vy + (1.0 if vy == 0 else -1.0) * t
            u_star, q_star = evaluate_kernels(xi, eta, ex, ey, nodes, elements, n_ey)

            phi_i = 1.0 - xi if local_i == 0 else xi
            phi_0 = 1.0 - eta
            phi_1 = eta

            factor = w * J * Lx * Ly
            H0 += phi_i * q_star * phi_0 * factor
            H1 += phi_i * q_star * phi_1 * factor
            G0 += phi_i * u_star * phi_0 * factor
            G1 += phi_i * u_star * phi_1 * factor

            s = u * v; t = u
            xi = vx + (1.0 if vx == 0 else -1.0) * s
            eta = vy + (1.0 if vy == 0 else -1.0) * t
            u_star, q_star = evaluate_kernels(xi, eta, ex, ey, nodes, elements, n_ey)

            phi_i = 1.0 - xi if local_i == 0 else xi
            phi_0 = 1.0 - eta
            phi_1 = eta

            H0 += phi_i * q_star * phi_0 * factor
            H1 += phi_i * q_star * phi_1 * factor
            G0 += phi_i * u_star * phi_0 * factor
            G1 += phi_i * u_star * phi_1 * factor
    else:
        for k in range(64):
            xi = u2[k]
            eta = v2[k]
            w = w2[k]
            u_star, q_star = evaluate_kernels(xi, eta, ex, ey, nodes, elements, n_ey)

            phi_i = 1.0 - xi if local_i == 0 else xi
            phi_0 = 1.0 - eta
            phi_1 = eta

            factor = w * Lx * Ly
            H0 += phi_i * q_star * phi_0 * factor
            H1 += phi_i * q_star * phi_1 * factor
            G0 += phi_i * u_star * phi_0 * factor
            G1 += phi_i * u_star * phi_1 * factor

    return H0, H1, G0, G1

@njit(parallel=True)
def assemble_system(N_nodes, N_elems, nodes, elements, L, n_arr, u2, v2, w2):
    H = np.zeros((N_nodes, N_nodes))
    G = np.zeros((N_nodes, N_nodes))

    for i in prange(N_nodes):
        for ex in range(N_elems):
            local_i = -1
            if elements[ex, 0] == i:
                local_i = 0
            elif elements[ex, 1] == i:
                local_i = 1

            if local_i != -1:
                for ey in range(N_elems):
                    H0, H1, G0, G1 = compute_pair(ex, ey, local_i, nodes, elements, L, n_arr, u2, v2, w2)

                    n0y = elements[ey, 0]
                    n1y = elements[ey, 1]

                    H[i, n0y] += H0
                    H[i, n1y] += H1
                    G[i, n0y] += G0
                    G[i, n1y] += G1

    return H, G

@njit(parallel=True)
def points_in_polygon(points, vertices):
    n_pts = points.shape[0]
    n_v = vertices.shape[0]
    inside = np.zeros(n_pts, dtype=np.bool_)
    for p in prange(n_pts):
        x, y = points[p, 0], points[p, 1]
        wn = 0
        for i in range(n_v - 1):
            x1, y1 = vertices[i, 0], vertices[i, 1]
            x2, y2 = vertices[i+1, 0], vertices[i+1, 1]
            if y1 <= y:
                if y2 > y:
                    is_left = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
                    if is_left > 0:
                        wn += 1
            else:
                if y2 <= y:
                    is_left = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
                    if is_left < 0:
                        wn -= 1
        inside[p] = (wn != 0)
    return inside

@njit(parallel=True)
def min_dist_to_boundary(points, vertices):
    n_pts = points.shape[0]
    n_edges = vertices.shape[0] - 1
    dists = np.zeros(n_pts)
    for p in prange(n_pts):
        px, py = points[p, 0], points[p, 1]
        min_d = 1e9
        for i in range(n_edges):
            x1, y1 = vertices[i, 0], vertices[i, 1]
            x2, y2 = vertices[i+1, 0], vertices[i+1, 1]
            l2 = (x2 - x1)**2 + (y2 - y1)**2
            if l2 == 0.0:
                d = np.hypot(px - x1, py - y1)
            else:
                t = max(0.0, min(1.0, ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / l2))
                proj_x = x1 + t * (x2 - x1)
                proj_y = y1 + t * (y2 - y1)
                d = np.hypot(px - proj_x, py - proj_y)
            if d < min_d:
                min_d = d
        dists[p] = min_d
    return dists

@njit(parallel=True)
def evaluate_interior(eval_pts, elements, nodes, u_bnd, q_bnd, u1, w1):
    n_pts = eval_pts.shape[0]
    n_elems = elements.shape[0]
    u_eval = np.zeros(n_pts)
    for p in prange(n_pts):
        px, py = eval_pts[p, 0], eval_pts[p, 1]
        val = 0.0
        for e in range(n_elems):
            n0, n1 = elements[e]
            x0, y0 = nodes[n0, 0], nodes[n0, 1]
            x1, y1 = nodes[n1, 0], nodes[n1, 1]
            L = np.hypot(x1 - x0, y1 - y0)
            if L == 0.0: continue
            nx = (y1 - y0) / L
            ny = -(x1 - x0) / L

            u0, u1_val = u_bnd[n0], u_bnd[n1]
            q0, q1_val = q_bnd[n0], q_bnd[n1]

            integral_u = 0.0
            integral_q = 0.0
            for g in range(len(u1)):
                eta = u1[g]
                w = w1[g]
                phi0 = 1.0 - eta
                phi1 = eta

                yx = x0 * phi0 + x1 * phi1
                yy = y0 * phi0 + y1 * phi1

                dx = px - yx
                dy = py - yy
                r2 = dx*dx + dy*dy

                u_star = -0.5 / np.pi * np.log(np.sqrt(r2))
                q_star = 0.5 / np.pi * (dx * nx + dy * ny) / r2

                q_y = q0 * phi0 + q1_val * phi1
                u_y = u0 * phi0 + u1_val * phi1

                integral_u += u_star * q_y * w * L
                integral_q += q_star * u_y * w * L

            val += integral_u - integral_q
        u_eval[p] = val
    return u_eval

def run_bem(N_elems, u1, w1, u2, v2, w2, warmup=False):
    t_start = time.time()

    nodes, elements = generate_naca4318(N_elems)
    N_nodes = nodes.shape[0]

    L = np.zeros(N_elems)
    n_arr = np.zeros((N_elems, 2))
    for i in range(N_elems):
        n0, n1 = elements[i]
        dx = nodes[n1, 0] - nodes[n0, 0]
        dy = nodes[n1, 1] - nodes[n0, 1]
        L[i] = np.hypot(dx, dy)
        n_arr[i, 0] = dy / L[i]
        n_arr[i, 1] = -dx / L[i]

    node_normals = np.zeros((N_nodes, 2))
    node_normals[0] = n_arr[0]
    node_normals[-1] = n_arr[-1]
    for i in range(1, N_nodes - 1):
        n_avg = n_arr[i-1] + n_arr[i]
        node_normals[i] = n_avg / np.linalg.norm(n_avg)

    node_bcs = np.zeros(N_nodes, dtype=np.bool_)
    for i in range(N_nodes):
        node_bcs[i] = nodes[i, 0] >= 0.8

    H, G = assemble_system(N_nodes, N_elems, nodes, elements, L, n_arr, u2, v2, w2)

    for i in range(N_nodes):
        H[i, i] = 0.0
        H[i, i] = -np.sum(H[i, :])

    A = np.zeros((N_nodes, N_nodes))
    b = np.zeros(N_nodes)

    u_exact = nodes[:, 0]**3 - 3 * nodes[:, 0] * nodes[:, 1]**2
    q_exact = (3 * nodes[:, 0]**2 - 3 * nodes[:, 1]**2) * node_normals[:, 0] - (6 * nodes[:, 0] * nodes[:, 1]) * node_normals[:, 1]

    for j in range(N_nodes):
        if node_bcs[j]:
            A[:, j] = -G[:, j]
            b -= H[:, j] * u_exact[j]
        else:
            A[:, j] = H[:, j]
            b += G[:, j] * q_exact[j]

    t_setup = time.time() - t_start

    t_solve_start = time.time()
    x_sol = np.linalg.solve(A, b)
    t_solve = time.time() - t_solve_start

    u_bnd = np.zeros(N_nodes)
    q_bnd = np.zeros(N_nodes)
    for j in range(N_nodes):
        if node_bcs[j]:
            u_bnd[j] = u_exact[j]
            q_bnd[j] = x_sol[j]
        else:
            u_bnd[j] = x_sol[j]
            q_bnd[j] = q_exact[j]

    t_eval_start = time.time()

    if warmup:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    perimeter = np.sum(L)
    h_coarse = perimeter / 100.0
    delta = 2 * h_coarse

    xs = np.linspace(-0.1, 1.1, 200)
    ys = np.linspace(-0.2, 0.2, 200)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack((XX.ravel(), YY.ravel()))

    interior = points_in_polygon(all_pts, nodes)
    pts_in = all_pts[interior]
    dist = min_dist_to_boundary(pts_in, nodes)
    grid_pts = pts_in[dist > delta]

    u_num = evaluate_interior(grid_pts, elements, nodes, u_bnd, q_bnd, u1, w1)
    u_ex_grid = grid_pts[:, 0]**3 - 3 * grid_pts[:, 0] * grid_pts[:, 1]**2

    err = np.linalg.norm(u_num - u_ex_grid) / np.linalg.norm(u_ex_grid)
    t_eval = time.time() - t_eval_start
    t_total = t_setup + t_solve + t_eval

    return N_nodes, err, t_setup, t_solve, t_eval, t_total

def main():
    u1_raw, w1_raw = np.polynomial.legendre.leggauss(8)
    u1 = 0.5 * (u1_raw + 1.0)
    w1 = 0.5 * w1_raw

    u2 = np.zeros(64)
    v2 = np.zeros(64)
    w2 = np.zeros(64)
    idx = 0
    for i in range(8):
        for j in range(8):
            u2[idx] = u1[i]
            v2[idx] = u1[j]
            w2[idx] = w1[i] * w1[j]
            idx += 1

    _ = run_bem(10, u1, w1, u2, v2, w2, warmup=True)

    N_values = [400, 800, 1600, 3200, 6400]
    errors = []

    print(f"{'N':<6} {'Nodes':<6} {'L2 Error':<13} {'Setup(s)':<10} {'Solve(s)':<10} {'Eval(s)':<10} {'Total(s)':<10}")
    print("-" * 71)

    for N in N_values:
        N_nodes, err, t_setup, t_solve, t_eval, t_total = run_bem(N, u1, w1, u2, v2, w2)
        errors.append(err)
        print(f"{N:<6} {N_nodes:<6} {err:<13.5e} {t_setup:<10.4f} {t_solve:<10.4f} {t_eval:<10.4f} {t_total:<10.4f}")

    slope, _ = np.polyfit(np.log(1.0 / np.array(N_values)), np.log(errors), 1)
    print(f"\nEstimated convergence order = {slope:.4f}")

if __name__ == "__main__":
    main()