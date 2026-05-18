import numpy as np
import time
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def generate_naca4318(N):
    m = 0.04
    p = 0.3
    t = 0.18

    theta = np.linspace(0, 2 * np.pi, N)
    xc = 0.5 * (1.0 + np.cos(theta))

    yt = 5.0 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1036 * xc**4)

    yc = np.zeros_like(xc)
    dyc = np.zeros_like(xc)
    for i in range(N):
        x = xc[i]
        if x < p:
            yc[i] = (m / p**2) * (2.0 * p * x - x**2)
            dyc[i] = (2.0 * m / p**2) * (p - x)
        else:
            yc[i] = (m / (1.0 - p)**2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2)
            dyc[i] = (2.0 * m / (1.0 - p)**2) * (p - x)

    alpha = np.arctan(dyc)

    x_nodes = np.zeros(N)
    y_nodes = np.zeros(N)
    for i in range(N):
        if theta[i] <= np.pi:
            x_nodes[i] = xc[i] - yt[i] * np.sin(alpha[i])
            y_nodes[i] = yc[i] + yt[i] * np.cos(alpha[i])
        else:
            x_nodes[i] = xc[i] + yt[i] * np.sin(alpha[i])
            y_nodes[i] = yc[i] - yt[i] * np.cos(alpha[i])

    return np.column_stack((x_nodes, y_nodes))

def compute_geometry(nodes):
    N = nodes.shape[0]
    Ne = N - 1
    elements = np.zeros((Ne, 2), dtype=np.int32)
    lengths = np.zeros(Ne)
    normals = np.zeros((Ne, 2))

    for i in range(Ne):
        elements[i, 0] = i
        elements[i, 1] = i + 1

        A = nodes[i]
        B = nodes[i + 1]
        dx = B[0] - A[0]
        dy = B[1] - A[1]

        lengths[i] = np.sqrt(dx**2 + dy**2)
        normals[i] = [dy / lengths[i], -dx / lengths[i]]

    node_normals = np.zeros((N, 2))
    node_normals[0] = normals[0]
    node_normals[-1] = normals[-1]
    for i in range(1, N - 1):
        nx = normals[i-1, 0] + normals[i, 0]
        ny = normals[i-1, 1] + normals[i, 1]
        mag = np.sqrt(nx**2 + ny**2)
        node_normals[i] = [nx/mag, ny/mag]

    return elements, lengths, normals, node_normals

@njit(parallel=True)
def points_in_polygon(pts, nodes):
    n_pts = pts.shape[0]
    n_v = nodes.shape[0]
    inside = np.zeros(n_pts, dtype=np.bool_)
    for i in prange(n_pts):
        x = pts[i, 0]
        y = pts[i, 1]
        c = False
        j = n_v - 1
        for k in range(n_v):
            vx_k = nodes[k, 0]
            vy_k = nodes[k, 1]
            vx_j = nodes[j, 0]
            vy_j = nodes[j, 1]
            if ((vy_k > y) != (vy_j > y)):
                intersect_x = (vx_j - vx_k) * (y - vy_k) / (vy_j - vy_k) + vx_k
                if x < intersect_x:
                    c = not c
            j = k
        inside[i] = c
    return inside

@njit(parallel=True)
def min_dist_to_boundary(pts, nodes, elements):
    N_pts = pts.shape[0]
    Ne = elements.shape[0]
    dists = np.zeros(N_pts)
    for i in prange(N_pts):
        px = pts[i, 0]
        py = pts[i, 1]
        min_d2 = 1e20
        for e in range(Ne):
            A = nodes[elements[e, 0]]
            B = nodes[elements[e, 1]]

            dx = B[0] - A[0]
            dy = B[1] - A[1]
            l2 = dx**2 + dy**2

            if l2 == 0.0:
                d2 = (px - A[0])**2 + (py - A[1])**2
            else:
                t = ((px - A[0])*dx + (py - A[1])*dy) / l2
                t = max(0.0, min(1.0, t))
                proj_x = A[0] + t * dx
                proj_y = A[1] + t * dy
                d2 = (px - proj_x)**2 + (py - proj_y)**2

            if d2 < min_d2:
                min_d2 = d2
        dists[i] = np.sqrt(min_d2)
    return dists

@njit(parallel=True)
def assemble_matrices(nodes, elements, lengths, normals, gauss_w, gauss_xi):
    N = nodes.shape[0]
    Ne = elements.shape[0]
    Nq = gauss_w.shape[0]
    H = np.zeros((N, N))
    G = np.zeros((N, N))

    for i in prange(N):
        xi = nodes[i, 0]
        yi = nodes[i, 1]

        for e in range(Ne):
            A_idx = elements[e, 0]
            B_idx = elements[e, 1]
            Ax = nodes[A_idx, 0]
            Ay = nodes[A_idx, 1]
            Bx = nodes[B_idx, 0]
            By = nodes[B_idx, 1]
            L = lengths[e]
            nx = normals[e, 0]
            ny = normals[e, 1]

            dist_A = np.sqrt((xi - Ax)**2 + (yi - Ay)**2)
            dist_B = np.sqrt((xi - Bx)**2 + (yi - By)**2)

            if dist_A < 1e-12:
                G[i, A_idx] += - (L / (4.0 * np.pi)) * (np.log(L) - 1.5)
                G[i, B_idx] += - (L / (4.0 * np.pi)) * (np.log(L) - 0.5)
            elif dist_B < 1e-12:
                G[i, A_idx] += - (L / (4.0 * np.pi)) * (np.log(L) - 0.5)
                G[i, B_idx] += - (L / (4.0 * np.pi)) * (np.log(L) - 1.5)
            else:
                g_A = 0.0
                g_B = 0.0
                h_A = 0.0
                h_B = 0.0
                for q in range(Nq):
                    xi_q = 0.5 * (1.0 + gauss_xi[q])
                    wq = 0.5 * gauss_w[q]

                    xq = Ax + xi_q * (Bx - Ax)
                    yq = Ay + xi_q * (By - Ay)

                    rx = xi - xq
                    ry = yi - yq
                    r2 = rx**2 + ry**2

                    u_star = -1.0 / (4.0 * np.pi) * np.log(r2)
                    q_star = 1.0 / (2.0 * np.pi) * (rx * nx + ry * ny) / r2

                    phi1 = 1.0 - xi_q
                    phi2 = xi_q

                    g_A += wq * phi1 * u_star * L
                    g_B += wq * phi2 * u_star * L
                    h_A += wq * phi1 * q_star * L
                    h_B += wq * phi2 * q_star * L

                G[i, A_idx] += g_A
                G[i, B_idx] += g_B
                H[i, A_idx] += h_A
                H[i, B_idx] += h_B

    for i in range(N):
        sum_H = 0.0
        for j in range(N):
            if i != j:
                sum_H += H[i, j]
        H[i, i] = -sum_H

    return H, G

@njit(parallel=True)
def evaluate_interior(pts, nodes, elements, lengths, normals, u, q, gauss_w, gauss_xi):
    N_pts = pts.shape[0]
    Ne = elements.shape[0]
    Nq = gauss_w.shape[0]
    u_int = np.zeros(N_pts)

    for i in prange(N_pts):
        px = pts[i, 0]
        py = pts[i, 1]
        val = 0.0

        for e in range(Ne):
            A_idx = elements[e, 0]
            B_idx = elements[e, 1]
            Ax = nodes[A_idx, 0]
            Ay = nodes[A_idx, 1]
            Bx = nodes[B_idx, 0]
            By = nodes[B_idx, 1]
            L = lengths[e]
            nx = normals[e, 0]
            ny = normals[e, 1]

            u_A = u[A_idx]
            u_B = u[B_idx]
            q_A = q[A_idx]
            q_B = q[B_idx]

            g_A = 0.0
            g_B = 0.0
            h_A = 0.0
            h_B = 0.0
            for k in range(Nq):
                xi_q = 0.5 * (1.0 + gauss_xi[k])
                wq = 0.5 * gauss_w[k]

                xq = Ax + xi_q * (Bx - Ax)
                yq = Ay + xi_q * (By - Ay)

                rx = px - xq
                ry = py - yq
                r2 = rx**2 + ry**2

                u_star = -1.0 / (4.0 * np.pi) * np.log(r2)
                q_star = 1.0 / (2.0 * np.pi) * (rx * nx + ry * ny) / r2

                phi1 = 1.0 - xi_q
                phi2 = xi_q

                g_A += wq * phi1 * u_star * L
                g_B += wq * phi2 * u_star * L
                h_A += wq * phi1 * q_star * L
                h_B += wq * phi2 * q_star * L

            val += (g_A * q_A + g_B * q_B) - (h_A * u_A + h_B * u_B)
        u_int[i] = val
    return u_int

def main():
    N_values = [400, 800, 1600, 3200, 6400]
    gauss_xi, gauss_w = np.polynomial.legendre.leggauss(16)

    l2_errors = []

    print(f"{'N nodes':>8} | {'Unknowns':>10} | {'L2 Error':>12} | {'Setup (s)':>10} | {'Solve (s)':>10} | {'Eval (s)':>10} | {'Total (s)':>10}")
    print("-" * 85)

    warmup_nodes = generate_naca4318(10)
    w_elem, w_len, w_norm, _ = compute_geometry(warmup_nodes)
    _, _ = assemble_matrices(warmup_nodes, w_elem, w_len, w_norm, gauss_w, gauss_xi)
    w_pts = np.array([[0.5, 0.0]])
    _ = min_dist_to_boundary(w_pts, warmup_nodes, w_elem)
    _ = points_in_polygon(w_pts, warmup_nodes)
    _ = evaluate_interior(w_pts, warmup_nodes, w_elem, w_len, w_norm, np.ones(10), np.ones(10), gauss_w, gauss_xi)

    for N in N_values:
        t0 = time.time()

        nodes = generate_naca4318(N)
        elements, lengths, normals, node_normals = compute_geometry(nodes)

        node_bcs = np.where(nodes[:, 0] >= 0.8, 1, 0)

        u_exact_bnd = np.zeros(N)
        q_exact_bnd = np.zeros(N)
        for i in range(N):
            x = nodes[i, 0]
            y = nodes[i, 1]
            nx = node_normals[i, 0]
            ny = node_normals[i, 1]
            u_exact_bnd[i] = x**3 - 3.0 * x * y**2
            q_exact_bnd[i] = (3.0 * x**2 - 3.0 * y**2) * nx - (6.0 * x * y) * ny

        H, G = assemble_matrices(nodes, elements, lengths, normals, gauss_w, gauss_xi)

        A = np.zeros((N, N))
        b = np.zeros(N)

        for j in range(N):
            if node_bcs[j] == 1:
                A[:, j] = -G[:, j]
                b -= H[:, j] * u_exact_bnd[j]
            else:
                A[:, j] = H[:, j]
                b += G[:, j] * q_exact_bnd[j]

        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        b[-1] = q_exact_bnd[-1]

        t1 = time.time()

        x_sol = np.linalg.solve(A, b)

        u_num = np.zeros(N)
        q_num = np.zeros(N)
        for i in range(N):
            if node_bcs[i] == 1:
                u_num[i] = u_exact_bnd[i]
                q_num[i] = x_sol[i]
            else:
                u_num[i] = x_sol[i]
                q_num[i] = q_exact_bnd[i]

        t2 = time.time()

        ngrid = 200
        xs = np.linspace(-0.1, 1.1, ngrid)
        ys = np.linspace(-0.2, 0.2, ngrid)
        XX, YY = np.meshgrid(xs, ys)
        all_pts = np.column_stack([XX.ravel(), YY.ravel()])

        interior_mask = points_in_polygon(all_pts, nodes)
        interior = all_pts[interior_mask]

        perimeter = np.sum(lengths)
        h_coarse = perimeter / min(N_values)
        delta = 2.0 * h_coarse

        dist = min_dist_to_boundary(interior, nodes, elements)
        grid_pts = interior[dist > delta]

        u_eval_num = evaluate_interior(grid_pts, nodes, elements, lengths, normals, u_num, q_num, gauss_w, gauss_xi)

        gx = grid_pts[:, 0]
        gy = grid_pts[:, 1]
        u_eval_exact = gx**3 - 3.0 * gx * gy**2

        err_l2 = np.linalg.norm(u_eval_num - u_eval_exact) / np.linalg.norm(u_eval_exact)
        l2_errors.append(err_l2)

        t3 = time.time()

        t_setup = t1 - t0
        t_solve = t2 - t1
        t_eval = t3 - t2
        t_total = t3 - t0

        print(f"{N:8d} | {N:10d} | {err_l2:12.4e} | {t_setup:10.4f} | {t_solve:10.4f} | {t_eval:10.4f} | {t_total:10.4f}")

        if N == N_values[-1]:
            error_field = np.abs(u_eval_num - u_eval_exact)
            triang = mtri.Triangulation(grid_pts[:, 0], grid_pts[:, 1])
            plt.figure(figsize=(10, 4))
            contour = plt.tricontourf(triang, error_field, levels=50, cmap='viridis')
            cbar = plt.colorbar(contour)
            cbar.set_label('Absolute Error |u_num - u_exact|')
            boundary_x = np.append(nodes[:, 0], nodes[0, 0])
            boundary_y = np.append(nodes[:, 1], nodes[0, 1])
            plt.plot(boundary_x, boundary_y, 'k-', linewidth=1.5, label='NACA 4318')
            plt.title(f'BEM Absolute Error Field (Linear, N = {N})')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis('equal')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()

    log_h = np.log(1.0 / np.array(N_values))
    log_err = np.log(np.array(l2_errors))
    slope, _ = np.polyfit(log_h, log_err, 1)

    print("-" * 85)
    print(f"Estimated convergence order (slope) = {slope:.4f}")

if __name__ == '__main__':
    main()