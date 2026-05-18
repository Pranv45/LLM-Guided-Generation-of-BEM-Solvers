import numpy as np
import time
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def generate_naca4318(N):
    m = 0.04
    p = 0.3
    t = 0.18

    beta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xc = 0.5 * (1.0 + np.cos(beta))
    xc = np.clip(xc, 0.0, 1.0)

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

    theta = np.arctan(dyc)

    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        if beta[i] <= np.pi:
            x[i] = xc[i] - yt[i] * np.sin(theta[i])
            y[i] = yc[i] + yt[i] * np.cos(theta[i])
        else:
            x[i] = xc[i] + yt[i] * np.sin(theta[i])
            y[i] = yc[i] - yt[i] * np.cos(theta[i])

    return np.column_stack((x, y))

def compute_geometry(nodes):
    Ne = nodes.shape[0]
    midpoints = np.zeros((Ne, 2))
    lengths = np.zeros(Ne)
    normals = np.zeros((Ne, 2))

    for i in range(Ne):
        p1 = nodes[i]
        p2 = nodes[(i + 1) % Ne]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        midpoints[i] = [p1[0] + dx / 2.0, p1[1] + dy / 2.0]
        lengths[i] = np.sqrt(dx**2 + dy**2)
        normals[i] = [dy / lengths[i], -dx / lengths[i]]

    return midpoints, lengths, normals

@njit(parallel=True)
def points_in_polygon(pts, poly):
    n_pts = pts.shape[0]
    n_v = poly.shape[0]
    inside = np.zeros(n_pts, dtype=np.bool_)
    for i in prange(n_pts):
        x = pts[i, 0]
        y = pts[i, 1]
        c = False
        j = n_v - 1
        for k in range(n_v):
            vx_k = poly[k, 0]
            vy_k = poly[k, 1]
            vx_j = poly[j, 0]
            vy_j = poly[j, 1]

            if ((vy_k > y) != (vy_j > y)):
                intersect_x = (vx_j - vx_k) * (y - vy_k) / (vy_j - vy_k) + vx_k
                if x < intersect_x:
                    c = not c
            j = k
        inside[i] = c
    return inside

@njit(parallel=True)
def min_dist_to_boundary(pts, nodes):
    N_pts = pts.shape[0]
    N_seg = nodes.shape[0]
    dists = np.zeros(N_pts)
    for i in prange(N_pts):
        px = pts[i, 0]
        py = pts[i, 1]
        min_d2 = 1e20
        for j in range(N_seg):
            nx1 = nodes[j, 0]
            ny1 = nodes[j, 1]
            nx2 = nodes[(j+1)%N_seg, 0]
            ny2 = nodes[(j+1)%N_seg, 1]

            dx = nx2 - nx1
            dy = ny2 - ny1
            l2 = dx**2 + dy**2

            if l2 == 0.0:
                d2 = (px - nx1)**2 + (py - ny1)**2
            else:
                t = ((px - nx1)*dx + (py - ny1)*dy) / l2
                t = max(0.0, min(1.0, t))
                proj_x = nx1 + t * dx
                proj_y = ny1 + t * dy
                d2 = (px - proj_x)**2 + (py - proj_y)**2

            if d2 < min_d2:
                min_d2 = d2
        dists[i] = np.sqrt(min_d2)
    return dists

@njit(parallel=True)
def assemble_matrices(midpoints, nodes, lengths, normals, gauss_w, gauss_xi):
    Ne = midpoints.shape[0]
    Nq = gauss_w.shape[0]
    H = np.zeros((Ne, Ne))
    G = np.zeros((Ne, Ne))

    for i in prange(Ne):
        mx = midpoints[i, 0]
        my = midpoints[i, 1]
        for j in range(Ne):
            if i == j:
                H[i, j] = 0.5
                G[i, j] = (lengths[i] / (2.0 * np.pi)) * (1.0 - np.log(lengths[i] / 2.0))
            else:
                nx1 = nodes[j, 0]
                ny1 = nodes[j, 1]
                nx2 = nodes[(j+1)%Ne, 0]
                ny2 = nodes[(j+1)%Ne, 1]

                dx = nx2 - nx1
                dy = ny2 - ny1
                mj_x = nx1 + dx / 2.0
                mj_y = ny1 + dy / 2.0
                nj_x = normals[j, 0]
                nj_y = normals[j, 1]
                J = lengths[j] / 2.0

                hij = 0.0
                gij = 0.0
                for q in range(Nq):
                    xi = gauss_xi[q]
                    wq = gauss_w[q]
                    xq = mj_x + xi * dx / 2.0
                    yq = mj_y + xi * dy / 2.0

                    rx = mx - xq
                    ry = my - yq
                    r2 = rx**2 + ry**2

                    u_star = -1.0 / (4.0 * np.pi) * np.log(r2)
                    q_star = 1.0 / (2.0 * np.pi) * (rx * nj_x + ry * nj_y) / r2

                    hij += wq * q_star * J
                    gij += wq * u_star * J

                H[i, j] = hij
                G[i, j] = gij
    return H, G

@njit(parallel=True)
def evaluate_interior(pts, nodes, lengths, normals, u, q, gauss_w, gauss_xi):
    N_pts = pts.shape[0]
    Ne = nodes.shape[0]
    Nq = gauss_w.shape[0]
    u_int = np.zeros(N_pts)

    for i in prange(N_pts):
        px = pts[i, 0]
        py = pts[i, 1]
        val = 0.0
        for j in range(Ne):
            nx1 = nodes[j, 0]
            ny1 = nodes[j, 1]
            nx2 = nodes[(j+1)%Ne, 0]
            ny2 = nodes[(j+1)%Ne, 1]

            dx = nx2 - nx1
            dy = ny2 - ny1
            mj_x = nx1 + dx / 2.0
            mj_y = ny1 + dy / 2.0
            nj_x = normals[j, 0]
            nj_y = normals[j, 1]
            J = lengths[j] / 2.0

            hij = 0.0
            gij = 0.0
            for k in range(Nq):
                xi = gauss_xi[k]
                wq = gauss_w[k]
                xq = mj_x + xi * dx / 2.0
                yq = mj_y + xi * dy / 2.0

                rx = px - xq
                ry = py - yq
                r2 = rx**2 + ry**2

                u_star = -1.0 / (4.0 * np.pi) * np.log(r2)
                q_star = 1.0 / (2.0 * np.pi) * (rx * nj_x + ry * nj_y) / r2

                hij += wq * q_star * J
                gij += wq * u_star * J

            val += gij * q[j] - hij * u[j]
        u_int[i] = val
    return u_int

def main():
    N_values = [400, 800, 1600, 3200, 6400]
    gauss_xi, gauss_w = np.polynomial.legendre.leggauss(8)

    l2_errors = []
    setup_times = []
    solve_times = []
    eval_times = []
    total_times = []

    print(f"{'N':>5} | {'Unknowns':>10} | {'L2 Error':>12} | {'Setup (s)':>10} | {'Solve (s)':>10} | {'Eval (s)':>10} | {'Total (s)':>10}")
    print("-" * 85)

    warmup_nodes = generate_naca4318(10)
    warmup_mid, warmup_len, warmup_norm = compute_geometry(warmup_nodes)
    _, _ = assemble_matrices(warmup_mid, warmup_nodes, warmup_len, warmup_norm, gauss_w, gauss_xi)
    warmup_pts = np.array([[0.5, 0.0]])
    _ = min_dist_to_boundary(warmup_pts, warmup_nodes)
    _ = points_in_polygon(warmup_pts, warmup_nodes)
    _ = evaluate_interior(warmup_pts, warmup_nodes, warmup_len, warmup_norm, np.ones(10), np.ones(10), gauss_w, gauss_xi)

    for N in N_values:
        t0 = time.time()

        nodes = generate_naca4318(N)
        midpoints, lengths, normals = compute_geometry(nodes)

        bcs_type = np.zeros(N, dtype=np.int32)
        bcs_type[midpoints[:, 0] >= 0.8] = 1

        u_exact_bnd = np.zeros(N)
        q_exact_bnd = np.zeros(N)
        for i in range(N):
            mx = midpoints[i, 0]
            my = midpoints[i, 1]
            nx = normals[i, 0]
            ny = normals[i, 1]
            u_exact_bnd[i] = mx**3 - 3.0 * mx * my**2
            q_exact_bnd[i] = (3.0 * mx**2 - 3.0 * my**2) * nx - (6.0 * mx * my) * ny

        H, G = assemble_matrices(midpoints, nodes, lengths, normals, gauss_w, gauss_xi)

        A = np.zeros((N, N))
        b = np.zeros(N)

        for j in range(N):
            if bcs_type[j] == 1:
                A[:, j] = -G[:, j]
                b -= H[:, j] * u_exact_bnd[j]
            else:
                A[:, j] = H[:, j]
                b += G[:, j] * q_exact_bnd[j]

        t1 = time.time()

        x_sol = np.linalg.solve(A, b)

        t2 = time.time()

        u_num = np.zeros(N)
        q_num = np.zeros(N)
        for i in range(N):
            if bcs_type[i] == 1:
                u_num[i] = u_exact_bnd[i]
                q_num[i] = x_sol[i]
            else:
                u_num[i] = x_sol[i]
                q_num[i] = q_exact_bnd[i]

        ngrid = 200
        xs = np.linspace(-0.1, 1.1, ngrid)
        ys = np.linspace(-0.2, 0.2, ngrid)
        XX, YY = np.meshgrid(xs, ys)
        all_pts = np.column_stack([XX.ravel(), YY.ravel()])

        interior_mask = points_in_polygon(all_pts, nodes)
        interior = all_pts[interior_mask]

        perimeter = np.sum(lengths)
        N_min = min(N_values)
        h_coarse = perimeter / N_min
        delta = 2.0 * h_coarse

        dist = min_dist_to_boundary(interior, nodes)
        grid_pts = interior[dist > delta]

        u_eval_num = evaluate_interior(grid_pts, nodes, lengths, normals, u_num, q_num, gauss_w, gauss_xi)

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

        setup_times.append(t_setup)
        solve_times.append(t_solve)
        eval_times.append(t_eval)
        total_times.append(t_total)

        print(f"{N:5d} | {N:10d} | {err_l2:12.4e} | {t_setup:10.4f} | {t_solve:10.4f} | {t_eval:10.4f} | {t_total:10.4f}")

    # --- PLACE THIS AT THE END OF THE `for N in N_values:` LOOP ---

        # Only plot for the highest resolution mesh to avoid blocking the loop
    if N == N_values[-1]:
            # Calculate absolute error
            error_field = np.abs(u_eval_num - u_eval_exact)

            # Create a triangulation of the unstructured interior points
            triang = mtri.Triangulation(grid_pts[:, 0], grid_pts[:, 1])

            plt.figure(figsize=(10, 4))

            # Plot the error field
            contour = plt.tricontourf(triang, error_field, levels=50, cmap='viridis')
            cbar = plt.colorbar(contour)
            cbar.set_label('Absolute Error |u_num - u_exact|')

            # Overlay the NACA airfoil boundary (close the loop for plotting)
            boundary_x = np.append(nodes[:, 0], nodes[0, 0])
            boundary_y = np.append(nodes[:, 1], nodes[0, 1])
            plt.plot(boundary_x, boundary_y, 'k-', linewidth=1.5, label='NACA 4318')

            # Formatting
            plt.title(f'BEM Absolute Error Field (N = {N} elements)')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis('equal')
            plt.legend(loc='upper right')
            plt.tight_layout()

            # Show the plot
            plt.show()

    log_h = np.log(1.0 / np.array(N_values))
    log_err = np.log(np.array(l2_errors))
    slope, _ = np.polyfit(log_h, log_err, 1)

    print("-" * 85)
    print(f"Estimated convergence order = {slope:.4f}")

if __name__ == '__main__':
    main()


## code is designed such that eval grid depends on n_min A larger N_min means the coarsest mesh has smaller elements.
# This results in a smaller safety buffer ($\delta$) near the boundary, meaning fewer points are excluded.
# Therefore, you evaluate the error over more interior points.