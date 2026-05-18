import numpy as np
import time
from numba import njit, prange

def generate_continuous_bumpy_mesh(N):
    u_pts = np.linspace(-1, 1, N+1)
    v_pts = np.linspace(-1, 1, N+1)

    node_dict = {}
    nodes_list = []
    elems_list = []

    def add_node(X, Y, Z):
        R = np.sqrt(X**2 + Y**2 + Z**2)
        X_sph, Y_sph, Z_sph = X/R, Y/R, Z/R
        Z_sph = max(min(Z_sph, 1.0), -1.0)

        theta = np.arccos(Z_sph)
        phi = np.arctan2(Y_sph, X_sph)
        r = 1.5 + 0.3 * np.sin(4*theta) * np.cos(5*phi)

        X_f = r * np.sin(theta) * np.cos(phi)
        Y_f = r * np.sin(theta) * np.sin(phi)
        Z_f = r * np.cos(theta)

        kx, ky, kz = round(X_f, 6), round(Y_f, 6), round(Z_f, 6)
        if (kx, ky, kz) not in node_dict:
            node_dict[(kx, ky, kz)] = len(nodes_list)
            nodes_list.append([X_f, Y_f, Z_f])
        return node_dict[(kx, ky, kz)]

    for face_id in range(6):
        for i in range(N):
            for j in range(N):
                u1, u2 = u_pts[i], u_pts[i+1]
                v1, v2 = v_pts[j], v_pts[j+1]

                if face_id == 0:
                    c0 = add_node(1, u1, v1)
                    c1 = add_node(1, u2, v1)
                    c2 = add_node(1, u2, v2)
                    c3 = add_node(1, u1, v2)
                elif face_id == 1:
                    c0 = add_node(-1, v1, u1)
                    c1 = add_node(-1, v1, u2)
                    c2 = add_node(-1, v2, u2)
                    c3 = add_node(-1, v2, u1)
                elif face_id == 2:
                    c0 = add_node(u1, 1, v1)
                    c1 = add_node(u2, 1, v1)
                    c2 = add_node(u2, 1, v2)
                    c3 = add_node(u1, 1, v2)
                elif face_id == 3:
                    c0 = add_node(v1, -1, u1)
                    c1 = add_node(v1, -1, u2)
                    c2 = add_node(v2, -1, u2)
                    c3 = add_node(v2, -1, u1)
                elif face_id == 4:
                    c0 = add_node(u1, v1, 1)
                    c1 = add_node(u2, v1, 1)
                    c2 = add_node(u2, v2, 1)
                    c3 = add_node(u1, v2, 1)
                elif face_id == 5:
                    c0 = add_node(v1, u1, -1)
                    c1 = add_node(v1, u2, -1)
                    c2 = add_node(v2, u2, -1)
                    c3 = add_node(v2, u1, -1)

                elems_list.append([c0, c1, c2])
                elems_list.append([c0, c2, c3])

    nodes = np.array(nodes_list, dtype=np.float64)
    elems = np.array(elems_list, dtype=np.int32)
    return nodes, elems

def compute_geometry(nodes, elems):
    v0 = nodes[elems[:, 0]]
    v1 = nodes[elems[:, 1]]
    v2 = nodes[elems[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0
    vec1 = v1 - v0
    vec2 = v2 - v0
    cross = np.cross(vec1, vec2)

    areas = 0.5 * np.linalg.norm(cross, axis=1)
    elem_normals = cross / (2.0 * areas[:, None])

    dots = np.sum(elem_normals * centroids, axis=1)
    flips = dots < 0
    elem_normals[flips] *= -1.0

    temp = elems[flips, 1].copy()
    elems[flips, 1] = elems[flips, 2]
    elems[flips, 2] = temp

    nodal_normals = np.zeros_like(nodes)
    for i in range(len(elems)):
        n0, n1, n2 = elems[i]
        area = areas[i]
        norm = elem_normals[i]
        nodal_normals[n0] += area * norm
        nodal_normals[n1] += area * norm
        nodal_normals[n2] += area * norm

    norms = np.linalg.norm(nodal_normals, axis=1)
    nodal_normals /= norms[:, None]

    return areas, centroids, elem_normals, nodal_normals, elems

def assign_mixed_bcs(nodes, nodal_normals):
    N_nodes = nodes.shape[0]
    bc_type = np.zeros(N_nodes, dtype=np.int32)
    bc_val = np.zeros(N_nodes, dtype=np.float64)

    X, Y, Z = nodes[:, 0], nodes[:, 1], nodes[:, 2]

    u_exact = np.sinh(X)*np.sin(Y) + np.cosh(Y)*np.cos(Z)
    grad_x = np.cosh(X)*np.sin(Y)
    grad_y = np.sinh(X)*np.cos(Y) + np.sinh(Y)*np.cos(Z)
    grad_z = -np.cosh(Y)*np.sin(Z)

    q_exact = grad_x*nodal_normals[:, 0] + grad_y*nodal_normals[:, 1] + grad_z*nodal_normals[:, 2]

    dirichlet_mask = X > 0
    bc_type[dirichlet_mask] = 0
    bc_val[dirichlet_mask] = u_exact[dirichlet_mask]

    neumann_mask = X <= 0
    bc_type[neumann_mask] = 1
    bc_val[neumann_mask] = q_exact[neumann_mask]

    return bc_type, bc_val

@njit(parallel=True, fastmath=True)
def assemble_system(nodes, elems, areas, centroids, elem_normals, gp7_u, gw7_u, gp12_u, gw12_u, dun_bary, dun_wts, bc_type, bc_val, A, b):
    N_nodes = nodes.shape[0]
    N_elems = elems.shape[0]

    for i in prange(N_nodes):
        Xi_0 = nodes[i, 0]
        Xi_1 = nodes[i, 1]
        Xi_2 = nodes[i, 2]

        H_row = np.zeros(N_nodes, dtype=np.float64)
        G_row = np.zeros(N_nodes, dtype=np.float64)

        for e in range(N_elems):
            n0 = elems[e, 0]
            n1 = elems[e, 1]
            n2 = elems[e, 2]

            V0_0, V0_1, V0_2 = nodes[n0, 0], nodes[n0, 1], nodes[n0, 2]
            V1_0, V1_1, V1_2 = nodes[n1, 0], nodes[n1, 1], nodes[n1, 2]
            V2_0, V2_1, V2_2 = nodes[n2, 0], nodes[n2, 1], nodes[n2, 2]

            area = areas[e]
            nx, ny, nz = elem_normals[e, 0], elem_normals[e, 1], elem_normals[e, 2]

            if i == n0 or i == n1 or i == n2:
                if i == n0: local_s = 0
                elif i == n1: local_s = 1
                else: local_s = 2

                for u_idx in range(7):
                    for v_idx in range(7):
                        u = gp7_u[u_idx]
                        v = gp7_u[v_idx]
                        w_uv = gw7_u[u_idx] * gw7_u[v_idx]

                        if local_s == 0:
                            L0 = 1.0 - u
                            L1 = u * (1.0 - v)
                            L2 = u * v
                        elif local_s == 1:
                            L1 = 1.0 - u
                            L2 = u * (1.0 - v)
                            L0 = u * v
                        else:
                            L2 = 1.0 - u
                            L0 = u * (1.0 - v)
                            L1 = u * v

                        Y0 = L0*V0_0 + L1*V1_0 + L2*V2_0
                        Y1 = L0*V0_1 + L1*V1_1 + L2*V2_1
                        Y2 = L0*V0_2 + L1*V1_2 + L2*V2_2

                        rx = Xi_0 - Y0
                        ry = Xi_1 - Y1
                        rz = Xi_2 - Y2
                        r = np.sqrt(rx*rx + ry*ry + rz*rz)

                        if r > 1e-14:
                            factor = 2.0 * area * u * w_uv / (4.0 * np.pi * r)
                            G_row[n0] += L0 * factor
                            G_row[n1] += L1 * factor
                            G_row[n2] += L2 * factor
            else:
                cx, cy, cz = centroids[e, 0], centroids[e, 1], centroids[e, 2]
                dist2 = (cx - Xi_0)**2 + (cy - Xi_1)**2 + (cz - Xi_2)**2

                if dist2 < 20.0 * area:
                    for u_idx in range(12):
                        for v_idx in range(12):
                            u = gp12_u[u_idx]
                            v = gp12_u[v_idx]
                            w_uv = gw12_u[u_idx] * gw12_u[v_idx]

                            L0 = 1.0 - u
                            L1 = u * (1.0 - v)
                            L2 = u * v

                            Y0 = L0*V0_0 + L1*V1_0 + L2*V2_0
                            Y1 = L0*V0_1 + L1*V1_1 + L2*V2_1
                            Y2 = L0*V0_2 + L1*V1_2 + L2*V2_2

                            rx = Xi_0 - Y0
                            ry = Xi_1 - Y1
                            rz = Xi_2 - Y2
                            r2 = rx*rx + ry*ry + rz*rz
                            r = np.sqrt(r2)

                            rdotn = rx*nx + ry*ny + rz*nz

                            h_factor = w_uv * 2.0 * area * u * (rdotn / (4.0 * np.pi * r2 * r))
                            g_factor = w_uv * 2.0 * area * u / (4.0 * np.pi * r)

                            H_row[n0] += L0 * h_factor
                            H_row[n1] += L1 * h_factor
                            H_row[n2] += L2 * h_factor
                            G_row[n0] += L0 * g_factor
                            G_row[n1] += L1 * g_factor
                            G_row[n2] += L2 * g_factor
                else:
                    for k in range(7):
                        L0 = dun_bary[k, 0]
                        L1 = dun_bary[k, 1]
                        L2 = dun_bary[k, 2]
                        w_k = dun_wts[k]

                        Y0 = L0*V0_0 + L1*V1_0 + L2*V2_0
                        Y1 = L0*V0_1 + L1*V1_1 + L2*V2_1
                        Y2 = L0*V0_2 + L1*V1_2 + L2*V2_2

                        rx = Xi_0 - Y0
                        ry = Xi_1 - Y1
                        rz = Xi_2 - Y2
                        r2 = rx*rx + ry*ry + rz*rz
                        r = np.sqrt(r2)

                        rdotn = rx*nx + ry*ny + rz*nz

                        h_factor = w_k * area * (rdotn / (4.0 * np.pi * r2 * r))
                        g_factor = w_k * area / (4.0 * np.pi * r)

                        H_row[n0] += L0 * h_factor
                        H_row[n1] += L1 * h_factor
                        H_row[n2] += L2 * h_factor
                        G_row[n0] += L0 * g_factor
                        G_row[n1] += L1 * g_factor
                        G_row[n2] += L2 * g_factor

        H_row[i] = 0.0
        sum_H = 0.0
        for j in range(N_nodes):
            sum_H += H_row[j]
        H_row[i] = -sum_H

        b_i = 0.0
        for j in range(N_nodes):
            if bc_type[j] == 0:
                A[i, j] = -G_row[j]
                b_i -= H_row[j] * bc_val[j]
            else:
                A[i, j] = H_row[j]
                b_i += G_row[j] * bc_val[j]
        b[i] = b_i

@njit(parallel=True, fastmath=True)
def evaluate_interior(eval_pts, nodes, elems, areas, centroids, elem_normals, gp12_u, gw12_u, dun_bary, dun_wts, u_full, q_full):
    N_pts = eval_pts.shape[0]
    N_elems = elems.shape[0]
    u_int = np.zeros(N_pts, dtype=np.float64)

    for i in prange(N_pts):
        Xi_0, Xi_1, Xi_2 = eval_pts[i, 0], eval_pts[i, 1], eval_pts[i, 2]
        val = 0.0

        for e in range(N_elems):
            n0, n1, n2 = elems[e, 0], elems[e, 1], elems[e, 2]
            V0_0, V0_1, V0_2 = nodes[n0, 0], nodes[n0, 1], nodes[n0, 2]
            V1_0, V1_1, V1_2 = nodes[n1, 0], nodes[n1, 1], nodes[n1, 2]
            V2_0, V2_1, V2_2 = nodes[n2, 0], nodes[n2, 1], nodes[n2, 2]

            area = areas[e]
            nx, ny, nz = elem_normals[e, 0], elem_normals[e, 1], elem_normals[e, 2]

            cx, cy, cz = centroids[e, 0], centroids[e, 1], centroids[e, 2]
            dist2 = (cx - Xi_0)**2 + (cy - Xi_1)**2 + (cz - Xi_2)**2

            if dist2 < 20.0 * area:
                for u_idx in range(12):
                    for v_idx in range(12):
                        u = gp12_u[u_idx]
                        v = gp12_u[v_idx]
                        w_uv = gw12_u[u_idx] * gw12_u[v_idx]

                        L0 = 1.0 - u
                        L1 = u * (1.0 - v)
                        L2 = u * v

                        Y0 = L0*V0_0 + L1*V1_0 + L2*V2_0
                        Y1 = L0*V0_1 + L1*V1_1 + L2*V2_1
                        Y2 = L0*V0_2 + L1*V1_2 + L2*V2_2

                        rx = Xi_0 - Y0
                        ry = Xi_1 - Y1
                        rz = Xi_2 - Y2
                        r2 = rx*rx + ry*ry + rz*rz
                        r = np.sqrt(r2)

                        rdotn = rx*nx + ry*ny + rz*nz

                        h_factor = w_uv * 2.0 * area * u * (rdotn / (4.0 * np.pi * r2 * r))
                        g_factor = w_uv * 2.0 * area * u / (4.0 * np.pi * r)

                        qY = L0*q_full[n0] + L1*q_full[n1] + L2*q_full[n2]
                        uY = L0*u_full[n0] + L1*u_full[n1] + L2*u_full[n2]

                        val += g_factor * qY - h_factor * uY
            else:
                for k in range(7):
                    L0 = dun_bary[k, 0]
                    L1 = dun_bary[k, 1]
                    L2 = dun_bary[k, 2]
                    w_k = dun_wts[k]

                    Y0 = L0*V0_0 + L1*V1_0 + L2*V2_0
                    Y1 = L0*V0_1 + L1*V1_1 + L2*V2_1
                    Y2 = L0*V0_2 + L1*V1_2 + L2*V2_2

                    rx = Xi_0 - Y0
                    ry = Xi_1 - Y1
                    rz = Xi_2 - Y2
                    r2 = rx*rx + ry*ry + rz*rz
                    r = np.sqrt(r2)

                    rdotn = rx*nx + ry*ny + rz*nz

                    h_factor = w_k * area * (rdotn / (4.0 * np.pi * r2 * r))
                    g_factor = w_k * area / (4.0 * np.pi * r)

                    qY = L0*q_full[n0] + L1*q_full[n1] + L2*q_full[n2]
                    uY = L0*u_full[n0] + L1*u_full[n1] + L2*u_full[n2]

                    val += g_factor * qY - h_factor * uY

        u_int[i] = val
    return u_int

def main():
    print("N    | Ne      | N_nodes | Rel L2 Error   | Setup (s) | Solve (s) | Eval (s) | Total (s)")

    gp7, gw7 = np.polynomial.legendre.leggauss(7)
    gp12, gw12 = np.polynomial.legendre.leggauss(12)
    gp7_u = (gp7 + 1.0) / 2.0
    gw7_u = gw7 / 2.0
    gp12_u = (gp12 + 1.0) / 2.0
    gw12_u = gw12 / 2.0

    dun_bary = np.array([
        [1/3, 1/3, 1/3],
        [0.470142064105115, 0.470142064105115, 0.059715871789770],
        [0.470142064105115, 0.059715871789770, 0.470142064105115],
        [0.059715871789770, 0.470142064105115, 0.470142064105115],
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        [0.797426985353087, 0.101286507323456, 0.101286507323456]
    ], dtype=np.float64)
    dun_wts = np.array([
        0.225000000000000,
        0.132394152788506, 0.132394152788506, 0.132394152788506,
        0.125939180544827, 0.125939180544827, 0.125939180544827
    ], dtype=np.float64)

    _n, _e = generate_continuous_bumpy_mesh(1)
    _a, _c, _en, _nn, _e = compute_geometry(_n, _e)
    _bt, _bv = assign_mixed_bcs(_n, _nn)
    _A = np.zeros((len(_n), len(_n)), dtype=np.float64)
    _b = np.zeros(len(_n), dtype=np.float64)
    assemble_system(_n, _e, _a, _c, _en, gp7_u, gw7_u, gp12_u, gw12_u, dun_bary, dun_wts, _bt, _bv, _A, _b)
    _uf, _qf = np.zeros(len(_n)), np.zeros(len(_n))
    _eval_pts = np.ascontiguousarray(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
    evaluate_interior(_eval_pts, _n, _e, _a, _c, _en, gp12_u, gw12_u, dun_bary, dun_wts, _uf, _qf)

    results = []

    for N in [8, 16, 32]:
        nodes, elems = generate_continuous_bumpy_mesh(N)
        areas, centroids, elem_normals, nodal_normals, elems = compute_geometry(nodes, elems)
        bc_type, bc_val = assign_mixed_bcs(nodes, nodal_normals)

        N_nodes = nodes.shape[0]
        Ne = elems.shape[0]

        A = np.zeros((N_nodes, N_nodes), dtype=np.float64)
        b = np.zeros(N_nodes, dtype=np.float64)

        t0 = time.time()
        assemble_system(nodes, elems, areas, centroids, elem_normals, gp7_u, gw7_u, gp12_u, gw12_u, dun_bary, dun_wts, bc_type, bc_val, A, b)
        t_setup = time.time() - t0

        t1 = time.time()
        x = np.linalg.solve(A, b)
        t_solve = time.time() - t1

        u_full = np.zeros(N_nodes, dtype=np.float64)
        q_full = np.zeros(N_nodes, dtype=np.float64)

        dir_mask = (bc_type == 0)
        neu_mask = (bc_type == 1)

        u_full[dir_mask] = bc_val[dir_mask]
        q_full[dir_mask] = x[dir_mask]

        u_full[neu_mask] = x[neu_mask]
        q_full[neu_mask] = bc_val[neu_mask]

        xi = np.linspace(-0.5, 0.5, 5)
        yi = np.linspace(-0.5, 0.5, 5)
        zi = np.linspace(-0.5, 0.5, 5)
        XX, YY, ZZ = np.meshgrid(xi, yi, zi)
        eval_pts = np.ascontiguousarray(np.vstack((XX.flatten(), YY.flatten(), ZZ.flatten())).T)

        t2 = time.time()
        u_int_num = evaluate_interior(eval_pts, nodes, elems, areas, centroids, elem_normals, gp12_u, gw12_u, dun_bary, dun_wts, u_full, q_full)
        t_eval = time.time() - t2

        u_int_exact = np.sinh(eval_pts[:, 0])*np.sin(eval_pts[:, 1]) + np.cosh(eval_pts[:, 1])*np.cos(eval_pts[:, 2])
        rel_l2_error = np.linalg.norm(u_int_num - u_int_exact) / np.linalg.norm(u_int_exact)
        t_total = t_setup + t_solve + t_eval

        print(f"{N:<4} | {Ne:<7} | {N_nodes:<7} | {rel_l2_error:.6e}   | {t_setup:.4f}    | {t_solve:.4f}    | {t_eval:.4f}   | {t_total:.4f}")
        results.append((N, rel_l2_error))

    slope = -(np.log(results[-1][1]) - np.log(results[0][1])) / (np.log(results[-1][0]) - np.log(results[0][0]))
    print("Convergence Analysis:")
    print(f"Computed Slope: {slope:.4f}")
    print("Expected Slope: ~2.0000 (O(h^2) for linear elements)")

if __name__ == '__main__':
    main()