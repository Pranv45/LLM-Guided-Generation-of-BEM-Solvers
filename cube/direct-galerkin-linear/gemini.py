import numpy as np
import time
import math
from numba import njit, prange

# ==========================================
# ADVANCED QUADRATURE RULES
# ==========================================
# 7 points Dunavant, degree 5
sqrt15 = math.sqrt(15.0)
a = (6.0 - sqrt15) / 21.0
b = (6.0 + sqrt15) / 21.0
w1 = 9.0 / 40.0
w2 = (155.0 + sqrt15) / 1200.0
w3 = (155.0 - sqrt15) / 1200.0

tri_pts_7 = np.array([
    [1.0/3.0, 1.0/3.0, 1.0/3.0],
    [a, a, 1.0 - 2.0*a],
    [a, 1.0 - 2.0*a, a],
    [1.0 - 2.0*a, a, a],
    [b, b, 1.0 - 2.0*b],
    [b, 1.0 - 2.0*b, b],
    [1.0 - 2.0*b, b, b]
], dtype=np.float64)
tri_wts_7 = np.array([w1, w2, w2, w2, w3, w3, w3], dtype=np.float64)

# ==========================================
# GEOMETRY AND MESH GENERATION
# ==========================================
def generate_mesh(N):
    nodes_dict = {}
    nodes_list = []
    elems = []

    def add_node(x, y, z):
        k = (round(x, 6), round(y, 6), round(z, 6))
        if k not in nodes_dict:
            nodes_dict[k] = len(nodes_list)
            nodes_list.append([x, y, z])
        return nodes_dict[k]

    def add_face(dim, fixed_val, u_dim, v_dim, normal_sign):
        for i in range(N):
            for j in range(N):
                u1, u2 = i/N, (i+1)/N
                v1, v2 = j/N, (j+1)/N

                coords = np.zeros((4, 3))
                coords[0, dim] = fixed_val; coords[0, u_dim] = u1; coords[0, v_dim] = v1
                coords[1, dim] = fixed_val; coords[1, u_dim] = u2; coords[1, v_dim] = v1
                coords[2, dim] = fixed_val; coords[2, u_dim] = u2; coords[2, v_dim] = v2
                coords[3, dim] = fixed_val; coords[3, u_dim] = u1; coords[3, v_dim] = v2

                n00 = add_node(*coords[0])
                n10 = add_node(*coords[1])
                n11 = add_node(*coords[2])
                n01 = add_node(*coords[3])

                vec1 = coords[1] - coords[0]
                vec2 = coords[2] - coords[0]
                cross = np.cross(vec1, vec2)

                if cross[dim] * normal_sign < 0:
                    elems.append([n00, n11, n10])
                    elems.append([n00, n01, n11])
                else:
                    elems.append([n00, n10, n11])
                    elems.append([n00, n11, n01])

    add_face(2, 0.0, 0, 1, -1) # z=0
    add_face(2, 1.0, 0, 1, 1)  # z=1
    add_face(1, 0.0, 0, 2, -1) # y=0
    add_face(1, 1.0, 0, 2, 1)  # y=1
    add_face(0, 0.0, 1, 2, -1) # x=0
    add_face(0, 1.0, 1, 2, 1)  # x=1

    return np.array(nodes_list, dtype=np.float64), np.array(elems, dtype=np.int32)

def compute_geometry(nodes, elems):
    Ne = elems.shape[0]
    areas = np.zeros(Ne, dtype=np.float64)
    normals = np.zeros((Ne, 3), dtype=np.float64)
    centroids = np.zeros((Ne, 3), dtype=np.float64)
    for i in range(Ne):
        p0, p1, p2 = nodes[elems[i, 0]], nodes[elems[i, 1]], nodes[elems[i, 2]]
        v1 = p1 - p0
        v2 = p2 - p0
        cross = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(cross)
        areas[i] = area
        normals[i] = cross / (2.0 * area)
        centroids[i] = (p0 + p1 + p2) / 3.0
    return areas, normals, centroids

def build_node_elems(nodes, elems):
    Nn = nodes.shape[0]
    node_elems = np.full((Nn, 12), -1, dtype=np.int32)
    node_elems_count = np.zeros(Nn, dtype=np.int32)
    for i in range(elems.shape[0]):
        for j in range(3):
            n = elems[i, j]
            idx = node_elems_count[n]
            node_elems[n, idx] = i
            node_elems_count[n] += 1
    return node_elems, node_elems_count

def assign_bcs(nodes):
    Nn = nodes.shape[0]
    bc_type = np.zeros(Nn, dtype=np.int32)
    bc_val = np.zeros(Nn, dtype=np.float64)
    for i in range(Nn):
        x, y, z = nodes[i]
        if abs(x - 1.0) < 1e-5:
            bc_type[i] = 1; bc_val[i] = y + z
        elif abs(y - 1.0) < 1e-5:
            bc_type[i] = 1; bc_val[i] = x + z
        elif abs(z - 1.0) < 1e-5:
            bc_type[i] = 1; bc_val[i] = x + y

        if abs(x) < 1e-5:
            bc_type[i] = 0; bc_val[i] = y * z
        elif abs(y) < 1e-5:
            bc_type[i] = 0; bc_val[i] = z * x
        elif abs(z) < 1e-5:
            bc_type[i] = 0; bc_val[i] = x * y
    return bc_type, bc_val

def precompute_quadrature(nodes, elems, areas):
    Ne = elems.shape[0]
    # 3-Point Quadrature Arrays
    quad_y_3 = np.zeros((Ne, 3, 3), dtype=np.float64)
    quad_wJ_3 = np.zeros((Ne, 3), dtype=np.float64)
    tri_phi_3 = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]], dtype=np.float64)

    # 7-Point Quadrature Arrays
    quad_y_7 = np.zeros((Ne, 7, 3), dtype=np.float64)
    quad_wJ_7 = np.zeros((Ne, 7), dtype=np.float64)
    tri_phi_7 = np.zeros((3, 7), dtype=np.float64)

    for k in range(7):
        tri_phi_7[0, k] = tri_pts_7[k, 0]
        tri_phi_7[1, k] = tri_pts_7[k, 1]
        tri_phi_7[2, k] = tri_pts_7[k, 2]

    for i in range(Ne):
        p0, p1, p2 = nodes[elems[i, 0]], nodes[elems[i, 1]], nodes[elems[i, 2]]

        # 3-point
        for q in range(3):
            quad_y_3[i, q] = tri_phi_3[0, q]*p0 + tri_phi_3[1, q]*p1 + tri_phi_3[2, q]*p2
            quad_wJ_3[i, q] = areas[i] / 3.0

        # 7-point
        for q in range(7):
            quad_y_7[i, q] = tri_phi_7[0, q]*p0 + tri_phi_7[1, q]*p1 + tri_phi_7[2, q]*p2
            quad_wJ_7[i, q] = areas[i] * tri_wts_7[q]

    return quad_y_3, quad_wJ_3, tri_phi_3, quad_y_7, quad_wJ_7, tri_phi_7

# ==========================================
# NUMBA ACCELERATED SYMMETRIC BEM ASSEMBLY
# ==========================================
@njit(parallel=True, fastmath=True)
def assemble_system(nodes, elems, areas, normals, centroids, quad_y_3, quad_wJ_3, tri_phi_3, quad_y_7, quad_wJ_7, tri_phi_7, node_elems, node_elems_count, gauss_pts, gauss_wts):
    Nn = nodes.shape[0]
    Ne = elems.shape[0]
    H = np.zeros((Nn, Nn), dtype=np.float64)
    G = np.zeros((Nn, Nn), dtype=np.float64)
    PI4 = 12.566370614359172

    for i_node in prange(Nn):
        for e_idx in range(node_elems_count[i_node]):
            e_outer = node_elems[i_node, e_idx]
            on0, on1, on2 = elems[e_outer]

            loc_i = 0
            if on1 == i_node: loc_i = 1
            elif on2 == i_node: loc_i = 2

            cx_o, cy_o, cz_o = centroids[e_outer, 0], centroids[e_outer, 1], centroids[e_outer, 2]

            for e_inner in range(Ne):
                in0, in1, in2 = elems[e_inner]

                # --- CASE 1: SELF INTEGRATION (Duffy Transformation) ---
                if e_outer == e_inner:
                    Atot = areas[e_inner]
                    v0_0, v0_1, v0_2 = nodes[in0, 0], nodes[in0, 1], nodes[in0, 2]
                    v1_0, v1_1, v1_2 = nodes[in1, 0], nodes[in1, 1], nodes[in1, 2]
                    v2_0, v2_1, v2_2 = nodes[in2, 0], nodes[in2, 1], nodes[in2, 2]

                    for x_q in range(3):
                        x0 = quad_y_3[e_outer, x_q, 0]
                        x1 = quad_y_3[e_outer, x_q, 1]
                        x2 = quad_y_3[e_outer, x_q, 2]
                        Wx = quad_wJ_3[e_outer, x_q] * tri_phi_3[loc_i, x_q]

                        L0_x, L1_x, L2_x = tri_phi_3[0, x_q], tri_phi_3[1, x_q], tri_phi_3[2, x_q]

                        for sub in range(3):
                            if sub == 0:
                                Bx, By, Bz = v0_0, v0_1, v0_2
                                Cx, Cy, Cz = v1_0, v1_1, v1_2
                                L_sub = L2_x
                            elif sub == 1:
                                Bx, By, Bz = v1_0, v1_1, v1_2
                                Cx, Cy, Cz = v2_0, v2_1, v2_2
                                L_sub = L0_x
                            else:
                                Bx, By, Bz = v2_0, v2_1, v2_2
                                Cx, Cy, Cz = v0_0, v0_1, v0_2
                                L_sub = L1_x

                            if L_sub < 1e-12: continue

                            CB_x = Cx - Bx
                            CB_y = Cy - By
                            CB_z = Cz - Bz

                            Bx_x = Bx - x0
                            Bx_y = By - x1
                            Bx_z = Bz - x2

                            for q1 in range(3):
                                eta1, w1 = gauss_pts[q1], gauss_wts[q1]
                                for q2 in range(3):
                                    eta2, w2 = gauss_pts[q2], gauss_wts[q2]

                                    Vx = Bx_x + CB_x * eta2
                                    Vy = Bx_y + CB_y * eta2
                                    Vz = Bx_z + CB_z * eta2

                                    V_len = math.sqrt(Vx*Vx + Vy*Vy + Vz*Vz)
                                    if V_len < 1e-14: continue

                                    u_star_J = (L_sub * Atot) / (6.283185307179586 * V_len)
                                    W_duffy = Wx * w1 * w2 * u_star_J

                                    if sub == 0:
                                        phi0 = L0_x*(1-eta1) + eta1*(1-eta2)
                                        phi1 = L1_x*(1-eta1) + eta1*eta2
                                        phi2 = L2_x*(1-eta1)
                                    elif sub == 1:
                                        phi0 = L0_x*(1-eta1)
                                        phi1 = L1_x*(1-eta1) + eta1*(1-eta2)
                                        phi2 = L2_x*(1-eta1) + eta1*eta2
                                    else:
                                        phi0 = L0_x*(1-eta1) + eta1*eta2
                                        phi1 = L1_x*(1-eta1)
                                        phi2 = L2_x*(1-eta1) + eta1*(1-eta2)

                                    G[i_node, in0] += W_duffy * phi0
                                    G[i_node, in1] += W_duffy * phi1
                                    G[i_node, in2] += W_duffy * phi2

                # --- CASE 2: NON-SELF INTEGRATION (Distance-Based) ---
                else:
                    cx_i, cy_i, cz_i = centroids[e_inner, 0], centroids[e_inner, 1], centroids[e_inner, 2]
                    dist_sq = (cx_o - cx_i)**2 + (cy_o - cy_i)**2 + (cz_o - cz_i)**2
                    threshold = 2.0 * (math.sqrt(areas[e_outer]) + math.sqrt(areas[e_inner]))
                    nx, ny, nz = normals[e_inner, 0], normals[e_inner, 1], normals[e_inner, 2]

                    if dist_sq < threshold*threshold:
                        # SYMMETRIC 7x7 NEAR-FIELD
                        for x_q in range(7):
                            x0 = quad_y_7[e_outer, x_q, 0]
                            x1 = quad_y_7[e_outer, x_q, 1]
                            x2 = quad_y_7[e_outer, x_q, 2]
                            Wx = quad_wJ_7[e_outer, x_q] * tri_phi_7[loc_i, x_q]

                            for y_q in range(7):
                                y0 = quad_y_7[e_inner, y_q, 0]
                                y1 = quad_y_7[e_inner, y_q, 1]
                                y2 = quad_y_7[e_inner, y_q, 2]
                                Wy = quad_wJ_7[e_inner, y_q]

                                dx = x0 - y0
                                dy = x1 - y1
                                dz = x2 - y2
                                r_sq = dx*dx + dy*dy + dz*dz

                                if r_sq > 1e-28:
                                    r = math.sqrt(r_sq)
                                    ndot = dx*nx + dy*ny + dz*nz
                                    ustar = 1.0 / (PI4 * r)
                                    qstar = ndot / (PI4 * r_sq * r)

                                    val_G = Wx * Wy * ustar
                                    val_H = Wx * Wy * qstar

                                    p0, p1, p2 = tri_phi_7[0, y_q], tri_phi_7[1, y_q], tri_phi_7[2, y_q]

                                    G[i_node, in0] += val_G * p0
                                    G[i_node, in1] += val_G * p1
                                    G[i_node, in2] += val_G * p2
                                    H[i_node, in0] += val_H * p0
                                    H[i_node, in1] += val_H * p1
                                    H[i_node, in2] += val_H * p2
                    else:
                        # SYMMETRIC 3x3 FAR-FIELD
                        for x_q in range(3):
                            x0 = quad_y_3[e_outer, x_q, 0]
                            x1 = quad_y_3[e_outer, x_q, 1]
                            x2 = quad_y_3[e_outer, x_q, 2]
                            Wx = quad_wJ_3[e_outer, x_q] * tri_phi_3[loc_i, x_q]

                            for y_q in range(3):
                                y0 = quad_y_3[e_inner, y_q, 0]
                                y1 = quad_y_3[e_inner, y_q, 1]
                                y2 = quad_y_3[e_inner, y_q, 2]
                                Wy = quad_wJ_3[e_inner, y_q]

                                dx = x0 - y0
                                dy = x1 - y1
                                dz = x2 - y2
                                r_sq = dx*dx + dy*dy + dz*dz

                                if r_sq > 1e-28:
                                    r = math.sqrt(r_sq)
                                    ndot = dx*nx + dy*ny + dz*nz
                                    ustar = 1.0 / (PI4 * r)
                                    qstar = ndot / (PI4 * r_sq * r)

                                    val_G = Wx * Wy * ustar
                                    val_H = Wx * Wy * qstar

                                    p0, p1, p2 = tri_phi_3[0, y_q], tri_phi_3[1, y_q], tri_phi_3[2, y_q]

                                    G[i_node, in0] += val_G * p0
                                    G[i_node, in1] += val_G * p1
                                    G[i_node, in2] += val_G * p2
                                    H[i_node, in0] += val_H * p0
                                    H[i_node, in1] += val_H * p1
                                    H[i_node, in2] += val_H * p2

    for i in range(Nn):
        sum_H = 0.0
        for j in range(Nn):
            if i != j:
                sum_H += H[i, j]
        H[i, i] = -sum_H

    return H, G

# ==========================================
# NUMBA ACCELERATED INTERIOR EVALUATION
# ==========================================
@njit(parallel=True, fastmath=True)
def eval_interior(pts, nodes, elems, u, q, quad_y_3, quad_wJ_3, normals, tri_phi_3):
    N_pts = pts.shape[0]
    Ne = elems.shape[0]
    u_eval = np.zeros(N_pts, dtype=np.float64)

    for i in prange(N_pts):
        px, py, pz = pts[i, 0], pts[i, 1], pts[i, 2]
        val = 0.0
        for e in range(Ne):
            n0, n1, n2 = elems[e]
            u0, u1, u2 = u[n0], u[n1], u[n2]
            q0, q1, q2 = q[n0], q[n1], q[n2]
            nx, ny, nz = normals[e, 0], normals[e, 1], normals[e, 2]

            for y_q in range(3):
                y0 = quad_y_3[e, y_q, 0]
                y1 = quad_y_3[e, y_q, 1]
                y2 = quad_y_3[e, y_q, 2]
                wy = quad_wJ_3[e, y_q]

                dx = px - y0
                dy = py - y1
                dz = pz - y2
                r_sq = dx*dx + dy*dy + dz*dz

                r = math.sqrt(r_sq)
                ndot = dx*nx + dy*ny + dz*nz
                ustar = 1.0 / (12.566370614359172 * r)
                qstar = ndot / (12.566370614359172 * r * r_sq)

                phi0, phi1, phi2 = tri_phi_3[0, y_q], tri_phi_3[1, y_q], tri_phi_3[2, y_q]
                u_y = u0*phi0 + u1*phi1 + u2*phi2
                q_y = q0*phi0 + q1*phi1 + q2*phi2

                val += wy * (ustar * q_y - qstar * u_y)
        u_eval[i] = val
    return u_eval

# ==========================================
# MAIN ROUTINE
# ==========================================
def solve_bem(N):
    t_start = time.time()
    nodes, elems = generate_mesh(N)
    areas, normals, centroids = compute_geometry(nodes, elems)
    node_elems, node_elems_count = build_node_elems(nodes, elems)
    quad_y_3, quad_wJ_3, tri_phi_3, quad_y_7, quad_wJ_7, tri_phi_7 = precompute_quadrature(nodes, elems, areas)

    gauss_pts = np.array([0.1127016653792583, 0.5, 0.8872983346207417], dtype=np.float64)
    gauss_wts = np.array([0.2777777777777778, 0.4444444444444444, 0.2777777777777778], dtype=np.float64)

    H, G = assemble_system(nodes, elems, areas, normals, centroids, quad_y_3, quad_wJ_3, tri_phi_3, quad_y_7, quad_wJ_7, tri_phi_7, node_elems, node_elems_count, gauss_pts, gauss_wts)

    bc_type, bc_val = assign_bcs(nodes)

    d_mask = bc_type == 0
    n_mask = bc_type == 1

    A = np.copy(H)
    B_mat = np.copy(G)

    A[:, d_mask] = -G[:, d_mask]
    B_mat[:, d_mask] = -H[:, d_mask]

    t_setup = time.time() - t_start

    t_solve_start = time.time()
    b = B_mat @ bc_val
    x_sol = np.linalg.solve(A, b)
    t_solve = time.time() - t_solve_start

    u = np.zeros(nodes.shape[0])
    q = np.zeros(nodes.shape[0])
    u[d_mask] = bc_val[d_mask]
    q[d_mask] = x_sol[d_mask]
    q[n_mask] = bc_val[n_mask]
    u[n_mask] = x_sol[n_mask]

    t_eval_start = time.time()
    ev_x = np.linspace(0.2, 0.8, 5)
    X, Y, Z = np.meshgrid(ev_x, ev_x, ev_x, indexing='ij')
    eval_pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    u_eval = eval_interior(eval_pts, nodes, elems, u, q, quad_y_3, quad_wJ_3, normals, tri_phi_3)
    u_exact = eval_pts[:, 0]*eval_pts[:, 1] + eval_pts[:, 1]*eval_pts[:, 2] + eval_pts[:, 2]*eval_pts[:, 0]

    rel_l2 = np.linalg.norm(u_eval - u_exact) / np.linalg.norm(u_exact)
    t_eval = time.time() - t_eval_start

    t_total = t_setup + t_solve + t_eval
    return elems.shape[0], rel_l2, t_setup, t_solve, t_eval, t_total

def main():
    # Warmup Numba compiler
    _ = solve_bem(2)

    Ns = [8, 16, 32]
    results = []

    print("-" * 89)
    print("N    | Ne     | Rel L2 Error   | Setup (s) | Solve (s) | Eval (s) | Total (s)")
    print("-" * 89)
    for N in Ns:
        Ne, err, ts, tsol, te, tt = solve_bem(N)
        print(f"{N:<4} | {Ne:<6} | {err:.6e}   | {ts:.4f}    | {tsol:.4f}    | {te:.4f}   | {tt:.4f}")
        results.append(err)

    slope = math.log(results[0] / results[2]) / math.log(4.0)
    print("-" * 89)
    print("Convergence Analysis:")
    print(f"Computed Slope: {slope:.4f}")
    print("Expected Slope: ~2.0000 (O(h^2) for linear elements)")

if __name__ == "__main__":
    main()