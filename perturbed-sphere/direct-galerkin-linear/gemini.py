import numpy as np
import time
from numba import njit, prange

# =====================================================================
# Quadrature Rules & Adaptive Subdivision
# =====================================================================
DUNAVANT_W = np.array([0.1125,
                       0.06296959027241358, 0.06296959027241358, 0.06296959027241358,
                       0.06619707639425308, 0.06619707639425308, 0.06619707639425308])
DUNAVANT_PT = np.array([
    [1.0/3.0, 1.0/3.0],
    [0.10128650732345633, 0.10128650732345633],
    [0.79742698535308730, 0.10128650732345633],
    [0.10128650732345633, 0.79742698535308730],
    [0.47014206410511505, 0.47014206410511505],
    [0.05971587178976989, 0.47014206410511505],
    [0.47014206410511505, 0.05971587178976989]
])

def build_subdivided_rule(level):
    if level == 0: return DUNAVANT_PT, DUNAVANT_W
    prev_pts, prev_ws = build_subdivided_rule(level - 1)
    pts = np.zeros((len(prev_pts)*4, 2)); ws = np.zeros(len(prev_ws)*4)
    tris = [((0.0, 0.0), (0.5, 0.0), (0.0, 0.5)), ((0.5, 0.0), (1.0, 0.0), (0.5, 0.5)),
            ((0.0, 0.5), (0.5, 0.5), (0.0, 1.0)), ((0.5, 0.0), (0.5, 0.5), (0.0, 0.5))]
    idx = 0
    for tri in tris:
        u0, v0 = tri[0]; u1, v1 = tri[1]; u2, v2 = tri[2]
        for i in range(len(prev_pts)):
            u, v = prev_pts[i]
            pts[idx, 0] = u0 + u*(u1-u0) + v*(u2-u0)
            pts[idx, 1] = v0 + u*(v1-v0) + v*(v2-v0)
            ws[idx] = prev_ws[i] / 4.0
            idx += 1
    return pts, ws

SUB0_PT, SUB0_W = DUNAVANT_PT, DUNAVANT_W
SUB1_PT, SUB1_W = build_subdivided_rule(1)
SUB2_PT, SUB2_W = build_subdivided_rule(2)

GAUSS_PT_1D = (np.array([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]) + 1.0) / 2.0
GAUSS_W_1D = np.array([0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]) / 2.0
G4_W = np.zeros(256); G4_P = np.zeros((256, 4)); idx = 0
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                G4_W[idx] = GAUSS_W_1D[i] * GAUSS_W_1D[j] * GAUSS_W_1D[k] * GAUSS_W_1D[l]
                G4_P[idx] = [GAUSS_PT_1D[i], GAUSS_PT_1D[j], GAUSS_PT_1D[k], GAUSS_PT_1D[l]]
                idx += 1

# =====================================================================
# Mesh Generation & Topology
# =====================================================================
def generate_continuous_bumpy_mesh(N):
    node_dict = {}; nodes_list = []; elems_list = []
    def get_node_id(x, y, z):
        R = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/R); phi = np.arctan2(y/R, x/R)
        r = 1.5 + 0.3 * np.sin(4*theta) * np.cos(5*phi)
        xf = r * np.sin(theta) * np.cos(phi)
        yf = r * np.sin(theta) * np.sin(phi)
        zf = r * np.cos(theta)
        key = (round(xf, 6), round(yf, 6), round(zf, 6))
        if key not in node_dict:
            node_dict[key] = len(nodes_list)
            nodes_list.append([xf, yf, zf])
        return node_dict[key]

    faces = [lambda u, v: (1, u, v), lambda u, v: (-1, v, u), lambda u, v: (u, 1, v),
             lambda u, v: (v, -1, u), lambda u, v: (u, v, 1), lambda u, v: (v, u, -1)]

    ls = np.linspace(-1, 1, N+1)
    for face_fn in faces:
        for i in range(N):
            for j in range(N):
                u0, v0 = ls[i], ls[j]; u1, v1 = ls[i+1], ls[j+1]
                n00 = get_node_id(*face_fn(u0, v0)); n10 = get_node_id(*face_fn(u1, v0))
                n01 = get_node_id(*face_fn(u0, v1)); n11 = get_node_id(*face_fn(u1, v1))
                elems_list.extend([[n00, n10, n11], [n00, n11, n01]])

    return np.array(nodes_list, dtype=np.float64), np.array(elems_list, dtype=np.int32)

def compute_geometry(nodes, elems):
    Ne = len(elems); Nn = len(nodes)
    areas = np.zeros(Ne, dtype=np.float64); normals = np.zeros((Ne, 3), dtype=np.float64)
    nodal_normals = np.zeros((Nn, 3), dtype=np.float64); centroids = np.zeros((Ne, 3), dtype=np.float64)

    for i in range(Ne):
        n0, n1, n2 = elems[i]
        p0, p1, p2 = nodes[n0], nodes[n1], nodes[n2]
        cross = np.cross(p1 - p0, p2 - p0)
        centroid = (p0 + p1 + p2) / 3.0
        centroids[i] = centroid

        if np.dot(cross, centroid) < 0:
            cross = -cross; elems[i] = [n0, n2, n1]

        norm = np.linalg.norm(cross); areas[i] = 0.5 * norm
        normals[i] = cross / (norm + 1e-14)
        for n in elems[i]: nodal_normals[n] += normals[i] * areas[i]

    for i in range(Nn):
        nn = np.linalg.norm(nodal_normals[i])
        if nn > 0: nodal_normals[i] /= nn
    return areas, normals, nodal_normals, centroids

# =====================================================================
# Exact Solutions
# =====================================================================
def exact_solution(X, Y, Z): return np.sinh(X)*np.sin(Y) + np.cosh(Y)*np.cos(Z)

def assign_mixed_bcs(nodes, nodal_normals):
    Nn = len(nodes); bc_type = np.zeros(Nn, dtype=np.int32); bc_val = np.zeros(Nn, dtype=np.float64)
    for i in range(Nn):
        x, y, z = nodes[i]
        if x > 0:
            bc_type[i] = 0; bc_val[i] = exact_solution(x, y, z)
        else:
            bc_type[i] = 1
            grad = np.array([np.cosh(x)*np.sin(y), np.sinh(x)*np.cos(y) + np.sinh(y)*np.cos(z), -np.cosh(y)*np.sin(z)])
            bc_val[i] = np.dot(grad, nodal_normals[i])
    return bc_type, bc_val

# =====================================================================
# Optimized Node-Centric Numba Acceleration
# =====================================================================
@njit(fastmath=True)
def build_connectivity(elems, Nn):
    Ne = len(elems); counts = np.zeros(Nn, dtype=np.int32)
    for i in range(Ne):
        counts[elems[i, 0]] += 1; counts[elems[i, 1]] += 1; counts[elems[i, 2]] += 1
    node_to_elems = np.empty((Nn, np.max(counts)), dtype=np.int32)
    current = np.zeros(Nn, dtype=np.int32)
    for i in range(Ne):
        for j in range(3):
            n = elems[i, j]
            node_to_elems[n, current[n]] = i
            current[n] += 1
    return node_to_elems, counts

@njit(fastmath=True)
def _ss_eval(p_t, p_s, u_t, v_t, u_s, v_s, weight, jac, area_t, area_s, kt_target, g0, g1, g2):
    phi_t = 1.0 - u_t - v_t if kt_target == 0 else (u_t if kt_target == 1 else v_t)
    phi_s0 = 1.0 - u_s - v_s; phi_s1 = u_s; phi_s2 = v_s

    xt0 = p_t[0,0]*(1.0-u_t-v_t) + p_t[1,0]*u_t + p_t[2,0]*v_t
    xt1 = p_t[0,1]*(1.0-u_t-v_t) + p_t[1,1]*u_t + p_t[2,1]*v_t
    xt2 = p_t[0,2]*(1.0-u_t-v_t) + p_t[1,2]*u_t + p_t[2,2]*v_t

    xs0 = p_s[0,0]*phi_s0 + p_s[1,0]*phi_s1 + p_s[2,0]*phi_s2
    xs1 = p_s[0,1]*phi_s0 + p_s[1,1]*phi_s1 + p_s[2,1]*phi_s2
    xs2 = p_s[0,2]*phi_s0 + p_s[1,2]*phi_s1 + p_s[2,2]*phi_s2

    r_dist = np.sqrt((xt0-xs0)**2 + (xt1-xs1)**2 + (xt2-xs2)**2)
    u_star = 1.0 / (4.0 * np.pi * r_dist) if r_dist > 1e-12 else 0.0
    wtot = weight * jac * 4.0 * area_t * area_s * phi_t * u_star

    return g0 + wtot*phi_s0, g1 + wtot*phi_s1, g2 + wtot*phi_s2

@njit(fastmath=True)
def sauter_schwab_identical_node(p_t, p_s, area_t, area_s, kt_target):
    g0 = g1 = g2 = 0.0
    for i in range(256):
        w1 = G4_P[i,0]; w2 = G4_P[i,1]; w3 = G4_P[i,2]; w4 = G4_P[i,3]; wt = G4_W[i]
        g0, g1, g2 = _ss_eval(p_t, p_s, w1, w1*(1-w2+w2*w3), w1*(1-w2*w4), w1*(1-w2), wt, w1**3 * w2, area_t, area_s, kt_target, g0, g1, g2)
        g0, g1, g2 = _ss_eval(p_t, p_s, w1*(1-w2*w3*w4), w1*(1-w2), w1, w1*(1-w2+w2*w3), wt, w1**3 * w2**2 * w3, area_t, area_s, kt_target, g0, g1, g2)
        g0, g1, g2 = _ss_eval(p_t, p_s, w1, w1*(w2*(1-w3+w3*w4)), w1*(1-w2*w3), w1*(w2*(1-w3)), wt, w1**3 * w2**2 * w3, area_t, area_s, kt_target, g0, g1, g2)
        g0, g1, g2 = _ss_eval(p_t, p_s, w1*(1-w2*w3), w1*(w2*(1-w3)), w1, w1*(w2*(1-w3+w3*w4)), wt, w1**3 * w2**2 * w3, area_t, area_s, kt_target, g0, g1, g2)
        g0, g1, g2 = _ss_eval(p_t, p_s, w1*(1-w2*w3*w4), w1*(w2*(1-w3*w4)), w1, w1*(w2*(1-w3)), wt, w1**3 * w2**2 * w3, area_t, area_s, kt_target, g0, g1, g2)
        g0, g1, g2 = _ss_eval(p_t, p_s, w1, w1*(w2*(1-w3)), w1*(1-w2*w3*w4), w1*(w2*(1-w3*w4)), wt, w1**3 * w2**2 * w3, area_t, area_s, kt_target, g0, g1, g2)
    return 0.0, 0.0, 0.0, g0, g1, g2

@njit(fastmath=True)
def integrate_pair_node(p_t, p_s, norm_s, area_t, area_s, pt_t, w_t, pt_s, w_s, kt_target):
    h0 = h1 = h2 = g0 = g1 = g2 = 0.0
    for it in range(len(w_t)):
        ut = pt_t[it, 0]; vt = pt_t[it, 1]
        phi_t = 1.0 - ut - vt if kt_target == 0 else (ut if kt_target == 1 else vt)
        xt0 = p_t[0,0]*(1.0-ut-vt) + p_t[1,0]*ut + p_t[2,0]*vt
        xt1 = p_t[0,1]*(1.0-ut-vt) + p_t[1,1]*ut + p_t[2,1]*vt
        xt2 = p_t[0,2]*(1.0-ut-vt) + p_t[1,2]*ut + p_t[2,2]*vt

        for is_ in range(len(w_s)):
            us = pt_s[is_, 0]; vs = pt_s[is_, 1]
            phi_s0 = 1.0 - us - vs; phi_s1 = us; phi_s2 = vs
            xs0 = p_s[0,0]*phi_s0 + p_s[1,0]*phi_s1 + p_s[2,0]*phi_s2
            xs1 = p_s[0,1]*phi_s0 + p_s[1,1]*phi_s1 + p_s[2,1]*phi_s2
            xs2 = p_s[0,2]*phi_s0 + p_s[1,2]*phi_s1 + p_s[2,2]*phi_s2

            rx = xt0-xs0; ry = xt1-xs1; rz = xt2-xs2
            r_dist = np.sqrt(rx**2 + ry**2 + rz**2)

            u_star = 1.0 / (4.0 * np.pi * r_dist)
            q_star = (rx*norm_s[0] + ry*norm_s[1] + rz*norm_s[2]) / (4.0 * np.pi * r_dist**3)

            wtot = w_t[it] * w_s[is_] * 4.0 * area_t * area_s * phi_t
            vg = wtot * u_star; vh = wtot * q_star

            g0 += vg*phi_s0; g1 += vg*phi_s1; g2 += vg*phi_s2
            h0 += vh*phi_s0; h1 += vh*phi_s1; h2 += vh*phi_s2

    return h0, h1, h2, g0, g1, g2

@njit(parallel=True, fastmath=True)
def assemble_system(nodes, elems, areas, normals, centroids, node_to_elems, node_elem_counts):
    Nn = len(nodes); Ne = len(elems)
    H = np.zeros((Nn, Nn), dtype=np.float64); G = np.zeros((Nn, Nn), dtype=np.float64)

    for nt in prange(Nn):
        p_t = np.empty((3, 3), dtype=np.float64)
        p_s = np.empty((3, 3), dtype=np.float64)

        for idx in range(node_elem_counts[nt]):
            t = node_to_elems[nt, idx]
            n_t0 = elems[t,0]; n_t1 = elems[t,1]; n_t2 = elems[t,2]
            p_t[0,0] = nodes[n_t0,0]; p_t[0,1] = nodes[n_t0,1]; p_t[0,2] = nodes[n_t0,2]
            p_t[1,0] = nodes[n_t1,0]; p_t[1,1] = nodes[n_t1,1]; p_t[1,2] = nodes[n_t1,2]
            p_t[2,0] = nodes[n_t2,0]; p_t[2,1] = nodes[n_t2,1]; p_t[2,2] = nodes[n_t2,2]

            kt_target = 1 if n_t1 == nt else (2 if n_t2 == nt else 0)
            ct = centroids[t]

            for s in range(Ne):
                n_s0 = elems[s,0]; n_s1 = elems[s,1]; n_s2 = elems[s,2]
                p_s[0,0] = nodes[n_s0,0]; p_s[0,1] = nodes[n_s0,1]; p_s[0,2] = nodes[n_s0,2]
                p_s[1,0] = nodes[n_s1,0]; p_s[1,1] = nodes[n_s1,1]; p_s[1,2] = nodes[n_s1,2]
                p_s[2,0] = nodes[n_s2,0]; p_s[2,1] = nodes[n_s2,1]; p_s[2,2] = nodes[n_s2,2]

                dist_sq = (ct[0]-centroids[s,0])**2 + (ct[1]-centroids[s,1])**2 + (ct[2]-centroids[s,2])**2
                Ls_sq = areas[s] * 2.25

                if t == s: h0, h1, h2, g0, g1, g2 = sauter_schwab_identical_node(p_t, p_s, areas[t], areas[s], kt_target)
                elif dist_sq < Ls_sq: h0, h1, h2, g0, g1, g2 = integrate_pair_node(p_t, p_s, normals[s], areas[t], areas[s], SUB1_PT, SUB1_W, SUB2_PT, SUB2_W, kt_target)
                elif dist_sq < Ls_sq * 5.44: h0, h1, h2, g0, g1, g2 = integrate_pair_node(p_t, p_s, normals[s], areas[t], areas[s], SUB0_PT, SUB0_W, SUB1_PT, SUB1_W, kt_target)
                else: h0, h1, h2, g0, g1, g2 = integrate_pair_node(p_t, p_s, normals[s], areas[t], areas[s], SUB0_PT, SUB0_W, SUB0_PT, SUB0_W, kt_target)

                H[nt, n_s0] += h0; G[nt, n_s0] += g0
                H[nt, n_s1] += h1; G[nt, n_s1] += g1
                H[nt, n_s2] += h2; G[nt, n_s2] += g2

    return H, G

@njit(fastmath=True)
def enforce_rbm(H):
    Nn = H.shape[0]
    for i in range(Nn):
        diag_sum = 0.0
        for j in range(Nn):
            if i != j: diag_sum += H[i, j]
        H[i, i] = -diag_sum

@njit(fastmath=True)
def apply_bcs(H, G, bc_type, bc_val):
    Nn = H.shape[0]; A = np.zeros((Nn, Nn), dtype=np.float64); b = np.zeros(Nn, dtype=np.float64)
    for i in range(Nn):
        for j in range(Nn):
            if bc_type[j] == 0: A[i, j] = -G[i, j]; b[i] -= H[i, j] * bc_val[j]
            else: A[i, j] = H[i, j]; b[i] += G[i, j] * bc_val[j]
    return A, b

@njit(fastmath=True)
def extract_results(x, bc_type, bc_val):
    Nn = len(x); u = np.zeros(Nn); q = np.zeros(Nn)
    for i in range(Nn):
        if bc_type[i] == 0: u[i] = bc_val[i]; q[i] = x[i]
        else: u[i] = x[i]; q[i] = bc_val[i]
    return u, q

@njit(parallel=True, fastmath=True)
def evaluate_interior(eval_pts, nodes, elems, u, q, areas, normals):
    Np = len(eval_pts); Ne = len(elems); u_int = np.zeros(Np, dtype=np.float64)
    for p in prange(Np):
        ptx, pty, ptz = eval_pts[p,0], eval_pts[p,1], eval_pts[p,2]; val = 0.0
        for e in range(Ne):
            n0, n1, n2 = elems[e]
            px0, py0, pz0 = nodes[n0]; px1, py1, pz1 = nodes[n1]; px2, py2, pz2 = nodes[n2]
            nx, ny, nz = normals[e]; area = areas[e]
            for is_ in range(7):
                us = DUNAVANT_PT[is_, 0]; vs = DUNAVANT_PT[is_, 1]; ws = 1.0 - us - vs
                xs_x = px0*ws + px1*us + px2*vs; xs_y = py0*ws + py1*us + py2*vs; xs_z = pz0*ws + pz1*us + pz2*vs
                u_interp = u[n0]*ws + u[n1]*us + u[n2]*vs
                q_interp = q[n0]*ws + q[n1]*us + q[n2]*vs
                rx = ptx - xs_x; ry = pty - xs_y; rz = ptz - xs_z
                r_dist = np.sqrt(rx**2 + ry**2 + rz**2)
                r_dot_n = rx*nx + ry*ny + rz*nz
                u_star = 1.0 / (4.0 * np.pi * r_dist)
                q_star = r_dot_n / (4.0 * np.pi * r_dist**3)
                val += DUNAVANT_W[is_] * 2.0 * area * (u_star * q_interp - q_star * u_interp)
        u_int[p] = val
    return u_int

# =====================================================================
# Main Execution
# =====================================================================
if __name__ == "__main__":
    print("Pre-compiling Numba functions (Warmup Run)...")
    nodes_w, elems_w = generate_continuous_bumpy_mesh(2)
    areas_w, normals_w, nodal_normals_w, centroids_w = compute_geometry(nodes_w, elems_w)
    bc_type_w, bc_val_w = assign_mixed_bcs(nodes_w, nodal_normals_w)
    node_to_elems_w, node_elem_counts_w = build_connectivity(elems_w, len(nodes_w))
    H_w, G_w = assemble_system(nodes_w, elems_w, areas_w, normals_w, centroids_w, node_to_elems_w, node_elem_counts_w)
    enforce_rbm(H_w)
    A_w, b_w = apply_bcs(H_w, G_w, bc_type_w, bc_val_w)
    x_sol_w = np.linalg.solve(A_w, b_w)
    u_bound_w, q_bound_w = extract_results(x_sol_w, bc_type_w, bc_val_w)
    evaluate_interior(np.array([[0.0,0.0,0.0]]), nodes_w, elems_w, u_bound_w, q_bound_w, areas_w, normals_w)

    print("\nN    | Ne      | N_nodes | Rel L2 Error   | Setup (s) | Solve (s) | Eval (s) | Total (s)")

    N_vals = [8, 16, 32]
    eval_pts = np.array([[x, y, z] for x in np.linspace(-0.5, 0.5, 5) for y in np.linspace(-0.5, 0.5, 5) for z in np.linspace(-0.5, 0.5, 5)], dtype=np.float64)
    u_exact_eval = np.array([exact_solution(p[0], p[1], p[2]) for p in eval_pts])

    for N in N_vals:
        t0 = time.time()

        nodes, elems = generate_continuous_bumpy_mesh(N)
        areas, normals, nodal_normals, centroids = compute_geometry(nodes, elems)
        bc_type, bc_val = assign_mixed_bcs(nodes, nodal_normals)
        node_to_elems, node_elem_counts = build_connectivity(elems, len(nodes))

        H, G = assemble_system(nodes, elems, areas, normals, centroids, node_to_elems, node_elem_counts)
        enforce_rbm(H)
        A, b = apply_bcs(H, G, bc_type, bc_val)
        t_setup = time.time() - t0

        t1 = time.time(); x_sol = np.linalg.solve(A, b); t_solve = time.time() - t1
        u_bound, q_bound = extract_results(x_sol, bc_type, bc_val)

        t2 = time.time()
        u_num_eval = evaluate_interior(eval_pts, nodes, elems, u_bound, q_bound, areas, normals)
        t_eval = time.time() - t2

        err_l2 = np.linalg.norm(u_num_eval - u_exact_eval) / np.linalg.norm(u_exact_eval)
        print(f"{N:<4} | {len(elems):<7} | {len(nodes):<7} | {err_l2:<14.6e} | {t_setup:<9.4f} | {t_solve:<9.4f} | {t_eval:<8.4f} | {time.time() - t0:<9.4f}")