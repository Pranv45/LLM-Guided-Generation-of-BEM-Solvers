import numpy as np
from numba import njit, prange
import time
import math

def generate_mesh(N):
    nodes = []
    node_map = {}
    elems = []

    def get_node_id(x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append([x, y, z])
        return node_map[key]

    # Defines faces by Origin, U_vec, V_vec to ensure OUTWARD normal on unit cube
    faces = [
        ((0,0,0), (0,1,0), (1,0,0)), # z=0
        ((0,0,1), (1,0,0), (0,1,0)), # z=1
        ((0,0,0), (1,0,0), (0,0,1)), # y=0
        ((0,1,0), (0,0,1), (1,0,0)), # y=1
        ((0,0,0), (0,0,1), (0,1,0)), # x=0
        ((1,0,0), (0,1,0), (0,0,1)), # x=1
    ]

    for origin, u_vec, v_vec in faces:
        o = np.array(origin, dtype=np.float64)
        u = np.array(u_vec, dtype=np.float64)
        v = np.array(v_vec, dtype=np.float64)
        for i in range(N):
            for j in range(N):
                p00 = o + (i/N)*u + (j/N)*v
                p10 = o + ((i+1)/N)*u + (j/N)*v
                p01 = o + (i/N)*u + ((j+1)/N)*v
                p11 = o + ((i+1)/N)*u + ((j+1)/N)*v

                n00 = get_node_id(*p00)
                n10 = get_node_id(*p10)
                n01 = get_node_id(*p01)
                n11 = get_node_id(*p11)

                elems.append([n00, n10, n11])
                elems.append([n00, n11, n01])

    nodes = np.array(nodes, dtype=np.float64)
    elems = np.array(elems, dtype=np.int32)

    # Skewed mapping: X = x + 0.5y, Y = y + 0.5z, Z = z
    X = nodes[:, 0] + 0.5 * nodes[:, 1]
    Y = nodes[:, 1] + 0.5 * nodes[:, 2]
    Z = nodes[:, 2]

    mapped_nodes = np.column_stack((X, Y, Z))
    return mapped_nodes, elems

def compute_geometry(nodes, elems):
    v0 = nodes[elems[:, 0]]
    v1 = nodes[elems[:, 1]]
    v2 = nodes[elems[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (2.0 * areas[:, np.newaxis])

    return normals, areas

def exact_solution(X, Y, Z):
    return np.sinh(np.sqrt(2) * X) * np.sin(Y) * np.cos(Z)

def get_dunavant_7():
    w1 = 0.225
    p1 = (1/3, 1/3, 1/3)
    w2 = 0.132394152788506
    a2 = 0.059715871789770
    b2 = 0.470142064105115
    w3 = 0.125939180544827
    a3 = 0.797426985353087
    b3 = 0.101286507323456

    pts = np.array([
        p1,
        (a2, b2, b2), (b2, a2, b2), (b2, b2, a2),
        (a3, b3, b3), (b3, a3, b3), (b3, b3, a3)
    ])
    wts = np.array([w1, w2, w2, w2, w3, w3, w3])
    return pts, wts

def precompute_quadrature(nodes, elems, areas):
    pts, wts = get_dunavant_7()
    Ne = len(elems)
    qx = np.zeros((Ne, 7), dtype=np.float64)
    qy = np.zeros((Ne, 7), dtype=np.float64)
    qz = np.zeros((Ne, 7), dtype=np.float64)
    qw = np.zeros((Ne, 7), dtype=np.float64)

    phi0 = np.zeros(7, dtype=np.float64)
    phi1 = np.zeros(7, dtype=np.float64)
    phi2 = np.zeros(7, dtype=np.float64)

    for k in range(7):
        u, v, w = pts[k]
        phi0[k] = 1.0 - u - v
        phi1[k] = u
        phi2[k] = v

    v0 = nodes[elems[:, 0]]
    v1 = nodes[elems[:, 1]]
    v2 = nodes[elems[:, 2]]

    for i in range(Ne):
        for k in range(7):
            qx[i, k] = phi0[k]*v0[i, 0] + phi1[k]*v1[i, 0] + phi2[k]*v2[i, 0]
            qy[i, k] = phi0[k]*v0[i, 1] + phi1[k]*v1[i, 1] + phi2[k]*v2[i, 1]
            qz[i, k] = phi0[k]*v0[i, 2] + phi1[k]*v1[i, 2] + phi2[k]*v2[i, 2]
            qw[i, k] = wts[k] * areas[i]

    return qx, qy, qz, qw, phi0, phi1, phi2

@njit
def build_connectivity(elems, Nn):
    node_elem_count = np.zeros(Nn, dtype=np.int32)
    for e in range(len(elems)):
        for k in range(3):
            node_elem_count[elems[e, k]] += 1

    max_elems = np.max(node_elem_count)
    node_to_elems = np.full((Nn, max_elems), -1, dtype=np.int32)
    current_count = np.zeros(Nn, dtype=np.int32)

    for e in range(len(elems)):
        for k in range(3):
            n = elems[e, k]
            idx = current_count[n]
            node_to_elems[n, idx] = e
            current_count[n] += 1

    return node_to_elems, current_count

@njit
def compute_nodal_areas(elems, areas, Nn):
    nodal_areas = np.zeros(Nn, dtype=np.float64)
    for e in range(len(elems)):
        A3 = areas[e] / 3.0
        nodal_areas[elems[e, 0]] += A3
        nodal_areas[elems[e, 1]] += A3
        nodal_areas[elems[e, 2]] += A3
    return nodal_areas

@njit(parallel=True, fastmath=True)
def assemble_system(elems, normals, qx, qy, qz, qw, phi0, phi1, phi2, node_to_elems, node_elem_count, Nn, nodal_areas):
    Ne = len(elems)
    H = np.zeros((Nn, Nn), dtype=np.float64)
    pi4 = 4.0 * math.pi

    for i in prange(Nn):
        count = node_elem_count[i]
        for c in range(count):
            e_test = node_to_elems[i, c]

            if elems[e_test, 0] == i:
                local_i = 0
            elif elems[e_test, 1] == i:
                local_i = 1
            else:
                local_i = 2

            for e_trial in range(Ne):
                if e_test == e_trial:
                    continue

                n0_s = elems[e_trial, 0]
                n1_s = elems[e_trial, 1]
                n2_s = elems[e_trial, 2]

                njx = normals[e_trial, 0]
                njy = normals[e_trial, 1]
                njz = normals[e_trial, 2]

                sum_0 = 0.0
                sum_1 = 0.0
                sum_2 = 0.0

                for u in range(7):
                    xx = qx[e_test, u]
                    xy = qy[e_test, u]
                    xz = qz[e_test, u]
                    wx = qw[e_test, u]

                    if local_i == 0:
                        phi_t = phi0[u]
                    elif local_i == 1:
                        phi_t = phi1[u]
                    else:
                        phi_t = phi2[u]

                    for v in range(7):
                        yx = qx[e_trial, v]
                        yy = qy[e_trial, v]
                        yz = qz[e_trial, v]
                        wy = qw[e_trial, v]

                        dx = xx - yx
                        dy = xy - yy
                        dz = xz - yz

                        r2 = dx*dx + dy*dy + dz*dz
                        r = math.sqrt(r2)
                        r3 = r2 * r
                        dot = dx*njx + dy*njy + dz*njz

                        kernel = (wx * wy * dot / r3) * phi_t

                        sum_0 += kernel * phi0[v]
                        sum_1 += kernel * phi1[v]
                        sum_2 += kernel * phi2[v]

                H[i, n0_s] += sum_0 / pi4
                H[i, n1_s] += sum_1 / pi4
                H[i, n2_s] += sum_2 / pi4

        # CRITICAL FIX: The Galerkin row sum must exactly equal the negative row sum
        # of the Mass matrix (-int phi_i dGamma) to deflate the null space.
        sum_H = 0.0
        for j in range(Nn):
            if i != j:
                sum_H += H[i, j]
        H[i, i] = -nodal_areas[i] - sum_H

    return H

def assemble_mass_matrix(elems, areas, Nn):
    M = np.zeros((Nn, Nn), dtype=np.float64)
    for e in range(len(elems)):
        n0, n1, n2 = elems[e]
        A = areas[e]
        diag = A / 6.0
        off = A / 12.0
        M[n0, n0] += diag; M[n1, n1] += diag; M[n2, n2] += diag
        M[n0, n1] += off; M[n0, n2] += off
        M[n1, n0] += off; M[n1, n2] += off
        M[n2, n0] += off; M[n2, n1] += off
    return M

@njit(parallel=True, fastmath=True)
def evaluate_interior(eval_pts, mu, elems, normals, qx, qy, qz, qw, phi0, phi1, phi2):
    N_eval = len(eval_pts)
    Ne = len(elems)
    u_eval = np.zeros(N_eval, dtype=np.float64)
    pi4 = 4.0 * math.pi

    for i in prange(N_eval):
        ex = eval_pts[i, 0]
        ey = eval_pts[i, 1]
        ez = eval_pts[i, 2]

        u_val = 0.0
        for e in range(Ne):
            n0 = elems[e, 0]
            n1 = elems[e, 1]
            n2 = elems[e, 2]

            njx = normals[e, 0]
            njy = normals[e, 1]
            njz = normals[e, 2]

            for k in range(7):
                dx = ex - qx[e, k]
                dy = ey - qy[e, k]
                dz = ez - qz[e, k]

                r2 = dx*dx + dy*dy + dz*dz
                r = math.sqrt(r2)
                r3 = r2 * r
                dot = dx*njx + dy*njy + dz*njz

                kernel = qw[e, k] * dot / r3
                u_val += kernel * (mu[n0]*phi0[k] + mu[n1]*phi1[k] + mu[n2]*phi2[k]) / pi4

        u_eval[i] = u_val

    return u_eval

def run_simulation():
    # Warmup Numba kernels
    nodes, elems = generate_mesh(2)
    normals, areas = compute_geometry(nodes, elems)
    qx, qy, qz, qw, phi0, phi1, phi2 = precompute_quadrature(nodes, elems, areas)
    Nn = len(nodes)
    node_to_elems, node_elem_count = build_connectivity(elems, Nn)
    nodal_areas = compute_nodal_areas(elems, areas, Nn)
    H = assemble_system(elems, normals, qx, qy, qz, qw, phi0, phi1, phi2, node_to_elems, node_elem_count, Nn, nodal_areas)
    eval_pts = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
    mu = np.zeros(Nn, dtype=np.float64)
    evaluate_interior(eval_pts, mu, elems, normals, qx, qy, qz, qw, phi0, phi1, phi2)

    N_list = [8, 16, 32]
    results = []

    lin = np.linspace(0.2, 0.8, 5)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing='ij')
    X_eval = xx.ravel() + 0.5 * yy.ravel()
    Y_eval = yy.ravel() + 0.5 * zz.ravel()
    Z_eval = zz.ravel()
    eval_pts = np.column_stack((X_eval, Y_eval, Z_eval))
    u_eval_exact = exact_solution(X_eval, Y_eval, Z_eval)

    for N in N_list:
        t0 = time.time()
        nodes, elems = generate_mesh(N)
        Nn = len(nodes)
        normals, areas = compute_geometry(nodes, elems)
        qx, qy, qz, qw, phi0, phi1, phi2 = precompute_quadrature(nodes, elems, areas)
        node_to_elems, node_elem_count = build_connectivity(elems, Nn)
        nodal_areas = compute_nodal_areas(elems, areas, Nn)

        u_bc = exact_solution(nodes[:, 0], nodes[:, 1], nodes[:, 2])
        M = assemble_mass_matrix(elems, areas, Nn)
        b = M @ u_bc

        H = assemble_system(elems, normals, qx, qy, qz, qw, phi0, phi1, phi2, node_to_elems, node_elem_count, Nn, nodal_areas)

        t1 = time.time()

        mu = np.linalg.solve(H, b)
        t2 = time.time()

        u_eval_num = evaluate_interior(eval_pts, mu, elems, normals, qx, qy, qz, qw, phi0, phi1, phi2)
        t3 = time.time()

        setup_time = t1 - t0
        solve_time = t2 - t1
        eval_time = t3 - t2
        total_time = t3 - t0

        err_l2 = np.linalg.norm(u_eval_num - u_eval_exact) / np.linalg.norm(u_eval_exact)
        results.append((N, len(elems), err_l2, setup_time, solve_time, eval_time, total_time))

    slope = math.log(results[2][2] / results[0][2]) / math.log(8.0 / 32.0)

    print("N    | Ne      | Rel L2 Error   | Setup (s) | Solve (s) | Eval (s) | Total (s)")
    for r in results:
        sn = str(r[0]).ljust(4)
        sne = str(r[1]).ljust(7)
        serr = f"{r[2]:.6e}".ljust(14)
        sst = f"{r[3]:.4f}".ljust(9)
        ssol = f"{r[4]:.4f}".ljust(9)
        sev = f"{r[5]:.4f}".ljust(8)
        stot = f"{r[6]:.4f}"
        print(f"{sn} | {sne} | {serr} | {sst} | {ssol} | {sev} | {stot}")

    print("Convergence Analysis:")
    print(f"Computed Slope: {slope:.4f}")
    print("Expected Slope: ~2.0000 (O(h^2) for linear elements)")

if __name__ == "__main__":
    run_simulation()