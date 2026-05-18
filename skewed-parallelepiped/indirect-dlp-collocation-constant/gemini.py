import numpy as np
from numba import njit, prange
import time
import math

def generate_mesh(N):
    nodes = []
    node_map = {}
    elems = []

    def get_node_id(x, y, z):
        key = (round(x, 5), round(y, 5), round(z, 5))
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append([x, y, z])
        return node_map[key]

    # Defines faces by Origin, U_vec, V_vec to ensure OUTWARD normal on unit cube
    faces = [
        ((0,0,0), (0,1,0), (1,0,0)), # z=0, outward normal: (0,0,-1)
        ((0,0,1), (1,0,0), (0,1,0)), # z=1, outward normal: (0,0,1)
        ((0,0,0), (1,0,0), (0,0,1)), # y=0, outward normal: (0,-1,0)
        ((0,1,0), (0,0,1), (1,0,0)), # y=1, outward normal: (0,1,0)
        ((0,0,0), (0,0,1), (0,1,0)), # x=0, outward normal: (-1,0,0)
        ((1,0,0), (0,1,0), (0,0,1)), # x=1, outward normal: (1,0,0)
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

                # 2 triangles per square face preserving orientation
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
    centroids = (v0 + v1 + v2) / 3.0

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (2.0 * areas[:, np.newaxis])

    return centroids, normals, areas, v0, v1, v2

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

def precompute_quadrature(v0, v1, v2, areas):
    pts, wts = get_dunavant_7()
    Ne = len(areas)
    qx = np.zeros((Ne, 7), dtype=np.float64)
    qy = np.zeros((Ne, 7), dtype=np.float64)
    qz = np.zeros((Ne, 7), dtype=np.float64)
    qw = np.zeros((Ne, 7), dtype=np.float64)

    for i in range(Ne):
        for k in range(7):
            u, v, w = pts[k]
            qx[i, k] = u * v0[i, 0] + v * v1[i, 0] + w * v2[i, 0]
            qy[i, k] = u * v0[i, 1] + v * v1[i, 1] + w * v2[i, 1]
            qz[i, k] = u * v0[i, 2] + v * v1[i, 2] + w * v2[i, 2]
            qw[i, k] = wts[k] * areas[i]
    return qx, qy, qz, qw

@njit(parallel=True, fastmath=True)
def assemble_system(centroids, normals, qx, qy, qz, qw):
    Ne = len(centroids)
    H = np.zeros((Ne, Ne), dtype=np.float64)
    pi4 = 4.0 * math.pi

    for i in prange(Ne):
        cx = centroids[i, 0]
        cy = centroids[i, 1]
        cz = centroids[i, 2]

        for j in range(Ne):
            if i == j:
                # CORRECTED JUMP TERM: -0.5 is required for the interior DLP limit
                # mapped against outward-facing normals.
                H[i, i] = -0.5
            else:
                njx = normals[j, 0]
                njy = normals[j, 1]
                njz = normals[j, 2]

                sum_h = 0.0

                for k in range(7):
                    # X - Y (Target - Source)
                    dx = cx - qx[j, k]
                    dy = cy - qy[j, k]
                    dz = cz - qz[j, k]
                    w = qw[j, k]

                    r2 = dx*dx + dy*dy + dz*dz
                    r = math.sqrt(r2)
                    r3 = r2 * r

                    dot = dx*njx + dy*njy + dz*njz
                    sum_h += (w * dot / r3)

                H[i, j] = sum_h / pi4

    return H

@njit(parallel=True, fastmath=True)
def evaluate_interior(eval_pts, mu, normals, qx, qy, qz, qw):
    N_eval = len(eval_pts)
    Ne = len(normals)
    u_eval = np.zeros(N_eval, dtype=np.float64)
    pi4 = 4.0 * math.pi

    for i in prange(N_eval):
        ex = eval_pts[i, 0]
        ey = eval_pts[i, 1]
        ez = eval_pts[i, 2]

        u_val = 0.0
        for j in range(Ne):
            njx = normals[j, 0]
            njy = normals[j, 1]
            njz = normals[j, 2]

            sum_h = 0.0

            for k in range(7):
                # X - Y (Target - Source)
                dx = ex - qx[j, k]
                dy = ey - qy[j, k]
                dz = ez - qz[j, k]
                w = qw[j, k]

                r2 = dx*dx + dy*dy + dz*dz
                r = math.sqrt(r2)
                r3 = r2 * r

                dot = dx*njx + dy*njy + dz*njz
                sum_h += (w * dot / r3)

            H_ij = sum_h / pi4
            u_val += H_ij * mu[j]

        u_eval[i] = u_val

    return u_eval

def run_simulation():
    # Warmup Numba kernels
    nodes, elems = generate_mesh(2)
    centroids, normals, areas, v0, v1, v2 = compute_geometry(nodes, elems)
    qx, qy, qz, qw = precompute_quadrature(v0, v1, v2, areas)
    H = assemble_system(centroids, normals, qx, qy, qz, qw)
    eval_pts = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
    mu = np.zeros(len(centroids), dtype=np.float64)
    evaluate_interior(eval_pts, mu, normals, qx, qy, qz, qw)

    N_list = [8, 16, 32]
    results = []

    # 5x5x5 mapped interior points
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
        centroids, normals, areas, v0, v1, v2 = compute_geometry(nodes, elems)
        qx, qy, qz, qw = precompute_quadrature(v0, v1, v2, areas)

        u_bc = exact_solution(centroids[:, 0], centroids[:, 1], centroids[:, 2])
        H = assemble_system(centroids, normals, qx, qy, qz, qw)

        t1 = time.time()

        mu = np.linalg.solve(H, u_bc)
        t2 = time.time()

        u_eval_num = evaluate_interior(eval_pts, mu, normals, qx, qy, qz, qw)
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
    print("Expected Slope: ~1.0000 (O(h) for constant elements)")

if __name__ == "__main__":
    run_simulation()