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

                elems.append([n00, n10, n11])
                elems.append([n00, n11, n01])

    nodes = np.array(nodes, dtype=np.float64)
    elems = np.array(elems, dtype=np.int32)

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

@njit(parallel=True, fastmath=True)
def assemble_system(nodes, elems, normals, areas, qx, qy, qz, qw, phi0, phi1, phi2):
    Nn = len(nodes)
    Ne = len(elems)
    H = np.zeros((Nn, Nn), dtype=np.float64)

    g_pts = np.array([0.1127016653792583, 0.5, 0.8872983346207417], dtype=np.float64)
    g_wts = np.array([0.2777777777777778, 0.4444444444444444, 0.2777777777777778], dtype=np.float64)

    pi4 = 4.0 * math.pi

    for i in prange(Nn):
        cx = nodes[i, 0]
        cy = nodes[i, 1]
        cz = nodes[i, 2]

        for e in range(Ne):
            n0 = elems[e, 0]
            n1 = elems[e, 1]
            n2 = elems[e, 2]

            njx = normals[e, 0]
            njy = normals[e, 1]
            njz = normals[e, 2]

            if i == n0 or i == n1 or i == n2:
                if i == n0:
                    pA, pB, pC = n0, n1, n2
                elif i == n1:
                    pA, pB, pC = n1, n2, n0
                else:
                    pA, pB, pC = n2, n0, n1

                ax = nodes[pA, 0]; ay = nodes[pA, 1]; az = nodes[pA, 2]
                bx = nodes[pB, 0]; by = nodes[pB, 1]; bz = nodes[pB, 2]
                cx_p = nodes[pC, 0]; cy_p = nodes[pC, 1]; cz_p = nodes[pC, 2]

                As = areas[e]
                sum_B = 0.0
                sum_C = 0.0

                for u in range(3):
                    eta1 = g_pts[u]
                    w1 = g_wts[u]
                    jac = 2.0 * As * eta1

                    for v in range(3):
                        eta2 = g_pts[v]
                        w2 = g_wts[v]

                        px = ax + eta1*(bx - ax) + eta1*eta2*(cx_p - bx)
                        py = ay + eta1*(by - ay) + eta1*eta2*(cy_p - by)
                        pz = az + eta1*(bz - az) + eta1*eta2*(cz_p - bz)

                        dx = cx - px
                        dy = cy - py
                        dz = cz - pz

                        r2 = dx*dx + dy*dy + dz*dz
                        if r2 > 1e-28:
                            r = math.sqrt(r2)
                            r3 = r2 * r
                            dot = dx*njx + dy*njy + dz*njz

                            kernel = w1 * w2 * jac * dot / r3
                            sB = eta1 * (1.0 - eta2)
                            sC = eta1 * eta2

                            sum_B += sB * kernel
                            sum_C += sC * kernel

                H[i, pB] += sum_B / pi4
                H[i, pC] += sum_C / pi4
            else:
                sum_0 = 0.0
                sum_1 = 0.0
                sum_2 = 0.0

                for k in range(7):
                    dx = cx - qx[e, k]
                    dy = cy - qy[e, k]
                    dz = cz - qz[e, k]

                    r2 = dx*dx + dy*dy + dz*dz
                    r = math.sqrt(r2)
                    r3 = r2 * r

                    dot = dx*njx + dy*njy + dz*njz
                    kernel = qw[e, k] * dot / r3

                    sum_0 += phi0[k] * kernel
                    sum_1 += phi1[k] * kernel
                    sum_2 += phi2[k] * kernel

                H[i, n0] += sum_0 / pi4
                H[i, n1] += sum_1 / pi4
                H[i, n2] += sum_2 / pi4

    for i in range(Nn):
        sum_H = 0.0
        for j in range(Nn):
            if i != j:
                sum_H += H[i, j]
        # CRITICAL FIX: Ensure row sum is strictly -1 for Interior DLP
        H[i, i] = -1.0 - sum_H

    return H

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
    nodes, elems = generate_mesh(2)
    normals, areas = compute_geometry(nodes, elems)
    qx, qy, qz, qw, phi0, phi1, phi2 = precompute_quadrature(nodes, elems, areas)
    H = assemble_system(nodes, elems, normals, areas, qx, qy, qz, qw, phi0, phi1, phi2)
    eval_pts = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
    mu = np.zeros(len(nodes), dtype=np.float64)
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
        normals, areas = compute_geometry(nodes, elems)
        qx, qy, qz, qw, phi0, phi1, phi2 = precompute_quadrature(nodes, elems, areas)

        u_bc = exact_solution(nodes[:, 0], nodes[:, 1], nodes[:, 2])
        H = assemble_system(nodes, elems, normals, areas, qx, qy, qz, qw, phi0, phi1, phi2)

        t1 = time.time()

        mu = np.linalg.solve(H, u_bc)
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