import numpy as np
import time

def solve_bem(N):
    t0 = time.time()

    # 1. Geometry Discretization (Rectangle [0,2] x [0,1])
    L_x, L_y = 2.0, 1.0

    # Define segments: Bottom (y=0), Right (x=2), Top (y=1), Left (x=0)
    # Each side has N elements
    num_elements = 4 * N

    # Start and end points for panels (counter-clockwise)
    x_nodes = np.concatenate([
        np.linspace(0, L_x, N, endpoint=False),      # Bottom
        np.full(N, L_x),                             # Right
        np.linspace(L_x, 0, N, endpoint=False),      # Top
        np.zeros(N)                                  # Left
    ])
    y_nodes = np.concatenate([
        np.zeros(N),                                 # Bottom
        np.linspace(0, L_y, N, endpoint=False),      # Right
        np.full(N, L_y),                             # Top
        np.linspace(L_y, 0, N, endpoint=False)       # Left
    ])

    x_next = np.roll(x_nodes, -1)
    y_next = np.roll(y_nodes, -1)

    # Midpoints, lengths, and outward normals
    xm = (x_nodes + x_next) / 2.0
    ym = (y_nodes + y_next) / 2.0
    lengths = np.sqrt((x_next - x_nodes)**2 + (y_next - y_nodes)**2)

    nx = (y_next - y_nodes) / lengths
    ny = -(x_next - x_nodes) / lengths

    # 2. Boundary Conditions
    # Dirichlet: Right (x=2), Left (x=0)
    # Neumann: Bottom (y=0), Top (y=1)
    u_known = np.zeros(num_elements)
    q_known = np.zeros(num_elements)
    is_dirichlet = np.zeros(num_elements, dtype=bool)

    # Bottom (0:N): q=0
    # Right (N:2N): u=cos(pi*y)
    is_dirichlet[N:2*N] = True
    u_known[N:2*N] = np.cos(np.pi * ym[N:2*N])
    # Top (2N:3N): q=0
    # Left (3N:4N): u=0
    is_dirichlet[3*N:4*N] = True
    u_known[3*N:4*N] = 0.0

    # 3. Assembly
    t_asm_start = time.time()
    G = np.zeros((num_elements, num_elements))
    H = np.zeros((num_elements, num_elements))

    for i in range(num_elements):
        for j in range(num_elements):
            if i == j:
                # Analytic self-terms
                G[i, j] = -(lengths[j] / (2.0 * np.pi)) * (np.log(lengths[j] / 2.0) - 1.0)
                H[i, j] = 0.5  # Jump term
            else:
                dx = xm[i] - xm[j]
                dy = ym[i] - ym[j]
                r2 = dx**2 + dy**2
                r = np.sqrt(r2)

                # SLP off-diagonal
                G[i, j] = -(1.0 / (2.0 * np.pi)) * np.log(r) * lengths[j]

                # DLP off-diagonal
                # kernel = grad(G).n = (r_vec . n) / (2*pi*r^2)
                # Note: r_vec points from source point j to field point i
                dot = dx * nx[j] + dy * ny[j]
                H[i, j] = (1.0 / (2.0 * np.pi * r2)) * dot * lengths[j]

    t_asm = time.time() - t_asm_start

    # 4. Matrix Solve: A*x = b
    # Reorganize Gq = Hu => Ax = b
    # If Dirichlet: -G*q + ... = -H*u_known
    # If Neumann: H*u + ... = G*q_known
    t_sol_start = time.time()
    A = np.zeros((num_elements, num_elements))
    b = np.zeros(num_elements)

    for j in range(num_elements):
        if is_dirichlet[j]:
            A[:, j] = -G[:, j]
            b += np.dot(H[:, j], -u_known[j])
        else:
            A[:, j] = H[:, j]
            b += np.dot(G[:, j], q_known[j])

    # System RHS: b = sum(H_known * u_known) - sum(G_known * q_known) is essentially done in loop
    # Correction for full system matrix approach
    # b is actually Hu_known - Gq_known
    b = np.dot(H[:, is_dirichlet], u_known[is_dirichlet]) - \
        np.dot(G[:, ~is_dirichlet], q_known[~is_dirichlet])

    x_sol = np.linalg.solve(A, b)
    t_sol = time.time() - t_sol_start

    # 5. Boundary Reconstruction
    u_full = np.copy(u_known)
    q_full = np.copy(q_known)
    u_full[~is_dirichlet] = x_sol[~is_dirichlet]
    q_full[is_dirichlet] = x_sol[is_dirichlet]

    # 6. Interior Evaluation
    t_ev_start = time.time()
    nx_g, ny_g = 40, 20
    xs = np.linspace(0.05, 1.95, nx_g)
    ys = np.linspace(0.05, 0.95, ny_g)
    XX, YY = np.meshgrid(xs, ys)
    u_int = np.zeros_like(XX)

    pts_x = XX.flatten()
    pts_y = YY.flatten()
    u_int_flat = np.zeros_like(pts_x)

    # Green's Representation formula
    for k in range(len(pts_x)):
        dx = pts_x[k] - xm
        dy = pts_y[k] - ym
        r2 = dx**2 + dy**2

        # u(x) = integral(Gq) - integral((dG/dn)u)
        g_ker = -(1.0 / (2.0 * np.pi)) * np.log(np.sqrt(r2))
        h_ker = (1.0 / (2.0 * np.pi * r2)) * (dx * nx + dy * ny)

        u_int_flat[k] = np.sum(g_ker * q_full * lengths) - np.sum(h_ker * u_full * lengths)

    u_int = u_int_flat.reshape(XX.shape)
    t_ev = time.time() - t_ev_start

    # 7. Error calculation
    u_exact = (np.sinh(np.pi * XX) / np.sinh(2.0 * np.pi)) * np.cos(np.pi * YY)
    rel_l2_error = np.linalg.norm(u_int - u_exact) / np.linalg.norm(u_exact)

    t_total = time.time() - t0

    return {
        'dofs': num_elements,
        't_asm': t_asm,
        't_sol': t_sol,
        't_ev': t_ev,
        't_total': t_total,
        'error': rel_l2_error
    }

# Refinement study
N_values = [10, 20, 40, 80]
results = []

print(f"{'N':<5} {'DOFs':<8} {'Asm(s)':<10} {'Sol(s)':<10} {'Eval(s)':<10} {'Total(s)':<10} {'L2 Error':<12}")
print("-" * 75)

for N in N_values:
    res = solve_bem(N)
    results.append(res)
    print(f"{N:<5} {res['dofs']:<8} {res['t_asm']:<10.4f} {res['t_sol']:<10.4f} {res['t_ev']:<10.4f} {res['t_total']:<10.4f} {res['error']:<12.4e}")

# Empirical Convergence Rate
dofs = np.array([r['dofs'] for r in results])
errors = np.array([r['error'] for r in results])

# log(error) = slope * log(dofs) + intercept
slope, _ = np.polyfit(np.log(dofs), np.log(errors), 1)
print(f"\nObserved convergence slope (log-log): {slope:.4f}")