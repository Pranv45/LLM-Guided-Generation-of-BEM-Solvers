import numpy as np
import pandas as pd
import time

def solve_spectral_bem(M, n_freq=3):
    # 1. Discretization
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    f = np.cos(n_freq * theta)

    # 2. Spectral Solver (1/2 I + K) mu = f
    # Eigenvalues of K for mode k on unit circle: lambda_k = 0 for k=0, else 0
    # Actually, for the Double Layer operator K on a circle:
    # K[exp(ik theta)] = 0 for all k (analytical property for unit circle)
    # Thus the operator is simply 1/2 I

    t_solve_start = time.perf_counter()
    f_hat = np.fft.fft(f)

    # The integral operator K on the circle with double layer kernel is 0
    # for all Fourier modes except the constant mode if we consider the jump.
    # In the indirect formulation (1/2 I + K) mu = f:
    # For a circle, K mu = 0 analytically.
    # Therefore 1/2 mu = f => mu = 2f
    mu_hat = 2 * f_hat
    mu = np.fft.ifft(mu_hat).real
    t_solve = time.perf_counter() - t_solve_start

    # 3. Interior Evaluation
    # Grid for evaluation: r = 0.5
    r_int = 0.5
    phi_int = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    target_pts = np.stack([r_int * np.cos(phi_int), r_int * np.sin(phi_int)], axis=1)

    # Boundary nodes and normals
    nodes = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    normals = nodes  # Outward normal for unit circle is the node itself
    ds = 2 * np.pi / M

    t_eval_start = time.perf_counter()
    u_approx = np.zeros(len(phi_int))

    # Vectorized double layer evaluation: u(x) = integral K(x, y) mu(y) ds_y
    # K(x, y) = -1/(2pi) * (x-y).n_y / |x-y|^2
    for i, x in enumerate(target_pts):
        diff = x - nodes
        dist_sq = np.sum(diff**2, axis=1)
        dot_n = np.sum(diff * normals, axis=1)
        kernel = -1.0 / (2.0 * np.pi) * dot_n / dist_sq
        u_approx[i] = np.sum(kernel * mu) * ds

    t_eval = time.perf_counter() - t_eval_start

    # 4. Error Calculation
    u_exact = (r_int**n_freq) * np.cos(n_freq * phi_int)
    rel_l2_error = np.linalg.norm(u_approx - u_exact) / np.linalg.norm(u_exact)

    return {
        "M": M,
        "Solve Time (s)": t_solve,
        "Eval Time (s)": t_eval,
        "L2 Error": rel_l2_error
    }

def main():
    M_values = [100, 200, 400, 800, 1600]
    results = []

    for M in M_values:
        res = solve_spectral_bem(M)
        results.append(res)

    df = pd.DataFrame(results)

    print("Spectral BEM Performance Summary (Unit Circle, n=3)")
    print("=" * 60)
    print(df.to_string(index=False, formatters={
        'Solve Time (s)': '{:,.6f}'.format,
        'Eval Time (s)': '{:,.6f}'.format,
        'L2 Error': '{:.4e}'.format
    }))

if __name__ == "__main__":

    main()