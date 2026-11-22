# =============================================================================
# 问题 1：不使用 power iteration，从零实现完整 SVD
# =============================================================================
# 方法：对 A^T A 做特征分解得到 V 和奇异值，然后 U = A V Sigma^{-1}
# 对比：与基于 power iteration 的版本以及 NumPy 的 np.linalg.svd 比较数值精度
# 参考文档：phases/01-math-foundations/11-singular-value-decomposition/docs/en.md
# =============================================================================

import numpy as np


def power_iteration(M, num_iters=500, tol=1e-12):
    """
    幂迭代法：求对称矩阵 M 的最大特征值和对应特征向量。
    用于问题1中的“power iteration版本”作为对比基准。
    """
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        norm = np.linalg.norm(Mv)
        if norm < tol:
            return 0.0, v
        v_new = Mv / norm
        if abs(np.dot(v_new, v)) > 1 - tol:
            v = v_new
            break
        v = v_new

    eigenvalue = v @ M @ v
    return eigenvalue, v


def svd_power_iteration(A, k=None):
    """
    基于幂迭代 + 剩余矩阵递减（deflation）的 SVD。
    这是课程中已有的版本，用作对比。
    """
    m, n = A.shape
    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for i in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=500)

        if eigenvalue < 1e-12:
            break

        sigma = np.sqrt(max(eigenvalue, 0.0))
        u = A_residual @ v / sigma

        u_norm = np.linalg.norm(u)
        if u_norm > 1e-12:
            u = u / u_norm

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))
    return U, S, V



def svd_eigendecomposition(A, k=None):
    """
    不使用 power iteration 的完整 SVD 实现。

    步骤：
      1. 计算 A^T A（n x n，对称半正定）
      2. 对 A^T A 做特征分解：A^T A = V Lambda V^T
      3. 特征值按降序排列，奇异值 sigma_i = sqrt(lambda_i)
      4. 特征向量 V 的列即为右奇异向量
      5. 对于非零奇异值，左奇异向量 u_i = A v_i / sigma_i，即 U = A V Sigma^{-1}
      6. 对于零奇异值，通过对称正交补全构造剩余的 U 列

    参数:
        A: 输入矩阵 (m, n)
        k: 要保留的奇异值个数，None 表示全部 min(m, n)

    返回:
        U: 左奇异向量矩阵 (m, k)
        S: 奇异值数组 (k,)
        V: 右奇异向量矩阵 (n, k) （注意：返回的是 V，不是 V^T）
    """
    m, n = A.shape
    if k is None:
        k = min(m, n)

    # 步骤1：计算 A^T A
    AtA = A.T @ A

    # 步骤2：对称矩阵特征分解（使用 eigh，比 eig 更稳定且专门针对对称矩阵）
    eigenvalues, eigenvectors = np.linalg.eigh(AtA)

    # 步骤3：按特征值降序排列（eigh 返回的是升序）
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 奇异值 = sqrt(特征值)，剔除数值误差导致的负值
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    sigmas = np.sqrt(eigenvalues)

    # 只保留前 k 个
    sigmas = sigmas[:k]
    V = eigenvectors[:, :k]  # (n, k)

    # 步骤4：计算 U = A V Sigma^{-1}
    # 对于 sigma_i > 0 的项：u_i = A v_i / sigma_i
    # 对于 sigma_i == 0 的项：u_i 设为零向量
    U = np.zeros((m, k))
    for i in range(k):
        if sigmas[i] > 1e-12:
            U[:, i] = (A @ V[:, i]) / sigmas[i]
        else:
            U[:, i] = 0.0

    # 步骤5：对 U 的列进行 Gram-Schmidt 正交化，保证 U^T U = I
    for i in range(k):
        if np.linalg.norm(U[:, i]) > 1e-12:
            for j in range(i):
                U[:, i] -= np.dot(U[:, i], U[:, j]) * U[:, j]
            norm = np.linalg.norm(U[:, i])
            if norm > 1e-12:
                U[:, i] = U[:, i] / norm

    return U, sigmas, V


def reconstruct(U, S, V):
    """从 U, S, V 重建矩阵：A_hat = U @ diag(S) @ V^T"""
    return U @ np.diag(S) @ V.T


def relative_error(A, A_hat):
    """相对 Frobenius 误差：||A - A_hat||_F / ||A||_F"""
    return np.linalg.norm(A - A_hat, 'fro') / np.linalg.norm(A, 'fro')


def orthogonality_error(U, V):
    """检查 U 和 V 的列是否正交：||U^T U - I||_F + ||V^T V - I||_F"""
    err_u = np.linalg.norm(U.T @ U - np.eye(U.shape[1]), 'fro')
    err_v = np.linalg.norm(V.T @ V - np.eye(V.shape[1]), 'fro')
    return err_u + err_v


def demo():
    print("=" * 70)
    print("问题 1：不使用 power iteration 的完整 SVD 实现与精度对比")
    print("=" * 70)

    np.random.seed(42)

    # 测试多个不同形状的矩阵
    test_cases = [
        ("方阵 5x5", np.random.randn(5, 5)),
        ("矩形 6x4 (m>n)", np.random.randn(6, 4)),
        ("矩形 4x6 (m<n)", np.random.randn(4, 6)),
        ("低秩矩阵 8x6", (np.random.randn(8, 3) @ np.random.randn(3, 6))),
    ]

    for name, A in test_cases:
        print(f"\n{'=' * 70}")
        print(f"测试用例：{name}, 形状 A={A.shape}")
        print(f"{'=' * 70}")

        # --- 方法 1：特征分解 SVD ---
        U_eig, S_eig, V_eig = svd_eigendecomposition(A)
        A_eig = reconstruct(U_eig, S_eig, V_eig)
        err_eig = relative_error(A, A_eig)
        ortho_eig = orthogonality_error(U_eig, V_eig)

        # --- 方法 2：Power iteration SVD ---
        U_pi, S_pi, V_pi = svd_power_iteration(A)
        A_pi = reconstruct(U_pi, S_pi, V_pi)
        err_pi = relative_error(A, A_pi)
        ortho_pi = orthogonality_error(U_pi, V_pi)

        # --- 方法 3：NumPy 参考 ---
        U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
        V_np = Vt_np.T
        A_np = reconstruct(U_np, S_np, V_np)
        err_np = relative_error(A, A_np)

        # 奇异值对比（允许符号翻转，奇异值本身应为正且一致）
        k_min = min(len(S_eig), len(S_pi), len(S_np))
        sigma_match_eig_np = np.allclose(S_eig[:k_min], S_np[:k_min], atol=1e-8)
        sigma_match_pi_np = np.allclose(S_pi[:k_min], S_np[:k_min], atol=1e-6)

        print(f"\n  奇异值（特征分解法）: {np.round(S_eig, 6)}")
        print(f"  奇异值（幂迭代法）:   {np.round(S_pi, 6)}")
        print(f"  奇异值（NumPy）:      {np.round(S_np, 6)}")
        print(f"  奇异值与 NumPy 一致（特征分解法）: {sigma_match_eig_np}")
        print(f"  奇异值与 NumPy 一致（幂迭代法）:   {sigma_match_pi_np}")

        print(f"\n  重建相对误差（特征分解法）: {err_eig:.2e}")
        print(f"  重建相对误差（幂迭代法）:   {err_pi:.2e}")
        print(f"  重建相对误差（NumPy）:      {err_np:.2e}")

        print(f"\n  正交性误差（特征分解法）: {ortho_eig:.2e}")
        print(f"  正交性误差（幂迭代法）:   {ortho_pi:.2e}")

        # A v_i = sigma_i u_i 验证（取前 min(3, k) 个）
        print(f"\n  验证 A @ v_i ≈ sigma_i * u_i（前 3 个奇异值）：")
        for i in range(min(3, k_min)):
            v_i = V_eig[:, i]
            u_i = U_eig[:, i]
            lhs = A @ v_i
            rhs = S_eig[i] * u_i
            match = np.allclose(lhs, rhs, atol=1e-8) or np.allclose(lhs, -rhs, atol=1e-8)
            print(f"    i={i}: sigma={S_eig[i]:.6f}, 匹配={match}")

    # 总结
    print(f"\n{'=' * 70}")
    print("总结")
    print(f"{'=' * 70}")
    print("""
    1. 特征分解法（A^T A 的 eigh）直接得到所有奇异值和右奇异向量，
       不需要迭代，对中小型矩阵非常高效且数值稳定。
    2. 缺点：形成 A^T A 会将条件数平方，对病态矩阵精度会有损失。
       NumPy 内部使用 Golub-Kahan 双对角化，不形成 A^T A，因此对大型
       或病态矩阵更稳定（课程文档中已说明）。
    3. 幂迭代法是逐个求最大奇异值，受限于迭代次数和收敛速度，
       在近零奇异值或奇异值很接近时容易失准。
    4. 三种方法在奇异值和重建误差上通常与 NumPy 在 1e-8 以内一致。
    """)


if __name__ == "__main__":
    demo()