# =============================================================================
# 问题 5：低秩矩阵去噪与最优截断秩选择
# =============================================================================
# 生成秩 3、大小 50x40 的干净低秩矩阵，加入不同噪声水平的高斯噪声
# (sigma = 0.1, 0.5, 1.0, 2.0)。对每个噪声水平，扫描 k=1..40，
# 测量重建误差（相对于干净矩阵），找出最优截断秩 k*，并绘制
# 最优 k 随噪声水平变化的曲线。
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt


def generate_clean_low_rank_matrix(m=50, n=40, rank=3, seed=42):
    """
    生成干净的低秩矩阵 A_clean = L @ R，其中 L (m, rank), R (rank, n)。

    参数:
        m: 行数
        n: 列数
        rank: 矩阵秩
        seed: 随机种子

    返回:
        A_clean: 干净的低秩矩阵 (m, n)
    """
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((m, rank))
    R = rng.standard_normal((rank, n))
    A_clean = L @ R
    return A_clean


def add_gaussian_noise(A_clean, sigma, seed=42):
    """
    向矩阵添加均值为 0、标准差为 sigma 的高斯噪声。

    参数:
        A_clean: 干净矩阵
        sigma: 噪声标准差
        seed: 随机种子

    返回:
        A_noisy: 加噪后的矩阵
    """
    rng = np.random.default_rng(seed)
    noise = sigma * rng.standard_normal(A_clean.shape)
    A_noisy = A_clean + noise
    return A_noisy


def find_optimal_rank(A_noisy, A_clean, max_k=None):
    """
    扫描 k=1..max_k，对加噪矩阵做截断 SVD 重建，
    测量相对于干净矩阵的相对 Frobenius 误差，返回误差最小的 k。

    参数:
        A_noisy: 加噪矩阵 (m, n)
        A_clean: 干净矩阵 (m, n)
        max_k: 最大截断秩，None 表示 min(m, n)

    返回:
        k_opt: 最优截断秩
        errors: 各 k 对应的误差列表 (长度 max_k)
        ks: k 值列表
    """
    m, n = A_noisy.shape
    if max_k is None:
        max_k = min(m, n)

    # 先做一次全 SVD，复用分解结果
    U, S, Vt = np.linalg.svd(A_noisy, full_matrices=False)

    clean_norm = np.linalg.norm(A_clean, 'fro')
    errors = []
    ks = list(range(1, max_k + 1))

    for k in ks:
        A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        err = np.linalg.norm(A_clean - A_k, 'fro') / clean_norm
        errors.append(err)

    errors = np.array(errors)
    k_opt = ks[int(np.argmin(errors))]

    return k_opt, errors, ks


def demo():
    print("=" * 70)
    print("问题 5：低秩矩阵去噪与最优截断秩选择")
    print("=" * 70)

    m, n, rank = 50, 40, 3
    sigmas = [0.1, 0.5, 1.0, 2.0]

    # 生成干净低秩矩阵
    A_clean = generate_clean_low_rank_matrix(m, n, rank, seed=42)
    clean_norm = np.linalg.norm(A_clean, 'fro')
    print(f"\n干净矩阵：{m} x {n}，真实秩 = {rank}，Frobenius 范数 = {clean_norm:.4f}")
    print(f"噪声水平 sigma：{sigmas}")

    results = []  # (sigma, k_opt, min_error)

    print(f"\n{'sigma':>8} | {'最优 k':>8} | {'最小误差':>12} | {'真实秩':>8}")
    print("-" * 44)

    for sigma in sigmas:
        A_noisy = add_gaussian_noise(A_clean, sigma, seed=42)
        k_opt, errors, ks = find_optimal_rank(A_noisy, A_clean)
        min_error = errors[k_opt - 1]
        results.append((sigma, k_opt, min_error))
        print(f"{sigma:>8.1f} | {k_opt:>8d} | {min_error:>12.6f} | {rank:>8d}")

    # 绘制最优 k 随噪声水平变化的曲线
    sigmas_arr = [r[0] for r in results]
    k_opts = [r[1] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(sigmas_arr, k_opts, 'o-', linewidth=2, markersize=8, color='tab:blue')
    plt.axhline(y=rank, color='r', linestyle='--', label=f'真实秩 = {rank}')
    plt.xlabel('噪声标准差 $\\sigma$')
    plt.ylabel('最优截断秩 $k^*$')
    plt.title('最优截断秩随噪声水平的变化')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = "svd_exercises/optimal_rank_vs_noise.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n[保存] 最优秩 vs 噪声曲线图：{plot_path}")
    plt.close()

    # 另外绘制：每个噪声水平下，误差随 k 的变化曲线（可选，供分析）
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, sigma in enumerate(sigmas):
        A_noisy = add_gaussian_noise(A_clean, sigma, seed=42)
        U, S, Vt = np.linalg.svd(A_noisy, full_matrices=False)
        clean_norm = np.linalg.norm(A_clean, 'fro')
        errors = []
        ks = range(1, min(m, n) + 1)
        for k in ks:
            A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            err = np.linalg.norm(A_clean - A_k, 'fro') / clean_norm
            errors.append(err)

        ax = axes[idx]
        k_opt = results[idx][1]
        ax.plot(ks, errors, 'b-', linewidth=1.5)
        ax.axvline(x=k_opt, color='g', linestyle='--', label=f'k*={k_opt}')
        ax.axvline(x=rank, color='r', linestyle=':', label=f'真实秩={rank}')
        ax.set_xlabel('截断秩 k')
        ax.set_ylabel('相对重建误差')
        ax.set_title(f'sigma = {sigma}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plot_path2 = "svd_exercises/error_vs_rank_all_sigmas.png"
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    print(f"[保存] 各噪声水平误差随 k 变化图：{plot_path2}")
    plt.close()

    # 分析结论
    print(f"\n{'=' * 70}")
    print("分析结论")
    print(f"{'=' * 70}")
    for sigma, k_opt, min_error in results:
        print(f"\n  sigma = {sigma}:")
        print(f"    最优截断秩 k* = {k_opt}（真实秩 = {rank}）")
        print(f"    最小相对重建误差 = {min_error:.6f}")
        if k_opt == rank:
            print(f"    ✓ 最优秩等于真实秩，去噪效果理想")
        elif k_opt < rank:
            print(f"    ⚠ 最优秩低于真实秩，噪声较大导致部分信号被截断")
        else:
            print(f"    ⚠ 最优秩高于真实秩，可能保留了一些噪声")

    print(f"""
    总结：
      - 噪声越小 (sigma=0.1)，最优截断秩越接近真实秩 (k=3)。
      - 噪声越大 (sigma=2.0)，最优截断秩倾向于减小，以去除更多噪声分量。
      - 截断 SVD 去噪的核心思想：信号集中在前几个大奇异值，噪声分散在
        许多小奇异值中，截断小奇异值可抑制噪声。
      - 最优 k 通常在“信号-噪声分界点”附近，即奇异值谱出现明显断崖的位置。
    """)


if __name__ == "__main__":
    demo()