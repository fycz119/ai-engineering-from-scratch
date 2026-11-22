# =============================================================================
# 问题 3：基于 SVD 的小型推荐系统
# =============================================================================
# 创建 10x8 用户-电影评分矩阵（部分条目已知，缺失设为 0）
# 用行均值填充缺失条目，计算秩-3 SVD 近似，预测缺失评分。
# =============================================================================

import numpy as np


def create_ratings_matrix(n_users=10, n_movies=8, seed=42):
    """
    创建用户-电影评分矩阵，评分范围 1-5。
    一些条目为已知评分，其余为缺失（标记为 0，后续填充）。

    为制造低秩结构，评分 = 用户偏好 + 电影类型 + 噪声。
    用户分为 3 类（动作片爱好者、喜剧片爱好者、文艺片爱好者），
    电影分为 3 类，同类用户对同类电影评分高。

    参数:
        n_users: 用户数
        n_movies: 电影数
        seed: 随机种子

    返回:
        R: 评分矩阵 (n_users, n_movies)，缺失处为 0
        mask: 布尔掩码，True 表示该条目是已知评分
    """
    rng = np.random.default_rng(seed)

    # 用户潜在偏好 (n_users, 3)：3 类用户
    user_pref = rng.standard_normal((n_users, 3))
    # 电影潜在特征 (n_movies, 3)：3 类电影
    movie_feat = rng.standard_normal((n_movies, 3))

    # 基础评分 = 用户偏好 @ 电影特征 + 偏置 + 噪声
    base = user_pref @ movie_feat.T  # (n_users, n_movies)
    base = (base - base.min()) / (base.max() - base.min())  # 归一化到 [0,1]
    ratings = 1 + 4 * base  # 映射到 [1, 5]
    ratings += 0.3 * rng.standard_normal((n_users, n_movies))  # 加噪声
    ratings = np.clip(ratings, 1, 5)

    # 随机标记一些条目为“已知”（约 40% 密度）
    mask = rng.random((n_users, n_movies)) < 0.4
    # 确保每行至少有一个已知评分（避免行均值为空）
    for i in range(n_users):
        if not mask[i].any():
            mask[i, rng.integers(n_movies)] = True

    R = np.zeros((n_users, n_movies))
    R[mask] = ratings[mask]

    return R, mask


def fill_missing_with_row_means(R, mask):
    """
    用每行的已知评分均值填充缺失条目（值为 0 的位置）。

    参数:
        R: 评分矩阵 (n_users, n_movies)，缺失处为 0
        mask: 布尔掩码，True 表示已知

    返回:
        R_filled: 填充后的矩阵 (n_users, n_movies)
    """
    R_filled = R.copy()
    for i in range(R.shape[0]):
        known = R[i, mask[i]]
        if known.size > 0:
            row_mean = known.mean()
            R_filled[i, ~mask[i]] = row_mean
        else:
            # 该用户没有任何已知评分，用全局均值
            R_filled[i, ~mask[i]] = 3.0
    return R_filled


def truncated_svd_reconstruct(R_filled, k=3):
    """
    对填充后的矩阵做 SVD，取前 k 个分量重建。

    返回:
        R_approx: 秩-k 近似矩阵
        U, S, Vt: SVD 分解结果
    """
    U, S, Vt = np.linalg.svd(R_filled, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    R_approx = U_k @ S_k @ Vt_k
    return R_approx, U, S, Vt


def predict_missing(R, mask, R_approx):
    """
    用秩-k 近似矩阵预测缺失条目（mask 为 False 的位置）。
    已知条目保持原值不变。

    返回:
        R_pred: 填充了预测值的完整矩阵
    """
    R_pred = R.copy()
    missing = ~mask
    R_pred[missing] = R_approx[missing]
    return R_pred


def demo():
    print("=" * 70)
    print("问题 3：基于 SVD 的小型推荐系统（秩-3 近似预测缺失评分）")
    print("=" * 70)

    n_users, n_movies = 10, 8
    R, mask = create_ratings_matrix(n_users, n_movies, seed=42)

    print(f"\n用户数：{n_users}，电影数：{n_movies}")
    print(f"已知评分密度：{mask.mean():.1%}")

    print(f"\n原始评分矩阵 R（0 表示缺失）：")
    print(np.round(R, 1))

    # 步骤 1：用行均值填充缺失值
    R_filled = fill_missing_with_row_means(R, mask)
    print(f"\n行均值填充后：")
    print(np.round(R_filled, 1))

    # 步骤 2：秩-3 SVD 重建
    k = 3
    R_approx, U, S, Vt = truncated_svd_reconstruct(R_filled, k=k)

    print(f"\n秩-{k} SVD 奇异值：{np.round(S, 4)}")
    print(f"前 {k} 个奇异值占比：{S[:k].sum() / S.sum():.2%}")

    # 步骤 3：预测缺失评分
    R_pred = predict_missing(R, mask, R_approx)

    print(f"\n预测后的完整矩阵 R_pred（缺失位已填预测值）：")
    print(np.round(R_pred, 1))

    # 步骤 4：验证——对已知条目，预测值应接近真实值
    print(f"\n{'=' * 70}")
    print("验证：在已知条目上，预测值 vs 真实值")
    print(f"{'=' * 70}")

    known_pred = R_approx[mask]
    known_true = R[mask]
    mae_known = np.mean(np.abs(known_pred - known_true))
    rmse_known = np.sqrt(np.mean((known_pred - known_true) ** 2))
    print(f"已知条目 MAE：{mae_known:.4f}")
    print(f"已知条目 RMSE：{rmse_known:.4f}")

    print(f"\n逐个已知条目对比：")
    for i in range(n_users):
        for j in range(n_movies):
            if mask[i, j]:
                true_val = R[i, j]
                pred_val = R_approx[i, j]
                print(f"  用户{i+1} -> 电影{j+1}: 真实={true_val:.1f}, 预测={pred_val:.2f}, 差={abs(true_val-pred_val):.2f}")

    # 步骤 5：查看预测的缺失评分（推荐结果）
    print(f"\n{'=' * 70}")
    print("缺失条目的预测评分（推荐结果）：")
    print(f"{'=' * 70}")

    for i in range(n_users):
        missing_movies = [j for j in range(n_movies) if not mask[i, j]]
        if missing_movies:
            preds = [(j + 1, R_pred[i, j]) for j in missing_movies]
            preds.sort(key=lambda x: -x[1])
            top = preds[:3]
            rec_str = ", ".join([f"电影{m}({p:.2f})" for m, p in top])
            print(f"  用户{i+1} 推荐：{rec_str}")

    # 合理性检查：预测值应在 [1, 5] 范围内
    pred_values = R_pred[~mask]
    print(f"\n{'=' * 70}")
    print("合理性检查")
    print(f"{'=' * 70}")
    print(f"预测值范围：[{pred_values.min():.2f}, {pred_values.max():.2f}]")
    print(f"所有预测值在 [1, 5] 之间：{np.all((pred_values >= 1) & (pred_values <= 5))}")
    print(f"""
    结论：
      - 秩-3 SVD 能够捕捉用户-电影评分矩阵的低秩结构。
      - 已知条目的预测误差较小（MAE 通常 < 1.0），说明模型拟合合理。
      - 缺失评分的预测值落在 [1, 5] 范围内，可作为推荐依据。
      - 行均值填充是一种简单的缺失值处理，实际系统中可用更复杂的方法
        （如仅对已知条目做 SVD 矩阵分解，见 Funk SVD）。
    """)


if __name__ == "__main__":
    demo()