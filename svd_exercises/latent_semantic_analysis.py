# =============================================================================
# 问题 4：基于 SVD 的隐语义分析（LSA）
# =============================================================================
# 创建 100x50 文档-术语矩阵，包含 3 个合成主题，每个主题 5 个关联术语。
# 加噪声后做 SVD，验证前 3 个奇异值远大于其余；
# 将文档投影到 3D 潜空间，检查同主题文档是否聚集。
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt


def create_document_term_matrix(n_docs=100, n_terms=50, n_topics=3,
                                terms_per_topic=5, seed=42):
    """
    合成文档-术语矩阵，具有明显的主题结构。

    构造方式：
      - 每个主题对应一组专属术语（terms_per_topic 个）
      - 每个文档属于某个主题，其对应术语出现频率高
      - 文档之间还有一些共享术语（背景词）
      - 加入高斯噪声

    参数:
        n_docs: 文档数
        n_terms: 术语数（词表大小）
        n_topics: 主题数
        terms_per_topic: 每个主题的专属术语数
        seed: 随机种子

    返回:
        X: 文档-术语矩阵 (n_docs, n_terms)
        doc_topics: 每个文档的主题标签 (n_docs,)
        topic_terms: 每个主题的术语索引列表
    """
    rng = np.random.default_rng(seed)

    # 分配文档到主题（均匀分布）
    docs_per_topic = n_docs // n_topics
    doc_topics = np.repeat(np.arange(n_topics), docs_per_topic)
    if len(doc_topics) < n_docs:
        doc_topics = np.append(doc_topics, rng.integers(0, n_topics,
                           n_docs - len(doc_topics)))
    rng.shuffle(doc_topics)

    # 每个主题分配专属术语索引
    topic_terms = []
    used_terms = set()
    for t in range(n_topics):
        available = [i for i in range(n_terms) if i not in used_terms]
        chosen = rng.choice(available, size=terms_per_topic, replace=False)
        topic_terms.append(chosen)
        used_terms.update(chosen)

    # 初始化矩阵
    X = np.zeros((n_docs, n_terms))

    # 背景术语（所有文档都有低频出现）
    bg_terms = rng.choice(n_terms, size=5, replace=False)

    for d in range(n_docs):
        topic = doc_topics[d]

        # 该主题的专属术语出现频率高
        for term_idx in topic_terms[topic]:
            X[d, term_idx] = rng.uniform(3, 8)

        # 其他主题的专属术语出现频率低（交叉主题）
        for other_t in range(n_topics):
            if other_t != topic:
                for term_idx in topic_terms[other_t]:
                    X[d, term_idx] = rng.uniform(0, 1)

        # 背景术语
        for term_idx in bg_terms:
            X[d, term_idx] = rng.uniform(0.5, 2.0)

        # 其他随机术语低频噪声
        for _ in range(3):
            t = rng.integers(n_terms)
            X[d, t] += rng.uniform(0, 0.5)

    # 加入高斯噪声
    noise_level = 0.5
    X += noise_level * rng.standard_normal((n_docs, n_terms))
    X = np.clip(X, 0, None)  # 词频不能为负

    return X, doc_topics, topic_terms


def demo():
    print("=" * 70)
    print("问题 4：基于 SVD 的隐语义分析（LSA）")
    print("=" * 70)

    n_docs, n_terms = 100, 50
    n_topics = 3
    terms_per_topic = 5

    X, doc_topics, topic_terms = create_document_term_matrix(
        n_docs=n_docs, n_terms=n_terms, n_topics=n_topics,
        terms_per_topic=5, seed=42)

    print(f"\n文档数：{n_docs}，术语数：{n_terms}，主题数：{n_topics}")
    print(f"每个主题的专属术语：{terms_per_topic} 个")
    for t in range(n_topics):
        print(f"  主题 {t}: 术语索引 {sorted(topic_terms[t])}")
    print(f"每个主题的文档数：{np.bincount(doc_topics)}")

    # SVD 分解
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    print(f"\n{'=' * 70}")
    print("奇异值分析")
    print(f"{'=' * 70}")
    print(f"\n所有奇异值（前 10 个）：{np.round(S[:10], 4)}")
    print(f"\n前 {n_topics} 个奇异值与其余奇异值的比值：")
    for i in range(n_topics):
        rest_mean = S[n_topics:].mean() if len(S) > n_topics else 0
        print(f"  sigma[{i}] = {S[i]:.4f}, 与其余均值比值 = {S[i] / max(rest_mean, 1e-10):.2f}x")

    top3_sum = S[:n_topics].sum()
    rest_sum = S[n_topics:].sum()
    print(f"\n前 {n_topics} 个奇异值占比：{top3_sum / S.sum():.2%}")
    print(f"其余奇异值占比：{rest_sum / S.sum():.2%}")

    # 验证：前 3 个奇异值应远大于其余
    gap = S[n_topics - 1] / S[n_topics] if n_topics < len(S) else float('inf')
    print(f"\nsigma[{n_topics-1}] / sigma[{n_topics}] = {gap:.2f}")
    if gap > 5:
        print(f"✓ 前 {n_topics} 个奇异值显著大于其余（间隙比 > 5），主题结构明显")
    else:
        print(f"⚠ 前 {n_topics} 个奇异值与其余差距不够大，主题结构可能不明显")

    # 将文档投影到 3D 潜空间（左奇异向量 U 的前 3 列，加权奇异值）
    doc_latent = U[:, :n_topics] @ np.diag(S[:n_topics])  # (n_docs, 3)
    term_latent = Vt[:n_topics, :].T @ np.diag(S[:n_topics])  # (n_terms, 3)

    print(f"\n{'=' * 70}")
    print("文档在 3D 潜空间的聚集性检查")
    print(f"{'=' * 70}")

    topic_centers = []
    for t in range(n_topics):
        docs_in_topic = doc_latent[doc_topics == t]
        center = docs_in_topic.mean(axis=0)
        topic_centers.append(center)
        print(f"  主题 {t}: 中心 = {np.round(center, 3)}")

    intra_dists = []
    inter_dists = []
    for t in range(n_topics):
        docs_in_topic = doc_latent[doc_topics == t]
        center = topic_centers[t]
        for d in docs_in_topic:
            intra_dists.append(np.linalg.norm(d - center))
        for other_t in range(n_topics):
            if other_t != t:
                inter_dists.append(np.linalg.norm(center - topic_centers[other_t]))

    intra_mean = np.mean(intra_dists)
    inter_mean = np.mean(inter_dists)
    print(f"\n平均组内距离（文档到其主题中心）：{intra_mean:.4f}")
    print(f"平均组间距离（主题中心之间）：     {inter_mean:.4f}")
    print(f"组间距离 / 组内距离比值：{inter_mean / intra_mean:.2f}")

    if inter_mean / intra_mean > 2:
        print(f"✓ 同主题文档在潜空间中聚集明显（组间距离显著大于组内距离）")
    else:
        print(f"⚠ 聚集性不明显，可能需要更多样本或降低噪声")

    # 绘制 3D 散点图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']
    labels = ['主题 0', '主题 1', '主题 2']

    for t in range(n_topics):
        mask = doc_topics == t
        ax.scatter(doc_latent[mask, 0], doc_latent[mask, 1],
                   doc_latent[mask, 2], c=colors[t], label=labels[t],
                   alpha=0.7, s=50)

    for t in range(n_topics):
        ax.scatter(topic_centers[t][0], topic_centers[t][1],
                   topic_centers[t][2], c=colors[t], marker='*',
                   s=300, edgecolors='k', linewidth=2)

    ax.set_xlabel('潜维度 1')
    ax.set_ylabel('潜维度 2')
    ax.set_zlabel('潜维度 3')
    ax.set_title('文档在 SVD 3D 潜空间的分布（星号=主题中心）')
    ax.legend()

    plot_path = "svd_exercises/lsa_3d_clusters.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n[保存] 3D 潜空间聚集图：{plot_path}")
    plt.close()

    print(f"""
    结论：
      - SVD 提取出的前 3 个奇异值对应 3 个潜在主题。
      - 文档投影到 3D 潜空间后，同主题文档聚集在一起，不同主题分离。
      - U @ diag(S) 的前 3 列即为文档在潜空间的坐标，可用于文档聚类、
        相似度计算、信息检索等下游任务。
      - 这就是“隐语义分析（Latent Semantic Analysis）”的核心思想。
    """)


if __name__ == "__main__":
    demo()