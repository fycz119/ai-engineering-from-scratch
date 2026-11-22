# =============================================================================
# 问题 2：基于 SVD 的灰度图像压缩
# =============================================================================
# 加载真实灰度图像（若无图像文件则生成合成图像），在不同秩 k 下压缩，
# 计算压缩比和相对重建误差，找出图像变得“视觉可接受”的秩。
# =============================================================================

import numpy as np
from PIL import Image
import os


def load_grayscale_image(path=None, size=(256, 256)):
    """
    加载灰度图像并转换为 numpy 数组 (H, W)，像素值归一化到 [0, 1]。

    若 path 为 None 或文件不存在，则生成一张合成“类真实”灰度图像：
    包含渐变、几何图形和纹理，模拟真实图像的低秩结构。

    参数:
        path: 图像文件路径（可选）
        size: 合成图像尺寸 (高, 宽)，仅在生成合成图像时使用

    返回:
        img: 灰度图像数组 (H, W)，浮点型，范围 [0, 1]
    """
    if path is not None and os.path.exists(path):
        print(f"[加载图像] {path}")
        img = Image.open(path).convert("L")  # 转换为灰度
        img = img.resize(size, Image.LANCZOS)
        arr = np.array(img, dtype=np.float64) / 255.0
        return arr
    else:
        if path is not None:
            print(f"[提示] 文件不存在：{path}，将生成合成灰度图像代替。")
        print(f"[生成合成图像] 尺寸={size}")

        h, w = size
        y = np.linspace(0, 1, h)[:, None]
        x = np.linspace(0, 1, w)[None, :]

        # 背景渐变
        img = 0.3 + 0.4 * y + 0.2 * x

        # 添加几个“物体”（椭圆/矩形块），制造低秩结构
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]

        # 一个大圆
        r = min(h, w) // 3
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
        img[mask] = 0.8

        # 一个小矩形
        img[h // 4:h // 4 + h // 8, w // 4:w // 4 + w // 8] = 0.2

        # 添加轻微纹理噪声
        rng = np.random.default_rng(42)
        img += 0.03 * rng.standard_normal((h, w))
        img = np.clip(img, 0.0, 1.0)

        return img


def svd_compress(U, S, Vt, k):
    """
    使用前 k 个奇异值/向量重建近似图像：A_k = U_k @ diag(S_k) @ Vt_k

    参数:
        U: 左奇异向量 (m, n)
        S: 奇异值 (n,)
        Vt: 右奇异向量的转置 (n, n)
        k: 截断秩

    返回:
        A_k: 秩-k 近似 (m, n)
    """
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    return U_k @ S_k @ Vt_k


def compression_ratio(m, n, k):
    """
    压缩比 = 压缩后存储量 / 原始存储量

    原始：m * n 个数字
    压缩后：U_k (m*k) + S_k (k) + Vt_k (k*n) = k*(m + n + 1)
    """
    original = m * n
    compressed = k * (m + n + 1)
    return compressed / original


def relative_error(A, A_hat):
    """相对 Frobenius 误差"""
    return np.linalg.norm(A - A_hat, 'fro') / np.linalg.norm(A, 'fro')


def save_compressed_image(A_hat, path):
    """将 [0,1] 浮点图像保存为 PNG"""
    img = Image.fromarray((np.clip(A_hat, 0, 1) * 255).astype(np.uint8))
    img.save(path)
    print(f"  [保存] {path}")


def demo():
    print("=" * 70)
    print("问题 2：基于 SVD 的灰度图像压缩")
    print("=" * 70)

    # 尝试加载真实图像，若无则生成合成图像
    image_path = None  # 可修改为实际图像路径，如 "photo.jpg"
    A = load_grayscale_image(image_path, size=(256, 256))
    m, n = A.shape
    print(f"\n图像尺寸：{m} x {n}，像素值范围 [{A.min():.3f}, {A.max():.3f}]")

    # NumPy SVD（全分解）
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    print(f"\n奇异值前 10 个：{np.round(S[:10], 4)}")
    print(f"奇异值总和（核范数）：{S.sum():.4f}")
    print(f"前 5 个奇异值占比：{S[:5].sum() / S.sum():.2%}")

    ranks = [1, 5, 10, 25, 50, 100]
    # 确保不超过 min(m, n)
    max_rank = min(m, n)
    ranks = [r for r in ranks if r <= max_rank]

    print(f"\n{'秩 k':>6} | {'压缩比':>10} | {'相对误差':>10} | {'存储节省':>10}")
    print("-" * 46)

    results = []
    for k in ranks:
        A_k = svd_compress(U, S, Vt, k)
        cr = compression_ratio(m, n, k)
        err = relative_error(A, A_k)
        savings = (1 - cr) * 100
        results.append((k, cr, err))
        print(f"{k:>6} | {cr:>10.4f} | {err:>10.6f} | {savings:>9.1f}%")

        # 保存压缩图像（可选，供视觉检查）
        out_path = f"svd_exercises/compressed_rank_{k}.png"
        save_compressed_image(A_k, out_path)

    # 找出“视觉可接受”的秩：相对误差低于 5% 且压缩比 < 0.5 的最小 k
    print(f"\n{'=' * 70}")
    print("视觉可接受性分析")
    print(f"{'=' * 70}")
    print("""
    判断标准（经验性）：
      - 相对误差 < 5%：主要结构已保留
      - 压缩比 < 0.5：存储量减少一半以上
    """)

    acceptable = None
    for k, cr, err in results:
        if err < 0.05 and cr < 0.5:
            acceptable = k
            break

    if acceptable:
        print(f"✓ 首个达到‘视觉可接受’的秩：k = {acceptable}")
        print(f"  压缩比：{compression_ratio(m, n, acceptable):.4f}")
        print(f"  相对误差：{relative_error(A, svd_compress(U, S, Vt, acceptable)):.6f}")
    else:
        print("✗ 未找到同时满足误差<5%且压缩比<0.5 的秩")
        # 推荐误差最小的那个
        best_k = min(results, key=lambda x: x[2])[0]
        print(f"  建议：使用 k = {best_k}（误差最小）")

    print(f"""
    视觉判断建议：
      - k=1  : 仅保留最强方向，图像几乎退化为条纹，不可接受。
      - k=5  : 能看到大致轮廓，但细节丢失严重，勉强可辨。
      - k=10 : 主要结构清晰，适合缩略图或预览。
      - k=25 : 大部分细节保留，对大多数用途已可接受。
      - k=50 : 几乎与原图无明显差异。
      - k=100: 接近无损，但压缩比较高。

    对 256x256 图像，通常 k=25 左右即可达到“视觉可接受”。
    """)

    # 保存原图供对比
    save_compressed_image(A, "svd_exercises/original.png")
    print(f"[保存] 原图：svd_exercises/original.png")


if __name__ == "__main__":
    demo()