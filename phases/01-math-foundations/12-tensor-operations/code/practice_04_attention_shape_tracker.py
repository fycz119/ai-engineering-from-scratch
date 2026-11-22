# -*- coding: utf-8 -*-
"""
练习 4：注意力形状追踪器 (Attention Shape Tracker)
=================================================

对应课程：phases/01-math-foundations/12-tensor-operations
练习描述：
    写一个函数，输入 batch_size、seq_len、embed_dim 和 num_heads，
    打印多头注意力每一步的精确形状：输入、Q/K/V 投影、头拆分、
    注意力分数、softmax 权重、加权求和、头合并、输出投影。
    与 demo_attention_einsum() 的输出对照验证。

核心概念：
    多头注意力的形状契约：
      输入 X: (B, T, E)
      Q/K/V 投影: (B, T, E)            [X @ W]
      头拆分: (B, H, T, D)             [reshape (B,T,H,D) + transpose]
      注意力分数: (B, H, T, T)         [Q @ K^T / sqrt(D)]
      softmax 权重: (B, H, T, T)
      加权求和: (B, H, T, D)           [weights @ V]
      头合并: (B, T, E)                [transpose + reshape]
      输出投影: (B, T, E)              [concat @ W_o]
    其中 E = embed_dim, D = E // num_heads, 要求 E % num_heads == 0。

运行方式：
    python practice_04_attention_shape_tracker.py
"""

import os
import sys
import numpy as np

_LESSON_CODE = os.path.dirname(os.path.abspath(__file__))
if _LESSON_CODE not in sys.path:
    sys.path.insert(0, _LESSON_CODE)


def attention_shape_tracker(batch_size, seq_len, embed_dim, num_heads):
    """打印多头注意力每一步的精确形状，返回形状记录列表。"""
    B, T, E, H = batch_size, seq_len, embed_dim, num_heads
    if E % H != 0:
        raise ValueError(f"embed_dim({E}) 必须能被 num_heads({H}) 整除")
    D = E // H

    records = []
    def rec(name, shape):
        records.append((name, tuple(shape)))
        print(f"  {name:22s} -> {tuple(shape)}")

    print("=" * 60)
    print("多头注意力形状追踪")
    print(f"  batch_size={B}, seq_len={T}, embed_dim={E}, num_heads={H}, head_dim={D}")
    print("=" * 60)

    # 1. 输入
    rec("1. 输入 X", (B, T, E))

    # 2. Q/K/V 投影
    rec("2. Q 投影", (B, T, E))
    rec("   K 投影", (B, T, E))
    rec("   V 投影", (B, T, E))

    # 3. 头拆分：(B, T, E) -> (B, T, H, D) -> (B, H, T, D)
    rec("3. 头拆分 Q", (B, H, T, D))
    rec("   头拆分 K", (B, H, T, D))
    rec("   头拆分 V", (B, H, T, D))

    # 4. 注意力分数：(B, H, T, D) x (B, H, T, D) -> (B, H, T, T)
    rec("4. 注意力分数", (B, H, T, T))

    # 5. softmax 权重
    rec("5. softmax 权重", (B, H, T, T))

    # 6. 加权求和：(B, H, T, T) x (B, H, T, D) -> (B, H, T, D)
    rec("6. 加权求和", (B, H, T, D))

    # 7. 头合并：(B, H, T, D) -> (B, T, H, D) -> (B, T, E)
    rec("7. 头合并", (B, T, E))

    # 8. 输出投影
    rec("8. 输出投影", (B, T, E))

    print("=" * 60)
    return records


def demo_attention_shapes(batch_size, seq_len, embed_dim, num_heads):
    """复现 demo_attention_einsum 的形状流，返回每步形状列表（用于对照）。"""
    B, T, E, H = batch_size, seq_len, embed_dim, num_heads
    D = E // H
    shapes = []
    X = np.random.randn(B, T, E)
    shapes.append(("input", X.shape))
    W_q = np.random.randn(E, E) * 0.02
    Q = X @ W_q  # (B,T,E) via matmul last two dims; numpy broadcasting over B
    # Actually use einsum like demo
    Q = np.einsum("bte,ek->btk", X, W_q)
    K = np.einsum("bte,ek->btk", X, W_q)
    V = np.einsum("bte,ek->btk", X, W_q)
    shapes.append(("qkv_proj", Q.shape))
    Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    shapes.append(("head_split", Q.shape))
    scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
    shapes.append(("scores", scores.shape))
    def softmax(x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
    weights = softmax(scores, axis=-1)
    shapes.append(("weights", weights.shape))
    attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)
    shapes.append(("weighted_sum", attn_output.shape))
    concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
    shapes.append(("head_merge", concat.shape))
    W_o = np.random.randn(E, E) * 0.02
    output = np.einsum("bte,ek->btk", concat, W_o)
    shapes.append(("output_proj", output.shape))
    return shapes


def main():
    # 默认参数与 demo_attention_einsum 一致：B=2, T=8, E=64 (H=4,D=16)
    B, T, E, H = 2, 8, 64, 4

    print("\n" + "#" * 60)
    print("# 步骤 A：运行注意力形状追踪器")
    print("#" * 60)
    records = attention_shape_tracker(B, T, E, H)

    print("\n" + "#" * 60)
    print("# 步骤 B：运行 demo_attention_einsum 等价形状流")
    print("#" * 60)
    ref_shapes = demo_attention_shapes(B, T, E, H)
    for name, shp in ref_shapes:
        print(f"  {name:22s} -> {shp}")

    # 对照验证：tracker 的关键形状应与 demo 一致
    print("\n" + "#" * 60)
    print("# 步骤 C：对照验证")
    print("#" * 60)
    # 从 records 中提取各步形状
    rec_map = {r[0]: r[1] for r in records}
    checks = [
        ("输入 X", (B, T, E), "input"),
    ]
    # 期望映射
    expected = {
        "input": (B, T, E),
        "qkv_proj": (B, T, E),
        "head_split": (B, H, T, D := E // H),
        "scores": (B, H, T, T),
        "weights": (B, H, T, T),
        "weighted_sum": (B, H, T, D),
        "head_merge": (B, T, E),
        "output_proj": (B, T, E),
    }
    # tracker records name -> shape
    tracker_shapes = {
        "input": rec_map["1. 输入 X"],
        "qkv_proj": rec_map["2. Q 投影"],
        "head_split": rec_map["3. 头拆分 Q"],
        "scores": rec_map["4. 注意力分数"],
        "weights": rec_map["5. softmax 权重"],
        "weighted_sum": rec_map["6. 加权求和"],
        "head_merge": rec_map["7. 头合并"],
        "output_proj": rec_map["8. 输出投影"],
    }
    ref_map = {name: shp for name, shp in ref_shapes}
    all_ok = True
    for key in expected:
        t_shp = tracker_shapes[key]
        r_shp = ref_map[key]
        status = "OK" if (t_shp == r_shp == expected[key]) else "FAIL"
        if status != "OK":
            all_ok = False
        print(f"  {key:14s}: tracker={t_shp} demo={r_shp} expected={expected[key]} -> {status}")
    print()
    if all_ok:
        print("对照验证通过：形状追踪器与 demo_attention_einsum 完全一致。")
    else:
        raise AssertionError("形状追踪器与 demo 不一致！")

    # 额外：测试不同参数
    print("\n" + "#" * 60)
    print("# 步骤 D：不同参数测试 (B=4, T=16, E=128, H=8)")
    print("#" * 60)
    attention_shape_tracker(4, 16, 128, 8)

    print("\n" + "=" * 60)
    print("结论：形状追踪器可快速验证任意配置下的注意力形状契约。")
    print("=" * 60)


if __name__ == "__main__":
    main()

