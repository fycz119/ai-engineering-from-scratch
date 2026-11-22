# -*- coding: utf-8 -*-
"""
练习 1：重塑往返（Reshape Round-Trip）
========================================

对应课程：phases/01-math-foundations/12-tensor-operations
练习描述：
    取一个形状为 (2, 3, 4) 的张量。先重塑为 (6, 4)，再为 (24,)，
    最后还原为 (2, 3, 4)。每一步打印扁平数据，验证元素顺序被保留。

核心概念：
    reshape 不移动数据，只改变"如何解读"同一块内存。
    C 顺序（行优先）下，扁平数据顺序始终不变，只有 shape/strides 元数据变化。
    因此只要总元素数一致，reshape 前后 _data 完全相同。

运行方式：
    python practice_01_reshape_roundtrip.py
"""

import os
import sys

# 把课程代码目录加入 sys.path，以便复用 Tensor 类
_LESSON_CODE = os.path.dirname(os.path.abspath(__file__))
if _LESSON_CODE not in sys.path:
    sys.path.insert(0, _LESSON_CODE)

from tensors import Tensor  # noqa: E402


def main():
    # 构造一个形状为 (2, 3, 4) 的张量，元素按 1..24 顺序填充
    # 这样扁平数据本身就是 1..24，便于观察顺序是否保留
    data = list(range(1, 2 * 3 * 4 + 1))
    t0 = Tensor(data, shape=(2, 3, 4))

    print("=" * 60)
    print("练习 1：重塑往返 (Reshape Round-Trip)")
    print("=" * 60)
    print("\n[步骤 0] 原始张量")
    print(f"  shape = {t0.shape}")
    print(f"  扁平数据 = {t0._data}")

    # 步骤 1：重塑为 (6, 4)
    t1 = t0.reshape((6, 4))
    print("\n[步骤 1] reshape -> (6, 4)")
    print(f"  shape = {t1.shape}")
    print(f"  扁平数据 = {t1._data}")
    assert t1._data == t0._data, "元素顺序被改变！"
    print("  OK 扁平数据与原始一致，元素顺序保留")

    # 步骤 2：重塑为 (24,)
    t2 = t1.reshape((24,))
    print("\n[步骤 2] reshape -> (24,)")
    print(f"  shape = {t2.shape}")
    print(f"  扁平数据 = {t2._data}")
    assert t2._data == t0._data, "元素顺序被改变！"
    print("  OK 扁平数据与原始一致，元素顺序保留")

    # 步骤 3：还原为 (2, 3, 4)
    t3 = t2.reshape((2, 3, 4))
    print("\n[步骤 3] reshape -> (2, 3, 4)  (还原)")
    print(f"  shape = {t3.shape}")
    print(f"  扁平数据 = {t3._data}")
    assert t3._data == t0._data, "元素顺序被改变！"
    print("  OK 扁平数据与原始一致，元素顺序保留")

    # 额外验证：嵌套结构也应完全相同
    assert t3.to_list() == t0.to_list(), "嵌套结构不一致！"
    print("\n[额外验证] to_list() 嵌套结构完全相同 OK")

    # 验证 strides 随 shape 变化而变化
    print("\n[步幅追踪]")
    print(f"  (2,3,4) strides = {t0.strides}")
    print(f"  (6,4)   strides = {t1.strides}")
    print(f"  (24,)   strides = {t2.strides}")

    print("\n" + "=" * 60)
    print("结论：reshape 只改 shape/strides 元数据，数据缓冲区原样保留。")
    print("=" * 60)


if __name__ == "__main__":
    main()
