# -*- coding: utf-8 -*-
"""
练习 3：从零实现 einsum (Einstein Summation)
"""
import os, sys, math
from itertools import product as iterproduct
import numpy as np
_LESSON_CODE = os.path.dirname(os.path.abspath(__file__))
if _LESSON_CODE not in sys.path:
    sys.path.insert(0, _LESSON_CODE)


def _compute_strides(shape):
    if len(shape) == 0:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _is_int_dtype(arr):
    return np.issubdtype(arr.dtype, np.integer)


def einsum(subscripts, *tensors):
    if "->" in subscripts:
        left, right = subscripts.split("->", 1)
        has_arrow = True
    else:
        left, right = subscripts, None
        has_arrow = False
    input_subs = [s.strip() for s in left.split(",")]
    if len(input_subs) != len(tensors):
        raise ValueError("input count mismatch")
    arrays = [np.asarray(t) for t in tensors]
    if has_arrow:
        output_sub = right.strip()
    elif right is None or right.strip() == "":
        all_idx = sorted({ch for s in input_subs for ch in s})
        output_sub = "".join(all_idx)
    else:
        output_sub = right.strip()
    size = {}
    for sub, arr in zip(input_subs, arrays):
        if len(sub) != arr.ndim:
            raise ValueError("ndim mismatch")
        for k, ch in enumerate(sub):
            sz = int(arr.shape[k])
            if ch in size and size[ch] != sz:
                raise ValueError("size mismatch")
            size[ch] = sz
    out_indices = list(dict.fromkeys(output_sub))
    contracted = [ch for ch in size if ch not in out_indices]
    all_indices = out_indices + contracted
    out_shape = tuple(size[i] for i in out_indices) if out_indices else ()
    out_size = 1
    for i in out_indices:
        out_size *= size[i]
    if all(_is_int_dtype(a) for a in arrays):
        result = [0] * (out_size if out_size > 0 else 1)
    else:
        result = [0.0] * (out_size if out_size > 0 else 1)
    out_strides = _compute_strides(out_shape) if out_shape else ()
    input_strides = [_compute_strides(tuple(a.shape)) for a in arrays]
    ranges = [range(size[i]) for i in all_indices]
    for combo in iterproduct(*ranges):
        assign = dict(zip(all_indices, combo))
        prod = 1
        for sub, arr, strides in zip(input_subs, arrays, input_strides):
            flat = 0
            for k, ch in enumerate(sub):
                flat += assign[ch] * strides[k]
            prod = prod * arr.reshape(-1)[flat]
        if out_indices:
            oflat = 0
            for k, ch in enumerate(out_indices):
                oflat += assign[ch] * out_strides[k]
        else:
            oflat = 0
        result[oflat] = result[oflat] + prod
    if not out_indices:
        return result[0]
    return np.array(result).reshape(out_shape)


def main():
    print("=" * 60)
    print("练习 3：从零实现 einsum")
    print("=" * 60)

    print("\n[用例 1] 点积：i,i->")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    mine = einsum("i,i->", a, b)
    ref = np.einsum("i,i->", a, b)
    print(f"  a={a.tolist()}, b={b.tolist()}")
    print(f"  mine={mine}, np.einsum={ref}")
    assert np.allclose(mine, ref)
    print("  OK")

    print("\n[用例 2] 矩阵乘法：ij,jk->ik")
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
    mine = einsum("ij,jk->ik", A, B)
    ref = np.einsum("ij,jk->ik", A, B)
    print(f"  A shape={A.shape}, B shape={B.shape}")
    print(f"  mine=\n{mine}")
    print(f"  np.einsum=\n{ref}")
    assert np.allclose(mine, ref)
    print("  OK")

    print("\n[用例 3] 外积：i,j->ij")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    mine = einsum("i,j->ij", a, b)
    ref = np.einsum("i,j->ij", a, b)
    print(f"  mine=\n{mine}")
    print(f"  np.einsum=\n{ref}")
    assert np.allclose(mine, ref)
    print("  OK")

    print("\n[用例 4] 转置：ij->ji")
    M = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    mine = einsum("ij->ji", M)
    ref = np.einsum("ij->ji", M)
    print(f"  原始=\n{M}")
    print(f"  mine=\n{mine}")
    print(f"  np.einsum=\n{ref}")
    assert np.allclose(mine, ref)
    print("  OK")

    print("\n[额外] 迹 ii-> 与对角 ii->i")
    Sq = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    t1 = einsum("ii->", Sq)
    t2 = np.einsum("ii->", Sq)
    print(f"  迹: mine={t1}, np={t2}")
    assert np.allclose(t1, t2)
    d1 = einsum("ii->i", Sq)
    d2 = np.einsum("ii->i", Sq)
    print(f"  对角: mine={d1.tolist()}, np={d2.tolist()}")
    assert np.allclose(d1, d2)
    print("  OK")

    print("\n[额外] 批量矩阵乘：bij,bjk->bik")
    rng = np.random.default_rng(0)
    BA = rng.standard_normal((2, 3, 4))
    BB = rng.standard_normal((2, 4, 5))
    m = einsum("bij,bjk->bik", BA, BB)
    r = np.einsum("bij,bjk->bik", BA, BB)
    print(f"  shape mine={m.shape}, np={r.shape}")
    assert np.allclose(m, r)
    print("  OK")

    print("\n" + "=" * 60)
    print("结论：朴素 einsum 通过遍历索引组合 + 收缩求和，复现 np.einsum。")
    print("=" * 60)


if __name__ == "__main__":
    main()

