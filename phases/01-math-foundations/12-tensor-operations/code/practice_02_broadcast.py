# -*- coding: utf-8 -*-
"""
练习 2：实现广播 (Broadcasting)
"""
import os, sys, math
from itertools import product as iterproduct
_LESSON_CODE = os.path.dirname(os.path.abspath(__file__))
if _LESSON_CODE not in sys.path:
    sys.path.insert(0, _LESSON_CODE)
from tensors import Tensor

def broadcast_to(self, shape):
    shape = tuple(shape)
    src_shape = (1,) * (len(shape) - self.rank) + self._shape
    if len(src_shape) != len(shape):
        raise ValueError("broadcast rank mismatch")
    for a, b in zip(src_shape, shape):
        if a != b and a != 1:
            raise ValueError(f"cannot broadcast {a} to {b}")
    result_data = [0] * (math.prod(shape) if shape else 1)
    for tgt_indices in iterproduct(*(range(s) for s in shape)):
        src_indices = tuple(0 if src_shape[i] == 1 else tgt_indices[i] for i in range(len(shape)))
        self_indices = src_indices[len(shape) - self.rank:] if self.rank > 0 else ()
        flat = 0
        for i, ind in enumerate(self_indices):
            flat += ind * self._strides[i]
        tgt_flat = 0
        tgt_strides = Tensor._compute_strides(shape)
        for i, ind in enumerate(tgt_indices):
            tgt_flat += ind * tgt_strides[i]
        result_data[tgt_flat] = self._data[flat]
    result = Tensor.__new__(Tensor)
    result._data = result_data
    result._shape = shape
    result._strides = Tensor._compute_strides(shape)
    return result
Tensor.broadcast_to = broadcast_to

def _broadcast_shape(shape_a, shape_b):
    n = max(len(shape_a), len(shape_b))
    a = (1,) * (n - len(shape_a)) + tuple(shape_a)
    b = (1,) * (n - len(shape_b)) + tuple(shape_b)
    out = []
    for x, y in zip(a, b):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            raise ValueError(f"cannot broadcast {shape_a} vs {shape_b}")
    return tuple(out)

def _elementwise_op(self, other, op):
    if isinstance(other, (int, float)):
        return Tensor([op(x, other) for x in self._data], shape=self._shape)
    if not isinstance(other, Tensor):
        raise TypeError(f"unsupported type {type(other)}")
    target = _broadcast_shape(self._shape, other._shape)
    a = self if self._shape == target else self.broadcast_to(target)
    b = other if other._shape == target else other.broadcast_to(target)
    return Tensor([op(x, y) for x, y in zip(a._data, b._data)], shape=target)
Tensor._elementwise_op = _elementwise_op

def main():
    print("=" * 60)
    print("练习 2：实现广播 (Broadcasting)")
    print("=" * 60)
    print("\n[测试1] broadcast_to 基本功能")
    t = Tensor([1, 2, 3], shape=(1, 3))
    bt = t.broadcast_to((4, 3))
    print(f"  原始 shape={t.shape}")
    print(f"  广播 shape={bt.shape}")
    assert bt.shape == (4, 3)
    for row in bt.to_list():
        assert row == [1, 2, 3]
    print("  OK")
    print("\n[测试2] (3,1) + (1,4) -> (3,4)")
    a = Tensor([1, 2, 3], shape=(3, 1))
    b = Tensor([10, 20, 30, 40], shape=(1, 4))
    print(f"  a.shape={a.shape}")
    print(f"  b.shape={b.shape}")
    result = a + b
    print(f"  result.shape={result.shape}")
    print(f"  result={result.to_list()}")
    assert result.shape == (3, 4)
    expected = [[a[i, 0] + b[0, j] for j in range(4)] for i in range(3)]
    assert result.to_list() == expected
    print("  OK")
    print("\n[测试3] 乘法广播")
    prod = a * b
    print(f"  shape={prod.shape}")
    assert prod.shape == (3, 4)
    print("  OK")
    print("\n[测试4] 与 NumPy 对比")
    import numpy as np
    na = np.array([[1.0],[2.0],[3.0]])
    nb = np.array([[10.0,20.0,30.0,40.0]])
    np_res = na + nb
    ours = Tensor([1.0,2.0,3.0], shape=(3,1)) + Tensor([10.0,20.0,30.0,40.0], shape=(1,4))
    assert ours.to_list() == np_res.tolist()
    print(f"  numpy shape={np_res.shape} ours shape={ours.shape}")
    print("  OK")
    print("\n" + "="*60)
    print("结论：broadcast_to + 自动广播使不同形状张量可直接运算。")
    print("="*60)

if __name__ == "__main__":
    main()
