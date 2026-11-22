# -*- coding: utf-8 -*-
"""
练习 3:鞍点逃逸(Saddle Point Escape)
=====================================
题目:定义 f(x, y) = x^2 - y^2(原点处是鞍点),从 (0.01, 0.01) 出发,
比较朴素梯度下降(vanilla GD)、带动量的 SGD、Adam 三者的行为,
看谁(们)能逃离鞍点。

为什么原点是鞍点:
    海森矩阵 H = [[∂²f/∂x², ∂²f/∂xy],
                   [∂²f/∂xy, ∂²f/∂y²]] = [[2, 0], [0, -2]]
    沿 x 方向曲率为 +2(向下弯曲,像"谷"),沿 y 方向曲率为 -2(向上弯曲,像"脊")。
    梯度为 0 但不是极小值 —— 这就是鞍点。

运行:python exercise3_saddle_point.py
"""

import math
import sys

# Windows 控制台默认代码页可能无法显示 UTF-8 中文,强制标准输出使用 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ----------------------------------------------------------------------
# 目标函数与其解析梯度
# ----------------------------------------------------------------------

def saddle(params):
    """鞍点函数 f(x, y) = x^2 - y^2。该函数无下界(向 y 方向可以一直降),
    所以我们的目标不是"收敛到最小值",而是"离开原点(鞍点)附近"。"""
    x, y = params
    return x * x - y * y


def saddle_gradient(params):
    """解析梯度:∂f/∂x = 2x,∂f/∂y = -2y。
    起点 (0.01, 0.01) 处梯度 = (0.02, -0.02):非常小,说明我们几乎贴着鞍点,
    但方向明确 —— 沿 y 轴正方向是最速下降方向(负曲率方向)。"""
    x, y = params
    return [2.0 * x, -2.0 * y]


# ----------------------------------------------------------------------
# 三种优化器(实现与 lessons/08-optimization 中的 Build It 部分一致)
# ----------------------------------------------------------------------

class GradientDescent:
    """朴素梯度下降。
    更新规则:p <- p - lr * g
    没有任何"记忆",每步位移完全由当前梯度大小决定:
    梯度越小走得越慢。在鞍点附近梯度趋近 0,因此爬行最慢。"""

    def __init__(self, lr=0.01):
        self.lr = lr  # 学习率(步长)

    def step(self, params, grads):
        # 每一步:参数沿梯度反方向走 lr * |g| 的距离
        return [p - self.lr * g for p, g in zip(params, grads)]


class SGDMomentum:
    """带动量的 SGD。
    更新规则:
        v <- beta * v + g    # 速度 = 历史速度的衰减 + 当前梯度
        p <- p - lr * v
    直觉:球沿山坡滚下,不会在每个小坑前停下重来。
    当梯度方向持续不变(如鞍点附近沿 y 方向)时,速度不断累积,
    等效步长按 1/(1-beta) 放大,从而显著加速逃离。"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.beta = momentum  # 动量系数:越大保留越多历史
        self.velocity = None  # 速度向量,首次 step 时初始化为 0

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        # 累积速度:保留 beta 比例的旧速度,叠加当前梯度
        self.velocity = [
            self.beta * v + g for v, g in zip(self.velocity, grads)
        ]
        # 用累积后的速度更新参数,而不是用单步梯度
        return [p - self.lr * v for p, v in zip(params, self.velocity)]


class Adam:
    """Adam(自适应矩估计)。
    更新规则:
        m <- beta1*m + (1-beta1)*g        # 一阶矩:梯度的滑动平均(方向)
        v <- beta2*v + (1-beta2)*g^2      # 二阶矩:梯度平方的滑动平均(幅度)
        m_hat = m / (1 - beta1^t)        # 偏置校正(初始时刻 m、v 偏小)
        v_hat = v / (1 - beta2^t)
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)
    关键:m_hat / sqrt(v_hat) 近似等于"梯度的单位方向",
    因此每个参数的有效步长都接近 lr,与梯度大小基本无关。
    这让它在小梯度区(鞍点附近)也能保持恒定步长,不会像 GD 那样越爬越慢。"""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1   # 一阶矩衰减率
        self.beta2 = beta2   # 二阶矩衰减率
        self.epsilon = epsilon  # 数值稳定项,防止除零
        self.m = None       # 一阶矩
        self.v = None       # 二阶矩
        self.t = 0          # 步数计数(用于偏置校正)

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        # 更新一阶矩(梯度方向的历史平均)与二阶矩(梯度幅度的历史平均)
        self.m = [
            self.beta1 * m + (1.0 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]
        self.v = [
            self.beta2 * v + (1.0 - self.beta2) * g * g
            for v, g in zip(self.v, grads)
        ]

        # 偏置校正:两矩都从 0 初始化,早期估计偏小,除以 (1-beta^t) 补偿
        m_hat = [m / (1.0 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1.0 - self.beta2 ** self.t) for v in self.v]

        # 自适应更新:每个参数用自己的 sqrt(v_hat) 归一,步长≈lr
        return [
            p - self.lr * mh / (math.sqrt(vh) + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]


# ----------------------------------------------------------------------
# 仿真与判定
# ----------------------------------------------------------------------

def run(optimizer, start=(0.01, 0.01), steps=300):
    """从 start 出发跑 steps 步,记录每一步的参数(第 0 步为初始点)。"""
    params = list(start)
    history = [params[:]]
    for _ in range(steps):
        grads = saddle_gradient(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history


def first_escape_step(history, radius=1.0):
    """返回首次"逃离"的步数:到原点距离 > radius(即离开鞍点邻域)。
    若跑完都没逃离,返回 None。"""
    for i, p in enumerate(history):
        if math.hypot(p[0], p[1]) > radius:
            return i
    return None


def fmt(v):
    """大数/小数统一用科学计数法,便于阅读。"""
    return f"{v:.4e}"


def main():
    start = (0.01, 0.01)
    steps = 300
    # 三种方法使用相同的学习率 lr=0.01,保证公平对比
    configs = [
        ("vanilla GD", GradientDescent(lr=0.01)),
        ("SGD + 动量", SGDMomentum(lr=0.01, momentum=0.9)),
        ("Adam", Adam(lr=0.01)),
    ]

    print("=" * 72)
    print("  练习 3:鞍点逃逸  f(x,y) = x^2 - y^2,起点 (0.01, 0.01),lr = 0.01")
    print("=" * 72)
    print(f"  起点: x={start[0]}, y={start[1]}, f={saddle(list(start)):.6f}, "
          f"梯度={saddle_gradient(list(start))}")
    print(f"  判定标准: 到原点距离 > 1.0 视为'逃离'鞍点邻域")
    print()

    escape_steps = {}
    for name, opt in configs:
        history = run(opt, start, steps)
        final = history[-1]
        esc = first_escape_step(history)
        escape_steps[name] = esc

        # 打印轨迹采样:每 25 步一行,展示 x、y、距离原点、函数值
        print(f"  --- {name} ---")
        print(f"  {'步':>5s}  {'x':>12s}  {'y':>12s}  {'距原点':>12s}  {'f(x,y)':>12s}")
        for i in range(0, len(history), 25):
            p = history[i]
            d = math.hypot(p[0], p[1])
            print(f"  {i:5d}  {p[0]:12.6f}  {p[1]:12.6f}  {d:12.6f}  {fmt(saddle(p))}")
        i = len(history) - 1
        p = history[-1]
        d = math.hypot(p[0], p[1])
        print(f"  {i:5d}  {p[0]:12.6f}  {p[1]:12.6f}  {d:12.6f}  {fmt(saddle(p))}")
        print()

    # 汇总对比表
    print("=" * 72)
    print("  汇总(判定标准: 距原点 > 1.0)")
    print("=" * 72)
    print(f"  {'方法':<14s}  {'逃离所需步数':>14s}  {'300 步后位置 (x, y)':>30s}")
    print(f"  {'-' * 64}")
    for name, _ in configs:
        esc = escape_steps[name]
        if esc is not None:
            esc_txt = f"第 {esc} 步"
        else:
            esc_txt = f"{steps} 步内未逃离"
        print(f"  {name:<14s}  {esc_txt:>14s}  (见上方轨迹末行)")

    # 理论对照:vanilla GD 在这道题上有精确解
    # x_k = x0*(1 - 2*lr)^k,y_k = y0*(1 + 2*lr)^k
    # lr = 0.01 时 y 每步只放大 1.02 倍,|y| 从 0.01 涨到 1 需要
    # k = ln(100)/ln(1.02) ≈ 233 步 —— 与上面实测应一致。
    k_gd = math.log(100.0) / math.log(1.0 + 2 * 0.01)
    print()
    print(f"  理论对照: GD 逃离所需步数 = ln(100)/ln(1.02) ≈ {k_gd:.1f} 步")
    print()
    print("  结论: 三者最终都逃离了鞍点(起点梯度非零),速度排序见上表。")
    print("  若恰好从 (0,0) 出发,梯度为 0,三种一阶优化器都会永远卡住,")
    print("  逃逸必须依赖扰动(非零初始偏移 / mini-batch 噪声)或二阶信息。")


if __name__ == "__main__":
    main()
