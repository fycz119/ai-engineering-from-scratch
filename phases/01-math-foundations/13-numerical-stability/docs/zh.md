# 数值稳定性

> 浮点数是一种"漏水的抽象"。它会在训练中咬你一口，而你根本不会提前看见。

**类型：** 构建
**语言：** Python
**先修知识：** 第1阶段，第01-04课
**时间：** 约120分钟

## 学习目标

- 用最大值减法技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的溢出、下溢和灾难性抵消
- 用中心有限差分验证解析梯度
- 解释为什么训练时偏好 bfloat16 而非 float16，以及损失缩放如何防止梯度下溢

## 问题

你的模型训练了三个小时，然后损失变成了 NaN。你加了一行 print。在第 9,000 步 logits 还好。第 9,001 步它们变成了 `inf`。第 9,002 步所有梯度都是 `nan`，训练死亡。

或者：模型训练跑完了，但准确率比论文低 2%。你检查了一切。架构匹配。超参数匹配。数据匹配。问题在于论文用了 float32，而你用了 float16 却没有正确的缩放。32 位累积的舍入误差悄悄吃掉了你的准确率。

或者：你从零实现了交叉熵损失。小 logits 上能工作。当 logits 超过 100 时，它返回 `inf`。softmax 溢出了，因为 `exp(100)` 比 float32 能表示的最大值还大。每个 ML 框架都用两行代码的技巧处理它。你却不知道这个技巧存在。

数值稳定性不是理论问题。它是训练跑成功和静默失败的分水岭。你最终调试的每个严肃 ML Bug，最终都归结于浮点数。

> 注：NaN 一旦出现就会"传染"——一个 NaN 的梯度更新让权重变成 NaN，接下来所有输出都是 NaN，一步之内训练就死。所以调试时要在每一步前向传播后检查 NaN/Inf，而不是等训练挂掉。

## 概念

### IEEE 754：计算机如何存储实数

计算机按 IEEE 754 标准把实数存为浮点值。一个 float 由三部分组成：符号位、指数位、尾数位（significand）。

```
Float32 布局（共 32 位）：
[1 位符号] [8 位指数] [23 位尾数]

值 = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度（能分辨多少位有效数字）。指数决定范围（能表示多大或多小的数）。

```
格式     位数   指数  尾数  十进制位数   范围（约）
float64  64     11    52    ~15-16      +/- 1.8e308
float32  32     8     23    ~7-8        +/- 3.4e38
float16  16     5     10    ~3-4        +/- 65,504
bfloat16 16     8     7     ~2-3        +/- 3.4e38
```

float32 给你大约 7 位十进制精度。也就是说它能分辨 1.0000001 和 1.0000002，但分不清 1.00000001 和 1.00000002。7 位之后，一切都是舍入噪声。

float16 只有大约 3 位精度。它能表示的最大数是 65,504。对于 ML 来说这小得吓人——logits、梯度、激活值 routinely 超过这个范围。

bfloat16 是 Google 对 float16 范围问题的回答。它和 float32 一样有 8 位指数（相同范围，最高 3.4e38），但只有 7 位尾数（精度比 float16 还低）。训练神经网络时，范围比精度重要，所以 bfloat16 通常胜出。

> 注：float 的三个部分可以类比科学计数法：`1.23 × 10^4` 里，尾数是 1.23（精度），指数是 4（范围），符号决定正负。float 本质上就是二进制的科学计数法。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 在二进制浮点里无法被精确表示。在 base 2 中它是循环小数：

```
0.1 的二进制 = 0.0001100110011001100110011...（无限循环）
```

Float32 把它截断到 23 位尾数。存储的值大约是 0.100000001490116。类似地，0.2 被存成大约 0.200000002980232。它们的和是 0.300000004470348，不是 0.3。

```
在 Python 中：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 的影响：

1. 像 `if loss < threshold` 这样的损失比较可能给出错误答案
2. 累积许多小值（数千步的梯度更新）会偏离真实和
3. 如果用 `==` 比较 float，校验和与可复现性测试会失败

修复：永远不要用 `==` 比较 float。用 `abs(a - b) < epsilon` 或 `math.isclose()`。

> 注：这不是 bug，是浮点数的固有属性。所有语言、所有硬件都一样。把 float 当成"近似值"而非精确值来思考，就能避免大部分陷阱。

### 灾难性抵消

当你减去两个几乎相等的浮点数时，有效数字会相互抵消，留下的舍入噪声被推到前导位。

```
a = 1.0000001    (在 float32 中存为 1.00000011920929)
b = 1.0000000    (在 float32 中存为 1.00000000000000)

真实差值：  0.0000001
计算结果：  0.00000011920929

相对误差： 19.2%
```

单次减法就产生 19% 的相对误差。在 ML 中，这发生在你：

- 计算大均值数据的方差时：`E[x^2] - E[x]^2` 当 E[x] 很大
- 减去几乎相等的对数概率
- 用太小的 epsilon 计算有限差分梯度

修复：重新排列公式，避免减去大的、几乎相等的数。方差用 Welford 算法，或先中心化数据。对数概率全程在 log 空间运算。

> 注：灾难性抵消的本质是"有效数字被吃掉"。两个 7 位数相减，结果可能只剩 1-2 位有效数字。这是数值计算里最隐蔽的精度杀手。

### 溢出与下溢

溢出：结果太大无法表示。下溢：结果太小（比最小可表示正数还接近零）。

```
Float32 边界：
  最大值：  3.4028235e+38
  最小正数（正规）：1.175e-38
  最小正数（非正规）：1.401e-45
  溢出：  大于 ~3.4e38 变成 inf
  下溢：  小于 ~1.4e-45 变成 0.0
```

`exp()` 函数是 ML 中溢出的首要来源：

```
exp(88.7)  = 3.40e+38   (勉强塞进 float32)
exp(89.0)  = inf         (溢出)
exp(-87.3) = 1.18e-38   (勉强高于下溢)
exp(-104)  = 0.0         (下溢为零)
```

`log()` 函数在另一个方向出问题：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (正常)
log(1e-46) = -inf        (输入下溢为 0，然后 log(0) = -inf)
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。组合 `log(exp(x))` 如果没有正确的技巧就是雷区。

> 注：记住这几个"魔法数字"：float32 的 exp 安全区间大约是 [-88, 88]，float16 是 [-11, 11]。超过就溢出。这就是为什么 softmax 必须先减 max。

### 对数-求和-指数技巧（Log-Sum-Exp Trick）

直接计算 `log(sum(exp(x_i)))` 在数值上很危险。如果某个 `x_i` 很大，`exp(x_i)` 溢出。如果所有 `x_i` 都很负，每个 `exp(x_i)` 都下溢为零，`log(0)` 是 `-inf`。

技巧：在指数化之前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效：减去 `max(x)` 后，最大的指数是 `exp(0) = 1`。不可能溢出。求和中至少有一项是 1，所以和至少为 1，`log(1) = 0`。不可能下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (加减 c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (提出公因子 exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

设 `c = max(x)`，溢出就被消除了。

这个技巧在 ML 中无处不在：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合
- 变分推断

> 注：这个技巧的本质是"把大数变成 0"。exp(0)=1 是安全的锚点。所有其他项都 ≤1，所以和在 [1, n] 之间，log 也安全。这是"重缩放"思想的典范。

### 为什么 Softmax 需要最大值减法技巧

Softmax 把 logits 转成概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有这个技巧，logits [100, 101, 102] 会导致溢出：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

这些会溢出 float32 吗？float32 最大 ~3.4e38，
而 exp(88.7) 就已经到 float32 的极限了，
exp(100) 在 float32 中 = inf。
```

用技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全相同。计算是安全的。这不是优化，这是正确性的要求。

> 注：减去 max 不改变结果，因为分子分母同乘一个因子 exp(-max)。这是"乘以 1"的变形——数学上等价，数值上安全。

### NaN 与 Inf：检测与预防

`nan`（Not a Number，非数字）和 `inf`（infinity，无穷大）会病毒式传播。梯度更新里一个 `nan` 让权重变成 `nan`，进而让后续所有输出变成 `nan`。一步之内训练就死。

`inf` 如何出现：
- `exp()` 作用于大正数
- 除以零：`1.0 / 0.0`
- float32 累加溢出

`nan` 如何出现：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` 作用于负数
- `log()` 作用于负数
- 任何涉及已有 `nan` 的运算

检测：

```python
import math

math.isnan(x)       # x 是 nan 时为 True
math.isinf(x)       # x 是 +inf 或 -inf 时为 True
math.isfinite(x)    # x 既不是 nan 也不是 inf 时为 True
```

预防策略：

1. 对 `exp()` 的输入做 clamp：`exp(clamp(x, -80, 80))`
2. 分母加 epsilon：`x / (y + 1e-8)`
3. `log()` 内部加 epsilon：`log(x + 1e-8)`
4. 使用稳定的实现（log-sum-exp、稳定 softmax）
5. 梯度裁剪防止权重爆炸
6. 调试时每次前向传播后检查 `nan`/`inf`

> 注：epsilon 的选择是一门学问。太小（1e-12）在 float32 里可能被舍入掉；太大（1e-3）会扭曲结果。常见折中是 1e-8 或 1e-5，取决于数值范围。

### 数值梯度检查

解析梯度（来自反向传播）可能有 Bug。数值梯度检查用有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 精度，远好于前向差分 `(f(x+h) - f(x)) / h`（只有 O(h)）。

选择 h：太大近似不准；太小灾难性抵消会毁掉答案。`h = 1e-5` 到 `1e-7` 是典型值。

检查：计算解析梯度和数值梯度之间的相对误差。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，大概率正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

实现新层或损失函数时一定要检查梯度。PyTorch 提供 `torch.autograd.gradcheck()` 做这件事。

> 注：中心差分比前向差分好，是因为它对称地采样函数，误差的奇数项相互抵消。代价是多算一次函数值。

### 混合精度训练

现代 GPU 有专用硬件（Tensor Cores），计算 float16 矩阵乘法比 float32 快 2-8 倍。混合精度训练利用这一点：

```
1. 维护 float32 的主权重副本
2. 前向传播用 float16（快）
3. 用 float32 计算损失（防止溢出）
4. 反向传播用 float16（快）
5. 把梯度缩放到 float32
6. 更新 float32 的主权重
```

纯 float16 训练的问题：梯度往往非常小（1e-8 或更小）。Float16 会把低于 ~6e-8 的值下溢为零。你的模型停止学习，因为所有梯度更新都是零。

修复是损失缩放：

```
1. 把损失乘以一个大缩放因子（如 1024）
2. 反向传播计算 (loss * 1024) 的梯度
3. 所有梯度都变成 1024 倍（被推到 float16 下溢阈值之上）
4. 更新权重前把梯度除以 1024
5. 净效果：更新相同，但没有下溢
```

动态损失缩放自动调整缩放因子。从一个大值（65536）开始。如果梯度溢出为 `inf`，就减半。如果 N 步没有溢出，就加倍。

> 注：损失缩放的妙处在于"乘除抵消"——数学上 net effect 是零，但数值上把梯度抬到了 float16 能表示的范围里。这是典型的"数值技巧不改变数学语义"的例子。

### bfloat16 vs float16：为什么训练时 bfloat16 胜出

```
float16:   [1 符号] [5 指数]  [10 尾数]
bfloat16:  [1 符号] [8 指数]  [7 尾数]
```

float16 精度更高（10 位尾数 vs 7 位），但范围受限（最大 ~65,504）。bfloat16 精度更低但和 float32 范围相同（最大 ~3.4e38）。

对于训练神经网络：
- 训练尖峰时激活值和 logits 经常超过 65,504。float16 溢出；bfloat16 能处理。
- float16 需要损失缩放，但 bfloat16 通常不需要，因为它的范围覆盖了梯度幅度谱。
- bfloat16 是 float32 的简单截断：丢掉尾数的低 16 位。转换简单，指数上无损。

float16 更适合推理（值有界，精度更重要）。bfloat16 更适合训练（范围更重要）。这就是为什么 TPU 和现代 NVIDIA GPU（A100、H100）有原生 bfloat16 支持。

> 注：bfloat16 的设计哲学是"为了训练，范围 > 精度"。神经网络对精度不敏感（它会学着适应），但对溢出极敏感（inf/nan 直接死亡）。Google 选 8 位指数就是为了和 float32 对齐，避免转换损失。

### 梯度裁剪

梯度爆炸发生在梯度通过很多层指数级增长时（RNN、深网络、Transformer 中常见）。一个大的梯度可以在一步内毁掉所有权重。

两种裁剪方式：

**按值裁剪：** 独立地 clamp 每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留梯度方向。这就是 `torch.nn.utils.clip_grad_norm_()` 做的事。这是标准选择。

典型值：Transformer 用 `max_norm=1.0`，RL 用 `max_norm=0.5`，简单网络用 `max_norm=5.0`。

梯度裁剪不是 hack。它是安全机制。没有它，一个离群批次就能产生大到足以毁掉数周训练的梯度。

> 注：按范数裁剪只在 ||grad|| 超过阈值时才缩放，平时不动。它保留方向，所以"更新方向"不变，只是步长被限制。这比按值裁剪更符合优化器的几何直觉。

### 归一化层作为数值稳定器

批归一化、层归一化和 RMS 归一化通常被当作帮助收敛的正则化器来呈现。它们也是数值稳定器。

没有归一化，激活值可能在层间指数级增长或缩小：

```
第1层：值在 [0, 1]
第5层：值在 [0, 100]
第10层：值在 [0, 10,000]
第50层：值在 [0, inf]
```

归一化在每一层重新中心化和重新缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常 1e-5）防止所有激活值相同时除以零。可学习参数 `gamma` 和 `beta` 让网络恢复它需要的任何尺度。

这使值在整个网络中保持在数值安全范围内，防止前向传播中的溢出和反向传播中的梯度爆炸。

> 注：归一化的 epsilon 不仅防除零，还能平滑梯度——std 很小时，分母不会爆炸。这也是为什么 LayerNorm 在 Transformer 里几乎不可或缺。

### 常见 ML 数值 Bug

**Bug：几轮之后损失变成 NaN。**
原因：logits 增长过大，softmax 溢出。或学习率太高导致权重发散。
修复：使用稳定 softmax（最大值减法），降低学习率，加梯度裁剪。

**Bug：损失卡在 log(num_classes)。**
原因：模型输出接近均匀概率。通常意味着梯度消失或模型根本没在学。
修复：检查数据标签是否正确，验证损失函数，检查死 ReLU。

**Bug：验证准确率比预期低 1-3%。**
原因：混合精度但没有正确的损失缩放。梯度下溢静默地把小更新置零。
修复：启用动态损失缩放，或切换到 bfloat16。

**Bug：某些层的梯度范数是 0.0。**
原因：死 ReLU 神经元（所有输入为负），或 float16 下溢。
修复：用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**Bug：同一模型在一块 GPU 上能跑，在另一块上结果不同。**
原因：非确定性的浮点累加顺序。GPU 并行规约在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
修复：接受小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度惩罚。

**Bug：损失计算中 `exp()` 返回 `inf`。**
原因：原始 logits 被传给 `exp()` 而没有最大值减法技巧。
修复：用 `torch.nn.functional.log_softmax()`，它内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的梯度幅度或高于 65,504 的激活值。
修复：使用带损失缩放的混合精度（AMP），或改用 bfloat16。

```figure
logsumexp-stability
```

## 构建它

### 步骤1：演示浮点精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤2：实现朴素 vs 稳定的 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

> 注：`softmax_naive` 在危险 logits 上会抛 OverflowError 或返回 inf/nan。稳定版本永远安全，且结果在数学上完全等价。

### 步骤3：实现稳定的 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### 步骤4：实现稳定的交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

> 注：稳定交叉熵避免了先算 softmax 再 log 的"往返"——直接在 log 空间里算 log_prob = shifted - logsumexp(shifted)。这不仅稳定，还更快（少一次 exp）。

### 步骤5：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用它

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

> 注：`simulate_bfloat16` 通过位掩码 `0xFFFF0000` 直接截断低 16 位，模拟 bfloat16 的"截断而非舍入"语义。这是 bfloat16 转换简单的原因。

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现与所有边界情况见 `code/numerical.py`。

## 部署它

本课产出：
- `code/numerical.py`：包含稳定 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`：用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现会在第3阶段构建训练循环时、第4阶段实现注意力机制时再次出现。

## 练习

1. **灾难性抵消。** 用朴素公式 `E[x^2] - E[x]^2` 在 float32 下计算 `[1000000.0, 1000001.0, 1000002.0]` 的方差。然后用 Welford 在线算法计算。与真实方差（0.6667）比较误差。

2. **精度寻猎。** 找到最小的正 float32 值 `x`，使得 Python 中 `1.0 + x == 1.0`。这就是机器 epsilon。验证它与 `numpy.finfo(numpy.float32).eps` 匹配。

3. **Log-sum-exp 边界情况。** 用以下输入测试你的 `logsumexp_stable`：(a) 所有值相等，(b) 一个值远大于其余，(c) 所有值都非常负（-1000）。验证在朴素版本失败的地方它给出正确结果。

4. **梯度检查一个神经网络层。** 实现单个线性层 `y = Wx + b` 及其解析反向传播。用 `numerical_gradient` 验证一个 3x2 权重矩阵的正确性。

5. **损失缩放实验。** 模拟 float16 训练：创建范围 [1e-9, 1e-3] 的随机梯度，转成 float16，测量有多少比例变成零。然后应用损失缩放（乘以 1024），转成 float16，缩放回去，再测量零的比例。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|------|----------------|----------------------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。每个现代 CPU 和 GPU 都实现它。 |
| 机器 epsilon | "精度极限" | 使得 1.0 + e != 1.0 的最小值 e（在给定 float 格式下）。float32 中约为 1.19e-7。 |
| 灾难性抵消 | "减法导致的精度损失" | 当减去几乎相等的浮点数时，有效数字抵消，舍入噪声主导结果。 |
| 溢出 | "数太大" | 结果超过最大可表示值，变成 inf。exp(89) 会溢出 float32。 |
| 下溢 | "数太小" | 结果比最小可表示正数还接近零，变成 0.0。exp(-104) 会下溢 float32。 |
| Log-sum-exp 技巧 | "先减最大值" | 通过提出 exp(max(x)) 来计算 log(sum(exp(x)))，防止溢出和下溢。用于 softmax、交叉熵和对数概率数学。 |
| 稳定 softmax | "不会爆炸的 softmax" | 指数化之前减去 max(logits)。数值上相同的结果，不可能溢出。 |
| 梯度检查 | "验证你的反向传播" | 把反向传播得到的解析梯度与有限差分得到的数值梯度比较，以捕获实现 Bug。 |
| 混合精度 | "float16 前向，float32 反向" | 速度敏感的操作用低精度 float，数值敏感的操作用高精度 float。典型加速 2-3 倍。 |
| 损失缩放 | "防止梯度下溢" | 反向传播前把损失乘以一个大常数，使梯度保持在 float16 可表示范围内，更新权重前再除以相同常数。 |
| bfloat16 | "Brain floating point" | Google 的 16 位格式：8 位指数（与 float32 相同范围）和 7 位尾数（精度低于 float16）。训练时首选。 |
| 梯度裁剪 | "限制梯度范数" | 缩放梯度向量使其范数不超过阈值。防止梯度爆炸毁掉权重。 |
| NaN | "Not a Number" | 未定义运算产生的特殊 float 值（0/0、inf-inf、sqrt(-1)）。会传播到后续所有运算。 |
| Inf | "无穷大" | 溢出或除以零产生的特殊 float 值。可组合产生 NaN（inf - inf、inf * 0）。 |
| 数值梯度 | "暴力导数" | 通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。慢但可靠，用于验证。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考，密度高但完整
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- NVIDIA 引入 float16 训练损失缩放的论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch 混合精度实践指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Google 为什么为 TPU 选择这个格式
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 减少浮点求和舍入误差的算法
