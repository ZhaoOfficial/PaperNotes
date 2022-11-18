# Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting

## 0 Abstract

我们介绍了一种新的算法--ReSTIR--能够以互动的方式高质量地渲染直接光，而且不需要维护复杂的数据结构。我们对一组候选光照样本进行反复重采样，并应用进一步的空间和时间重采样来利用附近相关样本的信息。
We introduce a new algorithm - ReSTIR - that renders direct lighting interactively, at high quality, and without needing to maintain complex data structures. We repeatedly resample a set of candidate light samples and apply further spatial and temporal resampling to leverage information from relevant nearby samples.

## 1 Introduction

做有许多发光物体的直接光仍然具有挑战性；追踪到所有灯的 shadow rays 是不可行的，找到在某一点上贡献最大的灯取决于每个灯对该点的可见度、散射函数（BSDF或相位函数）在该点的分布、以及光源的功率和发光特性。
Direct lighting with many emissive objects remains challenging; it’s not feasible to trace shadow rays to all of the lights, and finding the lights that contribute most at a given point depends on each light’s visibility to that point, the distribution of the scattering function (BSDF or phase function) at the point, and the light source’s power and emissive characteristics.

实时渲染增加了更多的挑战：要渲染的场景是动态的，而渲染器一般不知道未来场景会如何变化，因为这可能会受到用户互动的影响。此外，目前每个像素点只能追踪几条光线，所以找到重要的灯光就更加关键了，然而建立和更新数据结构以帮助灯光采样的时间是有限的。即使对于第一个相机顶点直接光的限制条件下也是如此。
Real-time rendering adds even more challenges: the scenes to be rendered are dynamic and the renderer generally has no future knowledge of how the scene will change, as that may be affected by user interaction. Furthermore, only a few rays can currently be traced at each pixel, so finding important lights is even more critical, yet there is a limited amount of time to build and update data structures to aid light sampling. This is true even for the restricted case of direct lighting at the first camera vertex.

我们介绍了一种从许多灯光中采样单次反射直接光的方法，该方法适用于具有完全动态场景的实时光追。我们的方法建立在重采样重要性采样 (RIS) 的基础上，该技术用于从一个分布中获取一组样本，并使用与被积函数更好匹配的另一个分布来选择它们的加权子集。
We introduce a method to sample one-bounce direct lighting from many lights that is suited to real-time ray tracing with fully dynamic scenes. Our approach builds on resampled importance sampling (RIS), a technique for taking a set of samples that are from one distribution and selecting a weighted subset of them using another distribution that better matches the function being integrated.

与以前的 RIS 应用程序不同，我们使用一个小的固定大小的数据结构 - 一个只存储接受的样本的“水库” - 和一个相关的采样算法（在非图形应用程序中经常使用）来帮助实现稳定的实时性能。
Unlike prior applications of RIS, we use a small fixed-size data structure - a "reservoir" that only stores accepted sample - and an associated sampling algorithm (used frequently in non-graphics applications) to help achieve stable, real-time performance.

我们的重用通知了渲染器内使用的采样概率，这反过来又使无偏的算法成为可能。
Our reuse informs the sampling probabilities used within the renderer, which in turn makes an unbiased algorithm possible.

## 2 Preliminaries

|                      Symbols                       |                         Descriptions                         |
| :------------------------------------------------: | :----------------------------------------------------------: |
|                        $L$                         |                           radiance                           |
|                      $\omega$                      |                          direction                           |
|                       $x,y$                        |                            point                             |
|                        $A$                         |                   light emitting surfaces                    |
| $\rho(y,\overrightarrow{yx}\leftrightarrow\omega)$ |      bidirectional scatter distribution function (BSDF)      |
|                   $L_e(x\to y)$                    |               emitted radiance from $x$ to $y$               |
|              $G(x\leftrightarrow y)$               | a geometry term: $\langle n_{x},\overrightarrow{xy}\rangle\langle n_y,\overrightarrow{yx}\rangle/r^2_{xy}$ |
|              $V(x\leftrightarrow y)$               |                      mutual visibility                       |

从 $y$ 点朝 $\omega$ 方向反射的辐射 $L$：
$$
L(y,\omega)=\int_A\rho(y,\overrightarrow{yx}\leftrightarrow\omega)L_e(x\to y)G(x\leftrightarrow y)V(x\leftrightarrow y)\mathrm{d}A_x\tag{1}
$$
为了简洁起见，丢弃观察方向 $\omega$ 和上色点 $y$，将面积微元表示为 $\mathrm{d}x$：
By dropping the viewing direction $\omega$ and shading point $y$ for brevity and denoting differential area as $\mathrm{d}x$:
$$
L=\int_Af(x)\mathrm{d}x\quad f(x)\equiv\rho(x)L_e(x)G(x)V(x)\tag{2}
$$

#### Importance Sampling (IS)

---

*Importance Sampling 是为了快速合理的估计积分值。只需要一个和被积函数形状相似的概率函数即可。*

---

IS 从一个已知 PDF 的分布里面采样 $N$ 次：
$$
\langle{L}\rangle^N_{\text{is}}=\frac{1}{N}\sum_{i=1}^N\frac{f(x_i)}{p(x_i)}\approx L\tag{3}
$$
如果当 $f(x)$ 非零时 $p(x)$ 为正，则 IS 保持无偏，并且理想情况下 $p(x)$ 与 $f(x)$ 相关以减少方差。
IS remains unbiased if $p(x)$ is positive whenever $f(x)$ is non-zero, and ideally $p(x)$ is correlated with $f(x)$ to reduce variance.

#### Multiple Importance Sampling (MIS)

----

*Multiple Importance Sampling 是因为获得一个和被积函数形状相似的概率函数比较困难，进而将被积函数拆成几个函数的乘积，随后用与这几个函数形状相似的概率函数做加权 Importance Sampling。其中权值函数可以是关于 $x$ 的函数。*

---

在实践中，与 $f(x)$ 成比例的直接采样是不可行的，部分原因是可见性因子 $V(x)$。但是，我们通常可以采样与被积函数中的各个成比例的项（例如，BSDF $\rho$ 或发射表面 $L_e$）。
In practice, directly sampling proportional to $f(x)$ is infeasible, in part due to the visibility factor $V(x)$. However, we can often draw samples proportional to individual terms in the integrand (e.g., the BSDF $\rho$ or the emissive surfaces $L_e$).  

给定 $M$ 个候选抽样策略 $p_s$，MIS 从每个策略 $s$ 中抽取 $N_s$ 个样本，并将其合并为一个单一的加权估计器：

Given $M$ such candidate sampling strategies $p_s$, MIS draws $N_s$ samples from each strategy $s$ and combines them into a single weighted estimator:
$$
\langle{L}\rangle^{M,N}_{\text{mis}}=\sum_{s=1}^M\frac{1}{N_s}\sum_{i=1}^{N_s}w_s(x_{s,i})\frac{f(x_{s,i})}{p_s(x_{s,i})}\tag{4}
$$
只要权重 $w_s$ 划分是归一化的，即 $\sum_{s=1}^{M}w_s(x)=1$，MIS 就保持无偏。平衡启发式 $w_s=N_sp_s(x)/(\sum_{j}N_jp_j(x))$ 是非负权重、流行、且可证明的好选择，它等效于从 $M$ 个策略的混合分布中采样。
As long as the weights $w_s$ form a partition of unity $\sum_{s=1}^{M}w(x)=1$, MIS remains unbiased. The balance heuristic, $w_s=N_sp_s(x)/(\sum_{j}N_jp_j(x))$, is a popular and provably good choice for non-negative weights, and is equivalent to sampling from the mixture distribution of the $M$ individual strategies.

---

Unbiased:
$$
\begin{align*}
E[\langle{L}\rangle^{M,N}_{\text{mis}}]&=\sum_{s=1}^M\frac{1}{N_s}E\biggl[\sum_{i=1}^{N_s}w_s(x_{s,i})\frac{f(x_{s,i})}{p_s(x_{s,i})}\biggr]\\
&=\sum_{s=1}^M\frac{1}{N_s}N_s\int w_s(x)\frac{f(x)}{p_s(x)}p_s(x)\mathrm{d}x\\
&=\sum_{s=1}^M\int w_s(x)f(x)\mathrm{d}x\\
&=\int \biggl[\sum_{s=1}^Mw_s(x)\biggr]f(x)\mathrm{d}x\\
&=\int f(x)\mathrm{d}x
\end{align*}
$$
Balance heuristc:
$$
\langle{L}\rangle^{M,N}_{\text{mis}}=\sum_{s=1}^M\sum_{i=1}^{N_s}\frac{f(x_{s,i})}{\sum_{j=1}^{M}N_jp_j(x_{s,i})}
$$

---

### 2.1 Resampled Importance Sampling (RIS)

---

*Resampled Importance Sampling 做的是先用一个已知分布的概率函数 $p$ 采样出一个比较复杂的分布 $f$ 的一些样本，再对这些样本采样得到归一化的 $\hat{f}$ 的样本。蕴含了 Bayesian 的思想。*

---

使用 MIS 从上色项的线性组合中采样的另一种方法是与某些项的乘积大致成比例地进行采样。
An alternative to sampling from a linear combination of shading terms using MIS is to sample approximately proportional to the product of some of the terms.

重采样重要性采样通过从**一个次优但容易抽样的源分布 $p$**（例如 $p\propto L_e$ 或均匀分布）中生成 $M\ge1$ 个候选样本 $\mathbf{x}=\{x_1,\dots,x_M\}$ 来实现。然后，它以离散的概率从这个候选池中随机选择一个下标 $z=\{1,\dots,M\}$，其离散概率由期望的目标 PDF $\hat{p}(x)$ 驱动，而 $\hat{p}(x)$ 可能不存在实际的采样算法（例如 $\hat{p}\propto\rho\cdot L_e\cdot G$）。
Resampled importance sampling achieves this by generating $M\ge1$ candidate samples $\mathbf{x}=\{x_1,\dots,x_M\}$ from **a source distribution $p$ that is sub-optimal, but easy to sample from** (e.g., $p\propto L_e$ or uniform distribution). It then randomly chooses an index $z=\{1,\dots,M\}$ from this pool of candidates with discrete probabilities driven by a desired target PDF $\hat{p}(x)$, for which no practical sampling algorithm may exist (e.g., $\hat{p}\propto\rho\cdot L_e\cdot G$).
$$
p(z\mid\mathbf{x})=\frac{\mathrm{w}(x_z)}{\sum_{i=1}^{M}\mathrm{w}(x_i)}\quad\text{with}\quad \mathrm{w}(x)=\frac{\hat{p}(x)}{p(x)}\tag{5}
$$
选择一个 $y\equiv x_z$ 的样本并用于单样本 RIS 估计器：
A sample $y\equiv x_z$ is selected and used in the 1-sample RIS estimator:
$$
\langle{L}\rangle^{1,M}_{\text{ris}}=\frac{f(y)}{\hat{p}(y)}\biggl(\frac{1}{M}\sum_{j=1}^{M}\mathrm{w}(x_j)\biggr)\tag{6}
$$

直观地说，估计器使用 $y$，就好像它是从 $\hat{p}$ 中提取的，然后使用括号中的因子来纠正 $y$ 的真实分布仅近似于 $\hat{p}$ 的事实。
Intuitively, the estimator uses $y$ as if it were drawn from $\hat{p}$ and then uses the parenthesized factor to correct for the fact that the true distribution of $y$ only approximates $\hat{p}$​.

---

*而他真正想做的是，通过一个概率函数得到了 $L_e$ 的分布 $p$，想用 $p$ 加上 RIS 得到归一化的被积函数的分布 $\hat{p}$。矫正后用 $\hat{p}$ 再做一次 IS 积分。*

*或者说，括号中的求和是得到了 $\hat{p}$ 的积分，作为对 $\hat{p}(y)$ 的归一化因子。因此 $f(y)/(\hat{p}(y)/\int\hat{p}\mathrm{d}x)$ 是一个 IS 估计器。只要 $M$ 越大，积分估计就越准，IS 也就越准。*

---

多次重复 RIS，并对结果进行平均，就可以得到一个 $N$ 样本的 RIS 估计器：
Repeating RIS multiple times and averaging the results yields an $N$-sample RIS estimator:
$$
\langle{L}\rangle^{N,M}_{\text{ris}}=\frac{1}{N}\sum_{i=1}^{N}\biggl(\frac{f(y_i)}{\hat{p}(y_i)}\cdot\biggl(\frac{1}{M}\sum_{j=1}^{M}\mathrm{w}(x_{ij})\biggr)\biggr)\tag{7}
$$
只要 $M,N\ge1$，函数 $p$ 和 $\hat{p}$ 在 $f$ 非零的地方都是正的，RIS 就是无偏的。虽然 $M$ 和 $N$ 可以自由选择，但存在一个由 $\hat{p}$ 和 $f$ 的方差和相对成本决定的 $M$ 与 $N$ 的最佳比例。
RIS is unbiased as long as $M,N\ge1$ and the functions $p$ and $\hat{p}$ are positive wherever $f$ is non-zero. While $M$ and $N$ can be chosen freely, there exists an optimal ratio of $M$ to $N$ determined by the variance and relative cost of $\hat{p}$ and $f$.

从现在开始，为了简单起见，我们将假设 $N=1$；我们的估计器可以通过平均 $N$ 个独立执行，每个独立执行都有 $M$ 的独立候选样本，从而简单地扩展到 $N>1$ 的情况。
From now on, we will assume $N = 1$ for simplicity; our estimators can be trivially extended to the $N > 1$ case by averaging $N$ independent executions, each with $M$ independent candidate samples.

```python
def ResampledImportanceSampling(M: int, p_q: Callable):
    # M: Number of candidates
    # p_q: The target PDF.

    # Generate proposals x = [x1, ..., xM].
    x, w, w_sum = [], [], 0
    for i in range():
        xi = sample(p)
        x.append(xi)
        wi = p_q(xi) / p(xi)
        w.append(wi)
        w_sum += wi
    # Compute the normalized CDF from w
    C = w / w_sum
    # Draw random index 0 <= z < M using C
    z = int(sample(C) * M)
    y = x[z]
    return y, w_sum
```

#### Combining RIS with MIS

上面我们假设了一个单一源 PDF $p$，但问题是存在多个合理的采样技术（例如 BSDF 或光采样）。只要 $p$ 在任何地方都是正的，$\hat{p}$ 都是正的，$y$ 的分布就会以 $M\to\infty$ 的方式接近 $\hat{p}$。
Above we assumed a single source PDF $p$, but many problems have several reasonable sampling techniques (e.g., BSDF or light sampling). As long as $p$ is positive anywhere $\hat{p}$ is positive, the distribution of $y$ approaches $\hat{p}$ as $M\to\infty$.

在 RIS 中利用 MIS 的多种竞争技术来减少方差：用 MIS 生成提案库，并在 RIS 程序的其余部分使用有效的 MIS（混合）PDF 作为源 PDF。
To leverage multiple competing techniques using MIS within RIS to reduce variance: generate the pool of proposals using MIS and use the effective MIS (mixture) PDF as the source PDF in the rest of the RIS procedure.

不幸的是，这种形式的MIS的成本随着技术数量的增加而呈四次方增长（因为需要为每个提议评估权重，而每个这样的权重都需要考虑所有提议的 PDF）。当 MIS 只使用两种技术（如照明和 BSDF）时，这不是一个问题，但随着策略数量的增加，它很快变得难以解决。
Unfortunately, the cost of this form of MIS increases quadratically with the number of techniques (since weights need to be evaluated for each proposal and each such weight needs to consider all proposal PDFs). This is not a problem when MIS is used with just two techniques (e.g., lighting and BSDF), but it quickly becomes intractable as the number of strategies increases.

### 2.2 Weighted Reservoir Sampling

加权水库抽样（WRS）是一系列的算法，用于从一个数据流 $\{x_1, x_2, x_3,\dots, x_M\}$ 中一次性抽取 $N$ 个随机元素。每个元素都有一个相关的权重 $w(x_i)$，这样 $x_i$ 就被选中的概率是：
Weighted reservoir sampling (WRS) is a family of algorithms for sampling $N$ random elements from a stream $\{x_1, x_2, x_3,\dots, x_M\}$ in a single pass over the data. Each element has an associated weight $w(x_i)$ such that $x_i$ should be selected with probability:
$$
P_i=\frac{\mathrm{w}(x_i)}{\sum_{j=1}^{M}\mathrm{w}(x_j)}\tag{8}
$$
水库采样对每个元素正好处理一次，只有蓄水池中的 $N$ 个元素必须保留在内存中。流的长度 $M$ 不需要事先知道。我们希望独立选择 $x_i$ 进行蒙特卡洛积分，所以我们在下面只考虑有放回加权水库采样。
Reservoir sampling processes each element exactly once, and only the $N$ items in the reservoir must remain in memory. The stream length $M$ need not be known in advance. We want independent selections $x_i$ for Monte Carlo integration, so we only consider weighted reservoir sampling with replacement below.

在下文中，我们重点讨论 $N=1$ 的情况，即水库里面只有一个样本。
In the following, we focus on the case where $N=1$, i.e. where the reservoir consists of one sample.

当处理一个新的流元素时，水库被更新以保持不变，即在 $m$ 个样本被处理后，样本 $x_i$ 以 $\mathrm{w}(x_i)/\sum_{j=1}^{m}\mathrm{w}(x_j)$ 的概率出现在水库。更新规则随机地用下一个样本 $x_{m+1}$ 替换水库中的 $x_i$，概率为：
When processing a new stream element, the reservoir is updated so as to maintain the invariant, which is that after $m$ samples have been processed, sample $x_i$ occurs in the reservoir with probability $\mathrm{w}(x_i)/\sum_{j=1}^{m}\mathrm{w}(x_j)$. The update rule stochastically replaces $x_i$ in the reservoir with the next sample $x_{m+1}$, with probability:
$$
\frac{\mathrm{w}(x_{m+1})}{\sum_{j=1}^{m+1}\mathrm{w}(x_j)}\tag{9}
$$
因此，此前的样本 $x_i$ 仍然留着水库的概率为：
Thus, any previous sample $x_i$ is in the reservoir with probability:
$$
\frac{\mathrm{w}(x_i)}{\sum_{j=1}^{m}\mathrm{w}(x_j)}\biggl(1-\frac{\mathrm{w}(x_{m+1})}{\sum_{j=1}^{m+1}\mathrm{w}(x_j)}\biggr)=\frac{\mathrm{w}(x_i)}{\cancel{\sum_{j=1}^{m}\mathrm{w}(x_j)}}\frac{\cancel{\sum_{j=1}^{m}\mathrm{w}(x_j)}}{\sum_{j=1}^{m+1}\mathrm{w}(x_j)}=\frac{\mathrm{w}(x_i)}{\sum_{j=1}^{m+1}\mathrm{w}(x_j)}\tag{10}
$$

```python
class Reservoir(object):
    def __init__(self):
        self.y, self.w_sum = 0.0, 0.0

    def update(self, xi: float, wi: float):
        self.w_sum += wi
        if random.random() < wi / self.w_sum:
            self.y = x_i

def ReservoirSampling(S: list):
    r = Reservoir()
    for sample in S:
        r.update(sample.value, sample.weight)
    return r
```

## 3 Streaming RIS With Spatiotemporal Reuse

不幸的是，原始的时空重采样方法是有偏见的，因为不同的像素根据不同的 BRDFs 和表面方向来选择样本。这就导致了图像中几何不连续点附近的能量损失，类似于后处理过滤中的典型问题。
Unfortunately, the naive approach to spatiotemporal resampling is biased, as different pixels select samples based on different BRDFs and surface orientations. This leads to energy loss near geometric discontinuities in images, similar to problems typical in post-process filtering.

### 3.1 Streaming RIS Using Reservoir Sampling

```python
def RIS(M: int, q: pixel):
    r = Reservoir()
    for i in range(M):
        xi = sample(p)
        r.update(xi, p_q(xi) / p(xi))
    # Record some information used in combination
    r.M = M
    r.W = 1 / p_q(r.y) * (r.w_sum / r.M)
    return r

def ShadePixel(r: Reservoir, q: Pixel):
    L = f_q(r.y) * r.W
    return L

for q in Image:
    Image[q] = ShadePixel(RIS(M, q), q)
```

流 RIS 的质量随着 $M$（候选数）和 $N$（上色的样本）的增加而提高。
Streaming RIS quality improves with increased $M$ (candidates) and $N$ (samples for shading).

### 3.2 Spatiotemporal Reuse

一个关键的观察是，相邻像素的目标 PDF 之间通常存在明显的相关性。例如，如果使用无阴影照明（$\hat{p}(x)=\rho(x)L_e(x)G(x)$），那么空间上的接近往往导致相邻像素的几何和 BSDF 因子相似。
A key observation is that significant correlation generally exists between target PDFs in neighboring pixels. For example, if using unshadowed illumination ($\hat{p}(x)=\rho(x)L_e(x)G(x)$), then spatial proximity often leads to the geometry and BSDF factors being similar at adjacent pixels.

水库的状态包含了当前选择的样本 $y$ 和迄今为止所有候选样本的权重之和 $\mathrm{w}_{\text{sum}}$。为了结合两个水库，我们将每个存储库的 $y$ 视为具有权重 $\mathrm{w}_{\text{sum}}$ 的新样本，并将其作为输入输入到一个新水库。其结果在数学上等同于对两个水库的合并输入流进行了水库采样。
A reservoir's state contains both the currently selected sample $y$ and the sum of weights $\mathrm{w}_{\text{sum}}$ of all candidates seen thus far. To combine two reservoirs, we treat each reservoir's $y$ as a fresh sample with weight $\mathrm{w}_{\text{sum}}$, and feed it as input to a new reservoir. The result is mathematically equivalent to having performed reservoir sampling on the two reservoirs' combined input streams.

然而，关键是这种操作只需要常数时间，并避免存储（或检索）任何一个输入流的元素，只需要访问每个存储库的当前状态。
However, crucially this operation only requires constant time and avoids storing (or retrieving) elements of either input stream, needing only access to each reservoir’s current state.

为了说明邻近像素 $q'$ 的样本是按照不同的目标分布 $\hat{p}_{q'}$ 重新采样的，我们用 $\hat{p}_{q}(r.y)/\hat{p}_{q'}(r.y)$ 的系数对样本进行重新加权，以说明与当前像素相比，邻近像素被过度采样或采样不足的区域。
To account for the fact that samples from the neighboring pixel $q'$ are resampled following a different target distribution $\hat{p}_{q'}$, we reweight the samples with the factor $\hat{p}_{q}(r.y)/\hat{p}_{q'}(r.y)$ to account for areas that were over- or undersampled at the neighbor compared to the current pixel.

```python
def CombineReservoirs(q: Pixel, rs: list[Reservoir]):
    s = Reservoir()
    for r in rs:
        # Re-weighted.
        s.update(r.y, p_q(r.y) * r.W * r.M)
   	s.M = sum(r.M for r in rs)
    s.W = 1 / p_q(s.y) * (s.w_sum / s.M)
    return s
```

#### Spatial Reuse

我们首先使用 $\mathrm{RIS}(q)$ 为每个像素 $q$ 生成 $M$ 个候选，并将生成的水库存储在一个图像大小的缓冲区中。在第二步中，每个像素选择它的邻居中的 $k$ 个，并将它们的水库与自己的水库结合在一起。
We first generate $M$ candidates for every pixel $q$ using $\mathrm{RIS}(q)$ and store the resulting reservoirs in an image-sized buffer. In a second step, each pixel selects $k$ of its neighbors and combines their reservoirs with its own.

每个像素的成本是 $O(k + M)$，但每个像素都能有效地找到 $k\cdot M$ 个候选。重要的是，空间重用可以重复进行，使用先前重用通道的输出作为输入。进行 $n$ 次迭代需要 $O(nk + M)$ 的计算，但假设每一步都使用不同的相邻像素，则每个像素有效地产生 $k^nM$ 个候选。
Per pixel costs are $O(k + M)$, but each pixel effectively sees $k\cdot M$ candidates. Crucially, spatial reuse can be repeated, using the outputs of the prior reuse pass as input. Performing $n$ iterations requires $O(nk + M)$ computation, but effectively yields $k^nM$ candidates per pixel, assuming distinct neighboring pixels are used at each step.

#### Temporal Reuse

图像往往不是孤立地呈现的，而是一个动画序列的一部分。在这种情况下，前一帧可以提供额外的重用对象。渲染完一帧后，我们将每个像素的最终储存器储存起来，以便在下一帧中重新使用。如果我们按顺序渲染帧，并将它们的储存库向前推送，那么一帧就不仅与前一帧的候选人结合在一起，而且与序列中的所有前一帧结合在一起，这就极大地提高了图像质量。
Images are often not rendered in isolation but are part of an animated sequence. In this case, the prior frame can provide additional candidates for reuse. After rendering a frame, we store each pixel’s final reservoir for reuse in the next frame. If we render frames sequentially and feed forward their reservoirs, a frame combines candidates not just with those of the previous frame, but all previous frames in the sequence, which dramatically improves image quality.

#### Visible Reuse

不幸的是，即使有无限数量的候选，RIS 也不能实现无噪声的渲染。尽管当 $M$ 接近无穷大时，样本的分布接近目标 PDF $\hat{p}$，但 $\hat{p}$ 对积分 $f$ 的采样并不完美。
Unfortunately, even with an unlimited number of candidates, RIS cannot achieve noise-free renderings. Although the distribution of samples approaches the target PDF $\hat{p}$ as $M$ approaches infinity, $\hat{p}$ does not sample the integrand $f$ perfectly.

在实践中，$\hat{p}$ 通常被设定为无阴影的路径贡献，这意味着随着 $M$ 的增长，可见度引起的噪声开始占主导地位。不幸的是，在大场景中，可见度噪声可能很严重。
In practice, $\hat{p}$ is usually set to the unshadowed path contribution, meaning that as $M$ grows large, noise due to visibility starts to dominate. Unfortunately, visibility noise can be severe in large scenes.

在进行空间或时间重用之前，我们评估每个像素的水库所选样本 $y$ 的可见性。如果 $y$ 被遮挡，我们就丢弃水库。这意味着被遮挡的样本不会传播到邻近的像素，如果可见度是局部一致的，那么空间重采样产生的最终样本很可能是不被遮挡的。
Before performing spatial or temporal reuse, we evaluate visibility of the selected sample $y$ for each pixel's reservoir. If $y$ is occluded, we discard the reservoir. This means that occluded samples will not propagate to neighboring pixels, and if visibility is locally coherent, the final sample produced by spatial resampling is likely to be unoccluded.

```python
def ReservoirReuse(prev_frame_reservoirs: Reservoir):
    rs = [Reservoir] * ImageSize
    # Generate initial candidates.
    for q in Image:
        rs[q] = RIS(q)
    # Evaluate visibility for initial candidates.
    for q in Image:
        if shadowed(rs[q].y):
            rs[q].W = 0
    # Temporal reuse.
    for q in Image:
        q_ = PickTemporalNeighbor(q)
        rs[q] = CombineReservoirs(q, [rs[q], prev_frame_reservoirs[q_]])
    # Spatial reuse.
    for i in range(n):
        for q in Image:
            Q = PickSpatialNeighbor(q)
            R = [rs[q_] for q_ in Q]
            rs[q] = CombineReservoirs(q, R + rs[q])
    # Compute pixel color
    for q in Image:
        q = ShadePixel(rs[q], q)
    # Used for next frame.
    return rs
```

## 4 (Eliminating) Bias in Multi-distribution RIS

每个像素使用不同的积分域和目标分布，重新使用相邻像素的候选者有可能引入偏差。这是因为，由于目标分布的不同，重新采样后的样本 PDF 在各个像素之间是不同的。
Each pixel uses a different integration domain and target distribution, and reusing candidates from adjacent pixels can potentially introduce bias. This is because the PDF of samples after resampling varies from pixel to pixel due to the different target distributions.

### 4.1 Analysis the RIS Weight

重新分析公式 $(6)$：
$$
\langle{L}\rangle^{1,M}_{\text{ris}}=f(y)\biggl(\frac{1}{\hat{p}(y)}\frac{1}{M}\sum_{j=1}^{M}\mathrm{w}(x_j)\biggr)=f(y)W(\mathbf{x},z)\tag{11}
$$

$$
W(\mathbf{x},z)=\frac{1}{\hat{p}(x_z)}\biggl[\frac{1}{M}\sum_{j=1}^{M}\mathrm{w}(x_j)\biggr]\tag{12}
$$

通常情况下，蒙特卡洛估计器的形式是 $f(y)/p(y)$。我们不知道 $p(y)$ --事实上，我们后来表明我们无法闭式地计算它--因此由 $W(\mathbf{x},z)$ 取代它在公式 $(11)$ 中的位置。
Normally, Monte Carlo estimators take on the form $f(y)/p(y)$. We do not know $p(y)$ - in fact, we later show that we cannot compute it in closed form - and $W(\mathbf{x},z)$ takes its place in equation $(11)$.

为了使方程 $(6)$ 是无偏的，$W(\mathbf{x},z)$ 的期望值应该等于 $1/p(y)$。
In order for equation $(6)$ to be unbiased, the expected value of $W(\mathbf{x},z)$  should be equal to $1/p(y)$.

#### Explanation of Reweighing Factor

$\hat{p}_q(r.y)\cdot r.W$ 简单地表示 $\hat{p}_q(r.y)/p_q(r.y)$ 的标准 RIS 权重，只是我们不知道确切的 PDF $p(r.y)$，而是使用 PDF 的逆的估计值 $r.W$。由于 $r.y$ 代表的是多个样本的组合结果，所以权重另外被产生 $r.y$ 的候选者 $r.M$ 的数量所缩放。
$\hat{p}_q(r.y)\cdot r.W$ simply represents the standard RIS weight of $\hat{p}_q(r.y)/p_q(r.y)$, except that we do not know the exact PDF $p(r.y)$ and use the estimate of the inverse PDF, $r.W$, instead. As $r.y$ represents the result of combining multiple samples, the weight is additionally scaled by the number of candidates $r.M$ that produced $r.y$.

### 4.2 Biased RIS

标准 RIS 假定所有候选样本都是由相同的 PDF $p$ 产生的。我们现在允许 $\mathrm{x}$ 中的每个样本$x_i$来自可能不同的源 PDF $p_i(x_i)$：
Standard RIS assumes that all candidate samples are produced by the same PDF $p$. We instead now allow each sample $x_i$ in $\mathrm{x}$ to come from a potentially different source PDF $p_i(x_i)$:
$$
p(\mathbf{x})=\prod_{i=1}^{M}p_i(x_i)\tag{13}
$$
因此 $\mathbf{x}$ 和 $z$ 的联合分布也可以得到：
$$
p(\mathbf{x},z)=p(\mathbf{x})p(z\mid\mathbf{x})=\biggl[\prod_{i=1}^{M}p_i(x_i)\biggr]\frac{\mathrm{w}_z(x_z)}{\sum_{i=1}^{M}\mathrm{w}_i(x_i)}\tag{15}
$$
定义可以产生 $y$ 的索引的集合：
$$
Z(y)=\{i\mid1\le i\le M\wedge p_i(y)>0\}
$$
因此 $p(y)$ 的 PDF 就可以用全概率公式得到：
$$
p(y)=\sum_{i\in Z(y)}\underbrace{\idotsint}_{M-1}p(\mathbf{x}^{i\to y}, i)\underbrace{\mathrm{d}x_1\dots\mathrm{d}x_M}_{M-1}\tag{17}
$$
其中 $\mathbf{x}^{i\to y}=\{x_1,\dots,x_{i-1},y,x_{i+1},\dots,x_M\}$。

#### Expected RIS Weight



### 4.3 Unbiased RIS

#### Combining with Multiple Importance Sampling

#### Comparison to RIS

### 4.4 A Practical Algorithm for Unbiased Reuse

## 5 Design and Implementation Choices

#### Candidate Generation

#### Target PDF

#### Neighbor Selection

#### Evaluated Sample Count

#### Reservoir Storage and Temporal Weighting

## 6 Results

## 7 Related Work

## 8 Conclusion

### 8.1 Limitations and Future Work
