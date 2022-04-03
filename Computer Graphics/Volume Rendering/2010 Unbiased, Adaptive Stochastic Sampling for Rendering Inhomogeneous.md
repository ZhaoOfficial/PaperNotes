# Unbiased, Adaptive Stochastic Sampling for Rendering Inhomogeneous Participating Media

## Abstract

寻找新的散射事件需要自由路径采样以确定与先前散射事件的距离，并且对于不均匀的参与介质通常是一个耗时的过程。 我们提出了一种使用基于 kd-tree 的空间划分的自适应和无偏采样技术。
Finding a new scattering event requires free path sampling to determine the distance from the previous scattering event, and is usually a time consuming process for inhomogeneous participating media. We propose an adaptive and unbiased sampling technique using kd-tree based space partitioning.

## 1 Introduction

设计这种方法的一个重要特征是无偏性，它带来了几个好处：1）解决方案可证明收敛到精确的解决方案； 2) 计算误差可以通过测量方差轻松评估，而有偏方法的计算误差难以评估 3) 无需担心由偏差引起的伪影。
An important feature in designing such a method is the unbiasedness, which brings several benefits: 1) the solution provably converges to the exact one; 2) the computational error can be easily evaluated by measuring the variance, whereas the computational error in biased methods is difficult to evaluate 3) there is no need to be concerned about bias-induced artifacts.

在预处理过程中，我们的方法根据介质中平均自由程的空间变化将分析空间（介质的边界框）划分为子空间（分区）。在渲染过程中，散射事件的位置使用 kd-tree 自适应地确定。
During preprocessing, our method partitions the analytical space (the bounding box of the medium) into sub-spaces (partitions) according to the spatial variation of the mean free path in the medium. During rendering, the locations of the scattering events are determined adaptively using the kd-tree.

我们方法的一个关键贡献是基于成本模型的自动分区方案，用于评估采样效率。我们通过解决*最大的空矩形问题*来找到关于成本模型的最优划分。
A key contribution of our method is an automatic partitioning scheme based on a cost model for evaluating the sampling efficiency. We find the optimal partitioning with respect to the cost model by solving the *largest empty rectangle problem*.

我们方法的另一个贡献是其可扩展性以处理广泛的场景，包括整个天空范围内的云和大气。
Another contribution of our method is its scalability to handle a wide range of scenes including clouds and atmosphere in whole-sky-scale.

## 3 Free Path Sampling

假设散射事件 $i$ 已经生成。散射事件 $(i+1)$ 的位置通常通过使用随机数对自由路径 $d_i$ 和散射方向 $\omega_i$ 进行采样来确定。对于散射方向，我们可以使用常规的重要性采样。自由路径 $d_i$ 应根据以下概率密度函数进行采样：
Assume the scattering event $i$ has already been generated. The location of scattering event $(i+1)$ is typically determined by sampling the free path $d_i$ and the scattering direction $\omega_i$ by using random numbers. For the scattering direction, we can use conventional importance sampling. The free path $d_i$ should be sampled according to the following probability density function:
$$
P(\mathbf x_{i+1}|\mathbf x_i,\omega_i)=T(\mathbf x_i,\mathbf x_{i+1})\mu_t(\mathbf x_{i+1})
$$
Delta 追踪里面每次采样点长度为：
$$
t=-\frac{\ln(1-\xi)}{\bar\mu}
$$
则期望长度为：
$$
\begin{align*}
\mathrm{E}[t]&=-\int_0^1\xi\frac{\ln(1-\xi)}{\bar\mu}\mathrm d\xi\\
&=-\frac{1}{\bar\mu}\left[\frac{1}{2}t^2\ln t-\frac{1}{4}t^2-t\ln t+t\right]_0^1\\
&=\frac{3}{4\bar\mu}
\end{align*}
$$


## 4 Unbiased Sampling and Space Partitioning

为了有效地对自由路径进行采样，我们将包含参与介质的边界框划分为一些分区。然后可以将每个分区的消光系数设置为小于全局的消光系数。这样做的结果是 1) 每次迭代 $(3/4\bar\mu)$ 中的平均距离变得更长，并且 2) 散射事件更有可能被接受。因此，可以减少预期的总迭代次数。
To efficiently sample the free path, we partition the bounding box containing the participating medium into some partitions. Then the majorant extinction coefficient for each partition can be set smaller than that for the entire domain. The results of this are that 1) the average distance in each iteration $(1/\bar\mu)$ becomes longer, and 2) scattering events are more likely to be accepted. Thus, the expected total number of iterations can be reduced.

### 4.1 Unbiased Sampling in Partitioned Space

光线 $\mathbf r(\mathbf x.\omega)$ 穿过两个相邻区域 $i,j$，分别为 $\bar\mu^{(i)}$ 和 $\bar\mu^{(j)}$，$s,q,t$ 是光线穿过边界时候的交点。在 $(s,t]$ 采样时候，先采样 $(s,q]$，可能会得到采样，也可能超过了范围。如果超过了，就在 $(q,t]$ 中采样。



### 4.2 Partitioning Participating Media

如果我们对一个近乎均匀的区间进行分区，则在进行到相邻区间时，需要进行一次额外的迭代以将自由路径倒回到 $q$，这与没有分区的情况相比是一种浪费。
If we partition a nearly homogeneous interval, an additional iteration is required to rewind the free path to $q$ when proceeding to the adjacent interval, and this would be a waste compared to the case with no partitioning.

#### One-dimensional case

如果 $(s,t]$ 中只有一个极大值点，那么采样次数就是 $(t-s)\bar\mu_{st}$。

这时候切分一次，采样次数就变成了 $(q-s)\bar\mu_{sq}+(t-q)\bar\mu_{st}+1$，$1$ 表示穿过切分的时候要重新计算距离。

做差：
$$
N_r(q)=(t-s)\bar\mu_{st}-[(q-s)\bar\mu_{sq}+(t-q)\bar\mu_{st}+1]=(q-s)(\bar\mu_{st}-\bar\mu_{sq})-1
$$
只要 $N_r>0$，就代表切分是有收益的。因此区间内最优做法为：
$$
q=\arg\max_{q'\in(s,t]}N_r(q')
$$
如果区间内有多个峰，观察到 $(q-s)(\bar\mu_{st}-\bar\mu_{sq})$ 代表了一个矩形的面积，而 $(t-s)\bar\mu_{st}$ 就是整个区间矩形的面积。则问题可以转换为在两个峰之间寻找最大的矩形然后挖去。因此：
$$
R=\arg\max_{R'\sub E}N^{\triangle}(R')=\arg\max_{R'\sub E}[A(R')-T(R')]
$$
其中 $E$ 是填充的区域，$N^{\triangle}(R')$ 表示迭代的减少次数。$A(R')$ 是挖去的矩形面积，$T(R')$ 是每次划分多出来的采集数。

对于解决方案，我们将空区域 $E$ 沿 $x$ 轴离散为 $n$ 个 bin，并使用动态规划找到 $E$ 中最大的空矩形。
For the solution, we discretize the empty region $E$ into $n$ bins along the $x$ axis and use dynamic programming to find the largest empty rectangle in $E$.

---

我们首先从左到右扫描 bin，并检查所有最大矩形。如果一个矩形不能沿任何方向延伸，则它是极大的。
We first scan the bins from left to right, and check all of the maximal rectangles. A rectangle is maximal if it cannot be extended along any axis.

子问题 $i$ 是找到其左边缘包含了第 i 个 bin 的最大矩形的问题，并且每个子问题都有一个关联的高度值。
A subproblem $i$ is the problem to find the largest rectangle whose left edge touches the left edge of the i-th bin, and each subproblem has an associated height value.

```python
def largestRec(bins):
    i = 0
    stack = []
    R = []
    max_area = 0
    prev_height = 0
    for i in range(len(bins)):
        h = bins[i]
        if prev_height < h:
            # expand the area
            stack.append((i, h))
        elif prev_height > h:
            (j, h_j) = stack[-1]
            while len(stack) > 0 and h_j >= h:
                stack.pop(-1)
                area = abs(bins[i] - bins[j]) * h_j
                if area > max_area:
                    max_area = area
                    R = [i, j, h_j]
            stack.append((j + 1, h))
        prev_height = h
    if R == []:
        R = [len(bins) - 1, 0, bins[0]]
    return max_area, R
```

$O(n)$ time and space complexity.

---

假设我们已经找到了最大的矩形，设 $q_1$ 和 $q_2$ 是矩形的两端，我们首先切分的是靠近中间的端，使得 $(s,q]$ 和 $(q,t]$ 尽量均匀。另一部分在下一次递归时候切开。

构造 kd-tree 使得内部节点代表对应的划分位置，叶子节点存储对应区间的 $\bar\mu$。
The kd-tree is constructed so that an inner node represents the corresponding partitioning location and a leaf node stores the majorant extinction coefficient of the corresponding interval.

#### Three-dimensional case

找到分割平面的一种直接方法是在由 $w = \mu(x)$、$w = \bar\mu$ 和 $V$ 界定的 4D 区域中找到最大的超矩形。 然而，目前 4D 空间中最大空矩形问题的解决方案具有很高的计算复杂度。
A straightforward way to find the partitioning plane is to find the largest hyper-rectangle in the 4D region bounded by $w = \mu(x)$, $w = \bar\mu$ and $V$. However, current solutions for the largest empty rectangle problem in 4D space have high computational complexity.

我们将问题视为沿三个轴中每一个轴的三个单独的一维问题，并尝试找到每个轴的分区候选者。然后，我们从这些候选人中选择最好的一个，比较他们的收益。
We regard the problem as three individual 1D problems along each of the three axes, and try to find the partitioning candidate for each of the axes. We then choose the best one from those candidates comparing their benefits.

我们首先解释如何找到 $x$ 轴的分区候选者。
We first explain how to find the partitioning candidate for the $x$ axis.
$$
\mu_{+}(x)=\max_{y,z,(x,y,z)\in V}\mu(x,y,z)\\
\mu_{\delta}(x)=\max_{y,z,(x,y,z)\in V}\mu(x,y,z)-\min_{y,z,(x,y,z)\in V}\mu(x,y,z)
$$
即 $\mu_+$ 是某一维度上的系数最大值，$\mu_{\delta}$ 是某一维度的跨度。

使用 $\mu_{\delta}(x)$ 使得在这种情况下检测非均匀区域成为可能，因为当平面中的介质分布时，$\mu_{\delta}(x)$ 变得很大，在垂直于 $x$ 轴的 $x$ 处是不均匀的，如果介质的分布几乎是均匀的，则它会变小。
Using $\mu_{\delta}(x)$ makes it possible to detect the non-uniform regions in such a case, because becomes $\mu_{\delta}(x)$ large when the distribution of the medium in the plane at $x$ perpendicular to the $x$ axis is inhomogeneous and becomes small if the distribution of the medium is nearly homogeneous.

随后在 $N_+^\triangle$ 和 $0.7*N_\delta^\triangle$ 中选大的值，以该种划分作为 kd 树划分。

为了处理物体表面，我们首先忽略表面执行上述过程。然后，对于每个叶节点，如果节点中的表面是对齐的，我们进一步在节点中为这些表面构建一个子 kd-tree。 与对表面和介质使用单独的 kd 树相比，我们可以减少多余的射线表面交叉点。
To handle object surfaces, we first perform the above process ignoring the surfaces. Then, for each leaf node, if surfaces are aligned in the node, we further construct a sub kd-tree in the node for the surfaces. Compared to using separate kd-trees for the surfaces and the medium, we can reduce redundant ray-surface intersections.

### 4.3 Ray Traversal

```python
def KDTreeFreePathSampling(x, omega):
    while True:
        p = node.left()
        if p == None:
            return math.inf
       	dmin, dmax = p.bound(x, omega)
        mu = p.mu
        d = deltaTracking(x, omega, mu, dmin, dmax)
        d_isect = math.inf
        if intersectWithSurface(x, omega, p):
            d_isect = distanceToSurface
        if d >= d_isec:
            return d_isec, surf
        if d < d_max:
            return d, volu
```

先求光线和节点的两端，然后求体碰撞和面碰撞，如果符合条件返回；否则去下一个节点。







