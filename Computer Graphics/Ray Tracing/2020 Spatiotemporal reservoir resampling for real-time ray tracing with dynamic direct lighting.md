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

与以前的 RIS 应用程序不同，我们使用一个小的固定大小的数据结构 - 一个只存储接受的样本的“存储库” - 和一个相关的采样算法（在非图形应用程序中经常使用）来帮助实现稳定的实时性能。
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

IS 从一个已知 PDF 的分布里面采样 $N$ 次：
$$
\langle{L}\rangle^N_{\text{is}}=\frac{1}{N}\sum_{i=1}^N\frac{f(x_i)}{p(x_i)}\approx L\tag{3}
$$
如果当 $f(x)$ 非零时 $p(x)$ 为正，则 IS 保持无偏，并且理想情况下 $p(x)$ 与 $f(x)$ 相关以减少方差。
IS remains unbiased if $p(x)$ is positive whenever $f(x)$ is non-zero, and ideally $p(x)$ is correlated with $f(x)$ to reduce variance.

#### Multiple Importance Sampling (MIS)

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

使用 MIS 从上色项的线性组合中采样的另一种方法是与某些项的乘积大致成比例地进行采样。
An alternative to sampling from a linear combination of shading terms using MIS is to sample approximately proportional to the product of some of the terms.

重采样重要性采样通过从一个次优但容易抽样的源分布 $p$（例如 $p\propto L_e$）中生成 $M\ge1$ 个候选样本 $\mathbf{x}=\{x_1,\dots,x_M\}$ 来实现。然后，它以离散的概率从这个候选池中随机选择一个下标 $z=\{1,\dots,M\}$：
Resampled importance sampling achieves this by generating $M\ge1$ candidate samples $\mathbf{x}=\{x_1,\dots,x_M\}$ from a source distribution $p$ that is sub-optimal, but easy to sample from (e.g., $p\propto L_e$). It then randomly chooses an index $z=\{1,\dots,M\}$ from this pool of candidates with discrete probabilities:
$$
p(z\mid\mathbf{x})=\frac{\mathrm{w}(x_z)}{\sum_{i=1}^{M}\mathrm{w}(x_i)}\quad\text{with}\quad \mathrm{w}(x)=\frac{\hat{p}(x)}{p(x)}\tag{5}
$$
由期望的目标 PDF $\hat{p}(x)$ 驱动，对于该目标可能不存在实际的采样算法（例如 $\hat{p}\propto\rho\cdot L_e\cdot G$）。
driven by a desired target PDF $\hat{p}(x)$, for which no practical sampling algorithm may exist (e.g., $\hat{p}\propto\rho\cdot L_e\cdot G$).

选择一个 $y\equiv x_z$ 的样本并用于单样本 RIS 估计器：
A sample $y\equiv x_z$ is selected and used in the 1-sample RIS estimator:
$$
\langle{L}\rangle^{M,1}_{\text{ris}}=\frac{f(y)}{\hat{p}(y)}\biggl(\frac{1}{M}\sum_{i=1}^{M}\mathrm{w}(x_i)\biggr)\tag{6}
$$

---

$$
\langle{L}\rangle^{M,1}_{\text{ris}}=\langle{L}\rangle^{M,1}_{\text{mis}}=\sum_{s=1}^M\frac{f(x_{s})}{\sum_{j=1}^{M}p_j(x_{s})}=\sum_{s=1}^M\frac{f(x_{s})}{\sum_{j=1}^{M}p_j(x_{s})}=\frac{1}{M}\sum_{i=1}^{M}\frac{f(x_i)}{\hat{p}(x_i)}\frac{\hat{p}(x_i)}{p(x_i)}
$$

---

直观地说，估计器使用 $y$，就好像它是从 $\hat{p}$ 中提取的，然后使用括号中的因子来纠正 $y$ 的真实分布仅近似于 $\hat{p}$ 的事实。
Intuitively, the estimator uses $y$ as if it were drawn from $\hat{p}$ and then uses the parenthesized factor to correct for the fact that the true distribution of $y$ only approximates $\hat{p}$.



### 2.2 Weighted Reservoir Sampling

## 3 Streaming RIS With Spatiotemporal Reuse

### 3.1 Streaming RIS Using Reservoir Sampling

### 3.2 Spatiotemporal Reuse

#### Spatial Reuse

#### Temporal Reuse

#### Visible Reuse

## 4 (Eliminating) Bias in Multi-distribution RIS

### 4.1 Analysis the RIS Weight

#### Explanation of Reweighing Factor

### 4.2 Biased RIS

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
