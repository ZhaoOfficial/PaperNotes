# Volume Path Guiding Based on Zero-Variance Random Walk Theory

## Abstract

路径构建受散射方向和距离采样、俄罗斯轮盘赌和分裂策略的影响。我们提出了一套一致的体积路径构建技术，其中所有这些采样决策都由伴随着传输解决方案的缓存估计指导。
Path construction is influenced by scattering direction and distance sampling, Russian roulette, and splitting strategies. We present a consistent suite of volumetric path construction techniques where all these sampling decisions are guided by a cached estimate of the adjoint transport solution.

该策略基于零方差路径采样方案的理论，考虑了体积运输的空间和方向变化。
The proposed strategy is based on the theory of zero-variance path sampling schemes, accounting for the spatial and directional variation in volumetric transport.

粒子碰撞距离与透射率和伴随传输解的乘积成正比。
The particle collision distance proportionally to the product of transmittance and the adjoint transport solution.

同样根据相位函数和入射辐射估计的乘积对散射方向进行采样。
Scattering directions are likewise sampled according to the product of the phase function and the incident radiance estimate.

## Introduction

MC 体积光传输必须以最佳方式考虑随机决策：所有空间、方向和路径长度的每个采样决策都应以一致的方式处理。
MC volumetric light transport must optimally account for stochastic decision: all spatial, directional, and path-length every sampling decisions should be treated in a consistent manner.

**引导传输路径**：零方差路径采样理论在这里作为理论框架，为渐进的路径构建期间使用的一组局部最优采样决策构建全局最优、联合路径空间采样分布提供了基本规则。
**Guiding transport path**: Zero-variance path sampling theory serves as the theoretical framework here, providing ground rules for building a globally optimal, joint path-space sampling distribution using but a set of locally optimal sampling decisions used during incremental path construction.

具体来说，我们提出了无偏方法来高效准确地进行所有体积采样决策：
Specifically, we propose unbiased methods to efficiently and accurately conduct all the volumetric sampling decisions:

- 我们的关键技术贡献是碰撞距离和散射/无散射决策的新技术，其执行与透射率和散射内辐射的乘积成正比。
  Our key technical contribution is a novel technique for the collision distance and scatter/no-scatter decisions performed in proportion to the full product of both transmittance and in-scattered radiance.
- 我们根据相位函数和入射辐射的乘积对散射方向进行采样。
  We sample scattering directions according to the product of the phase function and incident radiance.
- 最后，我们的路径终止/分割决策根据估计的贡献，在局部调整追踪路径的密度。
  Finally, our path termination/splitting decisions locally adjust the density of traced paths according to their contributions estimated.

由于实际实现需要伴随传输解决方案的知识，我们在粒子追踪预传递期间缓存近似伴随。
Since a practical implementation requires knowledge of the adjoint transport solution we cache approximate adjoints during a particle tracing pre-pass.

首先，它们在各个局部子域中与体积传输方程的完整被积函数成比例地采样。其次，所有局部采样决策始终由相同的潜在伴随解决方案指导。
First, they sample proportionally to the full integrand of the volume transport equation, across respective local sub-domains. Second, all local sampling decisions are consistently guided by the same underlying adjoint solution.

## Related Work

表格透射率采样引入了偏差，而其无偏差的对应物 delta 追踪对于具有显着光密度变化的介质变得效率低下。一个加权追踪方案而不是一个同样被证明模拟有效。
Tabulated transmittance sampling introduces bias, whereas its unbiased counterpartdelta tracking becomes inefficient for media with significant optical density variation. A weighted tracking scheme instead of an one has likewise proven analog effective.

## Background

### Volume Rendering Equation

不考虑自发光介质。
Not consider emissive media.
$$
L(x,\omega)=\underbrace{T(x,z)L_o(z,\omega)}_{L_s(x,\omega)}+\underbrace{\int_0^s T(x,y)\mu_s(y)\left[\int_{\mathcal S^2}f(y,\omega,\omega')L(y,\omega')d\omega'\right]dt}_{L_m(x,\omega)}
$$
其中：$z=x-s\omega,y=x-t\omega$.
$$
T(x_1,x_2)=\exp(-\tau(x_1,x_2))\quad\tau(x_1,x_2)=\int_0^{\|x_1-x_2\|}\mu_t(y)dt
$$

### Zero-variance Path Sampling Theory

假设非发光介质，发光只能发生在光源表面。“最后事件”估计器：粒子仅在发光位置收集辐射，然后立即终止。
Assuming non-emissive media, emission can only occur on a light source surface. "Last event" estimator: the particle only collects radiance at an emitting position and is then immediately terminated.

每个微粒路径 $r_i$ 的贡献由累积权重 $a$ 和路径末端的发射辐射 $L_e$ 给出：
The contribution of each particle's path $r_i$ is given by the cumulative particle weight $a$ and the emitted radiance $L_e$ at the end of the path:
$$
I=\frac{1}{N}\sum_{i}^{N}a(r_i)\cdot L_e(x_{i,M},\omega_{i,M-1})
$$
到达光源前共有 $M-1$ 次散射事件。
Each particle undergoes $M-1$ scattering events. At each event and its coupled propagation step, the particle performs a state change, its position and orientation change from $(x_j,\omega_j)$ to $(x_{j+1},\omega_{j+1})$
$$
a(r)=\prod_{j=1}^{M-1}a_j(r)
$$
如果只有体作用：
$$
a_j(r)=\frac{1}{q_{j-1}}\frac{T(x_{j-1},x_j)\sigma_s(x_j)f(x_j,\omega_{j-1},\omega_j)}{P(x_j,\omega_j|x_{j-1},\omega_{j-1})}
$$
$P,q_{j-1}$ 之后定义。

**Zero-variance path sampling strategy:**

这是因为每一个局部决策都依赖于全局的运输解决方案的信息。
This is enabled by the fact that each of these local decisions relies on (is "guided" by) information about the global transport solution.

1. 给定粒子状态 $(x_j,\omega_j)$，状态变化的第一步是确定新的粒子位置 $x_{j+1}$。 这包括两个步骤：
   Given a particle state $(x_j,\omega_j)$, the first step of the state change is to determine the new particle position $x_{j+1}$. This involves two sub-steps:

   首先，离散决策确定路径是探索来自最近表面的贡献 $L_s$，还是探索来自体介质贡献 $L_m$。
   First, a discrete decision determines whether the path explores the contribution $L_s$ from the nearest surface, or the volumetric contribution $L_m$ from the medium.

   在 $x_j$ 和下一个表面之间采样 $L_m$ 的最佳概率是：
   The optimal probability for sampling $L_m$ between $x_j$ and the next surface is:
   $$
   P_m(x_j,\omega_j)=\frac{L_m(x_j,\omega_j)}{L_s(x_j,\omega_j)+L_m(x_j,\omega_j)}
   $$
   接下来，假设选择了体贡献，必须采样到下一个散射位置 $x_{j+1}$ 的碰撞距离 $d_{j+1}$。从 $(x_j,\omega_j )$ 开始采样 $d_{j+1}$ 的最佳 PDF 是：
   Next, assuming that the volume contribution was selected, the collision distance $d_{j+1}$ to the next scattering position at $x_{j+1}$ has to be sampled. The optimal PDF for sampling $d_{j+1}$ starting at $(x_j,\omega_j )$​ is:
   $$
   P(d_{j+1}|x_j,\omega_j)=\frac{T(x_j,x_{j+1})\mu_s(x_{j+1})L_i(x_{j+1},\omega_j)}{L_m(x_j,\omega_j)}
   $$
   
2. 在新的散射位置 $x_{j+1}$ 处，随机游走需要探索入射散射光 $L_i$ 的积分，即散射到 $\omega_j$ 方向。 因此，需要采样一个新的方向 $\omega_{j+1}$，最优的 PDF：
    At the new scattering position $x_{j+1}$, the random walk needs to explore the integral of in-scattered radiance $L_i$, i.e., scattered into the direction $\omega_j$. Therefore, a new direction $\omega_{j+1}$ needs to be sampled, with the optimal PDF:
    $$
    P_{\omega}(\omega_{j+1}|x_{j+1},\omega_j)=\frac{f(x_{j+1},\omega_j,\omega_{j+1})L(x_{j+1},\omega_{j+1})}{L_i(x_{j+1},\omega_j)}
    $$
    
3. 获得 $L_i$ 本身涉及对入射光亮度 $L$ 的评估。因此重复步骤 1-2，直到我们到达位置 $x_j$，该位置沿 $\omega_{j-1}$ 方向发射能量，并决定终止随机游走并返回其加权辐射 $a(r )\cdot L_e$。$x_j$ 处的最优终止概率为：
     Obtaining $L_i$ itself involves an evaluation of the incoming radiance $L$. Steps 1–2 are thus repeated until we reach a position $x_j$, which emits energy in the direction $\omega_{j-1}$ and the decision is made to terminate the random walk and return its weighted emission $a(r)\cdot L_e$. The optimal termination probability at $x_j$ is:
     $$
     P(x_j,\omega_{j-1})=\frac{L_e(x_j,\omega_{j-1})}{L_e(x_j,\omega_{j-1})+L_r(x_j,\omega_{j-1})}=\frac{L_e(x_j,\omega_{j-1})}{L_o(x_j,\omega_{j-1})}
     $$
     由此产生的幸存概率为：
     with the resulting survival probability of:
     $$
     q_j=1-P(x_j,\omega_{j-1})
     $$
     在 $x_j$ 处没有辐射的情况下，继续前行的零方差概率为 1。
     In the case there is no emission at $x_j$, the zero-variance probability of continuing the walk is one.

将所有东西放在一起，状态转移 PDF 由这些采样决策中的每一个的条件 PDF 组成：
Putting everything together, the resulting PDF for a state change is composed of the conditional PDFs for each of these sampling decisions:
$$
\begin{align*}
P(x_{j+1},\omega_{j+1}|x_j,\omega_j)&=P_m(x_j,\omega_j)P(d_{j+1}|x_j,\omega_j)P_{\omega}(\omega_{j+1}|x_{j+1},\omega_j)\\
&=\frac{T(x_j,x_{j+1})\sigma_s(x_{j+1})f(x_{j+1},\omega_j,\omega_{j+1})L(x_{j+1},\omega_{j+1})}{L_s(x_j,\omega_j)+L_m(x_j,\omega_j)}
\end{align*}
$$

## Methodology

我们的方法需要依赖于估计引导随机路径构建过程的伴随传输解决方案，因为所有部分 PDF 都需要全局能量分布的先验。
Our method needs to rely on estimating the adjoint transport solution that guides the stochastic path construction process, since all the partial PDFs require the knowledge of the global energy distribution.

## Adjoint Solution Estimate

零方差采样需要光传输解决方案的先验知识。鉴于这种循环依赖，我们的方法求助于近似值，特别是入射辐射 $L$ 和内散射辐射 $L_i$。这些量存储在混合的空间和方向缓存结构中。
Zero-variance sampling requires prior knowledge of the light transport solution. Given this cyclic dependency, our method resorts to approximations, specifically of the incident radiance $L$ and in-scattered radiance $L_i$. These quantities are stored in a hybrid spatial and directional cache structure.

在简短的预处理阶段，我们初始化先验，其中从光源追踪光子，然后使用空间划分 (kD) 树进行聚类。每个叶节点中的光子用于为该节点中的体像构建入射辐射的方向。
We initialize our prior estimates in a short pre-processing phase, where photons are traced from the light sources and then clustered using a spatial subdivision (kD) tree. Photons in each leaf node are used to build the directional representation of the incoming radiance for the volume contained within that node.

我们使用基于 vMF 分布的参数混合模型来表示方向分布：定义在球体上，因此不需要额外的映射；内散射辐射的估计是通过卷积计算的；对于任何方向，入射辐射和相位函数之间的乘积分布也以闭式获得。
We represent directional distributions using a parametric mixture model based on the vMF distribution: defined on the sphere and therefore no additional mapping is required; the in-scattered radiance estimate is calculated through an efficient closed-form convolution; the product distribution between the incident radiance and the phase function for any directional configuration is also obtained in a closed-form.

### Incoming radiance estimate

为了表示介质内不同位置 $x$ 的传入辐射分布，我们使用由空间变化参数集 $\Theta(x)$ 描述的 vMF 混合 $\mathcal V$。这种混合通过一组 $K$ 个加权 vMF 波瓣 $\mathscr v$ 表示球形分布：
To represent the incoming radiance distributions at different locations $x$ inside a medium, we use vMF mixtures $\mathcal V$ described by the spatially-varying parameter set $\Theta(x)$. Such a mixture represents a spherical distribution via a set of $K$ weighted vMF lobes $\mathscr v$:
$$
\mathcal V(\omega|\Theta)=\sum_{i=1}^{K}\pi_i\cdot\mathscr v(\omega|\mu_i,\kappa_i)
$$
$\Theta$ 包含了每个组分的权重 $\{\pi_1,\dots\pi_K\}$，平均方向 $\{\mu_1,\dots,\mu_K\}$，精度 $\{\kappa_1,\dots,\kappa_K\}$。每个 vMF 波瓣是一个球面高斯，旋转到了某个方向 $\mu$，底面积和 $\kappa$ 成反比。

入射辐射估计值 $\tilde{L}$ 与归一化的 vMF 混合相关，通过标量辐照度缩放：
The incident radiance estimate $\tilde{L}$ is related to the normalized vMF mixture through scaling by the scalar irradiance:
$$
\tilde{L}(x,\omega)=\Phi(x)\cdot\mathcal V_L(\omega|\Theta_L(x))\\
\Phi(x)=\int_{\mathcal S^2}L(x,\omega')d\omega'
$$

### In-scattered radiance estimate

从功能上讲，入射辐射亮度与相位函数的球面卷积给出了入射辐射亮度 $L_i$
Functionally, the in-scattered radiance $L_i$ is given by spherically convolving the incident radiance with the phase function:
$$
\tilde{L}_i(x,\omega)=\Phi(x)\cdot\int_{\mathcal S^2}f(x,\omega,\omega')\mathcal V_L(\omega'|\Theta_L(x))d\omega'
$$
由于相位函数通常是旋转不变的，它们可以用一个或多个 vMF 波瓣表示。两个 vMF 波瓣的卷积产生另一个 vMF 波瓣，它可以通过解析计算提供一种高效的方法来获得散射辐射估计：
Since phase functions are typically rotationally invariant, they can be represented with one or more vMF lobes. The convolution of two vMF lobes results in another vMF lobe, which can be calculated analytically providing a highly efficient means to obtain the integrated in-scattered radiance estimate:
$$
\tilde{L}_i(x,\omega)=\Phi(x)\cdot\mathcal V_{L_i}(\omega'|\Theta_{L_i})\\
\mathcal V_{L_i}(\omega)=(\mathcal V_f*\mathcal V_L)(\omega)
$$

## Sampling Procedures

特定于体积的采样决策，涉及在给定粒子当前位置的情况下沿粒子的传播方向找到散射位置。
The volume-specific sampling decision that concerns finding a scattering location along a particle’s propagating direction, given its current position.

### Guided Product Distance Sampling

探索介质贡献 $L_m$ 或表面贡献 $L_s$ 可以合并为单个距离采样决策。
Exploring the medium contribution $L_m$ or the surface contribution $L_s$ can be merged into a single distance sampling decision.
$$
P(d|x,\omega)=\frac{T(x,x_d)\sigma_s(x_d)L_i(x_d,\omega)}{L(x,\omega)}
$$

#### Naive Solution

根据零方差 PDF 进行的采样使用我们的体积辐射估计一种直接方法是，通过对上述等式进行均匀采样，明确构建表格离散 PDF 及其相应的累积密度函数 (CDF)。
A straightforward way to use our volumetric radiance estimates to sample according to the zero-variance PDF is to explicitly build a tabulated discrete PDF and its corresponding cumulative density function (CDF) by a uniform sampling of Equation above.

这种方法的缺点是它需要沿着光线的整个长度前进进以评估离散 PDF 的所有区间，并在做出任何采样决定之前计算其归一化因子。
The downside of this method is that it requires stepping along the entire length of the ray to evaluate all the bins of the discrete PDF, and calculate its normalization factor—before making any sampling decision.

#### Proposed Incremental Distance Sampling

在追踪过程逐步通过介质时隐式地对其进行动态估计。
The tracking procedure implicitly estimates it on the fly while stepping through the medium.

由于采样从 $x$ 沿 $\omega$ 方向进行，我们将在本节的其余部分中在一维距离空间内操作，原点为 $x$。
Since the sampling proceeds along the direction -- $\omega$ starting at $x$, we will for the remainder of this section operate within the 1D distance spacewith the origin at $x$.

$D$：散射距离随机变量。
$$
P(D\le d)=\frac{\int_0^dT(0,d')\mu_s(d')L_i(d',\omega)\mathrm dd'}{L(0,\omega)}
$$

#### Per-bin sampling

我们现在提议转移到下一个点 $d_{i+1}$，目标是确定碰撞发生在 $d_i$ 和 $d_{i+1}$ 之间的概率，或者在 $d_{i+1}$。 如果发生前一种情况，则跟踪停止，而在后一种情况下，它会从距离 $d_{i+1}$ 重新开始。
We now propose the transition to the next point $d_{i+1}$, with the goal to determine the probability of the collision being between $d_i$ and $d_{i+1}$, or alternatively, after $d_{i+1}$. If the former case occurs, the tracking stops, whereas in the latter case, it restarts again from distance $d_{i+1}$.

假设区间内入射辐射度 $L_i$ 和 散射系数 $\mu_s$ 不变
$$
P(D\le d_{i+1}|d_i<D)\approx\frac{1-T(d_i,d_{i+1})}{\mu_t(d_i)}\cdot\frac{\mu_s(d_i)L_i(d_i)}{L(d_i,\omega)}
$$
且在之前都不碰撞的概率是：
$$
P(d_i<D)=\prod_{j=0}^{i-1}1-P(D\le d_{j+1}|d_j<D)
$$
因此在该区间发生碰撞的距离就是：
$$
P(d|x,\omega)=P(d_i<D)P(D\le d_{i+1}|d_i<D)P(d|d_i,d_{i+1},\mu_t)
$$

#### Sampling stability: local versus global estimates

#### Sampling stability: incoming radiance filter

为了在跟踪过程中改进入射辐射估计，我们可以重用来自先前区间的已经评估的估计值，以在 $d_i$ 处改进当前区间内的近似值。
To improve the incident radiance estimate during the tracking, we can reuse the already evaluated estimates from previous bins to refine our approximation within the current bin at $d_i$.
$$
\tilde{L}^I(d_i,\omega)=(1-A)\cdot \tilde{L}^{II}(d_i,\omega)+A\cdot\tilde{L}(d_i,\omega)\\
\tilde{L}^{II}(d_i,\omega)=\frac{\tilde{L}^I(d_{i-1},\omega)-(1-T(d_{i-1},d_i))\cdot\alpha(d_{i-1})\cdot\tilde{L}(d_{i-1},\omega)}{T(d_{i-1},d_i)}
$$

### Guided Product Directional Sampling

### Guided Russian Roulette and Splitting

在零方差理论采样的背景下，使用正确的 RR 引导概率尤其重要，因为过于激进的路径终止会显着增加方差。
Using the correct RR guided probability is especially important in the context of zero-variance-theoretical sampling, since an overly aggressive path termination can dramatically increase variance.

这表明一个理想的路径生存概率与路径的预期贡献成正比：低贡献路径更有可能被终止（拒绝），因此幸存的路径都贡献相等（即，在按生存概率归一化之后）。
This suggests an ideal path survival probability proportional to the path’s expected contribution: low-contribution paths are more likely to be terminated (rejected), so that surviving paths all contribute equally (that is, after normalizing by the survival probability).

#### Volumetric guided RR and splitting

GRR&S 计算俄罗斯轮盘赌和部分生成路径的分裂因子 $q\in[0,\infty]$ 作为路径的预期贡献 $E[r]$ 与像素的真实值 $I$ 之间的比率。
GRR&S computes the Russian roulette and splitting factor $q\in[0,\infty]$ for a partially-generated path as the ratio between the path's expected contribution $E[r]$, and pixel's true value $I$.

$q<1$：做俄罗斯轮盘赌。

$q>1$：路径预期贡献高于实际像素值，我们将其拆分为 $n = \lfloor q\rfloor + 1$ 子路径。

#### Distance GRR&S

#### Directional GRR&S







