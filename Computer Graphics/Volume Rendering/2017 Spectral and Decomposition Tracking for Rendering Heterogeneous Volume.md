# Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes

## 0 Abstract

两种在不均匀介质中无偏的自由程采样技术。
Two novel unbiased techniques for sampling free paths in heterogeneous participating media.

**Decomposition tracking**

我们的分解追踪将介质分成控制分量和残差分量，并分别对它们进行采样，来加速自由程的构建。
Our decomposition tracking accelerates free-path construction by splitting the medium into a control component and a residual component and sampling each of them separately.

控制分量解析地构建自由程。
Defining the control component to allow constructing free paths in closed form.

然后通过添加虚拟介质将剩余的不均匀介质分量均质化，并使用加权 delta 追踪进行处理，从而无需计算消光函数的严格界限。
The residual heterogeneous component is then homogenized by adding a fictitious medium and handled using weighted delta tracking, which removes the need for computing strict bounds of the extinction function.

**Spectral tracking**

光谱追踪可在彩色介质中实现高效的光传输模拟。
Spectral tracking enables efficient light transport simulation in chromatic media.

我们修改了自由程分布，最小化了路径吞吐量的波动，从而减少估计方差。
We modify free-path distributions to minimize the fluctuation of path throughputs and thereby reduce the estimation variance.

## 1 Introduction

当介质的外观以高阶散射为主时，每条光路可以包含多达数千个散射事件。**对这些散射事件进行建模--通过采样自由程--是合成图像成本的大头，也是本文的主要重点。**
When the appearance of the medium is dominated by high-order scattering, each such light path can consist of up to thousands of scattering events. **Modeling these scattering events -- by means of sampling free paths -- is the main contributor to the cost of synthesizing an image, and the main focus of this paper.**

虚拟介质的特性被定义为，在碰撞时，光向前散射，其方向和辐射强度不变。
The properties of the fictitious medium are defined such that, upon a collision, light scatters forward with its direction and radiant intensity unaltered.

然后仅使用其中一个分量构建单个自由程或其分段。如果使用控制分量，则计算是完全解析的，我们避免了昂贵的内存查找或程序评估，从而使构建路径的成本保持在较低水平。
Individual free paths, or their segments, are then constructed using only one component. If the control component is used, the calculation is fully analytic and we avoid expensive memory lookups or procedural evaluations, keeping the cost of constructing paths low.

我们还解决了有效处理与波长相关的碰撞系数的挑战。我们分析了加权追踪的几种矢量化变体，并提出了一种历史感知策略来设置碰撞概率，以防止路径吞吐量的几何增长。
We also address the challenges of efficiently handling wavelength-dependent collision coefficients. We analyze several vectorized variants of weighted tracking and propose a history-aware strategy for setting collision probabilities to prevent geometric growth of the path throughput.

## 2 Existing Tracking Algorithms

数值求解 RTE 的一个关键工具是随机构建由自由程组成的光粒子轨迹，即介质中两次碰撞之间的直线光路。
A key tool for numerically solving the RTE is a stochastic construction of light particle's trajectories that consist of free paths, i.e. straight flight paths between subsequent collisions with the medium.

依赖于构建此类路径的无偏估计器可分为，直接模拟实际物理过程的模拟方法；和非模拟方法，它们使自由路径分布和重新加权样本产生偏差，以产生具有较低方差的无偏结果或没有模拟方法的限制。

Unbiased estimators that rely on constructing such paths can be classified as either analog methods, that directly model the actual physical process, or non-analog methods, that bias the free-path distribution and re-weight samples to produce unbiased results with lower variance or without the constraints of the analog process.

*Closed-form Tracking*

那些具有常数、多项式或指数变化消光系数的自由程可以使用逆变换采样进行采样。
Those with constant, polynomial, or exponentially varying extinction coefficient, free paths can be sampled using the inverse transform sampling.
$$
T(t)=\exp({-\int_0^{t}\mu_t(\mathbf x(\tau))d\tau})
$$
可以看作光线穿过距离 $t$ 仍不与介质发生碰撞的概率。设 $X$ 是自由程长度的随机变量，则：
$$
P(X>t)=T(t)\\
P(X\le t)=1-T(t)=F(t)\\
$$
假设 $F(t)$ 存在逆映射，且消光系数为常数，则逆采样可得
$$
t'=-\ln(1-\xi)/\mu_t
$$
缺点：消光系数或光学厚度不能解析求出。

*Regular Tracking*

如果介质由分段均匀区域组成，则可以通过沿光线访问边界，并积分各个区域的光学厚度，直到其负指数值达到随机选择的透射率值来找到自由程。
If the medium consists of piecewise homogeneous regions, the free path can be found by visiting boundaries along the ray and integrating optical thickness of individual regions until its negative exponentiated value reaches a stochastically chosen value of transmittance.

缺点：边界寻找是比较耗时间的。

*Ray Marching*

降低常规追踪成本的一种直接方法是忽略边界，并以固定步幅沿射线行进。
A straightforward approach to reduce the cost of regular tracking is to ignore the boundaries and march along the ray with a fixed stride.

缺点：有偏的估计。

*Delta Tracking*

Delta 追踪的主要思想是引入一个虚构的介质，由空碰撞系数 $\mu_n(\mathbf x)$ 表示，它使总碰撞密度均匀化，以便能够对自由程进行分析采样。
The main idea of delta tracking is to introduce a fictitious medium, represented by the null-collision coefficient $\mu_n(\mathbf x)$ that homogenizes the total collision density in order to enable analytic sampling of free paths.

它的唯一目的是解析地采样距离。
Its sole purpose is to allow analytic distance sampling.

缺点：额外开销比较大。

*Weighted Tracking*

另一种减少空碰撞的方法是允许组合系数偶尔低估真正的消光系数。为了保持无偏，生成的非模拟估计器必须适当地重新加权那些进入空碰撞系数为负值的区域的样本。
Another way to reduce null collisions is to allow the combined coefficient to occasionally underestimate the real extinction. To remain unbiased, the resulting non-analog estimator must appropriately re-weight samples that step into regions with negative values of the null-collision coefficient.

*Multi-volume Tracking*

对于由于多次转移而可能发生吸收的材料，通过仅对总和中的一项进行概率评估来降低跟踪成本。
For materials where absorption can occur due to multiple transitions, reduce the tracking cost by probabilistically evaluating only a single term in the sum.

*Multi-wavelength Tracking*

## 3 Radiative Transfer with Null Collisions

### 3.1 Integral Formulation of RTE with Null Collisions

$$
(\omega\cdot\nabla)L(\mathbf x,\omega)=-\mu_t(\mathbf x)L(\mathbf x,\omega)+\mu_a(\mathbf x)L_e(\mathbf x,\omega)+\mu_s(\mathbf x)\int_{\mathcal S^2}f_p(\omega,\omega')L(\mathbf x,\omega')d\omega'
$$

等式右侧的第一项表示由于吸收和外散射造成的损失。由辐射发射和内散射引起的增益分别在第二项和第三项中描述。
The first term on the right side of Equation expresses the losses due to absorption and out-scattering. The gains due to radiant emission and in-scattering are described in the second and third term, respectively.



加入空碰撞：
$$
-\mu_n(\mathbf x)L(\mathbf x,\omega)+\mu_n(\mathbf x)\int_{\mathcal S^2}\delta(\omega-\omega')L(\mathbf x,\omega')d\omega'=0
$$
第一项为由空碰撞引起的能量耗散，第二项为平衡空碰撞而获得的内散射，称其为空碰撞辐射 null-collided radiance。因此总的渲染方程为：
$$
\begin{align*}
(\omega\cdot\nabla)L(\mathbf x,\omega)&=-[\mu_t(\mathbf x)+\mu_n(\mathbf x)]L(\mathbf x,\omega)+\mu_a(\mathbf x)L_e(\mathbf x,\omega)\\
&+\mu_s(\mathbf x)\int_{\mathcal S^2}f_p(\omega,\omega')L(\mathbf x,\omega')d\omega'\\
&+\mu_n(\mathbf x)\int_{\mathcal S^2}\delta(\omega-\omega')L(\mathbf x,\omega')d\omega'
\end{align*}
$$
积分可得：
$$
L(\mathbf x,\omega)=\int_0^{\infty}T(0,t)[\mu_a(\mathbf x_t)L_e(\mathbf x_t,\omega)+\mu_s(\mathbf x_t)L_s(\mathbf x_t,\omega)+\mu_n(\mathbf x_t)L(\mathbf x_t,\omega)]dt\\
T(0,t)=\exp({-\int_0^t[\mu_t(\mathbf x_s)+\mu_n(\mathbf x_s)]ds})\\
L_s(\mathbf x,\omega)=\int_{\mathcal S^2}f_p(\omega,\omega')L(\mathbf x,\omega')d\omega'\\
\mathbf x_t=\mathbf x-t\omega\quad \mathbf x_s=\mathbf x-s\omega
$$
如果 $\mu_n(\mathbf x)$ 是正的，会导致指数变小，但此时也会被空碰撞辐射补充；如果 $\mu_n(\mathbf x)$ 是负的，会导致指数变大，但是也会被空碰撞辐射削减。

### 3.2 Integral Formulation of Tracking Methods

我们松散地遵循他们对仅吸收介质的推导，并将其扩展到包括散射过程。
We loosely follow their derivation for absorptive-only media and extend it to include the process of scattering.

定义 PDF 和自由采样系数：
$$
P(t)=\bar{\mu}(\mathbf x_t)\exp({-\int_0^t\bar{\mu}(\mathbf x_s)ds})\\
\bar{\mu}(\mathbf x)=\mu_t(\mathbf x)+\mu_n(\mathbf x)
$$
用于解析地采样光线 $\mathbf r(\mathbf x,\omega)$ 的自由程的长度 $t\in(0,\infty)$。

带入到渲染方程里面：
$$
L(\mathbf x,\omega)=\int_0^{\infty}P(t)\left[\frac{\mu_a(\mathbf x_t)}{\bar{\mu}(\mathbf x_t)}L_e(\mathbf x_t,\omega)+\frac{\mu_s(\mathbf x_t)}{\bar{\mu}(\mathbf x_t)}L_s(\mathbf x_t,\omega)+\frac{\mu_n(\mathbf x_t)}{\bar{\mu}(\mathbf x_t)}L(\mathbf x_t,\omega)\right]dt\\
$$
由不同类型碰撞事件之间的许多自由程段组成的光路。
A light path consisting of a number of free-path segments between different types of collision events.

令 $\mathbf x_0\equiv\mathbf x,\mathbf x_{j+1}=\mathbf x_j-t_j\omega_j$，引入各项的概率来对各项估值：
$$
\begin{align*}
L(\mathbf x_j,\omega_j)=\int_0^{\infty}&P(t_j)[P_a(\mathbf x_{j+1})\frac{\mu_a(\mathbf x_{j+1})L_e(\mathbf x_{j+1},\omega_j)}{\bar{\mu}(\mathbf x_{j+1})P_a(\mathbf x_{j+1})}\\
&+P_s(\mathbf x_{j+1})\frac{\mu_s(\mathbf x_{j+1})L_s(\mathbf x_{j+1},\omega_j)}{\bar{\mu}(\mathbf x_{j+1})P_s(\mathbf x_{j+1})}\\
&+P_n(\mathbf x_{j+1})\frac{\mu_n(\mathbf x_{j+1})L(\mathbf x_{j+1},\omega_j)}{\bar{\mu}(\mathbf x_{j+1})P_n(\mathbf x_{j+1})}]dt_j
\end{align*}
$$
用 Russian Roulette 来对其进行采样：
$$
\int_0^1f(x)dx=\int_0^p\frac{f(x)}{p}dx=\int_0^1\mathcal H[x<p]\frac{f(x)}{p}dx
$$
带入
$$
\begin{align*}
L(\mathbf x_j,\omega_j)=\int_0^{\infty}&P(t_j)[\int_0^1\mathcal H[\xi_a<P_a(\mathbf x_{j+1})]\frac{\mu_a(\mathbf x_{j+1})L_e(\mathbf x_{j+1},\omega_j)}{\bar{\mu}(\mathbf x_{j+1})P_a(\mathbf x_{j+1})}d\xi_a\\
&+\int_0^1\mathcal H[\xi_s<P_s(\mathbf x_{j+1})]\frac{\mu_s(\mathbf x_{j+1})L_s(\mathbf x_{j+1},\omega_j)}{\bar{\mu}(\mathbf x_{j+1})P_s(\mathbf x_{j+1})}d\xi_s\\
&+\int_0^1\mathcal H[\xi_n<P_n(\mathbf x_{j+1})]\frac{\mu_n(\mathbf x_{j+1})L(\mathbf x_{j+1},\omega_j)}{\bar{\mu}(\mathbf x_{j+1})P_n(\mathbf x_{j+1})}d\xi_n]dt_j
\end{align*}
$$

通过用蒙特卡洛估计器替换积分，这个版本的 RTE 可以直接转化为大量递归算法，无论选择 $\mu_n(x)$ 和概率 $P_a(x)$，$P_s(x)$ 和 $P_n(x)$，这些算法都能产生正确的解决方案。
By replacing the integrals with Monte Carlo estimators, this version of the RTE can be directly translated into a multitude of recursive algorithms that all yield the correct solution regardless of the choice of $\mu_n(x)$ and probabilities $P_a(x)$, $P_s(x)$, and $P_n(x)$.

### 3.3 Delta Tracking

之前 $P_a,P_s,P_n$ 是任意选取的，如果 $\bar{\mu}$ 是 $\mu_t(\mathbf x)$ 的上界
$$
P_a(\mathbf x)=\frac{\mu_a(\mathbf x)}{\bar{\mu}(\mathbf x)}\quad
P_s(\mathbf x)=\frac{\mu_s(\mathbf x)}{\bar{\mu}(\mathbf x)}\quad
P_n(\mathbf x)=\frac{\mu_n(\mathbf x)}{\bar{\mu}(\mathbf x)}
$$
且都处于 $[0,1]$ 之间，且和为 1，则消去，合并分支：
$$
\begin{align*}
L(\mathbf x_j,\omega_j)=\int_0^{\infty}&P(t_j)[\int_0^1\mathcal H[\xi<P_a(\mathbf x_{j+1})]L_e(\mathbf x_{j+1},\omega_j)\\
&+\mathcal H[P_a(\mathbf x_{j+1})<\xi<1-P_n(\mathbf x_{j+1})]L_s(\mathbf x_{j+1},\omega_j)\\
&+\mathcal H[1-P_n(\mathbf x_{j+1})<\xi]L(\mathbf x_{j+1},\omega_j)d\xi]dt_j
\end{align*}
$$

```python
def deltaTracking(x, omega):
    while True:
        # sample distance
        t = -np.log(np.random.random_sample()) / mu_bar
        x = x - t * omega
        xi = np.random.random_sample()
        if xi < mu_a(x) / mu_bar:
            return Le(x, omega)
        else if xi < 1 - (mu_n(x) / mu_bar):
            # sample direction
            omega = fp(omega)
        else:
            # null collision
            continue
```

### 3.4 Weighted Delta Tracking

注意到不一定要满足 $\mu_t(\mathbf x)\le \bar{\mu}$，但对于负的 $\mu_n$ 需要特别处理：
$$
P_a(\mathbf x)=\frac{\mu_a(\mathbf x)}{\mu_t(\mathbf x)+|\mu_n(\mathbf x)|}\\
P_s(\mathbf x)=\frac{\mu_s(\mathbf x)}{\mu_t(\mathbf x)+|\mu_n(\mathbf x)|}\\
P_n(\mathbf x)=\frac{|\mu_n(\mathbf x)|}{\mu_t(\mathbf x)+|\mu_n(\mathbf x)|}\\
$$
路径追踪算法也因此导出，我们用 $w$ 记录每次采样的权重并且动态调整：

```python
def weightedDeltaTracking(x, omega):
    w = 1
    while True:
        # sample distance
        t = -np.log(np.random.random_sample()) / mu_bar
        x = x - t * omega
        xi = np.random.random_sample()
        if xi < Pa(x):
            return w * mu_a(x) * Le(x, omega) / mu_bar
        else if xi < 1 - :
            # sample direction
            omega = fp(omega)
            w = w * mu_s(x) / (mu_bar * Ps(x))
        else:
            w = w * mu_n(x) / (mu_bar * Pn(x))
```

## 4 Decomposition Tracking

符号表：

|                    Symbol & Value                    |                Description                |
| :--------------------------------------------------: | :---------------------------------------: |
|           $\mu_a(\mathbf x)\in[0,\infty)$            | absorption coefficient of original volume |
|           $\mu_s(\mathbf x)\in[0,\infty)$            | scattering coefficient of original volume |
| $\mu_t(\mathbf x)=\mu_a(\mathbf x)+\mu_s(\mathbf x)$ | extinction coefficient of original volume |
|                      $\mu_a^c$                       | absorption coefficient of control volume  |
|                      $\mu_s^c$                       | scattering coefficient of control volume  |
|              $\mu_t^c=\mu_a^c+\mu_s^c$               | extinction coefficient of control volume  |
|    $\mu_a^r(\mathbf x)=\mu_a(\mathbf x)-\mu_a^c$     | absorption coefficient of residual volume |
|    $\mu_s^r(\mathbf x)=\mu_s(\mathbf x)-\mu_s^c$     | scattering coefficient of residual volume |
|    $\mu_t^r(\mathbf x)=\mu_t(\mathbf x)-\mu_t^c$     | extinction coefficient of residual volume |
|             $\bar\mu\in(\mu_t^c,\infty)$             |      free-path-sampling coefficient       |
|  $\mu_n(\mathbf x)=\mu-\mu_t^c-\mu_t^r(\mathbf x)$   |        null-collision coefficient         |

### 4.1 Analog Decomposition Tracking

我们观察到原始介质中的自由程样本可以在每个成分中获取最小自由程样本来获得。
We observe that a free-path sample in the original medium can be obtained by taking the minimum of free-path samples in individual components.

<blockquote style="border-left: 5px solid #42b983; border-radius: 3px 0 0 3px; padding: 10px 15px; background-color: rgba(66, 185, 131, 0.1)">
	Theorem 1
</blockquote>

Let non-negative extinction coefficients $\mu_A(\mathbf x)$, $\mu_B(\mathbf x)$ be combined to $\mu_C(\mathbf x)=\mu_A(\mathbf x)+\mu_B(\mathbf x)$. Let $A,B,C$ be independent random variables distributed according to the following CDF $F_X(t)=1-\exp({-\int_0^t\mu_X(\mathbf x_s)ds})$ with $X\in\{A,B,C\}$. Then the CDFs if $C$ and $\min(A,B)$ are identical.

<blockquote style="border-left: 5px solid #4545aa; border-radius: 3px 0 0 3px; padding: 10px 15px; background-color: rgba(70, 70, 188, 0.1)">
    Proof
</blockquote>

$$
\begin{align*}
F_D(t)&=P(D\le t)\\
&=P(\min(A,B)\le t)\\
&=1-P(\min(A,B)> t)\\
&=1-P(A>t)P(B>t)\\
&=1-T_A(t)T_B(t)\\
&=1-\exp({-\int_0^t\mu_A(\mathbf x_s)ds})\exp({-\int_0^t\mu_B(\mathbf x_s)ds})\\
&=1-\exp({-\int_0^t\mu_C(\mathbf x_s)ds})\\
&=F_C(t)
\end{align*}
$$

```python
def analogDecompositionTraking(x, omega):
    while True:
        # sample analitically in control volume
        t_c = -np.log(np.random.random_sample()) / mu_c_t
        t_r = 0
        
        while True:
            t_r = -np.log(np.random.random_sample()) / (mu_bar - mu_c_t)
            if t_r > t_c:
                # sample control volume
                x = x - t_c * omega
                if np.random.random_sample() < mu_a_c(x) / mu_t_c:
                    return Le(x, omega)
                else:
                    omega = fp(omega)
                    break
            else:
                # sample residual volume
                # delta tracking
                xi = np.random.random_sample()
                if xi < mu_a_r(x) / (mu_bar - mu_t_c):
                    x = x - t_r * omega
                    return Le(x, omega)
                else if xi < 1 - mu_n(x) / (mu_bar - mu_t_c):
                    x = x - t_r * omega
                    omega = fp(omega)
                    break
```

首先对于控制分量采样一个距离 $t^c=-\ln(1-\zeta)/\mu_c^t$。对于残差分量采样的距离 $t^r$，首先类似 Delta 追踪补齐虚拟介质，随后采样。如果 $t^r>t^c$，则选取 $t^c$ 作为下一个点，并且查看是吸收或是散射。如果 $t^r<t^c$，则类似 Delta 追踪，分三种情况。

模拟分解跟踪要求残差消光系数 $\mu_t^r(\mathbf x)$ 和零碰撞系数 $\mu_n^r(\mathbf x)$ 始终为非负。因此，控制分量的消光系数 $\mu_t^c$ 需要是 $\mu_t(\mathbf x)$ 的下界，以确保 $\mu_t^r(\mathbf x)\ge0$，并且残差分量的自由路径采样系数 $\bar\mu^r$ 必须是 $\mu_t^r(\mathbf x)$ 的上界，以确保 $\mu_n(\mathbf x)\ge0$。
The analog decomposition tracking requires both the residual extinction coefficient $\mu_t^r(\mathbf x)$ and the null-collision coefficient $\mu_n^r(\mathbf x)$ to be always non-negative. The control extinction $\mu_t^c$ thus needs to be a minorant of $\mu_t(\mathbf x)$ to ensure $\mu_t^r(\mathbf x)\ge0$, and the free-path-sampling coefficient of the residual component $\bar\mu^r$ must be a majorant of $\mu_t^r(\mathbf x)$ to ensure $\mu_n(\mathbf x)\ge0$.

#### Limitation

当上下界都是紧的时候，性能最好。
The best performance requires both bounds to be tight.

只能处理单色光。
It can handle only a single wavelength.

### 4.2 Weighted Decomposition Tracking

分解跟踪的加权版本，可以处理无边界控制消光和自由路径采样系数，并能够有效处理色度碰撞系数——这两者在模拟版本中都是问题。
A weighted version of the decomposition tracker that can handle non-bounding control extinctions and free-path-sampling coefficients, and enables efficient handling of chromatic collision coefficients -- both of which are problematic with the analog version.

假设有 $k$ 个吸收和散射分量，则渲染方程就变成：
$$
L(\mathbf x,\omega)=\int_0^{\infty}T(0,t)[\sum_{i=1}^k\mu_a^i(\mathbf x_t)L_e(\mathbf x_t,\omega)+\sum_{i=1}^k\mu_s^i(\mathbf x_t)L_s(\mathbf x_t,\omega)+\mu_n(\mathbf x_t)L(\mathbf x_t,\omega)]dt\\
T(0,t)=\exp({-\int_0^t[\sum_{i=1}^k\mu_t^i(\mathbf x_s)+\mu_n(\mathbf x_s)]ds})\\
$$
带入概率：
$$
\begin{align*}
L(\mathbf x_j,\omega_j)=\int_0^{\infty}&P(t_j)[\sum_{i=1}^k\int_0^1\mathcal H[\xi_a^i<P_a^i(\mathbf x_{j+1})]\frac{\mu_a^i(\mathbf x_{j+1})L_e(\mathbf x_{j+1},\omega_j)}{\bar{\mu}(\mathbf x_{j+1})P_a^i(\mathbf x_{j+1})}d\xi_a^i\\
&+\sum_{i=1}^k\int_0^1\mathcal H[\xi_s^i<P_s^i(\mathbf x_{j+1})]\frac{\mu_s^i(\mathbf x_{j+1})L_s(\mathbf x_{j+1},\omega_j)}{\bar{\mu}(\mathbf x_{j+1})P_s^i(\mathbf x_{j+1})}d\xi_s^i\\
&+\int_0^1\mathcal H[\xi_n<P_n(\mathbf x_{j+1})]\frac{\mu_n(\mathbf x_{j+1})L(\mathbf x_{j+1},\omega_j)}{\bar{\mu}(\mathbf x_{j+1})P_n(\mathbf x_{j+1})}d\xi_n]dt_j
\end{align*}
$$
这个问题里面，我们只有两种分量，因此有 5 项。

*Control Probabilities*

最重要的是，我们定义了仅使用常数系数对控制分量进行采样的概率，避免了可能昂贵的空间变化系数查找：
Most importantly though, we define the probabilities of sampling the control component using only the constant coefficients, avoiding possibly expensive lookups of spatially varying coefficients:
$$
P_a^c=P_cP_{a|c}=\frac{\mu_t^c}{\bar\mu}\frac{\mu_a^c}{\mu_t^c}=\frac{\mu_a^c}{\bar\mu}\\
P_s^c=P_cP_{s|c}=\frac{\mu_t^c}{\bar\mu}\frac{\mu_s^c}{\mu_t^c}=\frac{\mu_s^c}{\bar\mu}
$$
$P_c$ 是合法的话，需要满足 $\bar\mu>\mu_t^c$。

*Residual and Null-collision Probabilities*

第一项总是表示不对控制分量进行抽样的边际概率。 第二项定义了选择残余吸收、残余散射和零碰撞的条件概率。
The first term always represents the marginal probability of not sampling the control component. The second term then defines the conditional probability of selecting the residual absorption, residual scattering, and null collision.
$$
P_a^r(\mathbf x)=\left(1-\frac{\mu_t^c}{\bar\mu}\right)\frac{|\mu_a^r(\mathbf x)|}{|\mu_a^r(\mathbf x)|+|\mu_s^r(\mathbf x)|+|\mu_n(\mathbf x)|}\\
P_s^r(\mathbf x)=\left(1-\frac{\mu_t^c}{\bar\mu}\right)\frac{|\mu_s^r(\mathbf x)|}{|\mu_a^r(\mathbf x)|+|\mu_s^r(\mathbf x)|+|\mu_n(\mathbf x)|}\\
P_n(\mathbf x)=\left(1-\frac{\mu_t^c}{\bar\mu}\right)\frac{|\mu_n(\mathbf x)|}{|\mu_a^r(\mathbf x)|+|\mu_s^r(\mathbf x)|+|\mu_n(\mathbf x)|}
$$

```python
def weightedDecompositionTraking(x, omega):
    w = 1
    while True:
        # sample analitically
        t = -np.log(np.random.random_sample()) / mu_bar

        x = x - t * omega
        F = 0
        xi = np.random.random_sample()
        if xi < (F = F + P_a_c(x)):
            # sample control volume, absorb
            return w * (u_a_c / (mu_bar * p_a_c(x))) * Le(x, omega)
        elif xi < (F = F + P_s_c(x)):
            # sample control volume, scatter
            omega = fp(omega)
            w = w * u_s_c / (mu_bar * p_s_c(x))
        elif xi < (F = F + P_a_r(x)):
            # sample residual volume, absorb
            return w * (u_a_r / (mu_bar * p_a_r(x))) * Le(x, omega)
        elif xi < (F = F + P_s_r(x)):
            # sample residual volume, scatter
            omega = fp(omega)
            w = w * u_s_r / (mu_bar * p_s_r(x))
        else:
            # sample fiction volume
            w = w * mu_n(x) / (mu_bar * p_n(x))
```

### 4.3 Analysis & Discussion

尽管加权公式使用不同的程序来构建自由路径，但它会产生与模拟版本相同的查找统计数据——统计数据来自体积参数，而不是过程本身。
Despite the weighted formulation using a different procedure for constructing the free path, it produces identical lookup statistics as the analog version -- the statistics stem from the parameters of the volume, not the procedure itself.

*Relative Cost*

$N_{\mathrm{dcmp}}$：分解追踪查看消光系数的次数。

$N_{\mathrm{delta}}$：Delta 追踪查看消光系数的次数。

相对花费：
$$
\rho=N_{\mathrm{dcmp}}/N_{\mathrm{delta}}=1-\mu_t^c/\bar\mu\quad(\mu_t^c<\min(\mu_t(\mathbf x)))
$$
当 $\mu_t^c>\min(\mu(\mathbf x))$ 时，相对成本会增加到最终超过 1，即分解跟踪变得比增量跟踪更昂贵。这是意料之中的，因为此时的控制介质显着高估了原始介质的密度。
When $\mu_t^c>\min(\mu(\mathbf x))$, the relative cost increases to eventually surpass 1, i.e. the decomposition tracking becomes more expensive than delta tracking. This is to be expected as the control medium at that point significantly overestimates the density of the original medium.

*Mean and Variance of $w(X)$*

因为跟踪器是无偏的，所以在第一次碰撞时路径吞吐量 $w(X)$ 的平均值始终为 1。此外，如果 $\mu_t^c\in[0,\min(\mu_t(\mathbf x)) ]$，则碰撞权重 $w_{\star}$ 始终为 1，因此 $w(X)$ 的方差为零。与相对成本的情况类似，当 $\mu_t^c>\min(\mu_t(\mathbf x))$ 时，跟踪器变得低效，在这种情况下，$w(X)$ 的方差迅速增加。
Because the tracker is unbiased, the mean of the path throughput $w(X)$ at the first collision is always 1. Furthermore, if $\mu_t^c\in[0,\min(\mu_t(\mathbf x))]$, then the collision weight $w_{\star}$ is always 1 and the variance of $w(X)$ is thus zero. Similarly to the case of the relative cost, the tracker becomes inefficient when $\mu_t^c>\min(\mu_t(\mathbf x))$, in which case the variance of $w(X)$ increase rapidly.

*Optimal Configuration*



## 5 SPECTRAL TRACKING

通过适当地重新加权样本来抵消分布中的偏差。我们利用这种能力来有效地处理光谱分辨的碰撞系数。
The bias in the distribution is counteracted by properly reweighting the samples. We leverage this ability to efficiently handle spectrally resolved collision coefficients.

我们为采样碰撞定义了一个单一分布，并通过重新加权来抵消该分布与真实的每波长自由路径分布之间的差异。
We define a single distribution for sampling collisions and counteract the discrepancy between this distribution and the true, per wavelength free-path distribution by reweighting.











