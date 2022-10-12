# Volume Rendering of Neural Implicit Surfaces

## 0 Abstract

我们将体积密度函数定义为拉普拉斯的累积分布函数（CDF），应用于有符号距离函数（SDF）表示。
We define the volume density function as Laplace’s cumulative distribution function (CDF) applied to a signed distance function (SDF) representation.

这种简单的密度表示有三个好处。(i)它为在神经体积渲染过程中学习到的几何图形提供了有用的归纳偏向；(ii)它有助于约束不透明度的近似误差，导致对观察光线的精确采样。准确的采样对于提供几何学和辐射度的精确耦合非常重要；(iii)它允许在体积渲染中有效地无监督地解除形状和外观的联系。
This simple density representation has three benefits: (i) it provides a useful inductive bias to the geometry learned in the neural volume rendering process; (ii) it facilitates a bound on the opacity approximation error, leading to an accurate sampling of the viewing ray. Accurate sampling is important to provide a precise coupling of geometry and radiance; and (iii) it allows efficient unsupervised disentanglement of shape and appearance in volume rendering.

此外，由于两者的解耦，在场景之间切换形状和外观是可能的。
Furthermore, switching shape and appearance between scenes is possible due to the disentanglement of the two.

## 1 Introduction

首先，它保证了一个产生密度的定义明确的表面的存在。这为分解密度和辐射场提供了有用的归纳偏向，这反过来又提供了更精确的几何近似。第二，我们表明这个密度公式允许约束沿光线的不透明度的近似误差。这个约束被用来对观察光线进行采样，以便在体积渲染积分中提供密度和辐射场的忠实耦合。例如，如果没有这样的约束，沿光线计算的辐射度（像素颜色）可能会遗漏或扩展表面部分，导致错误的辐射度近似。
First, it guarantees the existence of a well-defined surface that generates the density. This provides a useful inductive bias for disentangling density and radiance fields, which in turn provides a more accurate geometry approximation. Second, we show this density formulation allows bounding the approximation error of the opacity along rays. This bound is used to sample the viewing ray so to provide a faithful coupling of density and radiance field in the volume rendering integral. E.g., without such a bound the computed radiance along a ray (pixel color) can potentially miss or extend surface parts leading to incorrect radiance approximation.

## 2 Related Work

## 3 Method

### 3.1 Density As Transformed SDF

|            Symbols             |        Descriptions         |
| :----------------------------: | :-------------------------: |
|      $\Omega\subset\R^3$       |           object            |
| $\mathcal{M}=\partial{\Omega}$ |          boundary           |
|     $d_{\Omega}:\R^3\to\R$     | SDF function with $L2$ norm |

使用可学习的有符号距离函数（SDF）的某种变换对密度进行建模：
Model the density using a certain transformation of a learnable Signed Distance Function (SDF):
$$
\sigma(\mathbf{x})=\alpha\Psi_{\beta}(-d_{\Omega}(\mathbf{x}))
$$
其中 $\Psi_\beta$ 是拉普拉斯分布的 CDF：
$$
\Psi_\beta(s)=\begin{cases}\dfrac{1}{2}\exp\biggl(\dfrac{s}{\beta}\biggr)&s\le0\\1-\dfrac{1}{2}\exp\biggl(-\dfrac{s}{\beta}\biggr)&s>0\end{cases}
$$
当 $\beta$ 接近零时，密度 $\sigma$ 收敛为 $\Omega$ 的冲击函数。
As $\beta$ approaches zero, the density $\sigma$ converge to a scaled indicator function of $\Omega$.

直观地说，密度 $\sigma$ 模拟一个具有恒定密度 $\alpha$ 的均匀物体，在物体的边界附近平滑地减少，其中平滑量由 $\beta$ 控制。
Intuitively, the density $\sigma$ models a homogeneous object with a constant density $\alpha$ that smoothly decreases near the object’s boundary, where the smoothing amount is controlled by $\beta$.

### 3.2 Volume Rendering of $\sigma$

#### Sampling

我们希望计算其体积积分，然而，第二个网络的密度不一定忠实地代表第一个网络的密度。此外，正如我们在后面所显示的，一次抽样可能不足以产生一个准确的样本 $S$。使用一个粗略的不透明度近似会导致一个次优的样本集 $S$，它错过了或过度扩展了不可忽略的 $\tau$ 值。因此，会出现不正确的辐射度近似（即像素颜色），有可能损害学到的密度-辐射度场分解。
However, the second network’s density does not necessarily faithfully represents the first network’s density, for which we wish to compute the volume integral. Furthermore, as we show later, one level of sampling could be insufficient to produce an accurate sample $S$. Using a naive or crude approximation of opacity would lead to a sub-optimal sample set $S$ that misses, or over extends non-negligible $\tau$ values. Consequently, incorrect radiance approximations can occur (i.e., pixel color), potentially harming the learned density-radiance field decomposition.

注意 NeRF 渲染中由随机采样引起的黑白颗粒的伪影；在 NeRF 中使用固定（均匀间隔）采样会导致不同类型的伪影。
Note the salt and pepper artifacts in the NeRF rendering caused by the random samples; using fixed (uniformly spaced) sampling in NeRF leads to a different type of artifacts.

### 3.3 Bound On the Opacity Approximation Error

采样：$\mathcal{T}=\{t_i\}_{i=1}^{n}$, $t_1<t_2<\dots<t_n$, $\delta_i=t_{i+1}-t_i$, $\sigma_i=\sigma(\mathbf{x}(t_i))$:

假设 $t\in[t_{k},t_{k+1}]$ Riemann 和：
$$
\begin{align*}
\int_0^t\sigma(\mathbf{x}(s))\mathrm{d}s&=\hat{R}(t)+E(t)\\
\hat{R}(t)&=\sum_{i=1}^{k-1}\delta_i\sigma_i+(t-t_k)\sigma_k
\end{align*}
$$
对应的该点的不透明度为：
$$
\hat{O}(t)=1-\exp(-\hat{R}(t))
$$
**Theorem 1**

<img src="../../Note Image/volsdf.png" alt="volsdf" style="zoom: 67%;" />

在 $[t_i,t_{i+1}]$ 段内，密度 $\sigma$ 的导数满足：
$$
\biggl|\frac{\mathrm{d}}{\mathrm{d}s}\sigma(\mathbf{x}(s))\biggr|\le\frac{\alpha}{2\beta}\exp\biggl(-\frac{d^\star_i}{\beta}\biggr)
$$
其中：
$$
\begin{align*}
d^\star_i&=\min_{s\in[t_i.t_{i+1}],\mathbf{y}\notin B_i\cup B_{i+1}}\|\mathbf{x}(s)-\mathbf{y}\|\\
B_i&=\{\mathbf{x}\mid\|\mathbf{x}-\mathbf{x}(t_i)\|<|d_i|\}
\end{align*}
$$
$d_i$ 是 $\mathbf{x}(t_i)$ 处的 sdf 值。即 $B_i$ 是以采样点为中心，采样点 sdf 值为半径的球。

---

在这个设定下，密度 $\sigma$ 不是处处可微的，比如表面处，因此我们的误差限应该去限制对应的 Lipschitz 常数 $K_i$。在 $[t_i,t_{i+1}]$ 段内，密度 $\sigma$ 的 Lipschitz 常数 $K_i$ 满足：
$$
K_i\le\frac{\alpha}{2\beta}\exp\biggl(-\frac{d^\star_i}{\beta}\biggr)
$$
 $d^\star_i$ 可以这样计算：
$$
d^\star_i=\begin{cases}
0&|d_i|+|d_{i+1}|\le\delta_i\\
\min
\end{cases}
$$


---

它允许在每个 $[t_i,t_{i+1}]$ 区间内，仅根据区间端点的无符号距离 $|d_i|, |d_{i+1}|$，以及密度参数 $\alpha,\beta$ 来约束密度的导数。这个约束可以用来推导出矩形规则对不透明度的近似的误差约束：
It allows to bound the density's derivative in each interval $[t_i,t_{i+1}]$ based only on the unsigned distance at the interval's end points, $|d_i|, |d_{i+1}|$, and the density parameters $\alpha,\beta$. This bound can be used to derive an error bound for the rectangle rule's approximation of the opacity:
$$
|E(t)|\le\hat{E}(t)=\frac{\alpha}{4\beta}\Biggl(\sum_{i=1}^{k-1}\delta^2_i\exp\biggl(-\frac{d^\star_i}{\beta}\biggr)+(t-t_k)^2\exp\biggl(-\frac{d^\star_k}{\beta}\biggr)\Biggr)
$$

---



---

**Theorem 2**

对于光线上的采样，不透明度 $\hat{O}$ 的误差限为：
$$
|O(t)-\hat{O}(t)|\le\exp(-\hat{R}(t))(\exp(\hat{E}(t))-1)
$$




### 3.4 Sampling Algorithm

