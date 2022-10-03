# Practical Pigment Mixing for Digital Painting

## 0 Abstract

今天的绘画软件有一个重大缺陷：颜色不像实际的颜料那样混合。这是因为该软件是围绕 **RGB 表示**构建的，它对彩色光的混合进行建模。然而，油漆的颜色来自颜料，其混合行为由 Kubelka-Munk (K-M) 模型预测。
There is a significant flaw in today’s painting software: the colors do not mix like actual paints. This is because the software is built around the **RGB representation**, which models the mixing of colored lights. Paints, however, get their color from pigments, whose mixing behavior is predicted by the Kubelka–Munk (K–M) model.

我们介绍了一种实用的方法，可以使用 K-M 混合颜色，同时将所有内容保持为 RGB。
We introduce a practical approach that enables mixing colors with K–M while keeping everything in RGB.

我们通过建立一个潜在的颜色空间来实现这一点，其中 RGB 颜色表示为**原色**与**添加剂残差**的混合物。潜在的颜色空间可以线性操作，从而产生预期的、合理的结果。
We achieve this by establishing a latent color space, where RGB colors are represented as mixtures of **primary pigments** together with **additive residuals**. The latents can be manipulated with linear operations, leading to expected, plausible results.

## 1 Introduction

事实上，当我们在同一点照射蓝光和黄光时，它们的光谱线性相加，我们得到灰色。
Indeed, when we shine blue and yellow light at the same spot, their spectra linearly sum together, and we get gray.

颜料从大量颜料颗粒中获得颜色，这些颜料颗粒以复杂的方式吸收和散射光。这使得颜料混合物的结果高度非线性。
Paints get their color from the mass of pigment particles which absorb and scatter the light in a complex fashion. That makes the outcome of pigment mixture highly non-linear.

与 RGB 不同，颜色在与白色混合时往往会失去饱和度，实际涂料的饱和度会瞬间增加，从而揭示颜料的真实性质。
Unlike RGB, where colors tend to lose saturation when mixed with white, the saturation of actual paints momentarily increases, revealing the true nature of the pigment.

开发人员发现 K-M 模型不实用。这是因为 K-M 需要跟踪每个像素上每种颜料的浓度。另一种选择是跟踪每个波长的吸收和散射系数。然而，在这两种情况下，每像素通道的数量都会大幅增长。
The developers do not find the K–M model practical to implement. This is because K–M requires tracking the respective concentrations of all available pigments at each pixel. Another option is to track the per wavelength absorption and scattering coefficients instead. However, in both cases, the number of per-pixel channels grows substantially.

1. 当代绘画软件通常围绕 3 通道 RGB 表示构建，实现 K-M 需要将软件结构更改到核心。
    The contemporary painting software is usually built around the 3-channel RGB representation, and implementing K–M would require changing the software structure down to the core.
2. 在屏幕上显示一幅画需要评估每个像素的可见光谱积分，性能成本相当大。
    Displaying a painting on screen would require evaluating the visible spectrum integrals per pixel, which has a considerable performance cost.
3. 颜料混合物不覆盖整个 RGB 色域，这使与外界的交换变得复杂。
    Pigment mixtures do not cover the whole RGB gamut, which complicates the exchange with the outside world.

我们提出了任何颜色混合模型都需要满足的要求，以便在现实世界的绘画软件中实用：
We put forward the requirements that any color-mixing model needs to meet in order to be practical for a real–world painting software:

1. 直接在 RGB 上工作，无需额外的通道。
    Work directly on RGB without requiring additional channels.
2. 计算速度要足够快，以避免绘画过程中的延迟。
    Be fast enough to compute to avoid latencies during painting.
3. 处理所有 RGB 颜色，而不会导致剪裁或失真。
    Handle all RGB colors without causing clipping or distortion.

## 2 Background and Related Work

正是吸收和散射特性的差异导致颜料具有不同的颜色。正是这种次表面散射现象导致蓝色和黄色涂料的混合物呈现绿色。
It is the difference in absorption and scattering properties that causes pigments to have distinct colors. It is this subsurface scattering phenomenon that causes the mixture of blue and yellow paint to appear green.

$K(\lambda)$: 吸收系数 Absorption coefficient
$S(\lambda)$: 散射系数 Scatter coefficient
$\mathbf c=[c_1,\dots,c_N]$: 浓度 Concentrations, $c_i\ge0,\sum_{i=1}^{N}c_i=1$.
$\mathcal P=\{K_{i}(\lambda),S_{i}(\lambda)\}_{i=1}^{N}$: 颜料 Pigments
$$
K_{\text{mix}}(\mathbf c,\lambda)=\sum_{i=1}^{N}c_iK_i(\lambda)\quad S_{\text{mix}}(\mathbf c,\lambda)=\sum_{i=1}^{N}c_iS_i(\lambda)
$$
$R_{\text{mix}}(\mathbf c,\lambda)$: 反射光谱 Reflectance spectrum
$$
R_{\text{mix}}(\mathbf c,\lambda)=1+\frac{K_{\text{mix}}(\mathbf c,\lambda)}{S_{\text{mix}}(\mathbf c,\lambda)}-\sqrt{\left(\frac{K_{\text{mix}}(\mathbf c,\lambda)}{S_{\text{mix}}(\mathbf c,\lambda)}\right)^2+2\frac{K_{\text{mix}}(\mathbf c,\lambda)}{S_{\text{mix}}(\mathbf c,\lambda)}}
$$
$R'_{\text{mix}}(\mathbf c,\lambda)$: 考虑了表面反射的反射光谱
$$
R'_{\text{mix}}(\mathbf c,\lambda)=\frac{(1-k_1)(1-k_2)R_{\text{mix}}(\mathbf c,\lambda)}{1-k_2R_{\text{mix}}(\mathbf c,\lambda)}
$$
$k_1,k_2$: 测量的反射常数 Measured reflectance constants

为了在屏幕上显示 $R_{\text{mix}}$，我们用合适的光源照亮混合物，并将反射光谱与 CIE 标准观察者函数在可见波长 $\lambda\in[380, 750 ]$ 积分。这里我们使用 $D_{65}$ 照明，它对应于平均日光并与 sRGB 白点重合：
To display $R_{\text{mix}}$ on screen, we illuminate the mixture with a suitable light source and integrate the reflected spectrum with the CIE standard observer functions over the visible range of wavelengths $\lambda\in[380, 750]$. Here we use the $D_{65}$​ illumination, which corresponds to average daylight and coincides with the sRGB white point:
$$
X(\mathbf c)=\int_{\lambda}\bar{x}(\lambda)D_{65}(\lambda)R'_{\text{mix}}(\mathbf c,\lambda)\mathrm{d}\lambda\\
Y(\mathbf c)=\int_{\lambda}\bar{y}(\lambda)D_{65}(\lambda)R'_{\text{mix}}(\mathbf c,\lambda)\mathrm{d}\lambda\\
Z(\mathbf c)=\int_{\lambda}\bar{z}(\lambda)D_{65}(\lambda)R'_{\text{mix}}(\mathbf c,\lambda)\mathrm{d}\lambda\\
$$
最后，为了获得颜料混合物的 sRGB 颜色，我们将 XYZ 三色值与 sRGB 色度矩阵相乘，并通过 $Y_{D_{65}}$ 进行归一化。
Finally, to obtain an sRGB color of the pigment mixture, we multiply the XYZ tristimulus values with the matrix of sRGB chromaticities and normalize by $Y_{D_{65}}$​.
$$
\mathop{\text{mix}}_{\mathcal P}(\mathbf c)=\begin{bmatrix}R(\mathbf c)\\G(\mathbf c)\\B(\mathbf c)\end{bmatrix}=\frac{1}{\int_{\lambda}\bar{y}(\lambda)D_{65}(\lambda)\mathrm{d}\lambda}\begin{bmatrix}+3.2406&-1.5372&-0.4986\\-0.9689&+1.8758&+0.0415\\+0.0557&-0.2040&+1.0570\end{bmatrix}\begin{bmatrix}X(\mathbf c)\\Y(\mathbf c)\\Z(\mathbf c)\end{bmatrix}
$$
因此我们就可以根据颜料 $\mathcal P$ 和浓度 $\mathbf c$ 得到在屏幕上显示的 RGB 值。

RYB：他们提出了另一种颜色空间，专门设计用于使蓝色和黄色的平均值变为绿色。 他们通过在 3D 立方体的顶点插入八种手工挑选的颜色来实现这一点。
RYB: They proposed an alternate color space, designed specifically to make the average of blue and yellow turn out green. They achieve this by interpolating eight hand-picked colors placed at the vertices of a 3D cube.

一种数据驱动的方法，他们从提供的示例中插入真实颜料混合物的样本。 然而，他们的方法仅适用于预定的混合比例（例如，50: 50），这对于需要以任意比例混合颜色的绘画软件来说不够灵活。
A data-driven approach, where they interpolate samples of real pigment mixtures taken from a provided example. However, their approach only works with a predetermined mixing ratio (e.g., 50: 50), which is not flexible enough for the use in painting software, where colors need to be mixed in arbitrary ratios.

尽管减色法混合确实有能力从黄色和蓝色中产生绿色，但产生的颜色缺乏混合真实颜料时所达到的饱和度。
Although the subtractive mixing does have the ability to produce green out of yellow and blue, the resulting color lacks the saturation achieved when mixing real paints.

## 3 Our Method

当今绘画软件中使用的最基本的混合操作是线性插值：
The most basic mixing operation used in today’s painting software is linear interpolation:
$$
\mathrm{lerp}(\mathrm{RGB}_1,\mathrm{RGB}_2,t)=(1-t)\mathrm{RGB}_1+t\mathrm{RGB}_2\quad t\in[0,1]
$$
我们现在的目标是构建线性插值的替代版本，它采用相同的 RGB 输入，但会产生一种混合实际的颜料就会产生这种颜色。
Our aim now is to build an alternate version of linear interpolation, which takes the same RGB inputs but produces a color that would result if we mixed actual paints.
$$
\mathrm{kmerp}(\mathrm{RGB}_1,\mathrm{RGB}_2,t)
$$
颜料的不同组合可用于混合相同的 RGB 颜色，每种颜色与另一种颜色混合时会产生不同的结果。为了解决这个问题，我们需要确定一组原色 $\mathcal P^*=\{(K_i^*(\lambda),S_i^*(\lambda))\}_{i=1}^N$。
Different combinations of pigments can be used to mix the same RGB color, each leading to a different outcome when mixed with another color. To resolve this, we need to decide on a set of primary pigments $\mathcal P^*=\{(K_i^*(\lambda),S_i^*(\lambda))\}_{i=1}^N$.

主要颜料的设置是先验决定的，具体选择取决于 kmerp 用户。对于一般用途，我们使用四种现代有机颜料，提供广泛的色域：酞菁蓝、喹吖啶酮洋红色、汉莎黄和钛白。
The set of primary pigments is decided a priori, and the concrete choice is up to the kmerp user. For the general purpose, we use four modern organic pigments that provide a wide color gamut: Phthalo Blue, Quinacridone Magenta, Hansa Yellow, and Titanium White.

kmerp 背后的主要思想是首先将每种 RGB 颜色分解为这些主要颜料的混合物。 这相当于反转 [](#2 Background and Related Work) 中等式中概述的 K-M 混合过程。
The main idea behind kmerp is to first decompose each RGB color into a mixture of these primary pigments. This amounts to inverting the K–M mixing procedure outlined in equations in [](#2 Background and Related Work).

我们将此反演任务视为最小二乘优化问题并求解未知浓度 $\mathbf c$：
We pose this inversion task as a least-squares optimization problem and solve for the unknown concentrations $\mathbf c$:
$$
\mathop{\mathrm{unmix}}_{\mathcal P^*}(\mathrm{RGB})=\mathop{\arg\min}_{\mathbf c}\|\mathop{\mathrm{mix}}_{\mathcal P^*}(\mathbf c)-\mathrm{RGB}\|^2
$$
尽管 $\mathbf{c}$ 中的混合操作是非线的，但它是平滑且可微的，这允许我们使用牛顿型求解器来获得优化的解。
Even though the mix operation is non-linear in $\mathbf{c}$, it is smooth and differentiable, which allows us to use a Newton-type solver to obtain the solution of the optimization.

给定两个输入颜色 $\mathrm{RGB}_1,\mathrm{RGB}_2$，我们对它们的未混合浓度 $\mathbf c_1,\mathbf c_2$ 进行线性组合并混合组合浓度 $\hat{\mathbf c}$ 返回得到最终的 $\mathrm{RGB}$ 颜色。
Given the two input colors $\mathrm{RGB}_1,\mathrm{RGB}_2$, we take a linear combination of their unmixed concentrations $\mathbf c_1,\mathbf c_2$ and mix the combined concentrations $\hat{\mathbf c}$ back to obtain the final $\mathrm{RGB}$ color.

```python
def kmerp(RGB1, RGB2, t):
    c1 = unmix(RGB1)
    c2 = unmix(RGB2)
    c  = lerp(c1, c2, t)
    return mix(c)
```

它有一个小缺陷：当 $t=0$ 或 $t=1$ 时，不能保证重现输入颜色。 这种缺陷源于这样一个事实，即存在无法由原色混合得到的 RGB 颜色。事实上，一些 RGB 颜色不能从任何目前已知的颜料中混合出来。
It has a slight deficiency: it isn’t guaranteed to reproduce the input colors when $t=0$ or $t=1$. This deficiency stems from the fact that there are RGB colors that cannot be mixed from the primary pigments. In fact, some RGB colors cannot be mixed out of any currently known pigments.

我们的下一个想法是通过引入一个附加残差项来纠正这个缺陷。残差是 RGB 颜色与其投影对应颜色之间的差异：$\mathrm{RGB}-\mathrm{mix}(\mathrm{unmix}(\mathrm{RGB}))$。
Our next idea is to correct this deficiency by introducing an additive residual term. The residual is the difference between the RGB color and its projected counterpart: $\mathrm{RGB}-\mathrm{mix}(\mathrm{unmix}(\mathrm{RGB}))$.

```python
def kmerp(RGB1, RGB2, t):
    c1 = unmix(RGB1)
    c2 = unmix(RGB2)
    r1 = RGB1 - mix(c1)
    r2 = RGB2 - mix(c2)
    c  = lerp(c1, c2, t)
    r  = lerp(r1, r2, t)
    return mix(c) + r
```

将色素浓度 $\mathbf{c}$ 和 RGB 残差 $\mathbf{r}$ 连接到单个潜在向量 $\mathbf z=[c_1,c_2, c_3,c_4,r_R,r_G,r_B]^T$。
Establishing a homogeneous color representation by concatenating the pigment concentrations $\mathbf{c}$ and the RGB residuals $\mathbf{r}$ into a single latent vector $\mathbf z=[c_1,c_2,c_3,c_4,r_R,r_G,r_B]^T$​.
$$
\mathcal F(\mathrm{RGB})=\begin{bmatrix}\mathbf c\\\mathbf r\end{bmatrix}=\begin{bmatrix}\mathrm{unmix}(\mathrm{RGB})\\\mathrm{RGB}-\mathrm{mix}(\mathrm{unmix}(\mathrm{RGB}))\end{bmatrix}=\mathbf z\\
\mathcal G(\mathbf z)=\mathrm{mix}(\mathbf c)+\mathbf r
$$
$\mathcal F$ and $\mathcal G$ are so-called encoder and decoder.

该方案类似地扩展到两种以上颜色混合在一起的情况，例如在双线性插值中，在应用卷积时，或一般采用 $n$ 颜色的加权平均：
The scheme extends analogously to situations where more than two colors mix together, like in bilinear interpolation, when applying convolution, or taking a weighted average of $n$ colors in general:
$$
\mathcal{G}\left(\frac{\sum_{i=1}^{n}wi\mathcal{F}(\mathrm{RGB}_i)}{\sum_{i=1}^{n}wi}\right)
$$

```python
def f(RGB):
    c = unmix(RGB)
    r = RGB - c
    return [c, r]
def g(c, r):
    return mix(c) + r
def kmerp(RGBs, w):
    return g(sum(w * f(RGBs)) / sum(w))
```

## 3.1 Surrogate Pigments

上述基于颜料的潜在表示的一个实际问题是，一些颜料混合物会产生 sRGB 色域之外的颜色。
A practical issue with the pigment-based latent representation described above is that some pigment mixtures produce colors outside the sRGB gamut.

我们可能会限制超出范围的值以进行快速补救，但这会导致混合过程不再可逆。
We may clamp the out-of-range values for a quick remedy, but that causes the mixing process to no longer be invertible.

缓解此问题的一种方法是执行色域压缩而不是设阈值。这个想法是使色域内的颜色变形，为色域外的颜色腾出空间并将它们挤进去。
One way to alleviate this issue is to perform gamut compression instead of clamping. The idea is to deform the in-gamut colors to make room for the outside-gamut colors and squeeze them in.

我们不是在颜色混合后压缩颜色，而是调整主要颜料本身，使它们的混合物首先保持在 RGB 色域内。
Instead of compressing the colors after they have been already mixed, we tweak the primary pigments themselves to make their mixtures stay inside the RGB gamut in the first place.

$\mathcal{Q^*}$: 替代颜料 surrogate pigment.















