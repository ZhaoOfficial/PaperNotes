# PREF: Phasorial Embedding Fields for Compact Neural Representations

## 0 Abstract

基于多层感知器 (MLP) 的神经技术偏向于低频信号，并且依赖于深度或傅里叶编码来避免细节丢失。
Pure multi-layer perceptron (MLP) based neural techniques are biased towards low frequency signals and have relied on deep layers or Fourier encoding to avoid loss of details.

相反，PREF 采用基于傅里叶嵌入空间的相量公式的紧凑且物理上可解释的编码场。
PREF instead employs a compact and physically explainable encoding field based on the phasor formulation of the Fourier embedding space.

然后，我们使用 PREF 的近似傅里叶逆变换方案以及新颖的 Parseval 正则化器开发了一个高效的频率学习框架。
We then develop a highly efficient frequency learning framework using an approximated inverse Fourier transform scheme for PREF along with a novel Parseval regularizer.

## 1 Introduction

从低维、低频输入到高频目标函数的暴力映射没有充分考虑映射函数的基本特征。结果是，他们在很大程度上求助于大型 MLP（隐藏维度宽或层深）来可靠地学习相应的函数映射。大型 MLP 的缺点是训练时间长且推理速度慢。
Brute-force mappings from low dimension, low-frequency inputs onto high frequency target functions do not sufficiently consider the underlying characteristics of the mapping function. As a result, they have largely resorted to large MLPs (either wide in hidden dimensions or deep in layers) to reliably learn the corresponding function mapping. The downside of large MLPs is the long training time and slow inference speed.

通过用高维高频嵌入体积（其大小通常由体积分辨率和内核大小决定）替换输入，他们设法弥合了低维低频坐标输入和高维坐标输入之间的差距。频率输出在宽度和深度上都具有更小的 MLP。
By replacing the input field with a high-dimension high-frequency embedding volume (whose size is generally determined by the volume resolution and kernel size), they manage to bridge the gap between the low-dimension low-frequency coordinate inputs and the high-frequency outputs with a much smaller MLP in both width and depth.

然而，由于它们的嵌入空间也以离散形式表示，因此最先进的技术依赖于插值来查询高维嵌入特征。为了保持高效率，采用最多的方案仍然是线性插值。
However, as their embedding space is also represented in discrete forms, state-of-the-art techniques have relied on interpolations for querying high-dimensional embedded features. To maintain high efficiency, the most adopted schemes are still linear interpolation.

- 嵌入场的最高可恢复频率由体积的分辨率决定。为了保持高频，将体积离散化到非常精细的水平是至关重要的。因此，频率完全存储需要极高的内存消耗，即使是最先进的 GPU 也是不行。The highest recoverable frequency of the embedding fields is determined by the resolution of the volume. To preserve high frequencies, it is critical to discretize the volume at a very fine level. Fully frequency preservation hence requires demand extremely high memory consumption, prohibitive on even the most advanced GPUs.
- 从信号处理的角度来看，嵌入体积内的线性插值不仅会导致混叠，还会导致高阶导数消失，阻碍反向传播和整体收敛。From a signal processing perspective, linear interpolation within the embedding volume not only leads to aliasing but also causes higher-order derivatives to vanish, hindering back-propagation and the overall convergence.
- 与其频率对偶相比，空间离散嵌入对信号的了解有限。特别是，尽管用于编辑和风格化的傅里叶信号处理工具种类繁多，但很少有直接适用于空间嵌入的。Spatially discretized embedding, compared to its frequency dual, provides limited insights on the signal. In particular, despite a large variety of Fourier signal processing tools for both editing and stylization, few are directly applicable to spatial embedding.

首先，相量的非线性变换性质避免了线性插值空间嵌入中的导数消失问题。
First, the nonlinear transform nature of the phasor avoids the vanishing derivative problem in linearly interpolated space embedding.

其次，PREF 非常紧凑，不需要用高分辨率的体积来保持高频。
Second, PREF is highly compact and does not require using high resolution volumes to preserve high frequency.

最后，基于傅里叶变换，PREF 设法利用许多现有的傅里叶信号处理技术，如微分、卷积和 Parseval 定理，为 MLP 执行传统上昂贵的操作。
Finally, based on Fourier transforms, PREF manages to exploit many existing Fourier signal processing techniques such as differentiation, convolution, and Parseval's theorem to conduct traditionally expensive operations for the MLP.

## 2 Related Work

### Encoding vs. Embedding

在傅里叶编码下，通过 MLP 进行的特征优化可以映射到线性层中具有复值输入的复值矩阵的优化。
Under Fourier encoding, feature optimization through MLPs can be mapped to optimizing complex-valued matrices with complex-value inputs in the linear layer.

在嵌入的上下文中，重点是保持可学习的高阶输入嵌入，然后通过嵌入空间中的插值计算特征，该方案通过最简单但不比线性插值复杂的方案。
In the context of embedding, the focus is to maintain a learnable high-order input embedding and then compute the feature via interpolation in the embedding space via schemes as simple as nearest neighbor but no more complicated than linear interpolation.

虽然在特征查询方面效率很高，但它们具有消失梯度的共同限制，即由于线性插值导致的分段常数一阶导数和零二阶导数，导致比暴力纯 MLP 有更高的误差。
While efficiency in feature querying, they share a common limitation of vanishing gradient, that is, the piecewise constant first order derivative and zero second order derivative due to linear interpolation, leading to higher than usual errors than brute-force pure MLP implementations.

## 3 Background and Notations

|                      Notation                      |                         Description                          |
| :------------------------------------------------: | :----------------------------------------------------------: |
|                       $f(x)$                       |                    a continuous function                     |
|                 $x\in\mathbb{R}^n$                 |                    low dimensional inputs                    |
|               $\mathcal{M}(;\theta)$               |      an MLP with parameter $\theta$ to approximate $f$       |
|                        $N$                         |              number of grid along one dimension              |
| $\mathbf{P}\in\mathbb{C}^{k\times N\times\dots N}$ |         a complex-value volume with $n+1$ dimension          |
|               $\tilde{\mathcal{T}}$                |                  inverse Fourier Transform                   |
|               $l_{x}\in\mathbb{R}^k$               | $l_x=\tilde{\mathcal{T}}(\mathbb{P};x)$, $f(x)=\mathcal{M}(l_x;\theta)$ |
|                        $j$                         |                    unit of imaginary part                    |

事实上，作为基于频率的替代方案，PREF 明确地将频率与特征相关联：每个体像条目代表相应频率下的傅立叶系数。
In fact, as a frequency-based alternative, PREF explicitly associates frequencies with features: each volume entry represents the Fourier coefficients under the corresponding frequency.

#### Property in $\mathbb{R}^2$
$$
\begin{align*}
f(x,y)&=\tilde{\mathcal{T}}(F)\\
&=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}F(u,v)e^{j2\pi\left(u\frac{x}{M}+v\frac{y}{N}\right)}\mathrm{d}u\mathrm{d}v\\
&\approx\sum_{u=-\infty}^{\infty}\sum_{v=-\infty}^{\infty}F[u,v]e^{j2\pi\left(u\frac{x}{M}+v\frac{y}{N}\right)}
\end{align*}\tag{1}
$$

Differential property
$$
\frac{\partial^n{f}}{\partial{x^n}}(x,y)=\tilde{\mathcal{T}}((j2\pi u)^nF[u,v])
$$

Parserval's equation
$$
\iint|f(x,y)|^2\mathrm{d}x\mathrm{d}y=\iint|F(u,v)|^2\mathrm{d}u\mathrm{d}v
$$

## 4 Phasorial Embedding Fields

简而言之，PREF 使用 (1) 中的傅里叶变换将空间坐标 $x$ 映射到 $k$ 通道特征 $f(x)\in\mathbb{R}^k$。
In a nutshell, PREF employs the Fourier transforms in (1) to map spatial coordinate $x$ into $k$-channel feature $f(x)\in\mathbb{R}^k$.

### 4.1 Phasor Volume Decomposition

为了降低复杂性，我们使用一组稀疏的频率来编码 PREF。该过程相当于有选择地将矩阵 $F(u, v)$ 中的大部分条目标记为零。
To reduce complexity, we use a sparse set of frequencies to encode PREF. The process is equivalent to selectively marking a large portion of the entries in matrix $F(u, v)$ as zero.

特别是，我们着手通过沿每个维度的对数采样来分解 $\mathbf{P}[u,v]\in\mathbb{C}^{n\times n}$。
In particular, we set out to factorize $\mathbf{P}[u,v]\in\mathbb{C}^{N\times N}$ by logarithmic sampling along each dimensions.
$$
\begin{align*}
\mathbf{P}^u&=\mathbf{P}[u,k],k=\{0,2^0,2^1,\dots,2^{D-2}\}\\
\mathbf{P}^v&=\mathbf{P}[l,v],l=\{0,2^0,2^1,\dots,2^{D-2}\}
\end{align*}
$$
with $D$ a small number. $N^2\to 2ND$ or $O(N^{n})\to O(nN^{n-1}\log(N))$.

### 4.2 IFT Approximation

如果所有输入坐标等距，则傅里叶变换将简化为离散傅里叶变换 (DFT)，可以使用快速傅里叶变换 (FFT) 方法显式求值。
If all input coordinates are equally spaced, then the Fourier transform simplifies to the Discrete Fourier Transform (DFT) that can be explicitly evaluated using fast Fourier transforms (FFT) methods.

我们同时采用 FFT 和数值积分来实现高精度和相对较低的复杂性。具体来说，我们首先沿其中一个轴 $u$ 执行 1D FFT，以获得中间映射 $\mathbf{P}_v[x^*,v]$，其中 $x^*=\{0,1,\dots ,M-1\}$。
We employ both FFT and numerical integration to achieve both high accuracy and relatively low complexity. Specifically, we first perform 1D FFT along one of the axes $u$ to obtain a intermediate map $\mathbf{P}_v[x^*,v]$, with $x^*=\{0,1,\dots,M-1\}$.
$$
\begin{align*}
\mathbf{P}_{v}(x,v)&=\langle\mathbf{P}_v[x^*,v]\rangle\\
f(x,y)&\approx\sum_{v=0}^{D}e^{j2\pi v\frac{y}{N}}\mathbf{P}_{v}(x,v)=\tilde{T}_*(\mathbf{P})
\end{align*}
$$
其中 $\langle\cdot\rangle$ 是一个线性插值运算，对应于从中间映射 $\mathbf{P}_v[x^{*}, v]$ 插值 $\mathbf{P}_{v}(x,v)$。
where $\langle\cdot\rangle$ is a linear interpolation operation that corresponds to interpolating $\mathbf{P}_{v}(x,v)$ from the intermediate map $\mathbf{P}_v[x^{*},v]$.

### 4.3 Volume Regularization

A novel Parseval regularizer:
$$
\mathcal{L}_{\text{Parseval}}=\sqrt{\sum_{u=0}^{N-1}\sum_{v=0}^{N-1}(2\pi u\mathbf{P}[u,v])^2}+\sqrt{\sum_{u=0}^{N-1}\sum_{v=0}^{N-1}(2\pi v\mathbf{P}[u,v])^2}
$$

令 $f(x,y)$ 是可积的，$\mathbf{P}(u,v)$ 是它的傅里叶变换。$f(x,y)$ 的 Anisotropic TV loss 可以表示为 $...$。
Let $f(x,y)$ be integrable, and $\mathbf{P}(u,v)$ be its Fourier transform. The anisotropic TV loss of $f(x,y)$ can be represented by $...$.

> Anisotropic TV loss:
  $$
> \sum_{i}\|\partial_{x_i}f\|_1=\sum_{x}\sum_{y}\left|\Delta_{x}f[x,y]\right|+\sum_{x}\sum_{y}\left|\Delta_{y}f[x,y]\right|
> $$
> where $\Delta_{x}$ is the difference along $x$ axis.
> 
  Since $f(x,y)$ and $P(u,v)$ are Fourier pair, by Parseval equation:
> $$\iint|f(x,y)|^2\mathrm{d}x\mathrm{d}y=\iint|P(u,v)|^2\mathrm{d}u\mathrm{d}v
  $$ And $\partial_{x}f(x,y)$ and $j2\pi u P(u,v)$ are Fourier pair, then:
  $$
> \iint|\partial_{x}f(x,y)|^2\mathrm{d}x\mathrm{d}y=\iint|j2\pi u P(u,v)|^2\mathrm{d}u\mathrm{d}v
> $$
  Then we have:
  $$
  \begin{align*}&\quad\ \ \iint|\partial_{x}f(x,y)|^2+|\partial_{y}f(x,y)|^2\mathrm{d}x\mathrm{d}y\\
  &=\iint|2\pi u P(u,v)|^2+|2\pi v P(u,v)|^2\mathrm{d}u\mathrm{d}v
  \end{align*}
  $$

### 4.4 Complexity Analysis

Space complexity: $O(kN^3)\to O(3kN^2D)$.

