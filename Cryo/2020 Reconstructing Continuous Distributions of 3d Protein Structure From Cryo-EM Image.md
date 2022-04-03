# Reconstructing Continuous Distributions of 3D Protein Structure From cryo-EM Images (CryoDRGN)

## 0 Abstract

在单粒子冷冻电镜中，核心问题是从 $10^{4}\sim10^{7}$ 噪声和随机朝向的 2D 投影图像重建大分子的 3D 结构。
In single particle cryo-EM, the central problem is to reconstruct the 3D structure of a macromolecule from $10^{4}\sim10^{7}$ noisy and randomly oriented 2D projection images.

该方法使用基于坐标的深度神经网络对傅里叶空间中的结构进行编码，并通过将**图像方向上的精确推断**与**结构异质性的变分推断相**结合，从未标记的 2D 冷冻电镜图像中训练这些网络。
This method encodes structures in Fourier space using coordinate-based deep neural networks, and trains these networks from unlabeled 2D cryo-EM images by combining **exact inference over image orientation** with **variational inference for structural heterogeneity**.

## 1 Introduction

由于冰中分子的每个分子方向未知，重建变得复杂。此外，冷冻电镜重建算法必须应对诸如极低的信噪比 (SNR)、未知的平面内平移、显微镜光学导致的不完美信号传输以及测量离散化等挑战。
cryo-EM reconstruction is complicated by the unknown orientation of each copy of the molecule in the ice. Furthermore, cryo-EM reconstruction algorithms must handle challenges such as an extremely low signal to noise ratio (SNR), unknown in-plane translations, imperfect signal transfer due to microscope optics, and discretization of the measurements.

我们提出了一个端到端的学习框架，使用图像编码器-体像解码器神经网络架构在 3D 体积上生成模型。 扩展空间-VAE，我们将我们的解码器制定为 3D 坐标和无约束潜变量的函数，这些潜变量表示我们期望由蛋白质结构异质性引起的图像变化因素。所有推理都在傅立叶空间中执行，这使我们能够通过傅立叶切片定理有效地将 2D 投影与 3D 体像相关联。
We present an end-to-end learning framework for a generative model over 3D volumes using an image encoder-volume decoder neural network architecture. Extending spatial-VAE, we formulate our decoder as a function of 3D Cartesian coordinates and unconstrained latent variables representing factors of image variation that we expect to result from protein structural heterogeneity. All inference is performed in Fourier space, which allows us to efficiently relate 2D projections to 3D volumes via the Fourier slice theorem.

## 2 Background and Notation

### 2.1 Image Formation Model

Cryo-EM 旨在恢复目标结构 $V:\mathbb R^3\to\mathbb R$，该结构由空间中每个点的电子密度组成，基于产生的噪声图像 $X_1,\dots,X_n$ 通过沿成像轴沿未知方向进行体像投影（即积分）。
Cryo-EM aims to recover a structure of interest $V:\mathbb R^3\to\mathbb R$ consisting of an electron density at each point in space based on a collection of noisy image $X_1,\dots,X_n$ produced by projecting (i.e. integrating) the volume in an unknown orientation along the imaging axis.

图像 $X$ 可以由下式生成
$$
X(r_x,r_y)=g*\int_{\mathbb{R}}V(R^T\mathbf{r}+t)dr_z+\mathcal{N}\qquad\mathbf{r}=(r_x,r_y,r_z)^T
$$
$V$ 是电子密度。

$R\in\mathrm{SO}(3)$ 是未知的 3D 旋转矩阵。

$t$ 是相对于未校准的体像的中心的未知 2D 平移向量。

$g$ 是显微镜点扩散函数，被与频率相关的噪声污染了，大小是 $D\times D$。

由观察可得 $V$ 的二维投影的傅里叶变换是在傅里叶域中通过 $V$ 的原点的二维切片，以此来简化重建问题，其中切片垂直于投影方向。 这种对应被称为傅里叶切片定理。
The reconstruction problem is simplified by the observation that the Fourier transform of a 2D projection of $V$ is a 2D slice through the origin of $V$​ in the Fourier domain, where the slice is perpendicular to the projection direction. This correspondence is known as the Fourier slice theorem.

在 Fourier 频域中，从体像 $\hat{V}$ 中生成图像 $\hat{X}$ 是
$$
\hat{X}(k_x,k_y)=\hat{g}S(t)A(R)\hat{V}(k_x,k_y)+\epsilon
$$
$\hat{g}$ 是显微镜的 CTF 函数。

$S(t)$ 是图像平移 $t$ 后对应的相移。

$A(R)\hat{V}(k_x,k_y)$ 是在旋转为 $R$ 的情况下沿着 $z$ 轴投影的 Fourier 切片。

$\epsilon$ 是在频域的零中心的 Gaussian 噪声。

从体像 $\hat{V}$ 中观察到位姿为 $\phi(R,t)$ 的图像 $\hat{X}$ 的概率是
$$
P(\hat{X}\mid \phi(R,t),\hat{V})=P(\hat{X}\mid R,t,\hat{V})=\frac{1}{Z}\exp({\sum_{l}\frac{-1}{2\sigma^{2}_{l}}|\hat{g}_lA_l(R)\hat{V}-S_l(t)\hat{X}_l|^2})
$$
$l$ 是二维 Fourier 系数。

$\sigma_l$ 是对应频率的 Gaussian 噪声方差。

$Z$ 是归一化常数。

即在 Fourier 频域中，体像经过旋转，被 CTF 调制后和平移后的图像的差，即噪声满足 Gaussian 分布。

### 2.2 Traditional Cryo-EM Reconstruction

[cryoSPARC](2017 cryoSPARC.md) Using Maximum a posterior.

在第 $n$ 轮估计中，图像会和 $V^{(n)}$ 对齐再投影出 $V^{(n+1)}$。

## 3 Method

### 3.1 Generative Model

3D 体像 $\hat{V}$ 可以看成概率解码器 $P_{\theta}(\hat{V}|k,z)$，$\theta,k,z$ 分别是 MLP 参数，3D 坐标，连续的潜在变量。解码器输出 Gaussian 分布 $\hat{V}(k,z)$ 的参数的分布，即在傅立叶空间中频率 $k$ 处体积 $\hat V_z$ 的电子密度。
The volume $\hat V$ is modeled as a probabilistic decoder $P_{\theta}(\hat{V}|k,z)$, where $\theta$ are parameters of a MLP. Given Cartesian coordinates $k\in\mathbb R^3$ and continuous latent variable $z$, the decoder outputs distribution parameters for a Gaussian distribution over $\hat{V}(k,z)$, i.e. the electron density of volume $\hat V_z$ at frequency $k$ in Fourier space.

在这里，这些坐标被明确地视为每个像素在 3D 傅里叶空间中的位置，并通过傅里叶切片定理在 3D 中强制执行 2D 视图之间的拓扑约束。
Here, these coordinates are explicitly treated as each pixel's location in 3D Fourier space and thus enforce the topological constraints between 2D views in 3D via the Fourier slice theorem.

Each image corresponds to an oriented central slice of the 3D volume in the Fourier domain.
每张图像对应于 3D 体像在 Fourier 域中的某一个朝向的中心切片。

During training, the 3D coordinates of an image's pixels can be explicitly represented by the rotation of a $D\times D$ lattice initially on the $xOy$ plane.
初始在 $xOy$ 平面上的 $D\times D$ 的格点旋转后可以表示图像的 3D 坐标。即将 2D 图像插入到 3D 空间中。
$$
\log P(\hat{X}|R,t,z)=\log P(\hat{X}'|R,z)=\sum_{i}\log P_{\theta}(\hat{V}|R^Tc_0^{(i)},z)
$$
$\hat{X}$ 是一维 $D\times D$ 图像，$R,t,z,\theta$ 是旋转，平移，潜在变量，MLP 参数。

$\hat{X}'=S(-t)\hat{X}$ 是居中的图像。

$c_0\in[-0.5,0.5,0]^2$ 是格点，表示了图像像素的无朝向时候的坐标。

$i$ 是固定格点 $c_0$ 的索引。

我们不直接提供 $k$，而是位置编码器
Instead of directly supplying $k$, a fixed positional encoding of $k$ is supplied to the decoder:
$$
pe^{(2i)}(k_j)=\sin({k_j(\frac{2}{D})^{\frac{2i}{D}}D\pi}),i\in(1,D/2);k_j\in k\\
pe^{(2i+1)}(k_j)=\cos({k_j(\frac{2}{D})^{\frac{2i}{D}}D\pi}),i\in(1,D/2);k_j\in k
$$

在不失一般性的情况下，我们假设由 $c_0$ 定义的长度尺度，它将体积的支持限制在半径为 0.5 的球体上。 因此，位置编码的波长遵循跨越傅里叶基从波长 1 到图像数据的奈奎斯特极限 ($2/D$) 的几何级数。 虽然这种编码在经验上适用于无噪声数据，但我们通过稍微修改特征化的噪声数据集获得了更好的结果，噪声数据集由几何序列组成，排除了无噪声位置编码的最高频率分量的前 10 个百分位数。
Without loss of generality, we assume a length scale by our definition of $c_0$ which restricts the support of the volume to a sphere of radius 0.5. The wavelengths of the positional encoding thus follow a geometric series spanning the Fourier basis from wavelength 1 to the Nyquist limit ($2/D$) of the image data. While this encoding empirically works well for noiseless data, we obtain better results with a slightly modified featurization for noisy datasets consisting of a geometric series which excludes the top 10 percentile of highest frequency components of the noiseless positional encoding.

### 3.2 Inference

我们使用标准 VAE 来近似推断潜在变量 $z$，用分支界限算法全局搜索推断位姿 $\phi=(R,t)$。
We employ a standard VAE for approximate inference of the latent variable $z$, but use a global search to infer the pose $\phi=(R,t)$ using a branch and bound algorithm.

#### 3.2.1 Variational Encoder

输入一个 $D\times D$ 的图像 $\hat{X}$，概率编码器 $q_{\xi}(z|X)$ 会输出 Gaussian 分布的 $\mu_{z|\hat{X}}$ 和 $\sigma_{z|\hat{X}}$，即 $z$ 的后验分布 $P(z|\hat{X})$。(假设 $z$ 的先验分布是 $\mathcal N(0,\mathbf{I})$)
The probabilistic encoder $q_{\xi}(z|X)$ is a MLP with variational parameters $\xi$ and Gaussian output with diagonal covariance. Given an input cryo-EM image $\hat X$, represented as a $D\times D$ vector, the encoder MLP outputs $\mu_{z|\hat{X}}$ and $\sigma_{z|\hat{X}}$, statistics that parameterize an approximate posterior to the intractable true posterior $P(z|\hat{X})$. The prior on $z$ is a standard normal, $\mathcal N(0,\mathbf{I})$.

#### 3.2.2 Pose Inference

在 $SO(3)\times\mathbb{R}^2$ 中搜寻最大似然位姿
We perform a global search over $SO(3)\times\mathbb{R}^2$ for the maximum-likelihood pose.

1. 离散化搜索空间，并且在分支界限法修剪掉一些分支后细分空间。
    Discretizing the search space on a uniform grid and sub-dividing grid points after pruning candidate poses with branch and bound.
2. 初始带通较低，随后迭代时提高通过频率。
    Band pass limiting the objective to low frequency components and incrementally increasing the k-space limit at each iteration (frequency marching).

对于一个给定的图像 $\hat{X}$，图像编码出以 $\mu_{z|\hat{X}}$ 和 $\sigma_{z|\hat{X}}$ 为参数的 Gaussian 分布。随后从分布中采样出潜在变量 $z_i\sim\mathcal N(\mu_{z|\hat X_i},\Sigma_{z|\hat X_i})$，并作用 (广播 broadcast) 到每个像素点。随后在给定的 $z_i$ 和当前的解码器，分支界限算法找到最大似然位姿。给定位置编码 $R^Tc_0$ 和 $z_i$，解码器 $p_{\theta}$ 运用位移向量 $t_i$ 和 CTF 函数 $\hat g_i$ 进行图像重建。在标准 VAE 框架下，优化函数是
$$
\mathcal{L}(\hat{X}_i;\xi,\theta)=\mathbb{E}_{q_{\xi}(z|\hat{X}_i)}[\log P_{\theta}(\hat{X}_i|Z)]-KL(q_{\xi}(z|\hat{X}_i)||p(z))
$$
期望是通过 Monte Carlo 采样得到。



