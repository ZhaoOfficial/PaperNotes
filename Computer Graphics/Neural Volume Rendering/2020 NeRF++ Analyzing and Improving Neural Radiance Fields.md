# NeRF++ Analyzing and Improving Neural Radiance Fields

## 0 Abstract

我们首先评论辐射场及其潜在的歧义性，即形状-辐射歧义，并分析为什么 NeRF 能成功避免这方面的歧义。
We first remark on radiance fields and their potential ambiguities, namely the shape-radiance ambiguity, and analyze NeRF’s success in avoiding such ambiguities.

我们解决了一个参数化问题，涉及将 NeRF 应用于大规模、无边界 3D 场景中的 $360^\circ$ 物体捕获。
We address a parametrization issue involved in applying NeRF to $360^\circ$ captures of objects within large-scale, unbounded 3D scenes.

## 1 Introduction

通过在每个表面点选择合适的 2D 辐射输出，可以在任意不正确的几何形状的情况下完美地拟合一组训练图像。
One can fit a set of training images perfectly for an arbitrary incorrect geometry by a suitable choice of outgoing 2D radiance at each surface point.

对于 $360^\circ$ 的照片，NeRF 假设整个场景可以装入一个有界的体积，这对于大规模的场景来说是有问题的：要么我们把场景的一小部分装入体积并对其进行详细采样，但完全不能捕获背景元素；要么，我们把整个场景装入体积，但由于采样分辨率有限，到处都缺乏细节。 
For $360^\circ$ captures, NeRF assumes that the entire scene can be packed into a bounded volume, which is problematic for large scale scenes: either we fit a small part of the scene into the volume and sample it in detail, but completely fail to capture background elements; or, we fit the full scene into the volume and lack detail everywhere due to limited sampling resolution.  

## 2 Preliminaries

## 3 Shape-Radiance Ambiguity

我们假设，有两个相关的因素来拯救 NeRF。1）不正确的几何迫使辐射场具有更高的内在复杂性（即更高的频率），而相反，2）NeRF 的特定 MLP 结构隐含地对表面反射率的平滑 BRDF 先验进行编码。
We hypothesize that two related factors come to NeRF's rescue: 1) incorrect geometry forces the radiance field to have higher intrinsic complexity (i.e., much higher frequencies) while in contrast 2) NeRF’s specific MLP structure implicitly encodes a smooth BRDF prior on surface reflectance.

### Factor 1

由于 $\sigma$ 偏离正确的形状，$\mathbf{c}$ 通常必须成为关于 $\mathbf{d}$ 的高频函数来重建输入图像。对于正确的形状，表面光场通常会更加平滑（事实上，对于漫反射材料来说是常数）。不正确形状所需的更高复杂性更难以用有限容量的 MLP 表示。
As $\sigma$ deviates from the correct shape, $\mathbf{c}$ must in general become a high-frequency function with respect to $\mathbf{d}$ to reconstruct the input images. For the correct shape, the surface light field will generally be much smoother (in fact, constant for Lambertian materials). The higher complexity required for incorrect shapes is more difficult to represent with a limited capacity MLP.

### Factor 2

对于一个固定的 $\mathbf{x}$，辐射度 $\mathbf{c}(\mathbf{x};\mathbf{d})$ 相对于 $\mathbf{d}$ 的表达能力有限。
For a fixed $\mathbf{x}$, the radiance $\mathbf{c}(\mathbf{x};\mathbf{d})$ has limited expressibility with respect to $\mathbf{d}$.

我们进行了一个实验，我们用一个普通 MLP 来表示 $\mathbf{c}$，它对称地对待 $\mathbf{x}$ 和 $\mathbf{d}$ — 即，两者都作为第一层的输入和接受 $\gamma^{10}(\cdot)$ 编码 — 以消除由网络结构产生的任何涉及观察方向的隐式先验。如果我们用这个替代模型从头开始训练 NeRF，与 NeRF 的特殊 MLP 相比，我们观察到测试图像质量下降。这个结果与我们的假设一致，即 NeRF 的 MLP 辐射模型中反射率的隐式正则化有助于恢复正确的解决方案。
We perform an experiment where we instead represent $\mathbf{c}$ with a vanilla MLP that treats $\mathbf{x}$ and $\mathbf{d}$ symmetrically — i.e., accepting both as inputs to the first layer and encoding both with $\gamma^{10}(\cdot)$ — to eliminate any implicit priors involving viewing direction that arise from the network structure. If we train NeRF from scratch with this alternate model for $\mathbf{c}$, we observe reduced test image quality compared with NeRF’s special MLP. This result is consistent with our hypothesis that implicit regularization of reflectance in NeRF’s MLP model of radiance $\mathbf{c}$ helps recover correct solutions.

## 4 Inverted Sphere Parametrization

对于户外，$360^\circ$ 捕捉以附近物体为中心同时观察周围环境，动态深度范围可以非常大，因为背景（建筑物、山脉、云层等）可以任意远。
For outdoor, $360^\circ$ captures centered on nearby objects that also observe the surrounding environment, the dynamic depth range can be extremely large, as the background (buildings, mountains, clouds, etc.) can be arbitrarily far away.

在更受限制的场景中，所有相机都面向将相机与场景内容分开的平面，NeRF 通过将欧几里得空间的子集（即参考相机的视锥体）投影映射到归一化设备坐标（NDC），并整合到这个 NDC 空间中。
In a more restricted scenario where all cameras are forward-facing towards a plane separating the cameras from the scene content, NeRF addresses this resolution issue by projectively mapping a subset of the Euclidean space, i.e., a reference camera’s view frustum, toNormalized Device Coordinates (NDC), and integrating in this NDC space.

在我们的表示中，我们首先将场景空间划分为两个体积，一个内部单位球体和一个由倒置球体表示的外部体积，覆盖内部体积的补充。内部体积包含前景和所有摄像机，而外部体积包含环境的其余部分。
In our representation, we first partition the scene space into two volumes, an inner unit sphere and an outer volume represented by an inverted sphere covering the complement of the inner volume. The inner volume contains the foreground and all the cameras, while the outer volume contains the remainder of the environment.  

这两个卷是用两个独立的 NeRF 建模的。为了渲染光线的颜色，它们分别进行光线投射，然后进行最终合成。内部 NeRF 不需要重新参数化，因为场景的那部分有很好的界限。对于外部 NeRF，我们应用了倒置球体参数化。
These two volumes are modelled with two separate NeRFs. To render the color for a ray, they are raycast individually, followed by a final compositition. No re-parameterization is needed for the inner NeRF, as that part of the scene is nicely bounded. For the outer NeRF, we apply an inverted sphere parametrization.

在单位球外的点 $(x,y,z)$ 可以重新参数化为：$(x',y',z',1/r)$，其中 $r=\sqrt{x^2+y^2+z^2}$，$(x',y',z')$ 是单位化的，和 $(x,y,z)$ 同方向的方向向量。因此后景也被限制在一个单位球内。

这不仅提高了数值稳定性，而且还尊重了更远的对象应该获得更低分辨率的事实。
This not only improves numeric stability, but also respects the fact that farther objects should get less resolution.

这里我们需要把渲染方程分裂成两部分：对于光线 $\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$，假设 $t\in(0,t')$ 在内部球，$t\in(t',\infty)$ 在外部体积。
$$
\begin{align*}
\mathbf{C}(\mathbf{r})&=\int_0^{t'}T(\mathbf{r}(0),\mathbf{r}(t))\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})\mathrm{d}t\\
&+T(\mathbf{r}(0),\mathbf{r}(t'))\int_{t'}^{\infty}T(\mathbf{r}(t'),\mathbf{r}(t))\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})\mathrm{d}t\\
\end{align*}
$$
最后一个积分是在外部体积中计算的，计算所用的应为 $\sigma(\mathbf{r}(t),1/|\mathbf{r}(t)|)$ 和 $\mathbf{c}(\mathbf{r}(t),1/|\mathbf{r}(t)|,\mathbf{d})$。

<img src="../../Note Image/nerf++.png" style="zoom:50%;" />

首先在 $[0,1]$ 范围内对于 $1/r$ 进行采样。对于一个给定的光线 $\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$ 和一个距离 $1/r$，我们需要得到在三维空间的点。假设光线和球相交于点 $\mathbf{a}$，这个点只要解 $|\mathbf{o}+t\mathbf{d}|=1$ 就可以得到。球心在光线上的垂足在点 $\mathbf{b}$，这个点只要解 $\mathbf{d}^T(\mathbf{o}+t\mathbf{d})=0$ 就可以得到。要得到 $(x',y',z')$ 的方向，我们只需要把 $\mathbf{a}$ 绕着 $\mathbf{b}\times\mathbf{d}$ 旋转 $\omega=\arcsin|\mathbf{b}|-\arcsin(|\mathbf{b}|/r)$ 就得到了方向。

它可以看作是一个虚拟相机，它的像平面是以场景为原点的单位球体。
It can be viewed in terms of a virtual camera whose image plane is the unit sphere at the scene origin.

## 6 Open Challenges

还可以考虑自动曝光和渐晕等光度效应，以提高图像保真度。
Photometric effects such as auto-exposure and vignetting can also be taken into account to increase image fidelity.
