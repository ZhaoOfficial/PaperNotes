# Plenoxels: Radiance Fields without Neural Networks

## 0 Abstract

Plenoxels 将场景表示为具有球谐函数的稀疏 3D 网格。这种表示可以通过梯度方法和正则化从标定图像中优化，而无需任何神经网络。
Plenoxels represent a scene as a sparse 3D grid with spherical harmonics. This representation can be optimized from calibrated images via gradient methods and regularization without any neural components.

## 1 Introduction

我们提供了一个自定义 CUDA 实现，它利用模型的简单性来实现显着的加速。
We provide a custom CUDA implementation that capitalizes on the model simplicity to achieve substantial speedups.

尽管我们的实现并未针对快速渲染进行优化，但我们可以以 15 fps 的交互速率渲染新视角。如果需要更快的渲染，我们优化的 Plenoxel 模型可以转换为 PlenOctree。
Although our implementation is not optimized for fast rendering, we can render novel viewpoints at interactive rates 15 fps. If faster rendering is desired, our optimized Plenoxel model can be converted into a PlenOctree.

我们将我们的模型称为 Plenoxel 用于全光体积元素，因为它由稀疏体素网格组成，其中每个体素存储不透明度和球谐系数。这些系数被插值以在空间中连续模拟全光函数。
We call our model Plenoxel for plenoptic volume elements, as it consists of a sparse voxel grid in which each voxel stores opacity and spherical harmonic coefficients. These coefficients are interpolated to model the full plenoptic function continuously in space.

尽管我们的核心模型是有界体素网格，但我们可以通过使用 NDC（用于前向场景）或通过用多球图像围绕我们的网格来编码背景（用于 360 度场景）来建模无界场景。
Although our core model is a bounded voxel grid, we can model unbounded scenes by using normalized device coordinates (for forward-facing scenes) or by surrounding our grid with multisphere images to encode the background (for $360^{\circ}$ scenes).

我们的方法表明，可以使用来自逆问题的标准工具来实现逼真的体积重建：数据表示、正向模型、正则化函数和优化器。我们的方法表明，这些组件中的每一个都可以很简单，并且仍然可以实现最先进的结果。
Our method reveals that photorealistic volumetric reconstruction can be approached using standard tools from inverse problems: a data representation, a forward model, a regularization function, and an optimizer. Our method shows that each of these components can be simple and state of the art results can still be achieved.

## 2 Related Work

### Classical Volume Reconstruction

### Neural Volume Reconstruction

### Accelerating NeRF

在这些方法中，有许多方法通过将 3D 体积细分为可以更有效地处理的区域来实现加速。其他加速方法侧重于一系列计算和预处理或后处理方法，以消除原始 NeRF 公式中的瓶颈。
Among these methods are many that achieve speedup by subdividing the 3D volume into regions that can be processed more efficiently. Other speedup approaches have focused on a range of computational and pre- or post-processing methods to remove bottlenecks in the original NeRF formulation.

JAXNeRF, AutoInt, Learned Initializations.

另一种方法是预训练一个 NeRF（或类似模型），然后将其提取到可以支持快速推理的不同数据结构中
Another approach is to pretrain a NeRF (or similar model) and then extract it into a different data structure that can support fast inference.

PlenOctree.

## 3 Methods

![pipeline](../../Note Image/plenoxels.png)

### 3.1 Volume Rendering

虽然这个公式并不精确（它假设样本之间的单一散射和恒定值），但它是可微的，并且能够根据每个训练射线的误差更新 3D 模型。
Although this formula is not exact (it assumes single scattering and constant values between samples), it is differentiable and enables updating the 3D model based on the error of each training ray.

### 3.2 Voxel Grid with Spherical Harmonics

然而，为了简单和易于实现三线性插值，我们不使用八叉树作为我们的数据结构。相反，我们将带有指针的密集 3D 索引数组存储到一个单独的数据数组中，该数组仅包含占用体素的值。
However, for simplicity and ease of implementing trilinear interpolation, we do not use an octree for our data structure. Instead, we store a dense 3D index array with pointers into a separate data array containing values for occupied voxels only.

我们使用 2 次球谐函数，每个颜色通道需要 9 个系数，每个体素总共需要 27 个球谐系数。因为 PlenOctrees 发现高次球谐函数收益比较小。
We use spherical harmonics of degree 2, which requires 9 coefficients per color channel for a total of 27 harmonic coefficients per voxel. We use degree 2 harmonics because PlenOctrees found that higher order harmonics confer only minimal benefit.

我们的 Plenoxel 网格使用三线性插值来定义整个体积的连续全光函数。这与 PlenOctrees 不同，后者假设不透明度和球谐系数在每个体素内保持不变。
Our Plenoxel grid uses trilinear interpolation to define a continuous plenoptic function throughout the volume. This is in contrast to PlenOctrees, which assumes that the opacity and spherical harmonic coefficients remain constant inside each voxel.

### 3.3 Interpolation

插值的好处有两点：插值可以表示次体素级别的颜色和不透明度的变化，以此提高有效分辨率；插值产生连续函数近似，这对于成功优化至关重要。
The benefits of interpolation are twofold: interpolation increases the effective resolution by representing sub-voxel variations in color and opacity, and interpolation produces a continuous function approximation that is critical for successful optimization.

将最近邻插值 Plenoxel 的分辨率加倍可以在固定分辨率下缩小最近邻插值和三线性插值之间的大部分差距，但由于优化不连续模型的困难，仍然存在一些差距。
Doubling the resolution of a nearest-neighbor interpolating Plenoxel closes much of the gap between nearest neighbor and trilinear interpolation at a fixed resolution, yet some gap remains due to the difficulty of optimizing a discontinuous model.

### 3.4 Coarse to Fine

我们通过从粗到细的策略实现高分辨率，该策略从较低分辨率的密集网格开始，优化、修剪不必要的体素，通过在每个维度上将每个体素细分为两半来细化剩余体素，然后继续优化。
We achieve high resolution via a coarse-to-fine strategy that begins with a dense grid at lower resolution, optimizes, prunes unnecessary voxels, refines the remaining voxels by subdividing each in half in each dimension, and continues optimizing.

在每个体素细分步骤之后，我们使用三线性插值来初始化网格值。事实上，我们可以使用三线性插值在任意分辨率之间调整大小。
We use trilinear interpolation to initialize the grid values after each voxel subdivision step. In fact, we can resize between arbitrary resolutions using trilinear interpolation.

由于三线性插值，直接修剪会对表面附近的颜色和密度产生不利影响，因为这些点的值会与直接外部的体素进行插值。为了解决这个问题，我们执行了一个膨胀操作，这样一个体素只有在它自己和它的邻居都被认为没有被占用时才会被修剪。
Due to trilinear interpolation, naively pruning can adversely impact the the color and density near surfaces since values at these points interpolate with the voxels in the immediate exterior. To solve this issue, we perform a dilation operation so that a voxel is only pruned if both itself and its neighbors are deemed unoccupied.

### 3.5 Optimization

我们针对渲染像素颜色的 (MSE) 优化体素不透明度和球谐系数，并进行总变差 (TV) 正则化。
We optimize voxel opacities and spherical harmonic coefficients with respect to the mean squared error (MSE) over rendered pixel colors, with total variation (TV) regularization.
$$
\begin{align*}
\mathcal L&=\mathcal L_{\mathrm{reconstruction}}+\lambda_{\mathrm{TV}}\mathcal L_{\mathrm{TV}}\\
\mathcal L_{\mathrm{reconstruction}}&=\frac{1}{|\mathcal R|}\sum_{\mathbf r\in\mathcal R}\|C(\mathbf r)-\hat{C}(\mathbf r)\|^2_2\\
\mathcal L_{\mathrm{TV}}&=\frac{1}{|\mathcal V|}\sum_{v\in\mathcal V,d\in[D]}\sqrt{\Delta_x^2(\mathbf v, d)+\Delta_y^2(\mathbf v, d)+\Delta_z^2(\mathbf v, d)}
\end{align*}
$$
其中 $\Delta_x^2(\mathbf v,d)$ 是体素 $\mathbf v$ 在 $x$ 方向上第 $d$ 个分量的差分 $\Delta_x^2(\mathbf v,d)=\left[\mathbf v(i,j,k,d)-v(i+1,j,k,d)\right]^2$，$y,z$ 同理。

为了更快的迭代，我们使用射线的随机样本 $\mathcal R$ 来评估 MSE 项，并使用体素的随机样本 $\mathcal V$ 来评估每个优化步骤中的 TV 项。
For faster iteration, we use a stochastic sample of the rays $\mathcal R$ to evaluate the MSE term and a stochastic sample of the voxels $\mathcal V$ to evaluate the TV term in each optimization step.

直接优化体素系数是一个具有挑战性的问题，原因如下：需要优化的值很多（问题是高维的），由于渲染公式，优化目标是非凸的，并且目标条件不佳。
Directly optimizing voxel coefficients is a challenging problem for several reasons: there are many values to optimize (the problem is high-dimensional), the optimization objective is nonconvex due to the rendering formula, and the objective is poorly conditioned.

相反，我们使用 RMSProp 来缓解病态问题，而无需二阶方法的全部计算复杂性。
Instead, we use RMSProp to ease the ill-conditioning problem without the full computational complexity of a second-order method.

### 3.6 Unbounded Scenes

对于 $360^\circ $ 的场景，我们使用多球图像 (MSI) 背景模型来增强我们的稀疏体素网格前景表示，该模型还使用学习的体素颜色和不透明度以及球体内部和球体之间的三线性插值。
For $360^\circ$ scenes, we augment our sparse voxel grid foreground representation with a multis phere image (MSI) background model, which also uses learned voxel colors and opacities with trilinear interpolation within and between spheres.

### 3.7 Regularization

在真实、前向和 360 度场景中，我们使用基于 Cauchy 损失的稀疏先验
$$
\mathcal L_s=\lambda_s\sum_{i,j}\log(1+2\sigma(\mathbf r_i(t_k))^2)
$$
$\sigma(\mathbf r_i(t_k))$ 是光线 $i$ 上样本 $k$ 的体密度。

在真实、360 度场景，我们在每批每根光线的前景累积透射率上使用 beta 分布正则项：
$$
\mathcal L_{\beta}=\lambda_{\beta}\sum_{\mathbf r}[\log(T_{\mathrm{FG}}(\mathbf r))+\log(1-T_{\mathrm{FG}}(\mathbf r))]
$$
$\mathbf r$ 是光线，$T_{\mathrm{FG}}$ 是光线前景累积透射率。

## 4 Result

### 4.4 Ablation Studies

控制变量法还揭示了为什么我们的模型在真实的前向场景上的 TV 正则化比合成场景表现更好：真实场景的训练图像要少得多，而更强的正则化有助于我们的优化平滑地扩展到稀疏监督区域。
This ablation also sheds light on why our model performs better with higher TV regularization on the real forward-facing scenes compared to the synthetic scenes: the real scenes have many fewer training images, and the stronger regularizer helps our optimization extend smoothly to sparsely-supervised regions.

## 5 Discussion

我们的方法也非常简单明了，揭示了解决 3D 逆问题所必需的核心元素：可微分的正向模型、连续表示（在我们的例子中，通过三线性插值）和适当的正则化。
Our method is also strikingly straightforward, shedding light on the core elements that are necessary for solving 3D inverse problems: a differentiable forward model, a continuous representation (in our case, via trilinear interpolation), and appropriate regularization.

尽管我们使用一组固定的超参数报告每个数据集的所有结果，但没有最佳的 TV 权重先验设置。
Although we report all of our results for each dataset with a fixed set of hyper-parameters, there is no optimal a priori setting of the TV weight.

