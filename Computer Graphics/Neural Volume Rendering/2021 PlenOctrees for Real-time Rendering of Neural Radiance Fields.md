# PlenOctrees for Real-time Rendering of Neural Radiance Fields

## 0 Abstract

为了保留与视角相关的效果，例如镜面反射，我们通过闭式的球面基函数对外观进行分解。
In order to preserve view-dependent effects such as specularities, we factorize the appearance via closed-form spherical basis functions.

训练 NeRF 去预测一个球谐函数表示的辐射，移除了神经网络输入中的视角方向。
Train NeRFs to predict a spherical harmonic representation of radiance, removing the viewing direction as an input to the neural network.

PlenOctrees 可以直接优化以进一步减小重建损失。
PlenOctrees can be directly optimized to further minimize the reconstruction loss.

## 1 Introduction

由于极端的采样需求和昂贵的神经网络查询，渲染 NeRF 是非常缓慢的。
Due to the extreme sampling requirements and costly neural network queries, rendering a NeRF is agonizingly slow.

NeRF 渲染速度很慢，因为它需要对场景进行密集采样，其中每个采样点都需要神经网络推理。因为每次询问都和视角，采样点空间位置有关，很难简单地缓存所有视角的颜色值。
Naïve NeRF rendering is slow because it requires dense sampling of the scene, where every sample requires a neural network inference. Because these queries depend on the viewing direction as well as the spatial position, one cannot naively cache these color values for all viewing directions.

通过提前采样 NeRF，把结果放入一个视角相关的容器，也就是 PlenOctree 里面，实现了实时渲染。
Enable real-time rendering by pre-sampling the NeRF into a tabulated view-dependent volume which we refer to as a PlenOctree.

使用了一个稀疏的基于体素的八叉树，每个叶节点都存储了空间中某个点对于辐射进行建模所需的外观和密度值。
Use a sparse voxel-based octree where every leaf of the tree stores the appearance and density values required to model the radiance at a point in the volume.

提出用球谐函数来表示某个位置的 RGB 值。球谐函数是定义在球体表面的一组标准基函数。
Propose to represent the RGB values at a location with spherical harmonics (SH), a standard basis for functions defined on the surface of the sphere.

事实上改造 NeRF 用球谐函数显式地预测外观。训练了网络产生球谐函数的系数而不是原始的 RGB 值，使得预测结果可以直接存到 PlenOctree 的叶节点中。
In fact modify a NeRF network to predict appearances explicitly in terms of spherical harmonics. Train a network that produces coefficient for the SH functions instead of raw RGB values, so that the predicted values can later be directly stored within the leaves of the PlenOctree.

一旦结构形成了，PlenOctree 中的值可以被优化，因为渲染过程还是可微的。
Once the structure is created, the values stored in PlenOctree can be optimized because the rendering procedure remains differentiable.

## 2 Related Work

### Novel View Synthesis

用于朗伯和非朗伯场景的基于网格的方法。难以适应任意拓扑的复杂场景。
Mesh based methods for Lambertian and non-Lambertian scenes. Hard to fit a complex scene of arbitrary topology.

基于图像的渲染方法可以轻松捕获和照片般逼真的快速渲染。通常受限于视角，不允许轻松编辑底层场景。
Image-based rendering methods enable easy capture and photo-realistic and fast rendering. Often bounded in the viewing angle and do not allow easy editing of the underlying scene.

基于体积的表示，例如体素网格和多平面图像 (MPI)，由于其无拓扑性质，是网格表示的一种流行替代方案：因此，基于梯度的优化很简单，而渲染仍然可以是实时的。内存限制，限制可以捕获的最大分辨率。
Volume-based representations such as voxel grids and multi-plane images (MPIs) are a popular alternative to mesh representations due to their topology-free nature: gradient-based optimization is therefore straightforward, while rendering can still be real-time. Memory bound, limiting the maximum resolution that can be captured.

### Coordinate-based Neural Networks

这些网络表示了一个连续函数，可以在不增加内存占用的情况下以任意精细的分辨率进行采样。不幸的是，这种紧凑性是以牺牲计算效率为代价的，因为每个样本都必须由神经网络处理。结果，这种表示对于实时渲染通常很慢且不切实际。
These networks represent a continuous function that can be sampled at arbitrarily fine resolutions without increasing the memory footprint. Unfortunately, this compactness is achieved at the expense of computational efficiency as each sample must be processed by a neural network. As a result, these representations are often slow and impractical for real-time rendering.

### NeRF Acceleration

结合从类似场景的数据集中学习的先验。
Incorporating priors learned from a dataset of similar scenes.

稀疏体素网格允许渲染器在追踪光线时跳过空白区域，从而将渲染时间提高约 10 倍。
The sparse voxel grid allows the renderer to skip over empty regions when tracing a ray which improves the render time about 10 times.

## 3 Preliminaries

### 3.1 Neural Radiance Fields

## 4 Methods

颜色是通过对应光线方向上评估的球谐函数的权重和来计算的。
The color $c$ is calculated by summing the weighted spherical harmonic bases evaluated at the corresponding ray direction $(\theta,\phi)$。

### 4.1 NeRF-SH: NeRF with Spherical Harmonics

我们改造了 NeRF 网络 $f$ 去输出球谐函数的系数 $\mathbf k$。
We adapt the NeRF network $f$ to output spherical harmonics coefficient $\mathbf k$, rather than RGB values.
$$
f(\mathbf x)=(\mathbf k,\sigma)\quad\mathbf k=(k_{l}^{m})_{l:0\le l\le l_{\max}}^{m:-l\le m\le l}
$$
每个 $k_{l}^{m}\in\mathbb R^3$ 是 3 个 RGB 系数。即 $\mathbf k\text{.shape}=(l_{\max}^2,\ 3)$。
Each $k_{l}^{m}\in\mathbb R^3$ is a set of 3 coefficients corresponding to the RGB components.

$\mathbf x$ 处视角相关的颜色 $\mathbf c$ 可以通过查询在视角 $\mathbf d$ 处球谐函数。
The view-dependent colour $\mathbf c$ at a point $\mathbf x$ may be determined by querying the SH functions at the desired viewing angle $\mathbf d$.
$$
\mathbf c(\mathbf d;\mathbf k)=\mathrm{Sigmoid}\left(\sum_{l=0}^{l_{max}}\sum_{m=-l}^{l}k_{l}^{m}Y_{l}^{m}(\mathbf d)\right)\quad Y_{l}^{m}:\mathbb S^2\to\mathbb R
$$

我们还可以通过在随机方向上对 NeRF 进行采样并乘以球谐函数分量值以形成内积的蒙特卡罗估计，从而将经过训练的 NeRF 直接投影到球谐函数的每个点。（慢，损失大）
Note that we can also project a trained NeRF to spherical harmonics directly at each point by sampling NeRF at random directions and multiplying by the SH component values to form Monte Carlo estimates of the inner products.

#### Sparsity prior

在没有正则化的情况下，该模型可以在未观察到的区域中自由生成任意几何形状。虽然这不会直接恶化图像质量，但它会对我们的转换过程产生不利影响，因为额外的几何体占据了大量的体素空间。
Without any regularization, the model is free to generate arbitrary geometry in unobserved regions. While this does not directly worsen image quality, it would adversely impact our conversion process as the extra geometry occupies significant voxel space.

我们在训练 NeRF 的时候引入一个额外的稀疏先验。这个先验会使得 NeRF 在留空的和填充颜色都为可行解的时候选择留空。
We introduce an additional sparsity prior during NeRF training. This prior encourages NeRF to choose empty space when both space and solid colours are possible solutions.
$$
\mathcal L_{\text{sparsity}}=\frac{1}{K}\sum_{k=1}^{K}|1-\exp(-\lambda\sigma_k)|
$$
$\{\sigma_{k}\}$ 是 $K$ 个在 bounding box 均匀采样点处的密度值，$\lambda$ 是超参。
$$
\sigma_k\to0,|1-\exp(-\lambda\sigma_k)|\to0
$$

最后的损失函数是：
$$
\beta_{\text{sparsity}}\mathcal L_{\text{sparsity}}+\mathcal L_{\text{RGB}}
$$
$\beta_{\text{sparsity}}$ 是超参。

### 4.2 PlenOctree: Octree-based Radiance Fields

PlenOctree 存储的是密度和球谐函数的系数，在叶节点来建模视角相关的物体外观。
A PlenOctree stores density and SH coefficients modeling view-dependent appearance at each leaf.

#### Rendering

要渲染 PlenOctree，对于每根光线，计算八叉树中和光线相交的体素。
To render the PlenOctree, for each ray, we first determine ray-voxel intersections in the octree structure.

提前停止，当透过率小于某一个值时候可以停止向前。
We further accelerate this rendering process by applying early-stopping when the ray has accumulated transmittance $T_i$ less than $\gamma=0.01$.

#### Conversion from NeRF-SH

总体来说，我们在网格上求网络的值，只保留密度值，然后用阈值筛选体素。最后在剩下的体素中随机采样点，求平均值得到球谐函数的系数并存入八叉树叶节点中。
At a high level, we evaluate the network on a grid, retaining only density values, then filter the voxels via thresholding. Finally we sample random points within each remaining voxel and average them to obtain SH coefficients to store in the octree leaves.

<blockquote style="border-left: 5px solid #42b983; border-radius: 3px 0 0 3px; padding: 10px 15px; background-color: rgba(66, 185, 131, 0.1)">
    Evaluation
</blockquote>
首先在均匀的 3D 网格上对神经网络求值，网格大小会自动伸缩来贴和场景大小。
We first evaluate the NeRF-SH model to obtain values on a uniformly spaced 3D grid. The grid is automatically scaled to tightly fit the scene content.

<blockquote style="border-left: 5px solid #42b983; border-radius: 3px 0 0 3px; padding: 10px 15px; background-color: rgba(66, 185, 131, 0.1)">
    Filtering
</blockquote>
下一步我们筛选网格来获取足够呈现场景，以网格点为中心的稀疏体素。
Next, we filter this grid to obtain a sparse set of voxels centred at the grid points sufficient for representing the scene.

我们对所有的训练视角用这个体素网格渲染 alpha map，追踪每个体素格子的最大光线权重 $1-\exp(-\sigma_i\delta_i)$。然后把权重小于阈值 $\tau_w$ 的格子筛选除去。
We render alpha maps for all the training views using this voxel grid, keeping track of the maximum ray weight $1-\exp(-\sigma_i\delta_i)$ at each voxel. We then eliminate the voxels whose weights are lower than a threshold $\tau_w$.

<blockquote style="border-left: 5px solid #42b983; border-radius: 3px 0 0 3px; padding: 10px 15px; background-color: rgba(66, 185, 131, 0.1)">
    Sampling
</blockquote>
最后我们在剩下的体素格子中随机采样 256 个点，并且取八叉树相关的叶节点的平均值作为最终的值来抗锯齿。
Finally, we sample a set of random points 256 in each remaining voxel and set the associated leaf of the octree to the mean of these values to reduce aliasing.

### 4.3 PlenOctree Optimization

## 5 Results

### 5.1 Experimental Setup

### 5.2 Quality Evaluation

$l_{\max}=3,4\ (16,25)$ of spherical harmonic components.

$256^3,512^3$ grid size.

with/without auto box scaling.

Thresholds $\tau_w$

early-stopping, maybe overfitting.

### 5.3 Speed Trade-off Analysis

### 5.4 Indirect Acceleration of NeRF Training

事实上，我们发现在微调期间获得的图像质量改进通常比相同时间继续训练 NeRF-SH 要大。
Indeed, we have found that the image quality improvements gained during fine-tuning can often be greater than continuing to train the NeRF-SH an equivalent amount of time.

## 6 Discussion

八叉树表示比原来 NeRF 紧凑的表示大得多。
The octree representation is much larger than the compact representation of the original NeRF model and has a larger memory footprint.

## B Technical Details

### B.1 Spherical Basis Functions: SH and SG

$$
Y_{l}^{m}(\theta,\phi)=\sqrt{\frac{(2l+1)(l-|m|)!}{4\pi(l+|m|)!}}P_{l}^{|m|}(\cos\theta)\exp(im\phi)\\
l\in\{0,\dots,l_{max}\},m\in\{-l,\dots,l\},\theta\in[0,\pi],\phi\in[0,2\pi]\\
P_{l}^{m}(\cos\theta)=\sin^m\theta\frac{d^{m}P_{l}(\cos\theta)}{dx^{m}}\\
P_{l}(\cos\theta)=\frac{1}{2^l}\sum_{k=0}^{K}(-1)^{k}\frac{(2l-2k)!}{k!(l-k)!(l-2k)!}\cos^{l-2k}\theta\quad K=\lfloor l/2\rfloor
$$

### B.3 Analytic Derivatives of PlenOctree Rendering

#### B.3.1 Definitions

The NeRF rendering model considers a ray divided into $N$ consecutive segments with endpoint $\{t_i\}_{i=0}^{N}$, where $t_0$ and $t_N$ are the near and far bounds. The segments have constant densities $\{\sigma_i\}_{i=0}^{N-1}$ where each $\sigma_i\ge0$. If we shine a light of intensity $1$ at $t_i$, then at the camera position $t_0$, the light intensity is given by:
$$
T_i\gets\exp(\sum_{j=0}^{i-1}-\delta_j\sigma_j)
$$
where $\delta_j\gets t_{j+1}-t_j$ are segment lengths.

Let $\{c_i\}_{i=1}^{N-1}\in[0,1]^3$ be the colour associated within segments, and $c_N$ be the background light intensity. 

#### B.3.2 Derivation of the Derivatives

The rendered ray colour $\hat{C}(\sigma,c)$ is:
$$
\begin{align*}
\hat{C}(\sigma,c)&=T_Nc_N+\sum_{i=0}^{N-1}T_i(1-\exp(-\sigma_i\delta_i))c_i\\
&=\sum_{i=0}^{N}w_ic_i
\end{align*}
$$
where $w_i=T_i(1-\exp(-\sigma_i\delta_i))=T_i-T_{i+1}$ and $w_N=T_N$.

##### Colour derivative

Since the rendered colour are a convex combination of the segment colours, it is clear that:
$$
\frac{\partial\hat{C}(\sigma,c)}{\partial c_i}=w_i
$$
Handling spherical harmonics colours is straightforward by applying the chain rule, noting that the SH basis function values are constant across the ray.

##### Density derivative

Expand the equation 
$$
\frac{\partial\hat{C}(\sigma,c)}{\partial\sigma_i}&=\frac{\partial T_N}{\partial\sigma_i}c_N+\sum_{k=0}^{N-1}(\frac{\partial T_k}{\partial\sigma_i}-\frac{\partial T_{k+1}}{\partial\sigma_i})c_k\\
$$
where:
$$
\begin{align*}
\frac{\partial T_k}{\partial\sigma_i}&=\frac{\partial\exp(\sum_{j=0}^{k-1}-\delta_j\sigma_j)}{\partial\sigma_i}\\
&=-\delta_i\exp(\sum_{j=0}^{k-1}-\delta_j\sigma_j)\cdot I(k>i)\\
&=-\delta_iT_k\cdot I(k>i)
\end{align*}
$$
where $I(\cdot)$ is the indicating function. 

Then:
$$
\begin{align*}
\frac{\partial\hat{C}(\sigma,c)}{\partial\sigma_i}&=\delta_i\left(-T_Nc_N-\sum_{k=i+1}^{N-1}(T_k-T_{k+1})c_k\right)\\
&=-\delta_i\left(T_Nc_N+\sum_{k=i+1}^{N-1}w_kc_k\right)
\end{align*}
$$

