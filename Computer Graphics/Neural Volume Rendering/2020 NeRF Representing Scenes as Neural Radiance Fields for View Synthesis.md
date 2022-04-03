# NeRF: 利用神经辐射场表示场景进行视角合成 NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

> NeRF 的网络学习的是空间中各点的体密度和颜色。特别的，体密度是一个空间中连续的数量场，可以用全连接层进行学习，用梯度下降法进行优化。

## 0 Abstract

使用较少的图片输入，通过优化连续体场景函数，在复杂场景中合成新视角。
Synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views.

全连接层
Fully-connected deep network.

- 输入：5D 坐标，其中 3D 表示位置，2D 表示视线方向。
  *Input: 5D coordinate (spatial location $(x,y,z)$ and viewing direction $(\theta,\phi)$).*
- 输出：体密度和空间中某处和视角有关的辐射度。
  *Output: the volume density and view-dependent emitted radiance at that spatial location.*

## 1 Introduction

直接优化 5D 连续场景表示函数的参数来减少和输入图像间的误差。
Directly optimizing parameters of a continuous 5D scene representation to minimize the error of rendering a set of captured images.

各点密度的作用为一个可微的不透明度，用来控制穿过 $(x,y,z)$ 点光线的辐射累积度。
A density at each point which acts like a differential opacity controlling how much radiance is accumulated by a ray passing through $(x, y, z)$.

Rendering
1. 使相机光线前进去生成 3D 样本点集。
   March camera rays through the scene to generate a sampled set of 3D points.
2. 把这些点和对应的 2D 视角方向作为神经网络的输入，把颜色和密度作为输出。
   Use those points and their corresponding 2D viewing directions as input to the neural network to produce an output set of colors and densities.
3. 用经典的体渲染技术去累计这些颜色和密度到 2D 图像中。
   Use classical volume rendering techniques to accumulate those colors and densities into a 2D image.

用梯度下降法去优化模型，模型误差就是输入图像和从对应视角渲染出的图片的误差和。
Using gradient descent to optimize this model by minimizing the error between each observed image and the corresponding views rendered from our representation.

我们发现，优化用神经辐射场表示的复杂场景基本实现不能收敛到足够高分辨率的表示，并且在每条相机光线所需的样本数量方面效率低下。
We find that the basic implementation of optimizing a neural radiance field representation for a complex scene does not converge to a suffciently high-resolution representation and is inefficient in the required number of samples per camera ray.

把输入的 5D 坐标用*位置编码*变换了一下，使得 MLP 得以表现高频函数。
Transforming input 5D coordinates with a *positional encoding* enables the MLP to represent higher frequency functions.

有层次的采样过程以减少采样高频场景时所需的采样次数。
A hierarchical sampling procedure to reduce the number of queries required to adequately sample this high-frequency scene representation.

## 2 Related Work

CV 最近一个有前途的方向是在 MLP 的权重中编码对象和场景，该 MLP 直接从 **3D 空间位置映射到形状的隐式表示**，例如该位置的符号距离。
A promising recent direction in computer vision is encoding objects and scenes in the weights of an MLP that directly **maps from a 3D spatial location to an implicit representation of the shape**, such as the signed distance at that location.

## 3 Neural Radiance Field Scene Representation

Input: 3D location $\mathbf x = (x, y, z)$ and 2D viewing direction $(\theta, \phi)$ or 3D Cartesian unit vector $\mathbf d$.

Output: an emitted color $\mathbf c = (r, g, b)$ and volume density $\sigma$.

密度只与位置有关，颜色与位置和视角方向有关。
Restricting the network to predict the volume density as a function of only the location $\mathbf x$, while allowing the RGB color $\mathbf c$ to be predicted as a function of both location and viewing direction.
$$
F_{\Theta}:(\mathbf x, \mathbf d)\to(\mathbf c,\sigma)\\
   \sigma=\sigma(\mathbf x)\\
   \mathbf c=\mathbf c(\mathbf x, \mathbf d)
$$

把 $\mathbf x$ 用一个 8 层、每层 256 维的全连接层进行处理，输出 $\sigma$ 和 256 维的特征向量。然后把特征向量和相机视角方向合并，传到 128 维用 ReLU 的全连接层，最后输出和视角有关的 RGB 值。
Processes $\mathbf x$ with *8 fully-connected layers (using ReLU activations and 256 channels per layer)*, and *outputs $\sigma$ and a 256-dimensional feature vector*. This feature vector is then concatenated with the camera ray's viewing direction and passed to *one additional fully-connected layer (using a ReLU activation and 128 channels)* that *output the view-dependent RGB color*.

## 4 Volume Rendering with Radiance Fields

体密度 $\sigma(\mathbf x)$ 可以看作光线打到位于 $\mathbf x$ 的无限小微粒的可微概率。在远近平面的限制下，相机光线 $\mathbf r$ 颜色的**期望**是
The volume density $\sigma(\mathbf x)$ can be interpreted as the differential probability of a ray terminating at an infinitesimal particle at location $\mathbf x$. The **expected** color $C(\mathbf r)$ of camera ray $\mathbf r(t) = \mathbf o + t\mathbf d$ with near and far bounds $t_n$ and $t_f$ is:
$$
C(\mathbf r)=\int^{t_f}_{t_n}T(t)\sigma(\mathbf r(t))\mathbf c(\mathbf r(t),\mathbf d)\mathrm dt\\
T(t)=\exp(-\int^{t}_{t_n}{\sigma(\mathbf r(s))}\mathrm ds)\tag{1}
$$
$T(t)$ 是光线从 $t_n$ 到 $t$ 的累计透光度，即不打到任何一个微粒的概率。
The function $T(t)$ denotes the accumulated transmittance along the ray from $t_n$ to $t_f$. Or the probability that the ray travels from $t_n$ to $t_f$ without hitting any other particle.

---

假设光线穿入了 volume 中，$T(t)$ 表示 $t$ 位置前不被吸收的概率，体密度 $\sigma(t)$ 可以看作光线打到位于 $t$ 的无限小微粒的可微概率。
$$
\begin{align*}
T(t+dt)&=T(t)(1-\sigma(t)dt)\\
T(t+dt)-T(t)&=-T(t)\sigma(t)dt\\
T'(t)dt&=T(t)\sigma(t)dt\\
T(t)&=\exp\left(-\int\sigma(t)dt\right)
\end{align*}
$$

---

用数值积分。通常用于渲染的确定间隔的离散体素网格将限制我们表示的分辨率，因为 MLP 只会在固定的离散位置上被查询。
Numerically estimate this continuous integral using quadrature. Deterministic quadrature, which is typically used for rendering discretized voxel grids, would effectively limit our representation's resolution because the MLP would only be queried at a fixed discrete set of locations.

用分层采样方法把 $[t_n, t_f]$ 分成 $N$ 等分，然后均匀地从每一个小区间中采样。（足够均匀也足够随机）。
Use a stratified sampling approach where we partition $[t_n, t_f]$ into $N$ evenly-spaced bins and then draw one sample uniformly at random from within each bin:
$$
t_i\sim\mathcal U\left[t_n+\frac{i-1}{N}(t_f-t_n),t_n+\frac{i}{N}(t_f-t_n)\right]\tag{2}
$$
Use these samples to estimate $\hat{C}(\mathbf r)$:
$$
\hat{C}(\mathbf r)=\sum_{i=1}^{N}T_i\sigma_i\delta_i\mathbf c_i=\sum_{i=1}^{N}T_i(1-\exp(-\sigma_i\delta_i))\mathbf c_i\\
T_i=\exp(-\sum_{j=1}^{i-1}\sigma_j\delta_j)\tag{3}
$$
这里用到了等价无穷小 $1-e^{-x}\sim x$，在图形学里面叫 $\alpha$ 混合。
And $\delta_i=t_{i+1}-t_i$ is the distance between adjacent samples. This function reduces to **traditional alpha compositing** with alpha values $\alpha_i=1−exp(−\sigma_i\delta_i)$.

## 5 Optimizing a Neural Radiance Field

然而，我们观察到这些组件不足以实现最先进的质量。我们引入了两项改进以支持表示高分辨率复杂场景。
However, we observe that these components are not sufficient for achieving state-of-the-art quality. We introduce two improvements to enable representing high-resolution complex scenes.

### 5.1 Positional encoding

深度网络偏向于学习低频函数。在传入网络之前，用高频函数把输入映射到高维空间，可以使得网络更好地拟合包含高频信息的数据
Deep networks are biased towards learning lower frequency functions. Solution: mapping the inputs to a higher dimensional space using high frequency functions before passing them to the network enables better fitting of data that contains high frequency variation.

$F_{\Theta}=F'_{\Theta}\circ\gamma,\gamma:\mathbb R\to\mathbb R^{2L}$
$$
\gamma(p)=(\sin(2^0\pi p), \cos(2^0\pi p), \dots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))\tag{4}
$$
$\gamma$ 会作用于 $\mathbf x$ 和 $\mathbf d$ 的每一个分量上。
This function $\gamma(\cdot)$ is applied separately to each of the three coordinate values in $\mathbf x$ (which are normalized to lie in $[−1,1]$) and to the three components of the Cartesian viewing direction unit vector $\mathbf d$ (which by construction lie in $[−1,1]$). 

### 5.2 Hierarchical volume sampling

那些空的和被遮挡的地方，也就是对成像没有用的地方，也会被采样到。
Free space and occluded regions that do not contribute to the rendered image are still sampled repeatedly.

2 个网络：粗糙的和精细的

粗糙的：首先用分层采样法（比随机采样方差更小）在 $N_c$ 个点上采样，然后训练粗糙网络。根据输出，然后我们在与体密度高的区域再次采样。
Coarse: First sample a set of $N_c$ locations using **stratified sampling**, and evaluate the “coarse” network at these locations as described in Equations (2) and (3). Given the output of this "coarse" network, we then produce a more informed sampling of points along each ray where samples are biased towards the relevant parts of the volume. Rewrite the alpha composited color from the coarse network $\hat{C}_c(\mathbf r)$ as a weighted sum of all sampled colors $c_i$ along the ray
$$
\hat{C}_c(\mathbf r)=\sum_{i=1}^{N_c}w_ic_i\qquad w_i=T_i(1-\exp(-\sigma_i\delta_i))\tag{5}
$$
归一化这些权重，然后得到一个在光线上分段的 PDF 函数，即比较粗糙的密度分布。
Normalizing these weights as $\hat{w}_i=w_i/\sum_{j=1}^{N_c}w_j$ produces a **piecewise-constant PDF** along the ray. 

精细的：然后用逆采样从这个分布里面采集 $N_f$ 个点，把全部 $N_c+N_f$ 个点放入精细的网络中一起训练。
Fine: Second sample a set of $N_f$ locations from this distribution using **inverse transform sampling**, evaluate the “fine” network at the union of the first and second set of samples, and compute the final rendered color of the ray $\hat{C}_f(\mathbf r)$ and using all $N_c+N_f$ samples. 

### 5.3 Implementation details

这仅需要捕获的场景 RGB 图像数据集、相应的相机位姿和内在参数以及场景边界
This requires only a dataset of captured RGB images of the scene, the corresponding camera poses and intrinsic parameters, and scene bounds.

在每次优化迭代时，从数据集中所有像素的集合中随机采样一批相机射线，然后按照分层采样从粗网络中查询 $N_c$ 个样本，从精细网络中查询 $N_c+N_f$ 个样本。
At each optimization iteration, randomly sampling a batch of camera rays from the set of all pixels in the dataset, and then follow the hierarchical sampling to query $N_c$ samples from the coarse network and $N_c+N_f$ samples from the fine network.

Loss

$$
\mathcal L=\sum_{\mathbf r\in\mathcal R}\left[\|\hat{C}_c(\mathbf r)-C_c(\mathbf r)\|_2^2+\|\hat{C}_f(\mathbf r)-C_c(\mathbf r)\|_2^2\right]\tag{6}
$$
where $\mathcal R$ is the set of rays in each batch.

Batch size = 4096, $N_c=64$, $N_f=128$, learning rate = $5\times10^{-4}\to5\times10^{-5}$ exponentially decay, Adam ($\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-7}$).

## 7 Conclusion

诸如体素网格和网格之类的采样表示允许对渲染视图的预期质量和故障模式进行推理，但是当我们在深度神经网络的权重中编码场景时，如何分析这些问题尚不清楚。
Sampled representations such as voxel grids and meshes admit reasoning about the expected quality of rendered views and failure modes, but it is unclear how to analyse these issues when we encode scenes in the weights of a deep neural network.







