# Literature Review of Grid Scene Representation

## 2021 PlenOctrees for Real-time Rendering of Neural Radiance Fields

### 1 Motivation

Accelerating neural rendering in real-time. For NeRF, rendering an $800 * 800$ picture takes 30 seconds.

### 2 Methods

#### NeRF-SH:

Adapted the NeRF network to output spherical harmonics coefficient $\mathbf k$:
$$
f(\mathbf x)=(\mathbf k,\sigma)\quad\mathbf k=(k_{l}^{m})_{l:0\le l\le l_{\max}}^{m:-l\le m\le l}
$$
By querying the SH function at the desired viewing direction $\mathbf d$ , we can get the colour:
$$
\mathbf c(\mathbf d;\mathbf k)=\mathrm{Sigmoid}\left(\sum_{l=0}^{l_{max}}\sum_{m=-l}^{l}k_{l}^{m}Y_{l}^{m}(\mathbf d)\right)
$$

#### Sparsity prior:

Without any regularization, the model is free to generate arbitrary geometry in unobserved regions. While this does not directly worsen image quality, it would adversely impact our conversion process as the extra geometry occupies significant voxel space. Then they proposed **a sparsity prior**:
$$
\mathcal L_{\text{sparsity}}=\frac{1}{K}\sum_{k=1}^{K}|1-\exp(-\lambda\sigma_k)|
$$

#### PlenOctree:

At a high level, we evaluate the network on a grid, retaining only density values, then filter the voxels via thresholding. Finally, we sample random points within each remaining voxel and average them to obtain SH coefficients to store in the octree leaves.

### 3 Related work

2020 Neural Sparse Voxel Fields (NSVF)

### 4 Results

Outperform all the work before, achieve real time rendering.

### 5 Discussion

Querying information from neural network is time-consuming while from octree is quite fast. Then we can store them into the octree. And since rendering is differentiable, we can also optimize the octree.

### 6 Future work

Fast training (Plenoxel).

## 2021 Plenoxels: Radiance Fields without Neural Networks

### 1 Motivation

Acceleration of training a neural representing scene.

### 2 Methods

The data structure they used is a **3D dense grid**. But they only take out the grids that are occupied and store the pointers into an index array. Then they use trilinear interpolation for querying density and spherical harmonics coefficients.

First, they use a coarse grid. They after several steps, they optimize, prune unnecessary voxels, split the grid to higher resolution.

The optimizer they use is MSE re-render loss. Additionally, they use total variation (TV) regularization. For faster iteration, they use a stochastic sample of the rays $\mathcal R$ to evaluate the MSE term and a stochastic sample of the voxels $\mathcal V$ to evaluate the TV term in each optimization step. For different scenes, they also use Cauchy loss and beta regularization.
$$
\begin{align*}
\mathcal L&=\mathcal L_{\mathrm{reconstruction}}+\lambda_{\mathrm{TV}}\mathcal L_{\mathrm{TV}}\\
\mathcal L_{\mathrm{reconstruction}}&=\frac{1}{|\mathcal R|}\sum_{\mathbf r\in\mathcal R}\|C(\mathbf r)-\hat{C}(\mathbf r)\|^2_2\\
\mathcal L_{\mathrm{TV}}&=\frac{1}{|\mathcal V|}\sum_{v\in\mathcal V,d\in[D]}\sqrt{\Delta_x^2(\mathbf v, d)+\Delta_y^2(\mathbf v, d)+\Delta_z^2(\mathbf v, d)}
\end{align*}
$$

### 3 Related work

Use parallelization (GUPs and TPUs), meta learning.

2021 PlenOctrees for Real-time Rendering of Neural Radiance Fields

### 4 Results

Outperform all the work before, achieve training time within 10 minutes.

### 5 Discussion

The core elements that are necessary for solving 3D inverse problems: **a differentiable forward model**, **a continuous representation** (in our case, via trilinear interpolation), and **appropriate regularization**.

### 6 Future work

Find a more uniform regularizer (TV regularizer) for each scene.

2022 Instant Neural Graphics Primitives with a Multiresolution Hash Encoding

2022 TensoRF: Tensorial Radiance Fields

## 2022 Instant Neural Graphics Primitives with a Multiresolution Hash Encoding

### 1 Motivation

Acceleration of any coordinate-based neural representing scene.

### 2 Methods

The author split the mapping into two parts. The encoding stage $\mathbf y = \mathrm{enc}(x;\theta)$ and the approximating/fitting stage $m(\mathbf y;\Phi)$ where $\theta$ and $\Phi$ are the corresponding parameters. In this paper they proposed the multi-resolution hash encoding to improves the approximation quality and training speed across a wide range of applications without incurring a notable performance overhead.

The trainable encoding parameters $\theta$ are arranged into $L$ levels, each containing up to $T$ feature vectors with dimensionality $F$. The number of trainable encoding parameters $\theta$ is therefore $O(T)$ and bounded by $T\cdot L\cdot F$.

Each level is independent and stores feature vectors at the vertices of a grid, the resolution of which is chosen to be a geometric progression between the coarsest and finest resolutions $[N_{\min}, N_{\max}]$:
$$
N_l\gets\lfloor N_\min\cdot b^l\rfloor\\
b\gets\exp\left(\frac{\ln N_{\max}-\ln N_{\min}}{L-1}\right)
$$
Consider a single level $l$. The input coordinate $\mathbf x\in\mathbb R^d$ is scaled by that level's grid resolution $N_l$ before rounding down and up $\lfloor\mathbf x_l\rfloor\gets\lfloor\mathbf x\cdot N_{l}\rfloor,\lceil\mathbf x_l\rceil\gets\lceil\mathbf x\cdot N_l\rceil$.

For coarse levels where a dense grid requires fewer than $T$ parameters, i.e. $(N_l+1)^d\le T$, this mapping is $1:1$. At finer levels, we use a hash function $h:\mathbb Z^d\mapsto\mathbb Z_T$ to index into the array, effectively treating it as a hash table, although there is no explicit collision handling.
$$
y = 
$$


### 3 Related work

2021 Plenoxels: Radiance Fields without Neural Networks

### 4 Results

### 5 Discussion

Impressive and brilliant work! I was fully astonished at the first glance of this paper and its result.

### 6 Future work



## 2022 TensoRF: Tensorial Radiance Fields

### 1 Motivation

Training a NeRF, a pure MLP scene representation, is time-consuming. Training a Plenoxel, a pure grid representation, is memory-consuming. TensoRF pursues an approach that is both efficient in training time and compact in memory footprint, and at the same time achieves state-of-the-art rendering quality.

### 2 Methods

Factorize the tensor of radiance fields into multiple low rank tensor components.

#### CP Decomposition



### 3 Related work

...

### 4 Results

...

### 5 Discussion

...

### 6 Future work

...

## 2022 PREF: Phasorial Embedding Fields for Compact Neural Representations

### 0 My Question

> Specifically, we first perform 1D FFT along one of the axes $u$ to obtain a intermediate map $\mathbf{P}_v[x^*,v]$, with $x^*=\{0,1,\dots,M-1\}$.

对谁做一维 Fourier 变换？$\mathbf{P}[u,v]$ 吗？

> $$
\begin{align*}
\mathbf{P}_{v}(x,v)&=\langle\mathbf{P}_v[x^*,v]\rangle\\
f(x,y)&\approx\sum_{v=0}^{D}e^{j2\pi v\frac{y}{N}}\mathbf{P}_{v}(x,v)=\tilde{T}_*(\mathbf{P})
\end{align*}
$$

直接先 $x$ 后 $y$ 的 FFT 真的合理吗？(Maybe 离散形式还挺合理)

> Recall the TV loss can be computed as $\|\nabla_{x}f(x,y)\|_2+\|\nabla_{x}f(x,y)\|_2$.

我没有找到任何有关 ATV 长这样的文章。别的文章都是：$\|\nabla_{x}f\|_1+\|\nabla_{y}f\|_1$。二范数和一范数差远了。

### 1 Motivation

...

### 2 Methods

...

### 3 Related work

...

### 4 Results

...

### 5 Discussion

...

### 6 Future work

...
