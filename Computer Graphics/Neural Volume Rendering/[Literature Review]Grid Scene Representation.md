## 2021 PlenOctrees for Real-time Rendering of Neural Radiance Fields

### 1 Motivation

Accelerating neural rendering in real-time. For NeRF, rendering an 800 * 800 picture takes 30 seconds.

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

Without any regularization, the model is free to generate arbitrary geometry in unobserved regions. While this does not directly worsen image quality, it would adversely impact our conversion process as the extra geometry occupies significant voxel space. Then they proposed a sparsity prior:
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

Accelerating training a neural representing scene.

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

## 2022 Instant Neural Graphics Primitives with a Multiresolution Hash Encoding



## 2022 TensoRF: Tensorial Radiance Fields

## 2022 PREF: Phasorial Embedding Fields for Compact Neural Representations

## 0 My Question

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
