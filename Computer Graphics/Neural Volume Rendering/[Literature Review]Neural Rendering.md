# Literature Review of Neural Rendering

When reading, ask yourself: 

1. What do the author(s) want to know (motivation)?
2. What did they do (methods)?
3. Why was it done that way (related work)?
4. What do the results show (figures and data tables)?
5. How did the author(s) interpret the results (discussion)?
6. What should be done next?

## 2020 NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

### 1 What do the author(s) want to know (motivation)?

Static 3D scene reconstruction and novel view synthesis.

### 2 What did they do (methods)?

Representing a continuous scene as an **MLP network** $F_{\Theta}:(\mathbf x,\mathbf d)\mapsto(\mathbf c,\sigma)$. The input is a 3D spatial position $\mathbf x=(x,y,z)$ and 2D viewing direction $(\theta,\phi)$. The output is a view-dependent colour $\mathbf c$ and a volume density $\sigma$.

For rendering, they use **volume rendering equation**:
$$
C(\mathbf r)=\int_{t_n}^{t_f}T(t)\sigma(\mathbf r(t))\mathbf c(\mathbf r(t),\mathbf d)\mathrm dt
$$
where $\mathbf r(t)=\mathbf o+t\mathbf d$ is the camera ray, $t_n$ and $t_f$ are near and far bounds, $C(\mathbf r)$ is *the expected colour*, $T(t)$ is the transmittance along the ray from $t_n$ to $t$. For sampling, they used jittered stratified sampling.

These components are insufficient for achieving SOTA, there for they use **positional encoding** for input and **hierarchical volume sampling** for rendering.

### 3 Why was it done that way (related work)?

2019 Neural Volumes Learning Dynamic Renderable Volumes from Images (Neural Volume)
2019 DeepSDF Learning Continuous Signed Distance Functions for Shape Representation (DeepSDF)

### 4 What do the results show (figures and data tables)?

Outperform all the work before, but too slow.

### 5 How did the author(s) interpret the results (discussion)?

Representing the static scene is used before, but proper number of input images, positional encoding, view-dependent color are important. Others are less important.

### 6 What should be done next?

A graphics pipeline based on real world imagery. Make the neural network represented scene interpretable.

## 2021 PlenOctrees for Real-time Rendering of Neural Radiance Fields

### 1 What do the author(s) want to know (motivation)?

Accelerating neural rendering in real-time. For NeRF, rendering an 800 * 800 picture takes 30 seconds.

### 2 What did they do (methods)?

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

### 3 Why was it done that way (related work)?

2020 Neural Sparse Voxel Fields (NSVF)

### 4 What do the results show (figures and data tables)?

Outperform all the work before, achieve real time rendering.

### 5 How did the author(s) interpret the results (discussion)?

Querying information from neural network is time-consuming while from octree is quite fast. Then we can store them into the octree. And since rendering is differentiable, we can also optimize the octree.

### 6 What should be done next?

Real time training (Plenoxel).

## 2021 Plenoxels: Radiance Fields without Neural Networks

### 1 What do the author(s) want to know (motivation)?

Accelerating training a neural representing scene.

### 2 What did they do (methods)?

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

### 3 Why was it done that way (related work)?

Use parallelization (GUPs and TPUs), meta learning.

2021 PlenOctrees for Real-time Rendering of Neural Radiance Fields

### 4 What do the results show (figures and data tables)?

Outperform all the work before, achieve training time within 10 minutes.

### 5 How did the author(s) interpret the results (discussion)?

The core elements that are necessary for solving 3D inverse problems: **a differentiable forward model**, **a continuous representation** (in our case, via trilinear interpolation), and **appropriate regularization**.

### 6 What should be done next?

Find a more uniform regularizer (TV regularizer) for each scene.

## 2022 Instant Neural Graphics Primitives with a Multiresolution Hash Encoding



## 2022 TensoRF: Tensorial Radiance Fields



## 2021 NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination

### 1 What do the author(s) want to know (motivation)?

Recovering the **shape** and **spatially-varying reflectance** of an object from posed multi-view images of the object **illuminated by one unknown lighting condition**.

Convert volume rendering to surface rendering using MLP.

### 2 What did they do (methods)?

#### Shape:

The shape is the depth in NeRF. And the surface point is the expectation of the ray traveling distance:
$$
\mathbf x_{\text{surf}}=\mathbf o+\left(\int_{0}^{\infty}T(t)\sigma(\mathbf r(t))tdt\right)\mathbf d\\
T(t)=\exp({-\int_{0}^{t}\sigma(\mathbf r(s))ds})
$$
Since we have known the position of surface, by taking the gradient of the volume density, we can get the surface normal. But this normal is not smooth, so we use an MLP to repair it.



### 3 Why was it done that way (related work)?



### 4 What do the results show (figures and data tables)?

### 5 How did the author(s) interpret the results (discussion)?

### 6 What should be done next?
