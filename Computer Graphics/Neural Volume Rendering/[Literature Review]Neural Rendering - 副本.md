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


## 2021 NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination

### 1 Motivation

Recovering the **shape** and **spatially-varying reflectance** of an object from posed multi-view images of the object **illuminated by one unknown lighting condition**.

Convert volume rendering to surface rendering using MLP.

### 2 Methods

#### Shape:

The shape is the depth in NeRF. And the surface point is the expectation of the ray traveling distance:
$$
\mathbf x_{\text{surf}}=\mathbf o+\left(\int_{0}^{\infty}T(t)\sigma(\mathbf r(t))tdt\right)\mathbf d\\
T(t)=\exp({-\int_{0}^{t}\sigma(\mathbf r(s))ds})
$$
Since we have known the position of surface, by taking the gradient of the volume density, we can get the surface normal. But this normal is not smooth, so we use an MLP to repair it.



### 3 Related work



### 4 Results

### 5 Discussion

### 6 Future work
