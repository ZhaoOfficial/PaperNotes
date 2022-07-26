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
where $\mathbf r(t)=\mathbf o+t\mathbf d$ is the camera ray, $t_n$ and $t_f$ are near and far bounds, $C(\mathbf r)$ is *the expected colour*, $T(t)$ is the transmittance along the ray from $t_n$ to $t$. When discretize this formula, they also used alpha composite. For sampling, they used jittered stratified sampling.

These components are insufficient for achieving SOTA, there for they use **positional encoding** for input and **hierarchical volume sampling** for two stage sampling.

### 3 Why was it done that way (related work)?

2019 Neural Volumes Learning Dynamic Renderable Volumes from Images (Neural Volume)
2019 DeepSDF Learning Continuous Signed Distance Functions for Shape Representation (DeepSDF)

### 4 What do the results show (figures and data tables)?

Outperform all the work before, but too slow.

### 5 How did the author(s) interpret the results (discussion)?

First it represents the static scene as a pure MLP, any spatial property queries are converted to network queries. Besides, proper number of input images, positional encoding, view-dependent color are also important. Others are less important.

### 6 What should be done next?

A graphics pipeline based on real world imagery. Make the neural network represented scene interpretable.


## 2021 NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination

### 1 Motivation

Recovering the **shape** and **spatially-varying reflectance** of an object from posed multi-view images of the object **illuminated by one unknown lighting condition**.

Convert volume rendering to surface rendering using MLP.

### 2 Methods

#### Shape

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

## 2021 Ref-NeRF Structured View-Dependent Appearance for Neural Radiance Fields

### 1 Motivation

NeRF's renderings of specular highlights reveals spurious glossy artifacts that fade in and out between rendered views, rather than smoothly moving across surfaces in a physically-plausible manner.

First, NeRF's parameterization of the outgoing radiance at each point as a function of the viewing direction is poorly-suited for interpolation.

Second, NeRF tends to "fake" specular reflections using isotropic emitters inside the object instead of view-dependent radiance emitted by points at the surface, resulting in objects with semitransparent or "foggy" shells.

### 2 Methods

#### Reflection Radiance

Re-parameterize outgoing radiance as a function of the reflection of the view direction about the local normal vector:
$$
\hat{\omega}_{r}=2(\hat{\omega}_o\cdot\hat{\mathbf{n}})\hat{\mathbf{n}}-\hat{\omega}_o
$$
where $\hat{\omega}_o=-\mathbf{d}$ is incident direction, $\hat{\mathbf{n}}$ is normal vector, $\hat{\omega}_{r}$ is reflection direction.

#### Integrated Directional Encoding

A way to deal with the specular highlight. Encode the directions with a set of spherical harmonics $\{Y_l^m\}$. Encodes the distribution of reflection directions using the expected value of a set of spherical harmonics under this vMF distribution:
$$
\mathrm{IDE}(\hat{\omega}_r,\kappa)=\{\mathbb{E}_{\hat{\omega}\sim\mathrm{vMF}(\hat{\omega}_r,\kappa)}[Y_l^m(\hat{\omega})]\mid(l,m)\in\mathcal{M}_L\}\\
\mathcal{M}_L=\{(l,m)\mid l\in\{1,2,\dots,2^L\},m\in\{0,1,\dots,l\}\}
$$
The expected value of any spherical harmonic under a vMF distribution has the following simple closed-form expression:
$$
\mathbb{E}_{\hat{\omega}\sim\mathrm{vMF}(\hat{\omega}_r,\kappa)}[Y_l^m(\hat{\omega})]=A_l(\kappa)Y_l^m(\hat{\omega}_r)\\
A_l(\kappa)=\frac{\kappa}{2\sinh\kappa}\int_{-1}^{1}P_l(u)e^{\kappa u}\mathrm{d}u\approx\exp\left(-\frac{l(l+1)}{2\kappa}\right)
$$
Increasing the roughness of a material by lowering $\kappa$ corresponds to attenuating the encoding’s spherical harmonics with high orders $l$, resulting in a wider interpolation kernel that limits the high frequencies in the represented view-dependent color.

#### Diffuse Colors

Diffuse color is by definition a function of only position. We modify the spatial MLP to output a diffuse color $\mathbf{c}_d$ and a specular tint $\mathbf{s}$, and we combine this with the specular color $\mathbf{c}_s$​ provided by the directional MLP to obtain a single color value:
$$
\mathbf{c}=\gamma(\mathbf{c}_d+\mathbf{s}\odot\mathbf{c}_s)
$$

#### Regularization for Normal Vectors

For each position $\mathbf{x}_i$ along a ray we output a 3-vector from the spatial MLP, which we then normalize to get a predicted normal $\hat{\mathbf{n}}_i'$ . We tie these predicted normals to the underlying density gradient normal samples along each ray $\{\hat{\mathbf{n}}_i\}$ using a simple penalty:
$$
\mathcal{R}_p=\sum_{i}w_i\|\hat{\mathbf{n}}_i-\hat{\mathbf{n}}_i'\|^2
$$
These MLP-predicted normals tend to be smoother than gradient density normals because the gradient operator acts as a high-pass filter on the MLP’s effective interpolation kernel.

Introducing a novel regularization term that penalizes normals that are “back-facing”, i.e. oriented away from the camera, at samples along the ray that contribute to the ray’s rendered color:
$$
\mathcal{R}_o=\sum_{i}w_i\max(0,\hat{\mathbf{n}}_i'\cdot\hat{\mathbf{d}})^2
$$

### 3 Related work

2021 NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination

### 4 Results

A much better result for specular reflection. Also better normal predictions for opaque objects.

### 5 Discussion

To deal with the specular reflection, they advocate using reflection directions as the input directions. And they also propose IDE to encode the directions. To obtain a better prediction of reflection vector, they need a better prediction of normal. The normal and the scene are jointly optimized. The two regularizer are mean to deal this problem.

### 6 Future work

Accelerating the mip-NeRF and ref-NeRF.



