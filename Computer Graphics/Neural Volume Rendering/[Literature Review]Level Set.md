# Literature Review of Signed Distance Function

## NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction

### 1 Motivation

NeRF: Extracting high-quality surfaces from this learned implicit representation is difficult because there are not sufficient surface constraints in the representation.

Existing neural surface reconstruction approaches, such as DVR and IDR require foreground mask as supervision, easily get trapped in local minima, and therefore struggle with the reconstruction of objects with severe self-occlusion or thin structures.

### 2 Methods

|                        Symbol                         |         Description         |
| :---------------------------------------------------: | :-------------------------: |
|           $f:\mathbb{R}^3\mapsto\mathbb{R}$           |           SDF MLP           |
|               $\mathbf{x}:\mathbb{R}^3$               |       input position        |
|               $\mathbf{d}:\mathbb{S}^2$               |    input view direction     |
| $c:\mathbb{R}^3\times\mathbb{S}^2\mapsto\mathbb{R}^3$ | view-dependent color output |
|                                                       |                             |
|                                                       |                             |

The target of NeuS is to extract high-quality surfaces. SDF with coordinate-based neural network is proven to be efficient for this. Instead of point-cloud input, NeuS takes a set of posed images as input and outputs a re-rendered image.

#### Density

Unlike NeRF directly output volume density, NeuS outputs the signed distance in the space. Then the distance is manipulated by the logistic density distribution to produce **S-density**:
$$
\phi_s(f(\mathbf{x})),\text{ where }
\phi_{s}(x)=\frac{\mathrm{d}}{\mathrm{d}x}\Phi_{s}(x)=\frac{se^{-sx}}{(1+e^{-sx})^2}
$$
and $\Phi_s(x)$ is sigmoid function. $\phi_s(x)$ is controlled by its standard deviation $1/s$ where $s$ is also a trainable parameter.

#### Rendering

Since we have density and color, we can render by volume rendering.

#### Weight Function

The weight function in conventional volume rendering is $w(t)=T(t)\sigma(t)$. However, there are two requirement for weight function:

1. **Unbiased**. Given a camera ray $\mathbf{r}(t)$, the local maximal point $t^*$ of $w(t)$ is the also the ray-surface intersection point $\mathbf{r}(t^*)$, i.e. with $f(\mathbf{r}(t^*))=0$. This guarantees that the ray-surface intersection point contributes most to the pixel color. (In reality, the weight function is almost similar to a impulse function at the intersection point.)
2. **Occlusion-aware**. For two equal-SDF value points, the point nearer to the view point should have a larger contribution to the final output color than that of the other point.  This guarantees that when a ray sequentially passes multiple surfaces, the rendering procedure will correctly use the color of the surface nearest to the camera to compute the output color.

#### Occlusion-aware Solution

The weight function in volume rendering is occlusion-aware but is biased. In this way, the weight function is in the form of:
$$
w(t)=T(t)\sigma(t),\text{ where }T(t)=\exp\left(-\int_0^{\infty}\sigma(s)\mathrm{d}s\right)
$$

#### Unbiased Solution

Obviously, this is unbiased:
$$
w(t)=\frac{\phi_{s}(f(\mathbf{r}(t)))}{\int_{0}^{\infty}\phi_{s}(f(\mathbf{r}(s)))\mathrm{d}s}
$$

#### Unbiased & Occlusion-aware Solution

We can induce that the density along the ray is:
$$
\rho(t)=\max\left(\frac{-\dfrac{\mathrm{d}\Phi_s}{\mathrm{d}t}\bigg|_{t=f(\mathbf{p}(t))}}{\Phi_s(f(\mathbf{p}(t)))},0\right)
$$
But we will not use it directly. A trick is that, we can directly output the alpha value used in alpha-blending:
$$
\alpha_{i}=\max\left(\frac{\Phi_{s}(f(\mathbf{p}(t_{i})))-\Phi_{s}(f(\mathbf{p}(t_{i+1})))}{\Phi_{s}(f(\mathbf{p}(t_i)))},0\right)
$$

#### Other Regularizer

L1 render loss, Eikonal loss.

### Related work

2020 Multiview neural surface reconstruction by disentangling geometry and appearance

### Results

No foreground mask needed. Better geometry extraction than NeRF and IDR.

### Discussion

This model was successful by its multiview consistent and dense spatial sampling. NeRF gives it a good rendering effect.

### Future work

Gradient field.

