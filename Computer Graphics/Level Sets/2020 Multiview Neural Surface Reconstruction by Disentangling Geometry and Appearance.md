# Multi-view Neural Surface Reconstruction by Disentangling Geometry and Appearance

## 0 Abstract

我们引入了一个神经网络架构，它可以同时学习未知的几何体、相机参数和一个神经渲染器，该渲染器近似于从表面反射到相机的光线。几何图形被表示为神经网络的零级集，而从渲染方程中导出的神经渲染器能够（隐含地）对广泛的照明条件和材料进行建模。
We introduce a neural network architecture that simultaneously learns the unknown geometry, camera parameters, and a neural renderer that approximates the light reflected from the surface towards the camera. The geometry is represented as a zero level-set of a neural network, while the neural renderer, derived from the rendering equation, is capable of (implicitly) modeling a wide set of lighting conditions and materials.

## 1 Introduction

本文的目标是设计一个端到端的神经架构系统，该系统可以从被遮挡的二维图像和粗略的相机估计中学习三维几何图形，并且不需要额外的监督。
The goal of this paper is to devise an end-to-end neural architecture system that can learn 3D geometries from masked 2D images and rough camera estimates, and requires no additional supervision.

为此，我们将一个像素的颜色表示为场景中三个未知数的可微分函数：几何、外观和摄像机。
Towards that end we represent the color of a pixel as a differentiable function in the three unknowns of a scene: the geometry, its appearance, and the cameras.

- 处理未知几何、外观和相机的端到端架构。
    End-to-end architecture that handles unknown geometry, appearance, and cameras.
- 表达神经隐含表面对摄像机参数的依赖性。
    Expressing the dependence of a neural implicit surface on camera parameters.
- 从现实生活中的二维图像中产生具有广泛外观的不同物体的最先进的三维表面重建，包括准确的和嘈杂的相机信息。
    Producing state of the art 3D surface reconstructions of different objects with a wide range of appearances, from real-life 2D images, with both exact and noisy camera information.

## 2 Related Work

#### Implicit surface differentiable ray casting

IDR 利用精确和可微调的表面点和隐含表面的法线，并考虑了更普遍的外观模型，以及处理带噪的相机。
IDR utilize exact and differentiable surface point and normal of the implicit surface, and considers a more general appearance model, as well as handle noisy cameras.

#### Multi-view surface reconstruction

#### Neural representation for view synthesis

## 3 Method

|        Symbols         |                         Descriptions                         |
| :--------------------: | :----------------------------------------------------------: |
|    $\theta\in\R^m$     |                      geometry, unknown                       |
|    $\gamma\in\R^n$     |                     appearance, unknown                      |
|     $\tau\in\R^k$      |                       cameras, unknown                       |
|      $\mathbf{x}$      |                        a 3-D position                        |
|      $\mathbf{n}$      |                     a 3-D normal vector                      |
|      $\mathbf{c}$      |                       camera position                        |
|      $\mathbf{v}$      |                     camera ray direction                     |
| $\mathcal{S}_{\theta}$ | $\mathcal{S}_{\theta}=\{\mathbf{x}\in\R^3\mid f(\mathbf{x};\theta)=0\}$, zero level set |

首先，SDF 允许使用球体追踪算法进行有效的光线投射；其次，IGR 享有隐性正则化，有利于光滑和现实的表面。
First, SDF allows an efficient ray casting with the sphere tracing algorithm; and second, IGR enjoys implicit regularization favoring smooth and realistic surfaces.

#### IDR forward model

|                       Symbols                        |                 Descriptions                  |
| :--------------------------------------------------: | :-------------------------------------------: |
|                         $p$                          |                    a pixel                    |
| $R_p(\tau)=\{\mathbf{c}_p+t\mathbf{v}_p\mid t\ge0\}$ |      the ray pass through the pixel $p$       |
|          $\mathbf{c}_p=\mathbf{c}_p(\tau)$           |       the unknown center of the camera        |
|          $\mathbf{v}_p=\mathbf{v}_p(\tau)$           | the ray direction, from $\mathbf{c}_p$ to $p$ |
|       $\mathbf{x}_p=\mathbf{x}_p(\theta,\tau)$       | the first intersection of ray and the surface |
|   $\mathbf{z}_p=\mathbf{z}_p(\mathbf{x}_p,\theta)$   |           a global geometry feature           |
|            $L_p=L_p(\theta,\gamma,\tau)$             |         the render color of the pixel         |
|                         $M$                          |                      MLP                      |

$$
L_p(\theta,\gamma,\tau)=M(\mathbf{x}_p,\mathbf{n}_p(\theta),\mathbf{z}_p(\mathbf{x}_p,\theta),\mathbf{v}_p(\tau);\gamma)
$$

### 3.1 Differentiable intersection of viewing direction and geometry

第一步是将交点 $\mathbf{x}_p(\theta,\tau)$ 表示为一个具有参数 $\theta,\tau$ 的神经网络。这可以通过对几何网络 $f$ 的轻微修改来实现。
The first step is to represent the intersection point $\mathbf{x}_p(\theta,\tau)$ as a neural network with parameters $\theta,\tau$. This can be done with a slight modification to the geometry network $f$.

让 $\mathbf{x}_p(\theta,\tau)=\mathbf{c}$ 表示交点。由于我们的目的是在类似梯度下降的算法中使用 $\mathbf{x}_p$，我们需要确保的是我们的推导在当前参数（用 $\theta_0$、$\tau_0$ 表示）下的值和一阶导数是正确的。记：$\mathbf{c}_0=\mathbf{c}_p(\tau_0)$, $\mathbf{v}_0=\mathbf{v}_p(\tau_0)$, $t_0=t(\theta,\mathbf{c}_0,\mathbf{v}_0)$, $\mathbf{x}_0=\mathbf{x}_p(\theta_0,\tau_0)=\mathbf{c}_0+t_0\mathbf{v}_0$。
Let $\mathbf{x}_p(\theta,\tau)=\mathbf{c}$ denote the intersection point. As we are aiming to use $\mathbf{x}_p$ in a gradient descent-like algorithm, all we need to make sure is that our derivations are correct in value and first derivatives at the current parameters, denoted by $\theta_0$, $\tau_0$; accordingly we denote $\mathbf{c}_0=\mathbf{c}_p(\tau_0)$, $\mathbf{v}_0=\mathbf{v}_p(\tau_0)$, $t_0=t(\theta,\mathbf{c}_0,\mathbf{v}_0)$, $\mathbf{x}_0=\mathbf{x}_p(\theta_0,\tau_0)=\mathbf{c}_0+t_0\mathbf{v}_0$.

**Lemma 1**

让 $\mathcal{S}_{\theta}$ 定义为曲面。射线 $R(\tau)$ 与曲面 $\mathcal{S}_{\theta}$ 的交点可以用公式表示：
Let $\mathcal{S}_{\theta}$ be defined as the surface. The intersection of the ray $R(\tau)$ and the surface $\mathcal{S}_{\theta}$ can be represented by the formula:
$$
\mathbf{x}_p(\theta,\tau)=\mathbf{c}_p(\tau)+t_0\mathbf{v}_p(\tau)-\frac{\mathbf{v}_p(\tau)}{\nabla_{\mathbf{x}}f(\mathbf{x}_0;\theta_0)\cdot\mathbf{v}_0}f(\mathbf{c}_p(\tau)+t_0\mathbf{v}_p(\tau);\theta)\tag{3}
$$
并且在 $\theta=\theta_0$ 和 $\tau=\tau_0$ 时，$\theta$ 和 $\tau$ 的值和一阶导数是精确的。
and is exact in value and first derivatives of $\theta$ and $\tau$ at $\theta=\theta_0$ and $\tau=\tau_0$.

---

To find the function dependence of $\mathbf{x}_p$ on $\theta,\tau$​, we use implicit differentiation:
$$
f(\mathbf{x}_p(\theta,\tau);\theta)\equiv0\\
f(\mathbf{c}_p(\tau)+t(\theta,\mathbf{c}_p(\tau),\mathbf{v}_p(\tau))\mathbf{v}_p(\tau);\theta)\equiv0
$$
We differentiate w.r.t $\mathbf{c}_p(\tau)$:
$$
\underbrace{\frac{\partial f}{\partial\mathbf{x}_p}}_{\R^{1\times3}}\biggl(\underbrace{I}_{\R^{3\times3}}+\underbrace{\mathbf{v}_p}_{\R^{3\times1}}\underbrace{\frac{\partial t}{\partial\mathbf{c}_p}}_{\R^{1\times3}}\biggr)=\mathbf{0}^T
$$
Rearranging and evaluating at $\theta_0,\mathbf{c}_0,\mathbf{v}_0$ we get:
$$
\begin{align*}
\frac{\partial f}{\partial\mathbf{x}_p}\mathbf{v}_p\frac{\partial t}{\partial\mathbf{c}_p}&=-\frac{\partial f}{\partial\mathbf{x}_p}\\
\frac{\partial t}{\partial\mathbf{c}_p}&=-\frac{1}{\frac{\partial f}{\partial\mathbf{x}_p}\mathbf{v}_p}\frac{\partial f}{\partial\mathbf{x}_p}\\
\frac{\partial t}{\partial\mathbf{c}_p}(\theta_0,\mathbf{c}_0,\mathbf{v}_0)&=-\frac{1}{\frac{\partial f}{\partial\mathbf{x}_p}(\mathbf{x}_0;\theta_0)\mathbf{v}_0}\frac{\partial f}{\partial\mathbf{x}_p}(\mathbf{x}_0;\theta_0)
\end{align*}
$$
We differentiate w.r.t $\mathbf{v}_p(\tau)$:
$$
\frac{\partial f}{\partial\mathbf{x}_p}\biggl(tI+\mathbf{v}_p\frac{\partial t}{\partial\mathbf{v}_p}\biggr)=\mathbf{0}^T
$$
Rearranging and evaluating at $\theta_0,\mathbf{c}_0,\mathbf{v}_0$ we get:
$$
\begin{align*}
\frac{\partial f}{\partial\mathbf{x}_p}\mathbf{v}_p\frac{\partial t}{\partial\mathbf{v}_p}&=-t\frac{\partial f}{\partial\mathbf{x}_p}\\
\frac{\partial t}{\partial\mathbf{v}_p}&=-\frac{t}{\frac{\partial f}{\partial\mathbf{x}_p}\mathbf{v}_p}\frac{\partial f}{\partial\mathbf{x}_p}\\
\frac{\partial t}{\partial\mathbf{v}_p}(\theta_0,\mathbf{c}_0,\mathbf{v}_0)&=-\frac{t_0}{\frac{\partial f}{\partial\mathbf{x}_p}(\mathbf{x}_0;\theta_0)\mathbf{v}_0}\frac{\partial f}{\partial\mathbf{x}_p}(\mathbf{x}_0;\theta_0)
\end{align*}
$$
We differentiate w.r.t $\theta$:
$$
\frac{\partial f}{\partial\mathbf{x}_p}\mathbf{v}_p\frac{\partial t}{\partial\theta}+\frac{\partial f}{\partial\theta}=0
$$
Rearranging and evaluating at $\theta_0,\mathbf{c}_0,\mathbf{v}_0$ we get:
$$
\begin{align*}
\frac{\partial f}{\partial\mathbf{x}_p}\mathbf{v}_p\frac{\partial t}{\partial\theta}&=-\frac{\partial f}{\partial\theta}\\
\frac{\partial t}{\partial\theta}&=-\frac{1}{\frac{\partial f}{\partial\mathbf{x}_p}\mathbf{v}_p}\frac{\partial f}{\partial\theta}\\
\frac{\partial t}{\partial\theta}(\theta_0,\mathbf{c}_0,\mathbf{v}_0)&=-\frac{1}{\frac{\partial f}{\partial\mathbf{x}_p}(\mathbf{x}_0;\theta_0)\mathbf{v}_0}\frac{\partial f}{\partial\theta}(\mathbf{x}_0;\theta_0)
\end{align*}
$$


---

We implement equation $(3)$ as a neural network, namely, we add two linear layers (with parameters c; v): one before and one after the MLP f. Equation 3 unifies the sample network
formula in [1] and the differentiable depth in [40] and generalizes them to account for unknown
cameras. The normal vector to Sθ at x^ can be computed by:

Note that for SDF the denominator is 1, so can be omitted.



