# NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction

## 0 Abstract

现有的神经表面重建方法，如 DVR 和 IDR 需要前景掩码作为监督，很容易陷入局部最小值，因此难以重建具有严重自遮挡或薄结构的物体。
Existing neural surface reconstruction approaches, such as DVR and IDR require foreground mask as supervision, easily get trapped in local minima, and therefore struggle with the reconstruction of objects with severe self-occlusion or thin structures.

从这种学习到的隐式表示中提取高质量的表面是很困难的，因为表示中没有足够的对表面的约束。
Extracting high-quality surfaces from this learned implicit representation is difficult because there are not sufficient surface constraints in the representation.

我们观察到传统的体绘制方法会导致表面重建的固有几何误差（即偏差），因此提出了一种新的公式，该公式在一阶近似中没有偏差，因此即使没有掩模监督也能实现更准确的表面重建。
We observe that the conventional volume rendering method causes inherent geometric errors (i.e. bias) for surface reconstruction, and therefore propose a new formulation that is free of bias in the first order of approximation, thus leading to more accurate surface reconstruction even without the mask supervision.

## 1 Introduction

例如，IDR 产生了令人印象深刻的重建结果，但它无法重建具有导致突然深度变化的复杂结构的对象。造成这种限制的原因是 IDR 中使用的表面渲染方法只考虑了每条光线的单个表面交点。因此，梯度只存在于这个单一的点，这对于有效的反向传播来说太局部了，并且当图像的深度突然变化时，优化会陷入糟糕的局部最小值。
For example, IDR produces impressive reconstruction results, but it fails to reconstruct objects with complex structures that causes abrupt depth changes. The cause of this limitation is that the surface rendering method used in IDR only considers a single surface intersection point for each ray. Consequently, the gradient only exists at this single point, which is too local for effective back propagation and would get optimization stuck in a poor local minimum when there are abrupt changes of depth on images.

由于孔洞引起的深度变化，神经网络会错误地将前表面附近的点预测为蓝色，而无法找到远处的表面。
With the radical depth change caused by the hole, the neural network would incorrectly predict the points near the front surface to be blue, failing to find the far-back blue surface.

体绘制方法的优势在于它可以处理突然的深度变化，因为它考虑了沿射线的多个点，因此所有采样点，无论是靠近表面还是在远表面上，都会产生用于反向传播的梯度信号。
The advantage of the volume rendering approach is that it can handle abrupt depth changes, because it considers multiple points along the ray and so all the sample points, either near the surface or on the far surface, produce gradient signals for back propagation.

NeuS 使用有符号距离函数 (SDF) 进行表面表示，并使用一种新颖的体绘制方案来学习神经 SDF 表示。
NeuS uses the signed distance function (SDF) for surface representation and uses a novel volume rendering scheme to learn a neural SDF representation.

## 2 Related Works

### 2.1 Classical Multi-view Surface and Volumetric Reconstruction

**基于点和表面的重建**：通过利用图像间光度一致性估计每个像素的深度图，然后将深度图融合到全局密集点云中。表面重建通常作为后处理使用筛选泊松表面重建等方法完成。
**point- and surface-based reconstruction**: estimate the depth map of each pixel by exploiting inter-image photometric consistency and then fuse the depth maps into a global dense point cloud. The surface reconstruction is usually done as a post processing with methods like screened Poisson surface reconstruction.

重构质量很大程度上依赖于对应匹配的质量，对于没有丰富纹理的对象，对应匹配困难往往会导致重构结果中出现严重的伪影和缺失部分。
The reconstruction quality heavily relies on the quality of correspondence matching, and the difficulties in matching correspondence for objects without rich textures often lead to severe artifacts and missing parts in the reconstruction results.

**体像重建**：通过从多视图图像估计体素网格中的占用率和颜色并评估每个体素的颜色一致性来规避显式对应匹配的困难。
**volumetric reconstruction**: circumvent the difficulty of explicit correspondence matching by estimating occupancy and color in a voxel grid from multi-view images and evaluating the color consistency of each voxel.

### 2.2 Neural Implicit Representation

一些方法通过引入归纳偏差在深度学习框架中强制执行 3D 理解。 这些归纳偏差可以是显式表示，例如体素网格、点云、网格和隐式表示。
Some methods enforce 3D understanding in a deep learning framework by introducing inductive biases. These inductive biases can be explicit representations, such as voxel grids, point cloud, meshes, and implicit representations.

## 3 Method

### 3.1 Rendering Procedure

#### Scene Representation

|                             Symbols                             |                            Description                            |
| :-------------------------------------------------------------: | :---------------------------------------------------------------: |
|                       $\{\mathcal{I}_k\}$                       |                       a set of posed images                       |
|                $f:\mathbb{R}^3\mapsto\mathbb{R}$                |    maps a spatial position to its signed distance to the object   |
|      $c:\mathbb{R}^3\times\mathbb{S}^2\mapsto\mathbb{R}^3$      | encodes the color associated with a point and a viewing direction |
| $\mathcal{S}=\{\mathbf{x}\in\mathbb{R}^3\mid f(\mathbf{x})=0\}$ |            SDF represented surface to be reconstructed            |

*S-density or logistic density distribution*: $\phi_{s}(f(\mathbf{x})),\mathbf{x}\in\mathbb{R}^3$.
$$
\Phi_{s}(x)=\frac{1}{1+e^{-sx}}\\
\phi_{s}(x)=\frac{\mathrm{d}}{\mathrm{d}x}\Phi_{s}(x)=\frac{se^{-sx}}{(1+e^{-sx})^2}
$$

原则上 $\phi_{s}(x)$ 可以是任何以 0 为中心的单峰（即钟形）密度分布；在这里，我们选择逻辑密度分布是为了计算方便。
In principle $\phi_{s}(x)$ can be any unimodal (i.e. bell-shaped) density distribution centered at 0; here we choose the logistic density distribution for its computational convenience.

注意 $\phi_{s}(x)$ 的标准差由 $1/s$ 给出，这也是一个可训练的参数，即随着网络训练收敛，$1/s$ 接近于零。
Note that the standard deviation of $\phi_{s}(x)$ is given by $1/s$, which is also a trainable parameter, that is, $1/s$ approaches to zero as the network training converges.
$$
\begin{align*}
\mathbb{E}_{x}(\phi_{s}(x))&=\int_{-\infty}^{+\infty}x\phi_{s}(x)\mathrm{d}x=\int_{-\infty}^{+\infty}\frac{sxe^{-sx}}{(1+e^{-sx})^2}\mathrm{d}x\\
&=0\quad(\text{odd function})
\end{align*}
$$

$$
\begin{align*}
\mathbb{Var}_{x}(\phi_{s}(x))&=\int_{-\infty}^{+\infty}x^2\phi_{s}(x)\mathrm{d}x\\
&=\int_{-\infty}^{+\infty}\frac{sx^2e^{-sx}}{(1+e^{-sx})^2}\mathrm{d}x\\
&=\frac{1}{s^2}\int_{-\infty}^{+\infty}\frac{u^2e^{-u}}{(1+e^{-u})^2}\mathrm{d}u&(u=sx)\\
&=\frac{\pi^2}{3s^2}&(\text{With the help of Wolfram})
\end{align*}
$$

在 S 密度场 $\phi_{s}(x)$ 的帮助下，体绘制用于训练仅使用 2D 输入图像作为监督的 SDF 网络。基于这种监督成功最小化损失函数后，网络编码 SDF 的零级集有望形成一个准确重建的表面 $S$，其诱导的 S 密度 $\phi_{s}(x)$ 显着假设表面附近的高值。
With the aid of the S-density field $\phi_{s}(x)$, volume rendering is used to train the SDF network with only 2D input images as supervision. Upon successful minimization of a loss function based on this supervision, the zero-level set of the network-encoded SDF is expected to represent an accurately reconstructed surface $S$, with its induced S-density $\phi_{s}(x)$ assuming prominently high values near the surface.

#### Rendering

$$
C(\mathbf{o},\mathbf{v})=\int_{0}^{\infty}w(t)c(\mathbf{p}(t),\mathbf{v})\mathrm{d}t
$$
where $\mathbf{p}(t)=\mathbf{o}+t\mathbf{v}$ is the ray, $c(\mathbf{x},\mathbf{v})$ is the view dependent color, $w(t)$ is the weight.

#### Requirements on weight function

1. 无偏的。给定相机射线 $\mathbf{p}(t)$，$w(t)$ 在表面交点 $\mathbf{p}(t^*)$ 处获得局部最大值，即与 $f(\mathbf{p}(t^*))=0$，即点 $\mathbf{p}(t^*)$ 在 SDF 的零水平集上。
   Unbiased. Given a camera ray $\mathbf{p}(t)$, $w(t)$ attains a locally maximal value at a surface intersection point $\mathbf{p}(t^*)$, i.e. with $f(\mathbf{p}(t^*))=0$, that is , the point $\mathbf{p}(t^*)$ is on the zero-level set of the SDF.
2. 遮挡感知。给定任意两个满足 $f(t_0)=f(t_1)$，$w(t_0)>0,w(t_1)>0$ 和 $t_0<t_1$ 的深度值 $t_0$ 和 $t_1$，有 $w(t_0)>w(t_1)$。也就是说，当两个点具有相同的 SDF 值（相同的 SDF 诱导的 S-密度值）时，离视点较近的点应该比另一个点对最终输出颜色的贡献更大。
   Occlusion-aware. Given any two depth values $t_0$ and $t_1$ satisfying $f(t_0)=f(t_1)$, $w(t_0)>0,w(t_1)>0$, and $t_0<t_1$, there is $w(t_0)>w(t_1)$. That is, when two points have the same SDF value (thus the same SDF-induced S-density value), the point nearer to the view point should have a larger contribution to the final output color than does the other point.

无偏权重函数 $w(t)$ 保证相机光线与 SDF 的水平集的交点对像素颜色的贡献最大。
An unbiased weight function $w(t)$ guarantees that the intersection of the camera ray with the zero-level set of SDF contributes most to the pixel color.

遮挡感知属性确保当光线顺序通过多个表面时，渲染过程将正确使用最接近相机的表面的颜色来计算输出颜色。
The occlusion-aware property ensures that when a ray sequentially passes multiple surfaces, the rendering procedure will correctly use the color of the surface nearest to the camera to compute the output color.

#### Naive solution

$$
w(t)=T(t)\sigma(t)
$$
where $\sigma(t)$ is volume density and $T(t)=\exp(-\int_0^t\sigma(s)\mathrm{d}s)$ is transmittance along the ray.

然后 $\sigma(t)$ 被设置为 S-密度，权重是上面的方程。
Then $\sigma(t)$ is set to be S-density and weight is the above equation.

权函数 $w(t)$ 在光线到达表面点 $\mathbf{p}(t^*)$ 之前的某一点达到局部最大值，而 $f(\mathbf{p}(t^*)) = 0$。
The weight function $w(t)$ attains a local maximum at a point before the ray reaches the surface point $\mathbf{p}(t^*)$, satisfying $f(\mathbf{p}(t^*))=0$.

（此部分在 Supplementary B.2，但是请先看完这个部分。）

#### Our solution

使用归一化的 S 密度作为权重。
Use the normalized S-density as weights.

$$
w(t)=\frac{\phi_{s}(f(\mathbf{p}(t)))}{\int_{-\infty}^{\infty}\phi_{s}(f(\mathbf{p}(u)))\mathrm{d}u}
$$

这种权重函数的构造是无偏的，但不是遮挡感知的。
This construction of weight function is unbiased, but not occlusion-aware.

（此部分在 Supplementary B.1，但是请先看完这个部分。）

我们首先定义一个不透明的密度函数 $\rho(t)$，它是标准体绘制中体密度的对应物。
We first define an opaque density function $\rho(t)$​, which is the counterpart of the volume density in standard volume rendering.
$$
w(t)=T(t)\rho(t)\quad T(t)=\exp\left(-\int_0^t\rho(s)\mathrm{d}s\right)
$$

#### How we derive opaque density $\rho$

具体来说，在单平面相交的简单情况下，很容易看出有符号距离函数 $f(\mathbf{p}(t))=-|\cos(\theta)|\cdot(t-t^*)$，其中 $f(\mathbf{p}(t^*))=0$，$\theta$ 是观察方向 $\mathbf{v}$ 与外表面法线向量的夹角 $\mathbf{n}$。因为假设是局部表面，因此 $\theta$ 是一个常数。
Specifically, in the simple case of a single plane intersection, it is easy to see that the signed distance function $f(\mathbf{p}(t))=-|\cos(\theta)|\cdot(t-t^*)$, where $f(\mathbf{p}(t^*))=0$, and $\theta$ is the angle between the view direction $\mathbf{v}$ and the outward surface normal vector $\mathbf{n}$. Because the surface is assumed locally, $\theta$ is a constant.

Under this setting, let $w(t)$ be:
$$
\begin{align*}
w(t)&=\frac{\phi_{s}(f(\mathbf{p}(t)))}{\int_{-\infty}^{\infty}\phi_{s}(f(\mathbf{p}(u)))\mathrm{d}u}\\
&=\frac{\phi_{s}(f(\mathbf{p}(t)))}{\int_{-\infty}^{\infty}\phi_{s}(-|\cos(\theta)|\cdot(u-t^*))\mathrm{d}u}\\
&=|\cos\theta|\phi_{s}(f(\mathbf{p}(t)))
\end{align*}
$$

Since:
$$
\begin{align*}
&\int_{-\infty}^{\infty}\phi_{s}(-|\cos(\theta)|\cdot(u-t^*))\mathrm{d}u\\
&=-|\cos\theta|^{-1}\int_{-\infty}^{\infty}\phi_{-|\cos(\theta)|s}(u-t^*)\mathrm{d}u\\
&=-|\cos\theta|^{-1}\int_{-\infty}^{\infty}\phi_{-|\cos(\theta)|s}(u)\mathrm{d}u\\
&=-|\cos\theta|^{-1}[\Phi_{-|\cos(\theta)|s}(\infty)-\Phi_{-|\cos(\theta)|s}(-\infty)]\\
&=|\cos\theta|^{-1}
\end{align*}
$$
Since by definition and induction:
$$
w(t)=T(t)\sigma(t)=|\cos\theta|\phi_{s}(f(\mathbf{p}(t)))
$$
Note that we have:
$$
\begin{align*}
\frac{\mathrm{d}T}{\mathrm{d}t}\bigg|_{t}&=-T(t)\sigma(t)\\
\frac{\mathrm{d}\Phi_s}{\mathrm{d}t}\bigg|_{t=t}=\frac{\mathrm{d}\Phi_s}{\mathrm{d}f}\bigg|_{f=f(\mathbf{p}(t))}\frac{\mathrm{d}f}{\mathrm{d}t}\bigg|_{t=t}&=-|\cos\theta|\cdot\phi_s(f(\mathbf{p}(t)))
\end{align*}
$$

Integrating both sides, taking the logarithm and then differentiating both sides:
$$
\begin{align*}
&\Longrightarrow&\int_{-\infty}^{t}\mathrm{d}T(t)&=\int_{-\infty}^{t}\mathrm{d}\Phi_s(f(\mathbf{p}(t)))\\
&\Longrightarrow&T(t)-T(-\infty)&=\Phi_s(f(\mathbf{p}(t)))-\Phi_s(f(\mathbf{p}(-\infty)))\\
&\Longrightarrow&T(t)-1&=\Phi_s(f(\mathbf{p}(t)))-1\\
&\Longrightarrow&\int_{-\infty}^t\rho(u)\mathrm{d}u&=-\ln\Phi_s(f(\mathbf{p}(t)))\\
&\Longrightarrow&\rho(t)&=-\frac{-\dfrac{\mathrm{d}\Phi_s}{\mathrm{d}t}\bigg|_{t=f(\mathbf{p}(t))}}{\Phi_s(f(\mathbf{p}(t)))}
\end{align*}
$$
沿射线 $\mathbf{p}(t)$ 有多个表面交点，我们将其裁剪为零以确保 $\rho$ 的值始终为非负数。
There are multiple surface intersections along the ray $\mathbf{p}(t)$, we clip it against zero to ensure that the value of $\rho$ is always non-negative.
$$
\rho(t)=\max\left(\frac{-\dfrac{\mathrm{d}\Phi_s}{\mathrm{d}f}\bigg|_{f=f(\mathbf{p}(t))}}{\Phi_s(f(\mathbf{p}(t)))},0\right)
$$

<blockquote style="border-left: 5px solid #4545aa; border-radius: 3px 0 0 3px; padding: 10px 15px; background-color: rgba(70, 70, 188, 0.1)">
    Theorem 1
</blockquote>
假设光滑曲面 $\mathbb{S}$ 由有符号距离函数 $f(\mathbf{x})=0$ 的水平集定义，并且一条光线 $\mathbf{p}(t)=\mathbf {o}+t\mathbf{v}$ 由外向内进入曲面 $\mathbb{S}$，交点在 $t^{*}$，即 $f(\mathbf{p}(t^{*}))=0$。并且存在区间 $[t_l,t_r]$，使得 $t^*\in[t_l,t_r]$ 的时候曲面可以切向近似为足够小的平面，即 $\nabla{f}$ 被认为是固定的。然后，在 $[t_l,t_r]$ 中由不透明密度方程和透射方程计算的权重函数 $w(t)$ 在 $t^{*}$ 处达到极大值。
Suppose that a smooth surface $\mathbb{S}$ is defined by the zero-level set of the signed distance function $f(\mathbf{x})=0$, and a ray $\mathbf{p}(t)=\mathbf{o}+t\mathbf{v}$ enters the surface $\mathbb{S}$ from outside to inside, with the intersection point at $t^{*}$, that is, $f(\mathbf{p}(t^{*}))=0$ and there exists an interval $[t_l,t_r]$ such that $t^*\in[t_l,t_r]$, the surface can be tangentially approximated by a sufficiently small planar patch, i.e., $\nabla{f}$ is regarded as fixed. Then, the weight function $w(t)$ computed by opaque density equation and transmittance equation in $[t_l,t_r]$ attains its maximum at $t^{*}$.

#### Discretization

该方案采样 $n$ 个点 $\{\mathbf{p}_i=\mathbf{o}+t_i\mathbf{v}\mid i\in\{1, \dots,n\},t_{i}< t_{i+1}\}$ 沿射线计算射线的近似像素颜色为
This scheme samples $n$ points $\{\mathbf{p}_i=\mathbf{o}+t_i\mathbf{v}\mid i\in\{1, \dots,n\},t_{i}<t_{i+1}\}$ along the ray to compute the approximate pixel color of the ray as
$$
\hat{C}=\sum_{i=1}^{n}T_{i}\alpha_{i}c_{i}\\
T_i=\prod_{j=1}^{i-1}(1-\alpha_{j})\\
$$

$$
\alpha_{i}=1-\exp\left(-\int_{t_{i}}^{t_{i+1}}\rho(t)\mathrm{d}t\right)\Rightarrow\\
\alpha_{i}=\max\left(\frac{\Phi_{s}(f(\mathbf{p}(t_{i})))-\Phi_{s}(f(\mathbf{p}(t_{i+1})))}{\Phi_{s}(f(\mathbf{p}(t_i)))},0\right)
$$

（此部分在 Supplementary A。）

### 3.2 Training

为了训练 NeuS，我们在没有任何 3D 监督的情况下最小化渲染颜色和 gt 颜色之间的差异。除了颜色，如果提供，我们还可以使用掩码进行监督。
To train NeuS, we minimize the difference between the rendered colors and the ground truth colors, without any 3D supervision. Besides colors, we can also utilize the masks for supervision if provided.

|      Symbol       |     Description     |
| :---------------: | :-----------------: |
|      $C_{k}$      |     pixel color     |
| $M_{k}\in\{0,1\}$ | optional mask value |
|        $n$        | point sampling size |
|        $m$        |     batch size      |

Loss function:
$$
\mathcal{L}=\mathcal{L}_{\text{color}}+\lambda\mathcal{L}_{\text{reg}}+\beta\mathcal{L}_{\text{mask}}
$$
$\mathcal{L}_{\text{color}}$ is L1 re-render loss.

$\mathcal{L}_{\text{reg}}$ is Eikonal term:
$$
\mathcal{L}_{\text{reg}}=\frac{1}{mn}=\sum_{i,j}(\|\nabla{f}(\mathbf{\hat{p}}_{i,j})\|_2-1)^2
$$
$\mathcal{L}_{\text{mask}}$ is optional mask loss:
$$
\mathcal{L}_{\text{mask}}=\mathrm{BCE}(M_k,\hat{O}_k)
$$
$\hat{O}_k=\sum_{i=1}^{n}T_{k,i}\alpha_{k,i}$ is the sum of weights along the camera ray, and BCE is the binary cross entropy loss.

## 4 Experiments

## 5 Conclusion

## A Derivation for Computing Opacity $\alpha_i$

首先是自定义的 density：
$$
\rho(t)=\max\left(\frac{-\dfrac{\mathrm{d}\Phi_s}{\mathrm{d}t}\bigg|_{t=f(\mathbf{p}(t))}}{\Phi_s(f(\mathbf{p}(t)))},0\right)
$$
由于：
$$
\frac{\mathrm{d}\Phi_s}{\mathrm{d}t}\bigg|_{t=t}=\frac{\mathrm{d}\Phi_s}{\mathrm{d}f}\bigg|_{f=f(\mathbf{p}(t))}\frac{\mathrm{d}f}{\mathrm{d}\mathbf{p}}\bigg|_{\mathbf{p}=\mathbf{p}(t)}\frac{\mathrm{d}\mathbf{p}}{\mathrm{d}t}\bigg|_{t=t}=-\phi_s(f(\mathbf{p}(t)))(\nabla{f}(\mathbf{p}(t))\cdot\mathbf{v})
$$
因此 density 可以表示为：
$$
\rho(t)=\max\left(\frac{-\phi_s(f(\mathbf{p}(t)))(\nabla{f}(\mathbf{p}(t))\cdot\mathbf{v})}{\Phi_s(f(\mathbf{p}(t)))},0\right)
$$
首先考虑的情况是：采样 $[t_i,t_{i+1}]$ ，$-(\nabla{f}(\mathrm{p}(t))\cdot\mathbf{v})>0$，因此：
$$
\begin{align*}
\alpha_i&=1-\exp\left(-\int_{t_i}^{t_{i+1}}\rho(t)\mathrm{d}t\right)\\
&=1-\exp\left(\int_{t_i}^{t_{i+1}}\frac{\phi_s(f(\mathbf{p}(t)))(\nabla{f}(\mathbf{p}(t))\cdot\mathbf{v})}{\Phi_s(f(\mathbf{p}(t)))}\mathrm{d}t\right)\\
&=1-\exp\left(\ln[\Phi_s(f(\mathbf{p}(t)))]\bigl|_{t_i}^{t_{i+1}}\right)\\
&=1-\frac{\Phi_s(f(\mathbf{p}(t_{i+1})))}{\Phi_s(f(\mathbf{p}(t_i)))}\\
&=\frac{\Phi_s(f(\mathbf{p}(t_i)))-\Phi_s(f(\mathbf{p}(t_{i+1})))}{\Phi_s(f(\mathbf{p}(t_i)))}
\end{align*}
$$
因此 $\alpha_i>0$，且由于 $\Phi(x)\in(0,1)$，$\alpha_i<1$。

其次是 $-(\nabla{f}(\mathrm{p}(t))\cdot\mathbf{v})<0$，$\alpha_i=0$。因此得到：
$$
\alpha_i=\max\left(\frac{\Phi_s(f(\mathbf{p}(t_i)))-\Phi_s(f(\mathbf{p}(t_{i+1})))}{\Phi_s(f(\mathbf{p}(t_i)))},0\right)
$$

## B First-order Bias Analysis

### B.1 Proof of Unbiased Property of Our Solution

**定理 1**：假设光线从表面的外部进入内部。因此，我们有 $-(\nabla{f}(\mathrm{p}(t))\cdot\mathbf{v})>0$，因为按照惯例，SDF $f(x)$ 在外部为正，表面内部为负。
**Theorem 1**: Suppose that the ray is going from outside to inside of the surface. Hence, we have $-(\nabla{f}(\mathrm{p}(t))\cdot\mathbf{v})>0$, because by convention the signed distance function $f(x)$ is positive outside and negative inside of the surface.





### B.2 Bias in Naive Solution

证明 $w(t)=\sigma(t)T(t)$ 的最大点小于 $\sigma(t)=\phi(f(\mathbf{p}(t)))$ 的最大点。
Proof of the maximum point of $w(t)=\sigma(t)T(t)$ is smaller than that of $\sigma(t)=\phi(f(\mathbf{p}(t)))$.

列出已有的关系：
$$
T(t)=\exp\left(-\int_0^t\sigma(s)\mathrm{d}s\right)\quad\frac{\mathrm{d}T}{\mathrm{d}t}=-\sigma(t)\exp\left(-\int_0^t\sigma(s)\mathrm{d}s\right)=-T(t)\sigma(t)
$$
首先求 $\sigma(t)$ 达到最大值时候，对应的 $\mathrm{d}w/\mathrm{d}t$ 值：
$$
\begin{align*}
\frac{\mathrm{d}w}{\mathrm{d}t}&=\frac{\mathrm{d}\sigma}{\mathrm{d}t}T+\frac{\mathrm{d}T}{\mathrm{d}t}\sigma\\
&=\frac{\mathrm{d}\sigma}{\mathrm{d}t}T-T\sigma^2\\
&=T\biggl(\frac{\mathrm{d}\sigma}{\mathrm{d}t}-\sigma^2\biggr)
\end{align*}
$$

此时：
$$
\frac{\mathrm{d}\sigma}{\mathrm{d}t}(t^*)=0\Longrightarrow\sigma^2(t^*)>0\Longrightarrow\mathrm{d}w/\mathrm{d}t<0
$$









