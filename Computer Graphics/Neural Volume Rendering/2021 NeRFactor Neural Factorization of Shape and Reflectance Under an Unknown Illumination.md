# NeRFactor Neural Factorization of Shape and Reflectance Under an Unknown Illumination

## Abstract

我们解决了一个这样的问题：从单一未知照明条件的多视图构成的图像中，恢复物体的形状、恢复随着空间位置变化的反射率的问题
We address the problem of recovering the shape and spatially-varying reflectance of an object from posed multi-view images of the object illuminated by one unknown lighting condition.

这使得在任意环境照明下渲染物体的新视图和编辑物体的材料属性成为可能。
This enables the rendering of novel views of the object under arbitrary environment lighting and editing of the object’s material properties.

将 NeRF 表示的物体的体积几何提取为表面表示，然后在求解随着空间位置变化的反射率和环境照明的同时联合细化几何。
Distill the volumetric geometry of a Neural Radiance Field (NeRF) representation of the object into a surface representation and then jointly refine the geometry while solving for the spatially-varying reflectance and the environment lighting.

NeRFactor 无需任何监督即可恢复表面法线、光能见度、反照率和双向反射分布函数 (BRDF) 的 3D 神经场。
NeRFactor recovers 3D neural fields of surface normals, light visibility, albedo, and Bidirectional Reflectance Distribution Functions (BRDFs) without any supervision.

## Introduction

我们可以首先用输入图像优化神经辐射场 (NeRF) 以初始化我们模型的表面法线和光能见度，然后联合优化这些初始估计以及着空间位置变化的反射率和照明条件，以最好地解释观察到的图像。
We can first optimize a Neural Radiance Field (NeRF) from the input images to initialize our model’s surface normals and light visibility, and then jointly optimize these initial estimates along with the spatially-varying reflectance and the lighting condition to best explain the observed images.

使用 NeRF 生成的用于初始化的高质量几何估计有助于拆分形状、反射率和光照之间固有的模糊性，从而使我们能够恢复完整的 3D 模型，以便仅使用重新渲染损失，每个组分的简单空间平滑先验，以及一个新的数据驱动的 BRDF 先验即可进行令人信服的视图合成和重打光。
The use of NeRF to produce a high-quality geometry estimation for initialization helps break the inherent ambiguities among shape, reflectance, and lighting, thereby allowing us to recover a full 3D model for convincing view synthesis and relighting using just a re-rendering loss, simple spatial smoothness priors for each of these components, and a novel data-driven BRDF prior.

它能够从反照率估计中去除阴影，并在任意新的照明条件下合成逼真的软或硬阴影。
It is capable of removing shadows from albedo estimation and synthesizing realistic soft or hard shadows under arbitrary novel lighting conditions.

问题：首先，NeRF 模型的形状是一个体积场，计算整个半球照明中沿相机光线每个点的阴影和可见性在计算上是昂贵的。
First, NeRF models shape as a volumetric field and as such it is computationally expensive to compute shading and visibility at each point along a camera ray for a full hemisphere of lighting.

解决：使用 NeRF 几何的“硬表面”近似值，我们仅在沿着每条射线的单个点处执行着色计算，对应于体积的预期终止深度。
Using a "hard surface" approximation of the NeRF geometry, where we only perform shading calculations at a single point along each ray, corresponding to the expected termination depth of the volume.

问题：其次，NeRF 估计的几何包含无关的高频内容，虽然在视图合成结果中不明显，但将高频伪影引入了从 NeRF 几何计算的表面法线和光可见性中。
Second, the geometry estimated by NeRF contains extraneous high-frequency content that, while unnoticeable in view synthesis results, introduces high-frequency artifacts into the surface normals and light visibility computed from the geometry of NeRF.

解决：将此表面上任何 3D 位置的表面法线和光可见性表示为由 MLP 参数化的连续函数，并鼓励这些函数接近从预训练 NeRF 派生的值并在空间上平滑。
Representing the surface normal and light visibility at any 3D location on this surface as continuous functions parameterized by Multi-Layer Perceptrons (MLPs), and encourage these functions to be close to the values derived from the pre-trained NeRF and be spatially smooth.

假设：NeRFactor 认为物体由硬表面组成，每条光线都只有一个交点，因此不会对体积光传输效果（例如散射、透明度和半透明）进行建模。
Assumptions. NeRFactor considers objects to be composed of hard surfaces with a single intersection point per ray, so volumetric light transport effects such as scattering, transparency, and translucency are not modeled.

## Related Work

## Method

输入：位姿多视角图像，由单一但未知的光源照亮。

网络：MLP，预测物体形状和随空间位置变化的反射率。

输出：3D 表面某一点 $\mathbf x$ 处的表面法向 $\mathbf n$，任何方向的可见度 $v(\omega_i)$，反照率 $\mathbf a$ 和反射率 $\mathbf z_{\text{BRDF}}$。

### Shape

特别是，我们用优化过的 NeRF 来计算沿任何相机光线的预期表面位置、物体表面上每个点的法线以及从物体表面上每个点的任何方向到达的光的可见性。
In particular, we use the optimized NeRF to compute the expected surface location along any camera ray, the surface normal at each point on the object’s surface, and the visibility of light arriving from any direction at each point on the object’s surface.

#### Surface points

相机光线：$\mathbf r(t)=\mathbf o+t\mathbf d$

NeRF 优化过的体密度：$\sigma$

NeRF 预测的表面位置：
$$
\mathbf x_{\text{surf}}=\mathbf o+\left(\int_{0}^{\infty}T(t)\sigma(\mathbf r(t))tdt\right)\mathbf d\\
T(t)=\exp({-\int_{0}^{t}\sigma(\mathbf r(s))ds})
$$
我们没有保持完整的体积表示，而是将几何形状固定在从优化的 NeRF 中提取的这个表面上。这可以在训练和推理期间实现更高效的重打光，因为我们可以计算出每条相机光线的预期终止位置处的出射辐射，而不是沿着每条相机光线的每个点。
Instead of maintaining a full volumetric representation, we fix the geometry to lie on this surface distilled from the optimized NeRF. This enables much more efficient relighting during both training and inference, because we can compute the outgoing radiance just at each camera ray’s expected termination location instead of every point along each camera ray.

#### Surface normals

表面法向量：$\mathbf n_a(\mathbf x)$，单位化的体密度 $\sigma$ 的梯度的反方向。
$$
\mathbf n_a(\mathbf x)=-\frac{\nabla \sigma(\mathbf x)}{\|\nabla \sigma(\mathbf x)\|}
$$
但是由于 NeRF 的几何重构的不好，噪声很大。我们用 MLP $f_n:x_{\text{surf}}\to\mathbf n$ 重新参数化了这些法向量。它会把表面点 $x_{\text{surf}}$ 映射到一个去噪的法向量 $\mathbf n$ 上。
We re-parameterize these normals using an MLP $f_n$, which maps from any location $x_{\text{surf}}$ on the surface to a "denoised" surface normal $\mathbf n:f_n:x_{\text{surf}}\to\mathbf n$.

我们鼓励此 MLP 的输出 (1) 保持接近由预训练 NeRF 生成的法线，(2) 在 3D 空间中平滑变化，以及 (3) 再现观察到的对象外观。反映 (1) 和 (2) 的损失函数是：
We encourage the output of this MLP (1) to stay close to the normals produced from the pretrained NeRF, (2) to vary smoothly in the 3D space, and (3) to reproduce the observed appearance of the object.
$$
\ell_n=\sum_{x_{\text{surf}}}\left(\frac{\lambda_1}{3}\|f_n(x_{\text{surf}})-n_a(x_{\text{surf}})\|_2^2+\frac{\lambda_2}{3}\|f_n(x_{\text{surf}})-f_n(x_{\text{surf}}+\epsilon)\|_1\right)
$$
$\epsilon$：$x_{\text{surf}}$ 附近随机的 3D 位移，满足 $\mathcal N(0,0.01)$ 分布。

$\lambda_1=0.1,\lambda_2=0.05$。

L1 损失函数是 SmoothL1。

#### Light visibility

穿过从点到每个光源位置的 NeRF 的 $\sigma$-volume ，计算从任何点到每个光源的可见性 $v_a$。
We compute the visibility $v_a$ to each light source from any point by marching through NeRF’s $\sigma$-volume from the point to each light location.

我们通过将可见性函数重新参数化为另一个 MLP 来解决噪声问题，该 MLP 把表面位置 $x_{\text{surf}}$ 和光方向 $\omega_i$ 映射到光可见性 $v:f_v:( x_{\text{surf}}\to v)$。
We address this by re-parameterizing the visibility function as another MLP that maps from a surface location $x_{\text{surf}}$ and a light direction $\omega_i$ to the light visibility $v:f_v:( x_{\text{surf}}\to v)$.

我们鼓励恢复的可见性场 (1) 接近从 NeRF 追踪的可见性，(2) 在空间上平滑，以及 (3) 再现观察到的外观。反映 (1) 和 (2) 的损失函数是：
We encourage the recovered visibility field (1) to be close to the visibility traced from the NeRF, (2) to be spatially smooth, and (3) to reproduce the observed appearance.
$$
\ell_v=\sum_{x_{\text{surf}}}\sum_{\omega_i}\left(\lambda_3[f_v(x_{\text{surf}},\omega_i)-v(x_{\text{surf}},\omega_i)]^2+\lambda_4|f_v(x_{\text{surf}},\omega_i)-f_v(x_{\text{surf}}+\epsilon,\omega_i)|\right)
$$
$\epsilon$：$x_{\text{surf}}$ 附近随机的 3D 位移，满足 $\mathcal N(0,0.01)$ 分布。

$\lambda_3=0.1,\lambda_4=0.05$。

### Reflectance

我们完整的 BRDF 模型 $R$ 由一个完全由反照率 $a$ 确定的漫反射分量（Lambertian）和一个随着空间位置变化的镜面反射 BRDF $f_r$（为表面上的任何位置 $x_{\text{surf}}$ 定义入射光方向 $\omega_i$ 和出射方向 $\omega_o$) 从真实世界的反射中学习：
Our full BRDF model $R$ consist of a diffuse component (Lambertian) fully determined by albedo $a$ and a specular spatially-varying BRDF $f_r$ (defined for any location on the surface $x_{\text{surf}}$ with incoming light direction $\omega_i$ and outgoing direction $\omega_o$) learned from real-world reflectance:
$$
R(x_{\text{surf}},\omega_i,\omega_o)=\frac{a(x_{\text{surf}})}{\pi}+f_r(x_{\text{surf}},\omega_i,\omega_o)
$$
NeRFactor 从学习反射函数开始，该函数经过预训练，可以重现各种经验观察到的真实世界反射函数，同时还学习这些真实世界反射函数的潜在空间。这样，我们在现实世界的 BRDF 上学习了数据驱动的先验，这些先验鼓励优化程序恢复合理的反射函数。
NeRFactor starts with a learned reflectance function that is pretrained to reproduce a wide range of empirically observed real-world reflectance functions, while also learning a latent space for those real-world reflectance functions. By doing so, we learn data-driven priors on real-world BRDFs that encourage the optimization procedure to recover plausible reflectance functions.

#### Albedo

反射率 $a$：表面处的反射率 $f_a:x_{\text{surf}}\to a$。

损失函数：
$$
\ell_a=\frac{\lambda_5}{3}\sum_{x_{\text{surf}}}\|f_a(x_{\text{surf}})-f_a(x_{\text{surf}}+\epsilon)\|_1
$$
$\epsilon$：$x_{\text{surf}}$ 附近随机的 3D 位移，满足 $\mathcal N(0,0.01)$ 分布。

$\lambda_5=0.05$。

最后 $a$ 需要过一个 sigmoid，乘上 0.77 再加 0.03 使得 $a\in[0.03,0.8]$。

#### Learning priors from real-world BRDFs

对于 BRDF 的镜面反射分量，我们学习真实世界 BRDF 的潜在空间和成对的“解码器”，将学习空间 $z_{\text{BRDF}}$ 中的每个潜在编码转换为完整的 4D BRDF .
For the specular components of the BRDF, we seek to learn a latent space of real-world BRDFs and a paired "decoder" that translates each latent code in the learned space $z_{\text{BRDF}}$ to a full 4D BRDF.

模型的 $f_r$ 成分是用 MERL 数据集进行预训练的。
The $f_r$ component of our model is pretrained using the the MERL dataset.

Rusinkiewicz coordinates $g:(\mathbf n,\omega_i,\omega_o)\to(\phi_d,\theta_h,\theta_d)$

Finally, we train a function $f'_r$ (a re-parameterization of $f_r$) that maps from a concatenation of a latent code $z_{\text{BRDF}}$ (which represents a BRDF indentity) and a set of Rusinkiewicz coordinates $(\phi_d,\theta_h,\theta_d)$ to an achromatic reflectance $\mathbf r$:
$$
f'_r:(z_{\text{BRDF}},(\phi_d,\theta_h,\theta_d))\to\mathbf r
$$
因为我们的反射模型的颜色分量被假定由反照率预测网络处理，我们通过将其 RGB 反射值转换为消色差值来丢弃来自 MERL 数据集的所有颜色信息。 
Because the color component of our reflectance model is assumed to be handled by the albedo prediction network, we discard all color information from the MERL dataset by converting its RGB ones reflectance values into achromatic ones.

潜在的 BRDF 身份代码被参数化为无约束的 3D 向量，$z_{\text{BRDF}}$ 被初始化为标准差为 0.01 的零均值各向同性高斯分布。
The latent BRDF identity codes are parameterized as unconstrained *3D vectors*, and $z_{\text{BRDF}}$ are initialized with a zero-mean isotropic Gaussian with a standard deviation of 0.01.

在这个预训练之后，这个 BRDF MLP 的权重参数们在我们整个模型的联合优化过程中被冻结，我们通过从头训练只预测每个 $x_{\text{surf}}$ 的 $z_{\text{BRDF}}$ BRDF 身份的 MLP：$f_z:x_{\text{surf}}\to z_{\text{BRDF}}$。
After this pretraining, the weight of this BRDF MLP are frozen during the joint optimaization of our entire model, and we predict only $z_{\text{BRDF}}$ for each $x_{\text{surf}}$ by training from scratch a BRDF identity MLP: $f_z:x_{\text{surf}}\to z_{\text{BRDF}}$​.
$$
\ell_z=\lambda_6\sum_{x_{\text{surf}}}\frac{\|f_z(x_{\text{surf}})-f_z(x_{\text{surf}}+\epsilon)\|_1}{\dim(z_{\text{BRDF}})}
$$
$\lambda_6=0.01, \dim(z_{\text{BRDF}})=3$ in this model.

最终的 BRDF:
$$
R(x_{\text{surf}},\omega_i,\omega_o)=\frac{f_a(x_{\text{surf}})}{\pi}+f'_r(f_z(x_{\text{surf}}),g(f_n(x_{\text{surf}}),\omega_i,\omega_o))
$$

### Illumination

用于照明的经纬度格式的 HDR 光探测图像。
An HDR light probe image in the latitude-longitude format for lighting.

与球谐函数或球面高斯混合函数相反，这种表示允许我们的模型表示详细的高频照明，因此支持硬投射阴影。
In contrast to spherical harmonics or a mixture of spherical Gaussians, this representation allows our model to represent detailed high frequency lighting and therefore to support hard cast shadows.

根据经验，我们对照明环境使用 16 × 32 分辨率，因为我们不希望在超过该分辨率的光探测图像中恢复更高频率的内容。
Empirically, we use a 16 × 32 resolution for our lighting environments, as we do not expect to recover higherfrequency content in the light probe image beyond that resolution.
$$
\ell_i=\lambda_7\left(\left\|\begin{bmatrix}-1&1\end{bmatrix}*L\right\|_2^2+\left\|\begin{bmatrix}-1\\1\end{bmatrix}*L\right\|_2^2\right)
$$
$*$ 代表卷积操作，$\lambda_7=5\times10^{-6}$。

### Rendering

误差会反向回传到法向量，可见度，反照率，BRDF 和照明上面。

只做直接光照：
$$
\begin{align*}
L_o(\mathbf x,\omega_o)&=\int_{\Omega}R(\mathbf x,\omega_i,\omega_o)L_i(\mathbf x,\omega_i)(\omega_i\cdot\mathbf n(\mathbf x))d\omega_i\\
&=\sum_{\omega_i}R(\mathbf x,\omega_i,\omega_o)L_i(\mathbf x,\omega_i)(\omega_i\cdot f_n(\mathbf x))\Delta\omega_i\\
&=\sum_{\omega_i}\left(\frac{f_a(\mathbf x)}{\pi}+f'_r(f_z(\mathbf x),g(f_n(\mathbf x),\omega_i,\omega_o))\right)L_i(\mathbf x,\omega_i)(\omega_i\cdot f_n(\mathbf x))\Delta\omega_i\\
\end{align*}
$$
$L_i(\mathbf x,\omega_i)$ 已经用可见度 $f_v(\mathbf x,\omega_i)$ 处理过了。

重建的损失函数是 MSE。

因此全部的损失函数是：
$$
\ell_{\text{recon}}+\ell_n+\ell_v+\ell_a+\ell_z+\ell_i
$$

### Implementation Details

Adam optimizer with the default hyperparameters.

#### Staged training

首先，我们使用输入的姿势图像（每个场景一次）优化 NeRF，并在 MERL 数据集上训练 BRDF MLP（所有场景仅一次）。这两个 MLP 在最终联合优化期间都被冻结。
First, we optimize a NeRF using the input posed images (once per scene) and train a BRDF MLP on the MERL dataset (only once for all scenes). Both of these MLPs are frozen during the final joint optimization.

其次，我们使用这个经过训练的 NeRF 来初始化我们的几何，通过优化法线和可见性 MLP 来简单地重现 NeRF 值，而没有任何额外的平滑度损失或正则化。
Second, we use this trained NeRF to initialize our geometry by optimizing the normal and visibility MLPs to simply reproduce the NeRF values, without any additional smoothness loss or regularization.

最后，我们从头开始联合优化反照率 MLP、BRDF 身份 MLP 和光探针像素，以及预训练的法向和可见性 MLP。
Finally, we jointly optimize the albedo MLP, BRDF identity MLP, and light probe pixels from scratch, along with the pretrained normal and visibility MLPs.

#### Architecture and positional encoding

我们使用 NeRF 的默认架构，我们引入的所有其他 MLP 都包含四层，每层有 128 个隐藏单元。我们将位置编码应用于所有网络的输入坐标，其中 10 级编码用于 3D 位置，4 级编码用于方向。
We use the default architecture for NeRF, and all other MLPs that we introduce contain four layers, each with 128 hidden units. We apply positional encoding to the input coordinates of all networks with 10 encoding levels for 3D locations and 4 encoding levels for directions.

#### Runtime

We train NeRF for 2,000 epochs. Geometry pretraining is performed for 200 epochs. Final joint optimization is performed for 100 epochs.













