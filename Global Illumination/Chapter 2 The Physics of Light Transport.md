# Chapter 2 The Physics of Light Transport

## 2.1 Brief History

## 2.2 Models of Light

## 2.3 Radiometry

### 2.3.1 Radiometric Quantities

**光通量**：单位时间内流过表面的总能量，符号 $\Phi$，单位 $W$。
**Flux**: The total energy flows through a surface per unit time, symbol $\Phi$, unit $W$.

**辐照度**：单位表面积的*入射*辐射功率，符号 $E$，单位 $W/m^2$。
**Irradiance**: The *incident* radiant power on a surface, per unit surface area, symbol $E$, unit $W/m^2$. 
$$
E=\frac{\mathrm{d}\Phi}{\mathrm{d}A}
$$
**辐射度**：单位表面积的*出射*辐射功率，符号 $B$，单位 $W/m^2$。
**Radiosity**: The *exitant* radiant power per unit surface area, symbol $B$, unit $W/m^2$.
$$
B=\frac{\mathrm{d}\Phi}{\mathrm{d}A}
$$
**辐射**：从所有方向到达或离开表面某一点每单位立体角每单位投影面积的功率，符号 $L$，单位 $W/(m^2\cdot\text{rad})$
**Radiance**: All the direction of power arrives at or leaves from a certain point on surface, per unit solid angle, per unit projected area, symbol $L$, unit $W/(m^2\cdot\text{rad})$.
$$
L=\frac{\mathrm{d}^2\Phi}{\mathrm{d}\omega\mathrm{d}A^{\perp}}=\frac{\mathrm{d}^2\Phi}{\mathrm{d}\omega\mathrm{d}A\cos\theta}
$$
$\mathrm{d}A$ 是到达或离开表面某一点附近的面积，$\mathrm{d}\omega$ 是光源或照射面在某一点处的立体角微元。

辐射度是一个随位置 $x$ 和方向向量 $\Theta$ 而变化的五维量，表示为 $L(x,\Theta)$。
Radiance is a five-dimensional quantity that varies with position $x$ and direction vector $\Theta$, and is expressed as $L(x,\Theta)$.

#### Transport Theory

扩散理论处理物理量（例如能量、电荷和质量）的传输或流动。
Transport theory deals with the transport or flow of physical quantities such as energy, charge, and mass.

已知 $p(x)$ 是空间中某处的光子密度值，因此在微小空间内光子的数量是 $p(x)\mathrm{d}V$。考虑在微小时间 $\mathrm{d}t$ 内光子穿过了可微面 $\mathrm{d}A$，穿过方向和可微面的法向夹角为 $\theta$。已知光速为 $c$，则穿过可微面的粒子数为：
$$
p(x)(c\mathrm{d}t)\mathrm{d}A\cos\theta
$$
考虑所有入射方向 $\omega$ 和入射的波长 $\lambda$，粒子数目就表示为：
$$
N=p(x, \omega, \lambda)c\mathrm{d}t\mathrm{d}A\cos\theta\mathrm{d}\omega\mathrm{d}\lambda
$$
因此光通量就：
$$
\begin{align*}
\Phi&\propto p(x, \omega, \lambda)\mathrm{d}A\cos\theta\mathrm{d}\omega\mathrm{d}\lambda\\
\frac{\Phi}{\mathrm{d}A\cos\theta\mathrm{d}\omega}&\propto p(x, \omega, \lambda)\mathrm{d}\lambda
\end{align*}
$$
带入量子力学中单光子的性质 $E=h\nu,c=\lambda\nu$，则有：
$$
L(x,\omega)=\int p(x, \omega, \lambda)h\frac{c}{\lambda}\mathrm{d}\lambda
$$
由此，$\cos\theta$ 和辐射的关系可以有比较清晰的认识。

### 2.3.2 Relationships between Radiometric Quantities

设 $L(x\rightarrow\Theta)$ 表示从 $\Theta$ 方向上离开点 $x$ 的辐射，而 $L(x\leftarrow\Theta)$ 表示从方向 $\Theta$ 到达点 $x$ 的辐射：
Let $L(x\rightarrow\Theta)$ represents radiance leaving point $x$ in direction $\Theta$, and $L(x\leftarrow\Theta)$ represents radiance arriving at point $x$ from direction $\Theta$. Then
$$
\begin{align*}
\Phi&=\int_{A}\int_{\Omega}L(x\rightarrow\Theta)\cos\theta\mathrm{d}\omega_{\Theta}\mathrm{d}A_{x}\\
E&=\int_{\Omega}L(x\leftarrow\Theta)\cos\theta\mathrm{d}\omega_{\Theta}\\
B&=\int_{\Omega}L(x\rightarrow\Theta)\cos\theta\mathrm{d}\omega_{\Theta}\\
\end{align*}
$$
辐射度是通过对可见光的波长域上的光谱辐射积分来计算的：
Radiance is computed by integrating spectral radiance over the wavelength domain covering visible light. That is
$$
L(x\rightarrow\Theta)=\int_{\text{spectrum}}L(x\rightarrow\Theta,\lambda)\mathrm{d}\lambda
$$

### 2.3.3 Properties of Radiance

#### Radiance is invariant along straight paths

此属性假设光在真空中传播。
This property assumes that light is traveling through a vacuum.
$$
L(x\to y)=L(y\gets x)
$$
$L(x\to y)$ 表示为：
$$
L(x\to y)=\frac{\mathrm{d}^2\Phi}{(\cos\theta_x\mathrm{d}A_x)\mathrm{d}\omega_{x\gets\mathrm{d}A_y}}
$$
而 $L(y\gets x)$ 表示为：
$$
L(y\gets x)=\frac{\mathrm{d}^2\Phi}{(\cos\theta_y\mathrm{d}A_y)\mathrm{d}\omega_{y\gets\mathrm{d}A_x}}
$$
其中 $\theta_{x},\theta_{y}$ 分别代表 $N_x,N_y$ 与 $xy$ 连线的夹角。

由于：
$$
\mathrm{d}\omega_{x\gets\mathrm{d}A_y}=\frac{\cos\theta_y\mathrm{d}A_y}{r^2_{xy}}\\
\mathrm{d}\omega_{y\gets\mathrm{d}A_x}=\frac{\cos\theta_x\mathrm{d}A_x}{r^2_{xy}}
$$
因此带入可得到 $L(x\to y)=L(y\gets x)$。

#### Sensors, such as cameras and the human eye, are sensitive to radiance.

传感器（例如，相机或人眼）的响应与入射的辐射成正比，其中比例常数取决于传感器的几何形状。
The response of sensors (for example, cameras or the human eye) is proportional to the radiance incident upon them, where the constant of proportionality depends on the geometry of the sensor.

### 2.3.4 Examples 太阳到地球的辐照度

Diffuse Emitter

漫反射发射器从其所有表面点向所有方向发射相等的辐射
A diffuse emitter emits equal radiance in all directions from all its surface points:
$$
L(x\to\Theta)=L
$$

$$
\begin{align*}
\Phi&=\int_{A}\int_{\Omega}L(x\rightarrow\Theta)\cos\theta\mathrm{d}\omega_{\Theta}\mathrm{d}A_{x}\\
&=\int_{A}\int_{\Omega}L\cos\theta\mathrm{d}\omega_{\Theta}\mathrm{d}A_{x}\\
&=L\int_{A}\mathrm{d}A_{x}\int_{\Omega}\cos\theta\mathrm{d}\omega_{\Theta}\\
&=\pi LA\\
\end{align*}
$$

假设太阳是各向均匀的辐射，已知太阳的功率是 $\Phi=3.91\times10^{26}W$，太阳半径 $r=6.96\times10^{8}m$，则太阳的辐射为：
$$
L=\frac{\Phi}{\pi A_{\text{sun}}}=\frac{\Phi}{4\pi^2r^2}=\frac{3.91\times10^{26}W}{4\pi^2(6.96\times10^{8})^2m^2}=2.04\times10^{7}W/(m^2\cdot\text{sr})
$$
假设太阳在天顶，已知日地距离 $d=1.496\times10^{11}m$ 对于地球上的一块单位面积来说，从这块区域看，太阳的立体角为：
$$
\omega=\frac{A_{\text{plane}}}{d^2}=\frac{\pi r^2}{d^2}=6.80\times10^{-5}\text{sr}
$$
因此辐照度为：
$$
E=L\omega=2.04\times10^{7}W/(m^2\cdot\text{sr})\times6.80\times10^{-5}\text{sr}=1.39\times10^{3}W/m^2
$$

## 2.4 Light Emission

精确全局照明的计算需要为每个光源指定以下三种分布：空间、方向和光谱强度分布。
The computation of accurate global illumination requires the specification of the following three distributions for each light source: spatial, directional, and spectral intensity distribution.

## 2.5 Interaction of Light with Surfaces

<u>我们假设入射到表面的光以相同的波长和相同的时间出射。</u>
<u>We assume that light incident at a surface exits at the same wavelength and same time.</u>

### 2.5.1 Bidirectional Reflectance Distribution Function BRDF

在最一般的情况下，光可以从入射方向 $\Psi$ 进入表面上的点 $\mathbf p$，并且可以从出射方向 $\Theta$ 离开表面上的点 $\mathbf q$ 。双向表面散射反射率分布函数。
In the most general case, light can enter some surface at a point $\mathbf p$ and incident direction $\Psi$ and can leave the surface at some other point $\mathbf q$ and exitant direction $\Theta$. Bidirectional surface scattering reflectance distribution function **BSSRDF**.

点 $x$ 处的 BRDF 定义为从出射方向 $\Theta$ 上反射的辐射微分与通过立体角微分 $\mathrm{d}\omega_{\Psi}$ 入射的辐射微分之比：
The BRDF at a point $x$ is defined as the ratio of the differential radiance reflected in an exitant direction $\Theta$, and the differential irradiance incident through a differential solid angle $\mathrm{d}\omega_{\Psi}$, the BRDF is:
$$
\begin{align*}
f_r(x, \Psi\rightarrow\Theta)&=\frac{\mathrm{d}L(x\rightarrow\Theta)}{\mathrm{d}E(x\leftarrow\Psi)}\\
&=\frac{\mathrm{d}L(x\rightarrow\Theta)}{L(x\leftarrow\Psi)\cos(N_{x},\Psi)\mathrm{d}\omega_{\Psi}}
\end{align*}
$$

$N_{x}$ 是 $x$ 处的法向量。

<u>BRDF 的意义是衡量了表面某处将一束入射光线散射到整个空间的能力。</u>

### 2.5.2 Properties of the BRDF

#### Range

BRDF 可以取任何正值并且可以随波长变化。
The BRDF can take any positive value and can vary with wavelength.

#### Dimension

BRDF 是定义在表面上每个点的四维函数；二维对应入射方向，二维对应出射方向。
The BRDF is a four-dimensional function defined at each point on a surface; two dimensions correspond to the incoming direction, and two dimensions correspond to the outgoing direction.

BRDF 一般来说是各向异性的 (anisotropic)。也就是说，如果表面围绕表面法线旋转，$f_r$ 的值会发生变化。
That is, if the surface is rotated about the surface normal, the value of $f_r$ will change.

#### Reciprocity

$$
f_r(x, \Psi\rightarrow\Theta)=f_r(x, \Theta\rightarrow\Psi)
$$

#### Relation between incident and reflected radiance

因此，BRDF 在所有入射方向上表现为线性函数。
Therefore, the BRDF behaves as a linear function with respect to all incident directions.
$$
\begin{align*}
\mathrm{d}L(x\to\Theta)&=f_r(x, \Psi\rightarrow\Theta)\mathrm{d}E(x\gets\Psi)\\
L(x\to\Theta)&=\int_{\Omega}f_r(x, \Psi\rightarrow\Theta)L(x\gets\Psi)\cos(N_{x},\Psi)\mathrm{d}\omega_{\Psi}\\
\end{align*}
$$

#### Energy conservation

能量守恒定律要求所有方向反射的总能量必须小于或等于入射到表面的总能量（多余的能量转化为热或其他形式的能量）。
The law of conservation of energy requires that the total amount of power reflected over all directions must be less than or equal to the total amount of power incident on the surface (excess power is transformed into heat or other forms of energy).

辐照度：
$$
E=\int_{\Omega_x}L(x\gets\Psi)\cos(N_{x},\Psi)\mathrm{d}\omega_{\Psi}
$$
辐射度：
$$
B=\int_{\Omega_x}L(x\to\Theta)\cos(N_{x},\Theta)\mathrm{d}\omega_{\Theta}
$$
由 BRDF 可知：
$$
f_r(x, \Psi\rightarrow\Theta)L(x\leftarrow\Psi)\cos(N_{x},\Psi)\mathrm{d}\omega_{\Psi}=\mathrm{d}L(x\rightarrow\Theta)
$$
因此辐射度可以表示为：
$$
B=\int_{\Omega_x}\int_{\Omega_x}f_r(x, \Psi\rightarrow\Theta)L(x\leftarrow\Psi)\cos(N_{x},\Psi)\mathrm{d}\omega_{\Psi}\cos(N_{x},\Theta)\mathrm{d}\omega_{\Theta}
$$
由 $B\le E$ 可得：
$$
\frac{\int_{\Omega_x}\int_{\Omega_x}f_r(x, \Psi\rightarrow\Theta)L(x\leftarrow\Psi)\cos(N_{x},\Psi)\mathrm{d}\omega_{\Psi}\cos(N_{x},\Theta)\mathrm{d}\omega_{\Theta}}{\int_{\Omega_x}L(x\gets\Psi)\cos(N_{x},\Psi)\mathrm{d}\omega_{\Psi}}\le1
$$

### 2.5.3 BRDF Examples

#### Diffuse Surfaces

$$
f_r(x, \Psi\rightarrow\Theta)=c=\frac{\rho_d}{\pi}
$$

$\rho_d$ 是反照率。

#### Specular Surface

##### Specular Reflection

反射光方向 $R=2(N_{x}\cdot\Psi)N_{x}-\Psi$
$$
f_r(x, \Psi\rightarrow\Theta)\propto\delta(R-\Theta)
$$

##### Specular Refraction

Snell's law:
$$
\eta_1\sin\theta_1=\eta_2\sin\theta_2
$$
折射光方向：
$$
T=-\frac{\eta_1}{\eta_2}\Psi+N_{x}\left(\frac{\eta_1}{\eta_2}(N_{x}\cdot\Psi)-\sqrt{1-\left(\frac{\eta_1}{\eta_2}\right)^2(1-(N_{x}\cdot\Psi)^2)}\right)
$$

##### Reciprocity for transparent surfaces

当一束光从光疏介质进入光密介质时，它会被压缩。因此，垂直于折射方向的单位面积光能变高。
When a pencil of light enters a dense medium from a less dense medium, it gets compressed. Therefore, the light energy per unit area perpendicular to the pencil direction becomes higher.

光线密度的变化是介质折射率的平方比：$(\eta_2/\eta_1)^2$。
The change in ray density is the square ratio of the refractive indices of the media: $(\eta_2/\eta_1)^2$.

##### Fresnel Equations

当光线照射到一个完全光滑的表面时，反射的光能取决于光的波长、表面的几何形状以及光的入射方向。
When light hits a perfectly smooth surface, the light energy that is reflected depends on the wavelength of light, the geometry at the surface, and the incident direction of the light.

偏振光的两个分量，$r_p$ 和 $r_s$，分别指的是平行和垂直分量。
The two components of the polarized light, $r_p$ and $r_s$, referring to the parallel and perpendicular.
$$
r_p=\frac{\eta_2\cos\theta_1-\eta_1\cos\theta_2}{\eta_2\cos\theta_1+\eta_1\cos\theta_2}\\
r_p=\frac{\eta_1\cos\theta_1-\eta_2\cos\theta_2}{\eta_1\cos\theta_1+\eta_2\cos\theta_2}\\
$$

#### Glossy Surfaces

大多数表面既不是理想的漫反射也不是理想的镜面反射，而是表现出两种反射行为的组合；这些表面称为光泽表面。
Most surfaces are neither ideally diffuse nor ideally specular but exhibit a combination of both reflectance behaviors; these surfaces are called glossy surfaces.

### 2.5.4 Shading Models

##### Lambert’s model

The BRDF is a constant.

##### Phong model

反射光方向 $R=2(N_{x}\cdot\Psi)N_{x}-\Psi$
$$
f_r(x, \Psi\rightarrow\Theta)=\underbrace{k_s\frac{(R\cdot\Theta)^n}{N_{x}\cdot\Psi}}_{\text{Specular}}+\underbrace{k_d}_{\text{Diffuse}}
$$

##### Blinn-Phong model

中间向量 $H=(\Theta+\Psi)/2$
$$
f_r(x, \Psi\rightarrow\Theta)=\underbrace{k_s\frac{(N_{x}\cdot H)^n}{N_{x}\cdot\Psi}}_{\text{Specular}}+\underbrace{k_d}_{\text{Diffuse}}
$$
它不是能量守恒的，它不满足亥姆霍兹的互易性，也不能捕捉到大多数真实材料的行为。
It is not energy conserving, it does not satisfy Helmholtz’s reciprocity, and it does not capture the behavior of most real materials.

##### Modified Blinn-Phong model

$$
f_r(x, \Psi\rightarrow\Theta)=\underbrace{k_s(N_{x}\cdot H)^n}_{\text{Specular}}+\underbrace{k_d}_{\text{Diffuse}}
$$

#### Physically Based Shading Models

##### Cook-Torrance model

|    Symbols    |                   Descriptions                   |
| :-----------: | :----------------------------------------------: |
|  $F(\beta)$   |   Fresnel reflectance, $F=(|r_p|^2+|r_s|^2)/2$   |
|    $\beta$    | the angle between $\Psi$ and $H=(\Psi+\Theta)/2$ |
| $D(\theta_h)$ |           the microfacet distribution            |
|  $\theta_h$   |  the angle between $N$ and $H=(\Psi+\Theta)/2$   |
|      $G$      |            a geometric shadowing term            |

最常见的分布之一是 Beckmann 的分布：
One of the most common distributions is the distribution by Beckmann:
$$
D(\theta_h)=\frac{1}{m^2\cos^4\theta_h}\exp\left(-\left(\frac{\tan\theta_h}{m}\right)^2\right)
$$
$m$ 是微表面的均方根斜率，它表示表面粗糙度。
$m$ is the root-mean-square slope of the microfacets, and it captures surface roughness.
$$
G = \min\left\{1,\frac{2(N\cdot H)(N\cdot\Theta)}{\Theta\cdot H},\frac{2(N\cdot H)(N\cdot\Psi)}{\Theta\cdot H}\right\}
$$
Cook-Torrance 模型包括一个微表面模型，该模型假设一个表面是由一些小的光滑平面小平面的随机集合组成的。
The Cook-Torrance model includes a microfacet model that assumes that a surface is made of a random collection of small smooth planar facets.
$$
f_r(x, \Psi\rightarrow\Theta)=\frac{F(\beta)}{\pi}\frac{D(\theta_h)G}{(N\cdot\Psi)(N\cdot\Theta)}+k_d
$$

#### Empirical Models

| Symbols  |            Descriptions            |
| :------: | :--------------------------------: |
| $\rho_d$ |      the diffuse reflectance       |
| $\rho_s$ |      the specular reflectance      |
| $\alpha$ | a measure of the surface roughness |

$$
f_r(x, \Psi\rightarrow\Theta)=\frac{\rho_d}{\pi}+\rho_s\frac{\exp\left(-\tan^2\theta_h/\alpha^2\right)}{4\pi\alpha^2\sqrt{(N\cdot\Psi)(N\cdot\Theta)}}
$$

## 2.6 Rendering Equation

### 2.6.1 Hemispherical Formulation

我们假设 $L_e(x\to\Theta)$ 表示表面在 $x$ 处从 $\Theta$ 方向向外上**发出**的辐射，而 $L_r(x\to\Theta )$ 表示在 $\Theta$ 方向上由表面**反射**的辐射。
Let us assume that $L_e(x\to\Theta)$ represents the radiance **emitted** by the surface at $x$ and in the outgoing direction $\Theta$, and $L_r(x\to\Theta)$ represents the radiance that is **reflected** by the surface at in that direction $\Theta$.
$$
L(x\to\Theta)=L_e(x\to\Theta)+L_r(x\to\Theta)\\
L(x\to\Theta)=L_e(x\to\Theta)+\int_{\Omega_{x}}f_r(x, \Psi\rightarrow\Theta)L(x\gets\Psi)\cos(N_{x},\Psi)\mathrm{d}\omega_{\Psi}
$$

### 2.6.2 Area Formulation

|   Symbols   |                         Descriptions                         |
| :---------: | :----------------------------------------------------------: |
|     $A$     |                the collection of all surfaces                |
|   $\tau$    | $\tau=\min\{t:t>0,x+t\Psi\in A\}$ the closest intersetion point |
| $r(x,\Psi)$ |  $r(x,\Psi)=\{y:y=x+\tau\Psi\}$ the *ray-casting* operation  |
|  $V(x,y)$   |              the visibility between two points               |

可见性函数是使用光线投射操作计算的：如果存在一些 $\Psi$ 使得 $r(x,\Psi)=y$，则 $x$ 和 $y$ 是相互可见的。
The visibility function is computed using the ray-casting operation: $x$ and $y$ are mutually visible if there exists some $\Psi$ such that $r(x,\Psi)=y$.

则从曲面 $A$ 上的一点 $y$ 到 $x$ 上的渲染方程可以改写为：
$$
\begin{align*}
L(x\to\Theta)&=L_e(x\to\Theta)+\int_{\Omega_{x}}f_r(x, \Psi\rightarrow\Theta)L(x\gets\Psi)\cos(N_{x},\Psi)\mathrm{d}\omega_{\Psi}\\
&=L_e(x\to\Theta)\\
&+\int_{A}f_r(x,\Psi\rightarrow\Theta)L(y\to-\Psi)V(x,y)\frac{\cos(N_{x},\Psi)\cos(N_y,-\Psi)}{r^2_{xy}}\mathrm{d}A_y\\
\end{align*}
$$

### 2.6.3 Direct and Indirect Illumination Formulation

直接照明是直接从场景中的光源到达表面的照明；间接照明是在从场景中的另一个表面反射至少一次后到达的光。
Direct illumination is the illumination that arrives at a surface directly from the light sources in a scene; indirect illumination is the light that arrives after bouncing at least once off another surface in the scene.
$$
\begin{align*}
L_{r}(x\to\Theta)&=L_{\text{direct}}+L_{\text{indirect}}\\
L_{\text{direct}}&=\int_{A}f_r(x,\overrightarrow{xy}\rightarrow\Theta)L(y\to\overrightarrow{yx})V(x,y)\frac{\cos(N_{x},\Psi)\cos(N_y,-\Psi)}{r^2_{xy}}\mathrm{d}A_y\\
L_{\text{indirect}}&=\int_{\Omega_x}f_r(x,\Psi\to\Theta)L_r(r(x,\Psi)\to-\Psi)\cos(N_{x},\Psi)\mathrm{d}\omega_{\Psi}\\
\end{align*}
$$
$L_{\text{direct}}$ 是 $y$ 直接照射到 $x$ 的辐射。

$L_{\text{indirect}}$ 是各个方向的环境光。

## 2.7 Importance

响应函数：每个像素都充当传感器，并了解它如何响应落在传感器上的光能。
Response function: each pixel functions as a sensor with some notion of how it responds to the light energy that falls on the sensor.
$$
W(x\to\Theta)=W_e(x\to\Theta)+\int_{\Omega_x}{\color{red}f_r(x,\Psi\gets\Theta)}W(x\gets\Psi)\cos(N_x,\Psi)\mathrm{d}\omega_{\Psi}
$$
如果在特定图像中表面 $i$ 是肉眼可见的，则 $W_e(i)$ 将捕获表面对图像的重要程度（图像上表面投影面积的某种度量）。如果在图像中表面 $j$ 也是可见的，并且表面 $i$ 将光反射到表面 $j$，那么由于 $j$ 的重要性，$i$ 将间接地更加重要。因此，当能量从 $i$ 流向$j$ 时，重要性从 $j$ 流向 $i$。
If surface $i$ is visible to the eye in a particular image, then $W_e(i)$ will capture the extent to which the surface is important to the image (some measure of the projected area of the surface on the image). If surface $j$ is also visible in an image and surface $i$ reflects light to surface $j$, then, due to the importance of $j$, $i$ will indirectly be even more important. Thus, while energy flows from $i$ to $j$, importance flows from $j$ to $i$.

## 2.8 The Measurement Equation

渲染方程描述了场景中光能的稳态分布。重要性方程描述了表面对图像的相对重要性。测量方程描述了全局照明算法必须解决的问题。
The rendering equation formulates the steady-state distribution of light energy in the scene. The importance equation formulates the relative importanceof surfaces to the image. The measurement equation formulates the problem that a global illumination algorithm must solve.

对于图像中的每个像素 $j$，$M_j$ 表示通过该像素 $j$ 的辐射度测量值。
For each pixel $j$ in an image, $M_j$ represents the measurement of radiance through that pixel $j$​.
$$
M_j=\int W(x\gets\Psi)L(x\gets\Psi)\cos(N_x,\Psi)\mathrm{d}A_x\mathrm{d}_{\omega_{\Psi}}
$$
我们在这里假设传感器是场景的一部分，以便我们可以在它们的表面上积分。
We assume here that the sensors are part of the scene so that we can integrate over their surface.

