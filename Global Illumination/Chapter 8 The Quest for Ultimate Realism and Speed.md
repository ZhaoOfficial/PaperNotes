# Chapter 8 The Quest for Ultimate Realism and Speed

在之前的章节里，我们假设可以忽略波效应，并且在相互可见的表面之间的路径上辐射度是守恒的。我们还假设光散射是瞬间发生的；散射光与入射光束具有相同的波长；并且它从撞击表面的同一位置散射。
In the previous chapters, we assumed that wave effects could be ignored and that radiance is conserved along its path between mutually visible surfaces. We also assumed that light scattering happens instantaneously; that scattered light has the same wavelength as the incident beam; and that it scatters from the same location where it hits a surface.

## 8.1 Beyond the Rendering Equation

### 8.1.1 Participating Media

如果真空没有填充物体表面之间的空间，这将导致光子改变方向并转化为其他形式的能量。在蜡烛火焰的情况下，其他形式的能量也转化为可见光光子。
If a vacuum is not filling the space between object surfaces, this will cause photons to change direction and to transform into other forms of energy. In the case of the candle flame, other forms of energy are also transformed into visible light photons.

### 8.1.2 Volume Emission

介质（如火）发光的强度可以通过体积发射函数 $\epsilon(z)$（单位 [$W/m^3$]）来表征。它基本上告诉我们每单位体积和每单位时间在三维空间中的某个点 $z$ 发射了多少光子。
The intensity by which a medium, like fire, glows can be characterized by a volume emittance function $\epsilon(z)$ (units [$W/m^3$]). It tells us basically how many photons per unit of volume and per unit of time are emitted at a point in $z$ three-dimensional space.

通常，体积发射是各向同性的，这意味着在 $z$ 周围的任何方向发射的光子数等于 $\epsilon(z)/4\pi$（单位 [$W/m^3\cdot\text{sr }$]）。
Usually, volume emission is isotropic, meaning that the number of photons emitted in any direction around $z$ is equal to $\epsilon(z)/4\pi$ (units [$W/m^3\cdot\text{sr}$]).

在 $z$ 处无限小厚度 $\mathrm{d}s$ 的切片中的体积发射给 $\Theta$ 方向增加的辐射为：
The radiance added along direction $\Theta$ due to volume emission in a pencil slice of infinitesimal thickness $\mathrm{d}s$ at $z$ is:
$$
\mathrm{d}L_e(z\to\Theta)=\frac{\epsilon(z)}{4\pi}\mathrm{d}s
$$

### 8.1.3 Absorption

光子沿其传播方向每单位距离被体积吸收的概率称为吸收系数 $\sigma_{a}(z)$（单位 [$m^{-1}$]）。
The probability that a photon gets absorbed in a volume, per unit of distance along its direction of propagation, is called the absorption coefficie $\sigma_{a}(z)$ (units [$m^{-1}$]).

此外，吸收通常是各向同性的：一个光子无论其飞行方向如何，都有相同的机会被吸收。这对于单个粒子的吸收很少是正确的，但在大多数介质中，粒子是随机定向的，因此可以观察到它们的平均定向吸收（以及散射）特性。
In addition, absorption is usually isotropic: a photon has the same chance of being absorbed regardless of its direction of flight. This is rarely true for ab sorptionby a single particle, but in most media, particles are randomly oriented so that their average directional absorption (and also scattering) characteristics are observed.

Ray from $x$ to $y$, direction $\Theta$, $z=x+s\Theta$.
$$
\begin{align*}
L(z+\Delta{s}\cdot\Theta\to\Theta)&=L(z\to\Theta)-L(z\to\Theta)\sigma_a(z)\Delta{s}\\
\frac{\mathrm{d}L(z\to\Theta)}{\mathrm{d}s}&=-L(z\to\Theta)\sigma_a(z)\\
L(z\to\Theta)&=L(x\to\Theta)\exp\left(-\int_0^s\sigma_a(x+t\Theta)\mathrm{d}t\right)
\end{align*}
$$

### 8.1.4 Out-Scattering, Extinction Coefficient, and Albedo

Out-scattering: scattering coefficient $\sigma_s(z)$ (units [$m^{-1}$]).

消光系数 $\sigma_t(z)=\sigma_a(z)+\sigma_s(z)$ (units [$m^{-1}$]) 给出了光子沿着飞行路径的每单位距离与介质碰撞（吸收或散射）的概率。
The extinction coefficiet $\sigma_t(z)=\sigma_a(z)+\sigma_s(z)$ (units [$m^{-1}$]) gives us the probability per unit distance along the path of flight that a photon collides (absorbs or scatters) with the medium.
$$
L(z\to\Theta)=L(x\to\Theta)\exp\left(-\int_0^s\sigma_t(x+t\Theta)\mathrm{d}t\right)
$$

在均匀介质中，两个后续碰撞之间的平均距离可以显示为 $1/\sigma_t$（单位 [$m$]）。后续碰撞之间的平均距离称为平均自由程。
In a homogeneous medium, the average distance between two subsequent collisions can be shown to be $1/\sigma_t$ (units [$m$]). The average distance between subsequent collisions is called the mean free path.

反照率 $\alpha(z)=\sigma_s(z)/\sigma_t(z)$（无量纲）描述了散射与吸收的相对重要性。
The albedo $\alpha(z)=\sigma_s(z)/\sigma_t(z)$ (dimensionless) describes the relative importance of scattering versus absorption.

### 8.1.5 In-Scattering, Field- and Volume-Radiance, and the Phase Function

在 $z$ 位置进行内散射的第一个条件是在 $z$ 处完全存在散射，换句话说，$\sigma_s(z)=\alpha(z)\sigma_t(t)\ne0$。
A first condition for in-scattering at a location $z$ is that there is scattering at $z$ at all, in other words, that $\sigma_s(z)=\alpha(z)\sigma_t(t)\ne0$.

场辐射与消光系数 $L_v(z,\Psi)=L(z,\Psi)\sigma_t(z)$ 的乘积描述了每单位时间在 $z$ 处与介质碰撞的光子数。
The product of field radiance with the extinction coefficient $L_v(z,\Psi)=L(z,\Psi)\sigma_t(z)$ describes the number of photons entering collisions with the medium at $z$ per unit of time.

$z$ 处的相位函数 $p(z,\Psi\leftrightarrow\Theta)$（单位 [$\mathrm{sr}^{-1}$]）描述了从方向 $\Psi$ 散射到 $\Theta$ 的概率。通常，相位函数仅取决于 $\Psi$ 和 $\Theta$ 两个方向之间的夹角。
The phase function $p(z,\Psi\leftrightarrow\Theta)$ at $z$ (units [$\mathrm{sr}^{-1}$]) describes the probability of scattering from direction $\Psi$ into $\Theta$. Usually, the phase function only depends on the angle between the two directions $\Psi$ and $\Theta$.

volume scattering equation:
$$
\begin{align*}
\mathrm{d}L(z\to\Theta)&=\int_{\mathcal{S}^2}\alpha(z)p(z,\Psi\leftrightarrow\Theta)\sigma_{t}(z)L(z\to\Psi)\mathrm{d}\omega_{\Psi}\mathrm{d}t\\
&=\sigma_{s}(z)\int_{\mathcal{S}^2}p(z,\Psi\leftrightarrow\Theta)L(z\to\Psi)\mathrm{d}\omega_{\Psi}\mathrm{d}t
\end{align*}
$$
