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











