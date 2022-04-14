# Confocal non-line-of-sight imaging based on the light-cone transform

## Abstract

由于现有重建算法的内存和处理要求过高，以及多重散射光的极弱信号，NLOS 成像仍然不切实际。
NLOS imaging has remained impractical owing to the prohibitive memory and processing requirements of existing reconstruction algorithms, and the extremely weak signal of multiply scattered light.

共聚焦扫描程序可以通过促进光锥变换的推导来解决这些挑战，以解决 NLOS 重建问题。
A confocal scanning procedure can address these challenges by facilitating the derivation of the light-cone transform to solve the NLOS reconstruction problem.

## Introduction

首先，就内存要求和处理周期而言，重建步骤的计算要求极高。其次，多重散射光的通量极低，需要在黑暗环境中进行大量采集，或者需要足够高功率的激光器来克服环境光的影响。最后，NLOS 成像通常需要由昂贵组件制成的定制硬件系统，从而阻碍了其广泛使用。
First, the reconstruction step is prohibitively computationally demanding, in terms of both memory requirements and processing cycles. Second, the flux of multiply scattered light is extremely low, requiring either extensive acquisition times in dark environments or a sufficiently high-power laser to overcome the contribution of ambient light. Finally, NLOS imaging often requires a custom hardware system made with expensive components, thus preventing its widespread use.

首先，它有助于推导出 NLOS 问题的闭式解。
First, it facilitates the derivation of a closed-form solution to the NLOS problem.

其次，虽然间接散射光对于漫射物体仍然非常微弱，但反光物体通过将光反射回其源头而散射最小，从而大大增加了间接信号。
Second, whereas indirectly scattered light remains extremely weak for diffuse objects, retro-reflective objects considerably increase the indirect signal by reflecting light back to its source with minimal scattering.

第三，商业激光雷达系统可能能够以最少的硬件修改支持这里开发的算法。
Third, commercial LIDAR systems may be capable of supporting the algorithms developed here with minimal hardware modifications.

假设：墙后只有单次散射（即场景隐藏部分没有相互反射），光各向同性散射（即模型忽略兰伯特余弦项），隐藏场景内不发生遮挡。
Assumptions: there is only single scattering behind the wall (that is, no inter-reflections in the hidden part of the scene), light scatters isotropically (that is, the model ignores Lambert’s cosine terms), and no occlusions occur within the hidden scene.

$x',y'$ 是在 $z'=0$ 墙上的共焦扫描点。

$\tau$ 是在点 $(x',y')$ 和时间 $t$ 时刻的 3D 体积测量量。

$c$ 是光速。

$\rho(x,y,z)$ 是隐藏物体的反射率。

$r=\sqrt{(x'-x)^2+(y'-y)^2+z^2}=tc/2$。

$\delta(2\sqrt{(x'-x)^2+(y'-y)^2+z^2}-tc)$ 是一个四维时空的光锥，建模了光的墙-物体-墙的传播过程。
$$
\tau(x',y',t)=\iiint_{\Omega}\frac{1}{r^4}\rho(x,y,z)\delta(2\sqrt{(x'-x)^2+(y'-y)^2+z^2}-tc)dxdydz
$$
我们注意到这个 $\delta$ 函数在 x y 轴上具有平移不变性，但是在 z 轴上没有
We note that the function is shift-invariant in the x and y axes, but not in the
z axis.
$$
\delta(2\sqrt{(x'-x)^2+(y'-y)^2+z^2}-tc)=\frac{1}{r}\delta((x'-x)^2+(y'-y)^2+z^2-(tc/2)^2)
$$
所谓光锥变换：$z=\sqrt{u},v=(tc/2)^2$
$$
LHS=\tau(x',y',\frac{2\sqrt{v}}{c})\\
RHS=\iiint_{\Omega}\frac{1}{v^{3/2}}\frac{1}{2\sqrt{u}}\rho(x,y,\sqrt{u})\delta((x'-x)^2+(y'-y)^2+u-v)dxdydu\\
v^{3/2}\tau(x',y',\frac{2\sqrt{v}}{c})=\iiint_{\Omega}\frac{1}{2\sqrt{u}}\rho(x,y,\sqrt{u})\delta((x'-x)^2+(y'-y)^2+u-v)dxdydu\\
\mathcal R_t\{\tau\}(x',y',v)=\mathcal R_z\{\rho\}(x,y,u)*h(x'-x,y'-y,v-u)
$$
这里，函数 $h$ 是一个平移不变的 3D 卷积核，变换 $\mathcal R_z$ 非均匀地沿 z 轴重采样和衰减体积 $\rho$ 的元素，变换 $\mathcal R_t$ 非均匀地沿时间轴重采样和衰减的测量值 $\tau$ 。
Here, the function $h$ is a shift-invariant 3D convolution kernel, the transform $\mathcal R_z$ non-uniformly re-samples and attenuates the elements of volume $\rho$ along the z axis, and the transform $\mathcal R_t$ non-uniformly re-samples and attenuates the measurements $\tau$ along the time axis.





