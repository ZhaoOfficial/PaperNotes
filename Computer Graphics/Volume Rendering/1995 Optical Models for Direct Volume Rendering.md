# Optical Models for Direct Volume Rendering

## 0 Abstract

## 1 Introduction

直接体渲染是指直接从体数据产生投影图像的技术，无需诸如轮廓表面多边形之类的中间构造。
Direct volume rendering refers to techniques which produce a projected image directly from the volume data, without intermediate constructs such as contour surface polygons.

可以将颜色和不透明度等光学属性指定为插值 $f(X)$ 的函数。
Optical properties like color and opacity can then be assigned as functions of the interpolated value $f(X)$.

## 2 Absorption only

$r$: 假设微粒是个球体，微粒半径 radius
$A=\pi r^2$: 投影面积 projected area
$\rho$: 单位体积微粒数量 the number of particles per unit volume
$E$: 圆柱板底面面积 the area of the base of a cylindrical slab
$\Delta s$: 圆柱板厚度 the thickness of a cylindrical slab
$I(s)$: 在 $s$ 处的光线强度 intensity of light at distance $s$

因此圆柱板内带有：
$$
N=\rho E\Delta s
$$
的微粒数。当 $\Delta s$​ 充分小的时候，微粒在垂直于圆柱板的投影的重合区域很少，因此总投影面积为 $NA$。当一束光穿过后，损失的光强为：
$$
\frac{NA}{E}=\frac{\rho E\Delta sA}{E}=\rho A\Delta s
$$
因此：
$$
\begin{align*}
I(s+\Delta s)-I(s)&=-\rho(s)A\Delta sI(s)\\
\frac{I(s+\Delta s)-I(s)}{\Delta s}&=-\rho(s)AI(s)\\
\frac{\mathrm{d}I(s)}{\mathrm{d}s}&=-\rho(s)AI(s)=-\sigma_{A}(s)I(s)\\
\end{align*}
$$
$\sigma_{A}(s)$: 吸收系数 absorption coefficient.

解微分方程可得：
$$
I(s)=I(0)\exp({-\int_0^s\sigma_A(t)\mathrm{d}t})=I(0)T(s)
$$
$T(s)$: 从 0 到 s 处的透射率 transparent from 0 to s
$\alpha$: 边长为 $l$ 的体素从平行于某一边的视角看过去时候的不透明度 opacity of a voxel of side $l$, viewed parallel to one edge.
$$
\alpha=1-T(l)=1-\exp({-\int_0^l\sigma_A(t)\mathrm{d}t})
$$
如果 $\sigma_A$ 在体素内是常值，则：
$$
\alpha=1-\exp(-\sigma_Al)=\sigma_Al-\frac{(\sigma_Al)^2}{2}
$$
将像 $\sigma_A$ 这样的光学属性的值分配给被可视化的标量 $f$ 的每个值的映射称为传递函数。
The mapping which assigns a value for an optical property like $\sigma_A$ to each value of the scalar $f$ being visualized is called a transfer function.











