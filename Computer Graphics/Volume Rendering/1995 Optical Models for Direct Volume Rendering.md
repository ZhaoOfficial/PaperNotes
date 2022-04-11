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
$\sigma_{A}(s)=\rho(s)A$: 吸收系数 absorption coefficient.

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
将像 $\sigma_A$ 这样的光学属性的值分配给被可视化的标量 $f$ 的每个值的映射称为传递函数 $\sigma_A(f)$。
The mapping which assigns a value for an optical property like $\sigma_A$ to each value of the scalar $f$ being visualized is called a transfer function $\sigma_A(f)$.

## 3 Emission Only

$C$ 每个投影面积上的发光强度 intensity per unit projected area.

因此圆柱底面的光通量为：
$$
C\rho E\Delta sA
$$
因此：
$$
\begin{align*}
I(s+\Delta s)-I(s)&=C(s)\rho(s)\Delta sA=g(s)\Delta s\\
\frac{I(s+\Delta s)-I(s)}{\Delta s}&=g(s)\\
\frac{\mathrm{d}I(s)}{\mathrm{d}s}&=g(s)\\
\end{align*}
$$
$g(s)=C(s)\sigma_A(s)$: radiance.

解微分方程可得：
$$
I(s)=I(0)+\int_0^sg(t)\mathrm{d}t
$$

## 4 Absorption Plus Emission

$$
\frac{\mathrm{d}I(s)}{\mathrm{d}s}=g(s)-\sigma_A(s)I(s)
$$

解微分方程可得：
$$
\begin{align*}
I(s)&=I(0)\exp({-\int_0^s\sigma_A(t)\mathrm{d}t})+\left[\int_0^sg(t)\exp({\int_0^t\sigma_A(\tau)\mathrm{d}\tau})\mathrm{d}t\right]\exp({-\int_0^s\sigma_A(\tau)\mathrm{d}\tau})\mathrm{d}t\\
&=I(0)\exp({-\int_0^s\sigma_A(t)\mathrm{d}t})+\int_0^sg(t)\exp({-\int_t^s\sigma_A(\tau)\mathrm{d}\tau})\mathrm{d}t\\
&=\color{red}I(0)T(0,s)+\int_0^sg(t)T(t, s)\mathrm{d}t\\
\end{align*}
$$
$T(t,s)$: $t$ 到 $s$ 点的透射率 transmittance between $t$ and $s$.

注意 0 处是光线发射出位置，是远端，$s$ 处是相机所在位置，是近端。

### 4.1 Calculation Methods

将 $0$ 到 $s$ 均匀划分为 $n$ 等分，每段长度为 $\Delta{x}=s/n$，每段的采样为区间右端点 $x_i=i\Delta{x}$，因此：
$$
\exp({-\int_0^s\sigma_A(t)\mathrm{d}t})=\exp({-\sum_{i=1}^{n}\sigma_A(i\Delta{x})\Delta{x}})=\prod_{i=1}^{n}\exp({-\sigma_A(i\Delta{x})\Delta{x}})\\
\int_0^sg(t)\exp({-\int_t^s\sigma_A(\tau)\mathrm{d}\tau})\mathrm{d}t=\sum_{i=1}^ng(i\Delta{x})\Delta{x}\exp({-\int_{i\Delta{x}}^s\sigma_A(\tau)\mathrm{d}\tau})\\
\exp({-\int_{i\Delta{x}}^s\sigma_A(\tau)\mathrm{d}\tau})=\exp({-\sum_{j=i+1}^n\sigma_A(j\Delta{x})\Delta{x}})=\prod_{j=i+1}^n\exp({-\sigma_A(j\Delta{x})\Delta{x}})
$$
定义：$t_i=\exp({-\sigma_A(i\Delta{x})\Delta{x}})$, $g_i=g(i\Delta{x})\Delta{x}$

$j=i+1$ 是因为积分从 0 到 n 区间积分，但是采样从 1 到 n 端点采样。

因此：
$$
\color{red}\begin{align*}
I(s)&\approx I(0)\prod_{i=1}^{n}t_i+\sum_{i=1}^{n}g_i\prod_{j=i+1}^{n}t_j\\
&=g_n+t_n(g_{n-1}+t_{n-1}(\cdots(g_1+t_1I(0))\cdots))
\end{align*}
$$

```python
def front2Back(t, g, I0):
    # equation above
    I = 0.0
    T = 1.0
    for i in range(n, 0, -1):
        I = T * I + g[i]
        T = T * t[i]
    I = I + T * I0
    return I
def back2Front(t, g, I0):
    # equation below
    I = I0
    for i in range(1, n + 1):
        I = t[i] * I + g[i]
    return I
```

### 4.2 The Particle Model

如果 $g(s)=C(s)\sigma_A(s)$ 中的 $C$ 是常数的话：
$$
\begin{align*}
\int_0^sg(t)\exp({-\int_t^s\sigma_A(\tau)\mathrm{d}\tau})\mathrm{d}t&=C\int_0^s\sigma_A(t)\exp({-\int_t^s\sigma_A(\tau)\mathrm{d}\tau})\mathrm{d}t\\
&=C\int_0^s\frac{\mathrm{d}}{\mathrm{d}t}\exp({-\int_t^s\sigma_A(\tau)\mathrm{d}\tau})\mathrm{d}t\\
&=C\left[1-\exp({-\int_0^s\sigma_A(\tau)\mathrm{d}\tau})\right]\\
&=C(1-T(0,s))
\end{align*}
$$
因此公式改为：
$$
\color{red}I(s)=I(0)T(0,s)+C(1-T(0,s))
$$
如果 $C$​ 是分段常数的话：
$$
\color{red}I(s)=\sum_{i=1}^{n}C_i(1-T((i-1)\Delta{x},i\Delta{x}))T(i\Delta{x},s)+I(0)T(0,s)
$$
由于我们把 $\sigma_A(s)$ 也看做分段常数：
$$
\color{red}I(s)=\sum_{i=1}^{n}C_i(1-\exp(-\sigma_{Ai}\Delta{x}))T(i\Delta{x},s)+I(0)T(0,s)
$$


