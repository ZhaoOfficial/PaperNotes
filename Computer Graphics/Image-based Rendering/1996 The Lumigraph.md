# 光场 The Lumigraph

## Abstract

## Introduction

## 表示 Representation

### 从 5D 到 4D From 5D to 4D

### 参数化的 4D 光场 Parameterization of the 4D Lumigraph

### 离散化 4D 参数 Discretization of the 4D Parameterization

M 等分 st 平面，N 等分 uv 平面。

因此，选取一个基函数 $B_{i,j,p,q}$，可以重建连续的光场。
$$
\tilde{L}(s,t,u,v)=\sum_{i=0}^{M}\sum_{j=0}^{M}\sum_{p=0}^{N}\sum_{q=0}^{N}x_{i,j,p,q}B_{i,j,p,q}(s,t,u,v)
$$
假如 $B_{i,j,p,q}(s,t,u,v)$ 是返回和 $(s,t,u,v)$ 最近的一个格点的值，那 $\tilde{L}$ 是阶梯函数了。假如 $B_{i,j,p,q}(s,t,u,v)$ 是四线性插值，则 $\tilde{L}$ 是 0 阶连续的了。

用 $L^2$ 距离衡量参数化光场和实际的差距。
We choose the distance $L^2$ metric, then the projection is defined by integrating
against the L duals of the basis functions.

我们希望物体表面更接近 uv 面（近的平面），这会影响 M 和 N 的选取。
The choices for M and N are influenced by the fact that we expect the visible surfaces of the object to lie closer to the uv plane than the st plane.

N 选取为图像最终的分辨率最好。
A choice for N close to final image resolution works best.

M 比 N 的数量少很多比较好。（如果 M 更多的话，从 uv 平面过来的光会更分散更弱，会更不清晰。）
A significantly lower resolution for M than N can be expected to yield good results.

几何信息可以帮助确定基函数。
Knowledge about the geometry can be used to help define the shape of our basis functions.

假设一条光线和物体相交，从 st 平面的各点射出一条光线到交点，最后达到 uv 平面。到 uv 平面的交点可以用相似三角形得到（假设深度信息已知）。





