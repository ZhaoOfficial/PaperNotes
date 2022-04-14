# A Theory of Fermat Paths for Non-Line-of-Sight Shape Reconstruction

## Abstract

## Introduction

在本文中，我们仅使用几何约束而非强度约束的技术来克服上述限制，这些约束来自 NLOS 场景的瞬态测量。
In this paper, we overcome the above limitations by developing techniques that use only geometric, rather than intensity, constraints derived from transient measurements of an NLOS scene.

根据费马原理，我们观察到这些路径要么遵循镜面反射定律，要么在物体边界的特定点处反射。
Based on Fermat's principle, we observe that these paths follow either the law of specular reflection or reflect at specific points at the object's boundary.

## Fermat Paths in NLOS Transients

光源 $s$ 探测器 $d$ 可见场景 $\mathcal V$ 可见点 $v$ NLOS 场景 $\mathcal X$ NLOS 点 $x$

探测器记录了一个瞬态 $I(t;v)$ ，相当于在 $t$ 时间内的光子辐照度。光子路径是：
$$
s\to v\to  x\to v\to d
$$
这种三反射假设在 NLOS 成像应用中很常见，原因有两个：首先，NLOS 瞬态成像系统通常具有时间门控机制，可用于去除仅与可见场景交互的直接光子。 其次，与 NLOS 场景 $\mathcal X$ 有不止一种相互作用的光子大大降低了信噪比，实际上很难检测到。
This three-bounce assumption is commonplace in NLOS imaging applications, for two reasons: First, NLOS transient imaging systems typically have time-gating mechanisms that can be used to remove direct photons that only interact with the visible scene. Second, photons with more than one interactions with the NLOS scene $\mathcal X$ have greatly reduced signal-to-noise ratio, and in practice are difficult to detect.

校准了如下的距离量：
$$
\tau_{\mathcal V}(v)\triangleq\|s-v\|+\|d-v\|
$$
然后可以计算在 NLOS 场景的路径长度：
$$
\tau=ct-\tau_{\mathcal V}(v)
$$
然后我们可以写出：
$$
I(\tau;v)=\int_{\mathcal X}f(x;v)\delta(\tau-\tau(x;v))dA(p,q)
$$
其中 $\tau(x;v)=2\|x-v\|,(p,q)\in[0,1]$ 是参数化的 NLOS 场景。

## 2.1 Fermat paths

我们假设 NLOS 场景由光滑的表面组成，其边界法向量未知。

对于任何可见点 $v$

镜面反射集合 $\mathcal S(v)\in\mathcal X$ 包含了所有 NLOS 场景的非边界点































