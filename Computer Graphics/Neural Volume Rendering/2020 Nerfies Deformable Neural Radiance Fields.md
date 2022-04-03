# Nerfies：可形变的神经辐射场Nerfies: Deformable Neural Radiance Fields

## Abstract

图片来自移动端设备。
Using photos/videos captured casually from mobile phones.

## Introduction

手持摄像机人物建模的困难有：1）软体性，人不能一直一个姿势着不动（非静态场景），2）头发，眼镜，耳环等小物件难以在一般重建方法中重建出来。
Modeling people with hand-held cameras is challenging: 1) nonrigidity – our inability to stay perfectly still, and 2) challenging materials like hair, glasses, and earrings that violate assumptions used in most reconstruction methods.

为了给非刚性形变场建模，我们引入一个额外组建来泛化 NeRF。一个典范 NeRF 模型作为所有观测的模板，对于每个观测辅以一个形变场，这个形变场把观测参考系的点扭曲到典型模型参考系中。
To model non-rigidly deforming scenes, we generalize NeRF by introducing an additional component: A canonical NeRF model serves as a template for all the observations, supplemented by a deformation field for each observation that warps 3D points in the frame of reference of an observation into the frame of reference of the canonical model.

无限制情况下，形变场倾向于畸变和过拟合。
Without constraints, the deformation fields are prone to distortions and over-fitting.

## Deformable Neural Radiance Fields

我们把非刚性变形场分解为 NeRF 和每个观测一个变形场，这些观测场把内部的点与 NeRF 的点联系起来。
We decompose a non-rigidly deforming scene into a template volume represented as a NeRF and a per-observation deformation field that associates a point in observation coordinates to a point on the template.

### Neural Radiance Fields

我们给每个观测帧一个潜在编码 $\psi_i$ 来调和颜色输出，以解决输入帧之间物体外观的变化。
We provide an appearance latent code $\psi_i$ for each observed frame that modulates the color output to handle appearance variations between input frames, e.g., exposure and white balance.

### Neural Deformation Fields

每一个帧都有观测场景到典范场景的形变。
Observation-to-canonical deformation for every frame.
$$
T:(\mathbf x,\omega_i)\to\mathbf x',\mathbf x\in\text{obervation space}, \mathbf x'\in\text{deformation space}
$$
$\omega_i$ 是每一帧的潜在编码。
$$
F(T(\mathbf x, \omega_i),\mathbf d,\psi_i)
$$
我们把形变用一个旋转平移群表示。旋转平移群来编码刚体运动，允许我们用相同的参数旋转点集。
We formulate the deformation using a dense $SE(3)$ field $W:(\mathbf x, \omega_i)\to SE(3)$. An $SE(3)$ transform encodes rigid motion, allowing us to rotate a set of distant points with the same parameters.
$$
\mathbf x'=T(\mathbf x,\omega_i)=e^{[\mathbf r]_{\times}}\mathbf x+\mathbf G\mathbf v\\
e^{[\mathbf r]_{\times}}=\mathbf I+\frac{\sin\theta}{\theta}[\mathbf r]_{\times}+\frac{1-\cos\theta}{\theta^2}[\mathbf r]_{\times}^2\\
\mathbf G=\mathbf I+\frac{1-\cos\theta}{\theta^2}[\mathbf r]_{\times}+\frac{\theta-\sin\theta}{\theta^3}[\mathbf r]_{\times}^2\\
(\mathbf r,\mathbf v)=W(\mathbf x,\omega_i)
$$
我们用一个类似 NeRF 架构的 MLP 里面编码变换场，通过调节潜在编码 $\omega_i$ 来呈现每一帧的变换。
We encode the transformation field in an MLP $W:(\mathbf x, \omega_i)\to SE(3)$ using a NeRF-like architecture, and represent the transformation of every frame $i$ by conditioning on a latent code $\omega_i$.

### Elastic Regularization

用 Jacobian 控制局部形变。
Control the local behavior of the deformation through Jacobian $\mathbf J_T(\mathbf x)$.
$$
L_{\text{elastic}}(\mathbf x)=\|\log\Sigma-\log\mathbf I\|^2_{F}=\|\log\Sigma\|^2_{F}\\
\mathbf J_T(\mathbf x)=\mathbf U\Sigma\mathbf V^T
$$
尽管人类很像刚体，但表情变化，皮肤等动作仍然会破坏局部刚性。
Although humans are mostly rigid, there are some movements which can break our assumption of local rigidity, e.g., facial expressions which locally stretch and compress our skin.
$$
L_{\text{elastic-r}}(\mathbf x)=\rho(\|\log\Sigma\|_{F},c)=\frac{2\|\log\Sigma\|_{F}^2}{\|\log\Sigma\|_{F}^2+4c^2}\\
$$
用 Geman-McClure robust error function 优化一下。

### Background Regularization

防止背景移动的正则项。

给定场景中必须固定好的一组 3D 点，我们对于这些点的扭曲进行惩罚。
Given a set of 3D points in the scene which we know should be static, we can penalize any deformations at these points.
$$
L_{\text{bg}}=\frac{1}{K}\sum_{k=1}^{K}\|T(\mathbf x_k)-\mathbf x_k\|
$$

### Coarse-to-Fine Deformation Regularization







