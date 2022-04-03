# Phase-Functioned Neural Networks for Character Control

## Abstract

实时角色控制的机制
Real-time character control mechanism

在这个网络结构中，权重是通过循环函数计算的，该函数使用相位作为输入。
In this network structure, the weights are computed via a cyclic function which uses the phase as an input.

除了相位，我们的系统将用户控制、角色的先前状态、场景的几何形状作为输入，并自动生成实现所需用户控制的高质量动作。
Along with the phase, our system takes as input user controls, the previous state of the character, the geometry of the scene, and automatically produces high quality motions that achieve the desired user control.

## Introduction

RNN 和其他自回归模型更适合于视频游戏和在线运动生成，因为它们只需要未来的某一帧输入，然而在生成长序列的运动时，它们往往会失败，因为其预测的误差会反馈到输入中并累积。
RNNs and other autoregressive models are more appropriate for video games and online motion generation as they only require a single frame of future input, yet they tend to fail when generating long sequences of motion as the errors in their prediction are fed back into the input and accumulate.

## System Overview

它是一种神经网络结构（用 $\Phi$ 表示），其中权重是由一个关于相位 $p$ 的周期函数，称为相位函数（用 $\Theta$ 表示）计算的。
It is a neural network structure (denoted by $\Phi$) where the weights are computed by a periodic function of the phase $p$ called the phase function (denoted by $\Theta$).

网络的输入 $x$ 包括角色之前的姿势和用户的控制，而输出 $y$ 包括相位的变化、角色当前的姿势以及后面描述的一些附加参数。
The inputs $x$ to the network include both the previous pose of the character and the user control, while the outputs $y$ include the change in the phase, the character’s current pose, and some additional parameters described later.

## Data Acquisition & Processing

### Motion Capture and Control Parameters

#### Motion Capture

#### Phase Labeling

首先，脚部接触时间是通过计算脚跟和脚趾关节的速度大小来自动标注的，并观察这些速度何时低于某个阈值。
Firstly, foot contact times are automatically labeled by computing the magnitude of the velocity of the heel and toe joints and observing when these velocities go below some threshold.

一旦获得了这些接触时间，就可以通过观察右脚接触地面的帧并分配 $0$ 的相位来自动计算相位，观察左脚接触地面时的帧并分配 $\pi$ 的相位，观察下一次右脚接触发生的时间，并指定 $2\pi$ 的相位。
Once these contact times are acquired the phase can be automatically computed by observing the frames at which the right foot comes in contact with the ground and assigning a phase of $0$, observing the frames when the left foot comes in contact with the ground and assigning a phase of $\pi$, and observing when the next right foot contact happens and assigning a phase of $2\pi$.

#### Gait Labels

#### Trajectory and Terrain Height

### Terrain Fitting

### System Input/Output Parameters

## Phase-Function Neural Network

PFNN 是一个神经网络，其权重根据相位值而循环变化。我们把产生网络权重的函数称为相位函数，在这项工作中，它被定义为三次 Catmull-Rom 样条。
The PFNN is a neural network with weights that cyclically change according to the phase value. We call the function which generates the network weights the phase function, which in this work is defined as a cubic Catmull-Rom spline.

### Neural Network Structure

$\mathbf x\in\mathbb R^n$：输入

$\mathbf y\in\mathbb R^m$：输出

$p\in\mathbb R$：相位参数

$\alpha$：权重和偏置的集合
$$
\Phi(\mathbf x;\alpha)=W_2\text{ELU}(W_1\text{ELU}(W_0\mathbf x+b_0)+b_1)+b_2
$$

### Phase Function

在 PFNN 中，网络权重 $\alpha$ 由称为相位函数的单独函数计算每帧，该函数将相位 $p$ 和参数 $\beta$ 作为输入，如下所示： $\alpha=\Theta(p ;\beta)$。$\Theta$ 是三次 Catmull-Rom 样条。
In the PFNN, the network weight $\alpha$ are computed each frame by a separate function called the phase function, which takes as input the phase $p$ and parameters $\beta$ as follows: $\alpha=\Theta(p;\beta)$. $\Theta$ is cubic Catmull-Rom spline.

通过让开始和结束控制点相同，它很容易形成循环，参数数量与控制点数量成正比，并且相对于输入参数 $p$ 平滑变化。
It is easily made cyclic by letting the start and end control points be the same, the number of parameters is proportional to the number of control points, and it varies smoothly with respect to the input parameter $p$.

选择三次 Catmull-Rom 样条来表示相位函数意味着每个控制 $\alpha_k$ 点代表神经网络 $\alpha$ 的某个权重配置，函数 $\Theta$ 在这些神经网络之间平滑地插值权重配置。
Choosing a cubic Catmull-Rom spline to represent the phase function means each control $\alpha_k$ point represents a certain configuration of weights for the neural network $\alpha$, and the function $\Theta$ performs a smooth interpolation between these neural network weight configurations.

训练网络是在神经网络权重空间中找到合适的循环流形，它成功地执行从输入参数到输出参数的回归。
Training the network is finding an appropriate cyclic manifold in the space of neural network weights which performs the regression from input parameters to output parameters successfully.

4 个权重配置就足够了：
$$
\begin{align*}
\Theta(p;\beta)&=\alpha_{k_1}\\
&+w(\frac{1}{2}\alpha_{k_2}-\frac{1}{2}\alpha_{k_0})\\
&+w^2(\alpha_{k_0}-\frac{5}{2}\alpha_{k_1}+2\alpha_{k_2}-\frac{1}{2}\alpha_{k_3})\\
&+w^3(\frac{3}{2}\alpha_{k_1}-\frac{3}{2}\alpha_{k_2}+\frac{1}{2}\alpha_{k_3}-\frac{1}{2}\alpha_{k_0})\\
w&=\frac{4p}{2\pi}(\text{mod }1)\\
k_n&=\lfloor\frac{4p}{2\pi}\rfloor+n-1(\text{mod }4)\\
\end{align*}
$$


