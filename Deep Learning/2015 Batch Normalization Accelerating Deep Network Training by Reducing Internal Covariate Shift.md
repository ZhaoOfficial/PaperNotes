# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

## Abstract

问题：**内部协变量偏移**：由于每层输入的分布在训练过程中随着前一层的参数发生变化而发生变化，因此训练深度神经网络很复杂。由于较低的学习率和精细地初始化参数会减慢训练速度，并且使得训练具有饱和非线性函数（sigmoid...）的模型变得非常困难。
Problem: **internal covariate shift**: Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities.

我们的方法的优势在于将归一化作为模型架构的一部分，并为每个训练小批量执行归一化。
Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch.

批归一化允许我们使用更高的学习率，并且对初始化不那么在意。它还充当正则化器，在某些情况下消除了 Dropout 的需要。
Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout.

## Introduction

SGD 优化网络参数 $\Theta$，即优化损失函数：
SGD optimizes the parameters $\Theta$ of the network, so as to minimize the loss:
$$
\Theta^*=\arg\min_{\Theta}\frac{1}{N}\sum_{i=1}^{N}\ell(x_i,\Theta)
$$
$x_1,\dots,x_N$ 是训练集。

使用 SGD，训练分步进行，每一步我们只需考虑一个大小为 $m$ 的小批量。
With SGD, the training proceeds in steps, and at each step we consider a mini-batch of size $m$.
$$
\frac{1}{m}\frac{\partial\ell(x_i,\Theta)}{\partial\Theta}
$$
首先，小批量损失函数的梯度是对训练集损失函数的梯度的一个估计，其质量随着批量大小的增加而提高。其次，由于并行计算，计算一批量比计算 $m$ 次单个样本更高效。
First, the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases. Second, computation over a batch can be much more efficient than $m$ computations for individual examples, due to the parallelism afforded by the modern computing platforms.

当学习系统的输入分布发生变化时，就会发生协变量偏移。
When the input distribution to a learning system changes, it is said to experience covariate shift.

考虑一个二层网络：
$$
\ell=F_2(F_1(u,\Theta_1),\Theta_2)
$$
我们把 $x=F_1(u,\Theta_1)$ 作为内网络，也作为外网络 $\ell=F_2(x,\Theta_2)$ 的输入。如果 $F_2$ 的输入的分布是固定的，那么可以使得训练 $F_2$ 更快更简单。

$\Theta_2$ 不必重新调整以补偿 $x$ 分布的变化。
$\Theta_2$ does not have to readjust to compensate for the change in the distribution of $x$.

考虑：
$$
z=\text{sigmoid}(Wu+b),x=Wu+b
$$
$x$​ 的绝对值可能会比较大，导致 sigmoid 函数的导数值趋向于 0，收敛变慢。此时反向传播为：
$$
\frac{e^{-x}}{(1+e^{-x})^2}\approx e^{-x}
$$
固定分布的网络输入将对网络外的层产生积极影响。
Fixed distribution of inputs to a network would have positive consequences for the layers outside the network.

由于 $x$ 受 $W,b$ 和内部所有层的参数的影响，因此在训练期间对这些参数的更改可能会将 $x$ 的许多维度移动到非线性的饱和状态并减慢收敛速度。
Since $x$ is affected by $W, b$ and the parameters of all the layers below, changes to those parameters during training will likely move many dimensions of $x$ into the saturated regime of the nonlinearity and slow down the convergence.

以前是用 ReLU，精细的初始化和较小的学习率来避免。

## Towards Reducing Internal Covariate Shift

众所周知，如果输入是白化的——即线性变换为具有零均值和单位方差，并且去相关，网络训练就会收敛得更快。
It has been long known that the network training converges faster if its inputs are whitened - i.e., linearly transformed to have zero means and unit variances, and decorrelated.

假设输入是 $u$，偏置是 $b$，输出是 $\hat{x}=x-E[x]=u+b-E[u+b]$，损失函数是 $y=f(\hat{x})$​ 。如果忽略了 $E[x]$ 对 $b$ 的依赖：
$$
\frac{\partial y}{\partial b}=\frac{\partial y}{\partial \hat{x}}\frac{\partial \hat{x}}{\partial b}=\frac{\partial y}{\partial \hat{x}}
$$
则更新参数时候：
$$
b\gets b-\alpha\frac{\partial y}{\partial \hat{x}}=b-\Delta b
$$
但此时的输出为：
$$
\hat{x}\gets u+b-\Delta b-E[u+b-\Delta b]=\hat{x}
$$
为了解决这个问题，我们希望确保对于任何参数值，网络总是产生具有所需分布的激活。这样，模型参数的损失函数的梯度会考虑到正则化，以及它对模型参数 $\Theta $ 的依赖。
To address this issue, we would like to ensure that, for any parameter values, the network always produces activations with the desired distribution. Doing so would allow the gradient of the loss with respect to the model parameters to account for the normalization, and for its dependence on the model parameters $\Theta$.

把正则化写成如下形式，$x$ 是输入向量，$\mathcal X$ 是全体输入：
$$
\hat{x}=\text{Norm}(x,\mathcal X)
$$
反向传播时候需要计算：
$$
\frac{\partial \text{Norm}(x,\mathcal X)}{\partial x}\quad\frac{\partial \text{Norm}(x,\mathcal X)}{\partial\mathcal X}
$$


## Normalization via Mini-Batch Statistics

第一个是我们将独立地归一化每个标量特征，而不是联合地白化输入和输出中的特征，通过使其具有 0 均值和 1 方差。
The first is that instead of whitening the features in layer inputs and outputs jointly, we will normalize each scalar feature independently, by making it have the mean of 0 and the variance of 1.

对于一个 $d$ 维的输入 $x=(x^{(1)},\dots,x^{(n)})$，我们归一化得到：
$$
\hat{x}^{(k)}=\frac{x^{(k)}-\text{E}[x^{(k)}]}{\sqrt{\text{Var}[x^{(k)}]}}
$$

请注意，简单地对层的每个输入进行归一化可能会改变层可以表示的内容。例如，对 sigmoid 的输入进行归一化会将它们限制在近似线性范围内。
Note that simply normalizing each input of a layer may change what the layer can represent. For instance, normalizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity.



















