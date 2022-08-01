# Fourier Features Let Networks Learn High Frequency Functions in Low Dimension Domains

## 0 Abstract

我们表明，通过简单的傅立叶特征映射传递输入点使多层感知器 (MLP) 能够学习低维问题域中的高频函数。
We show that passing input points through a simple Fourier feature mapping enables a multilayer perceptron (MLP) to learn high-frequency functions in low dimensional problem domains.

## 1 Introduction

MLP 难以学习高频函数，这种现象在文献中被称为“频谱偏差”。
MLPs have difficulty learning high frequency functions, a phenomenon referred to in the literature as "spectral bias".

NTK 理论表明，这是因为标准的基于坐标的 MLP 对应于具有快速频率衰减的内核，这有效地阻止了它们能够表示自然图像和场景中存在的高频内容。
NTK theory suggests that this is because standard coordinate-based MLPs correspond to kernels with a rapid frequency falloff, which effectively prevents them from being able to represent the high-frequency content present in natural images and scenes.

Fourier features mapping:
$$
\gamma(\mathbf{v})=[a_1\sin(2\pi\mathbf{b}_1^T\mathbf{v}),a_1\cos(2\pi\mathbf{b}_1^T\mathbf{v}),\dots,a_m\sin(2\pi\mathbf{b}_m^T\mathbf{v}),a_m\cos(2\pi\mathbf{b}_m^T\mathbf{v})]
$$

> For NeRF's positional encoding:
> $$
> a_i=1,b_{3i+k}=(2^i\delta_{0k},2^i\delta_{1k},2^i\delta_{2k})^T
> $$

我们展示了这种映射将 NTK 转换为一个平稳（平移不变）的内核，并通过修改频率向量 $\mathbf{b}_j$ 来调整 NTK 的频谱，从而控制相应 MLP 可学习的频率范围。
We show that this mapping transforms the NTK into a stationary (shift-invariant) kernel and enables tuning the NTK's spectrum by modifying the frequency vectors $\mathbf{b}_j$, thereby controlling the range of frequencies that can be learned by the corresponding MLP.

我们表明，设置 $a_j = 1$ 并从各向同性分布中随机抽样 $\mathbf{b}_j$ 的简单策略可以获得良好的性能，并且该分布的尺度（标准差）比其特定形状更重要.
We show that the simple strategy of setting $a_j = 1$ and randomly sampling $\mathbf{b}_j$ from an isotropic distribution achieves good performance, and that the scale (standard deviation) of this distribution matters much more than its specific shape.

## 2 Related Work

## 3 Background and Notation

### Kernel regression

|                         Symbols                          |                         Descriptions                         |
| :------------------------------------------------------: | :----------------------------------------------------------: |
| $(\mathbf{X},\mathbf{y})=\{(\mathbf{x_i},y_i)\}_{i=1}^n$ |                     features and labels                      |
|                       $\mathbf{K}$                       | $\mathbf{K}_{ij}=k(\mathbf{x}_i,\mathbf{x}_j)$ where $k$ is a symmetric positive semi-definite kernel function which represents the "similarity" |
|                        $\hat{f}$                         |                           estimate                           |

Kernel regression is a classic nonlinear regression algorithm.
$$
\hat{f}(\mathbf{x})=\sum_{i=1}^n(\mathbf{K}^{-1}\mathbf{y})_ik(\mathbf{x}_i,\mathbf{x})
$$
直观地说，在任何点 $\mathbf{x}$ 的核回归估计可以被认为是使用相应 $\mathbf{x}_i$ 和 $\mathbf{x}$ 之间的相似性对训练标签 $y_i$ 的加权和。
Intuitively, the kernel regression estimate at any point $\mathbf{x}$ can be thought of as a weighted sum of training labels $y_i$ using the similarity between the corresponding $\mathbf{x}_i$ and $\mathbf{x}$.

### Approximating deep networks with kernel regression

|                           Symbols                            |                         Descriptions                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                             $f$                              | a fully-connected deep network with weights $\theta$ initialized from a Gaussian distribution $\mathcal{N}$ |
| $\hat{\mathbf{y}}^{(t)}=f(\mathbf{X}_{\mathrm{test}};\theta)$ | the network's predictions on input points $\mathbf{X}_{\mathrm{test}}$ at training iteration $t$ |
|                 $\mathbf{K}_{\mathrm{test}}$                 | the NTK matrix between all points in $\mathbf{X}_{\mathrm{test}}$ and all points in the training dataset |

当 $f$ 中层的宽度趋于无穷大并且 SGD 的学习率趋于零时，函数 $f(\mathbf{x};\theta)$ 在训练过程中使用神经正切核 (NTK) 收敛到核回归解决方案，NTK 定义为：
When the width of the layers in $f$ tends to infinity and the learning rate for SGD tends to zero, the function $f(\mathbf{x};\theta)$ converges over the course of training to the kernel regression solution using the neural tangent kernel (NTK), defined as:
$$
k_{\mathrm{NTK}}(\mathbf{x}_i,\mathbf{x}_j)=\mathbb{E}_{\theta\sim\mathcal{N}}\left\langle\frac{\partial{f}(\mathbf{x}_i;\theta)}{\partial\theta},\frac{\partial{f}(\mathbf{x}_j;\theta)}{\partial\theta}\right\rangle
$$
We consider a network trained with an L2 loss and a learning rate , where the network’s weights are initialized such that the output of the network at initialization is close to zero.

The network’s output for any data $\mathbf{X}_{\mathrm{test}}$ after $t$ training iterations can be approximated as:
$$
\hat{\mathbf{y}}^{(t)}\approx\mathbf{K}_{\mathrm{test}}\mathbf{K}^{-1}(\mathbf{I}-\exp(-\eta\mathbf{K}t))\mathbf{y}
$$

### Spectral bias when training neural networks

|                  Symbols                  |                         Descriptions                         |
| :---------------------------------------: | :----------------------------------------------------------: |
| $\hat{\mathbf{y}}^{(t)}_{\mathrm{train}}$ | the network’s predictions on the training dataset at iteration $t$ |
|           $\mathbf{Q},\Lambda$            | $\mathbf{K}=\mathbf{Q}\Lambda\mathbf{Q}^T$, eigen decomposition, $\mathbf{Q}$ orthogonal, $\Lambda$ diagonal matrix whose entries are the eigenvalues $\lambda_i\ge0$ |

$$
\exp(-\eta\mathbf{K}t))=\mathbf{Q}\exp(-\eta\Lambda t))\mathbf{Q}^T
$$

then:
$$
\begin{align*}
\mathbf{Q}^T(\hat{\mathbf{y}}^{(t)}_{\mathrm{train}}-\mathbf{y})&\approx\mathbf{Q}^T((\mathbf{I}-\exp(-\eta\mathbf{K}t))\mathbf{y}-\mathbf{y})\\
&=-\mathbf{Q}^T\exp(-\eta\mathbf{K}t)\mathbf{y}\\
&=-\exp(-\eta\Lambda t)\mathbf{Q}^T\mathbf{y}
\end{align*}
$$
这意味着如果我们在 NTK 的特征基中考虑训练收敛，则绝对误差 $|\mathbf{Q}^T(\hat{\mathbf{y}}^{(t) }_{\mathrm{train}}-\mathbf{y})|$ 将以 $\eta\lambda_i$ 的速率近似呈指数衰减。
This means that if we consider training convergence in the eigenbasis of the NTK, the $i$-th component of the absolute error $|\mathbf{Q}^T(\hat{\mathbf{y}}^{(t)}_{\mathrm{train}}-\mathbf{y})|$ will decay approximately exponentially at the rate $\eta\lambda_i$.

换句话说，目标函数中对应于具有较大特征值的核特征向量的分量将被更快地学习。
In other words, components of the target function that correspond to kernel eigenvectors with larger eigenvalues will be learned faster.

对于传统的 MLP，NTK 的特征值衰减很快。
For a conventional MLP, the eigenvalues of the NTK decay rapidly.

## 4 Fourier Features for a Tunable Stationary Neural Tangent Kernel



















