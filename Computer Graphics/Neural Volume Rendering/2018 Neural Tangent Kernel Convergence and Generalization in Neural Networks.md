# Neural Tangent Kernel Convergence and Generalization in Neural Networks

## 0 Abstract

在初始化时，人工神经网络 (ANN) 等效于无限宽度限制中的高斯过程，从而将它们连接到核方法。
At initialization, artificial neural networks (ANNs) are equivalent to Gaussian processes in the infinite-width limit, thus connecting them to kernel methods.

在 ANN 参数的梯度下降过程中，网络函数 $f_{\theta}$（将输入向量映射到输出向量）遵循函数成本的核梯度（与参数成本相比，它是凸的）关于新内核：神经正切核（NTK）。
During gradient descent on the parameters of an ANN, the network function $f_{\theta}$ (which maps input vectors to output vectors) follows the kernel gradient of the  functional cost (which is convex, in contrast to the parameter cost) w.r.t. a new kernel: the Neural Tangent Kernel (NTK).

虽然 NTK 在初始化时是随机的，并且在训练期间会发生变化，但在无限宽度限制中，它会收敛到显式限制核，并且在训练期间保持不变。
While the NTK is random at initialization and varies during training, in the infinite-width limit it converges to an explicit limiting kernel and it stays constant during training.

## 1 Introduction







































