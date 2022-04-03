# Light Stage Super-Resolution: Continuous High-Frequency Relighting

## Abstract

尽管灯光舞台是重打光主体的强大工具，但由于舞台的相邻灯光相隔一定距离，因此其效果会受到影响。
Though the light stage is a powerful tool for relighting human subjects, its renderings suffer because adjacent lights of the stage are separated by some distance.

## Introduction

灯光舞台：用程序激活和停用排列在球体中的多个 LED 灯来捕捉同步图像，获取主体全反射场，我们将其称为 OLAT 图像集。
Light stage: by programmatically activating and deactivating several LED lights arranged in a sphere while capturing synchronized images, the light stage acquires a full reflectance field for a human subject, which we refer to as a “one-light-at-a-time” (OLAT) image set.

因为光是可加的，而这个 OLAT 扫描代表了一个光的基函数。通过简单地将环境贴图投影到光舞台基函数上，我们可以用所需的环境贴图重新照亮对象。
Because light is additive, this OLAT scan represents a lighting “basis”, and the subject can be relit according to some desired environment map by simply projecting that environment map onto the light stage basis.

灯光舞台上的灯光通常设计得很小且远离拍摄对象，因此它们近似作为方向光源。
The lights on the light stage are usually designed to be small and distant from the subject, so that they are well-approximated as directional light sources.

来自灯光舞台的 OLAT 扫描相对于灯光的角度采样而言是欠采样的，并且使用传统方法渲染的图像可能会包含重影。
The OLAT scans from a light stage are undersampled with respect to the angular sampling of lights, and the rendered images using conventional approaches will likely contain ghosting.

尝试使用位于舞台真实灯光之间的“虚拟”光源渲染图像，通过对相邻 OLAT 图像应用加权平均不会产生软阴影或条纹镜面反射，而是会叠加出多个清晰的阴影和镜面反射点。
Attempting to render an image using a “virtual” light source that lies in between the real lights of the stage by applying a weighted average on adjacent OLAT images will not produce a soft shadow or a streaking specularity, but will instead produce the superposition of multiple sharp shadows and specular dots.

## Model

$n$: OLAT 数据集大小。

$\{I_i,\ell_i\}$: 输入，OLAT 数据集，包含了光基函数的图像和方向。

$\ell$: 输入，新的光方向。

$\Phi(\cdot)$: 深度神经网络。

$I$: 输出，新的图片。
$$
I(\ell)=\Phi(\{I_i,\ell_i\}_{i=1}^{n},\ell)
$$
$A(\ell)$: 激活集，$k$ 个和输入方向 $\ell$ 相邻的 OLAT 光基函数的图像和方向。

$\Phi_e(\cdot)$: 卷积神经网络的编码器，处理把激活集当成小批量输入。

随后，$k$ 个活性集卷积得到的活性层汇聚成 1 个活性层，汇聚的方式是权重和，权重 $W(\ell,\ell_i)$ 是用输入光方向和激活集中光的方向计算得到。

$\Phi_d(\cdot)$: 卷积神经网络的解码器，输入的光方向和活性层一起解码，得到输出 $I$。
$$
I(\ell)=\Phi_d\left(\sum_{i\in A(\ell)}W(\ell,\ell_i)\Phi_e(I_i,\ell_i),\ell\right)
$$
这种最近邻选择和神经网络处理的混合方法使我们能够学习产生高质量结果的单个神经网络，并在我们的 OLAT 数据集中的查询光方向和主体之间很好地泛化。
This hybrid approach of nearest-neighbor selection and neural network processing allows us to learn a single neural network that produces high quality results, and generalizes well across query light directions and across subjects in our OLAT dataset.

### Active Set Selection

灯光舞台通常是通过将灯光放置在球体的规则六边形镶嵌上来构建的。
Light stages are conventionally constructed by placing lights on a regular hexagonal tessellation of a sphere.

#### 困难之处

首先，我们只能使用与灯光舞台的真实灯光完全一致的“虚拟”灯光来监督我们的超分辨率模型，因为这些是我们拥有真实图像的唯一灯光方向。
First, we can only supervise our super-resolution model using “virtual” lights that exactly coincide with the real lights of the light stage, as these are the only light directions for which we have ground-truth images.

其次，这种规则的六边形采样意味着，对于舞台中的任何给定光线，它与其邻居之间的距离将始终表现出高度规则的模式。
Second, this regular hexagonal sampling means that, for any given light in the stage, the distances between it and its neighbors will always exhibit a highly regular pattern.

在测试时，我们希望能够生成对应于球体上任意点的查询光方向的渲染，这些点可能具有不规则的相邻光分布。
At test time we would like to be able to produce renderings for query light directions that correspond to arbitrary points on the sphere, and those points will likely have irregular distributions of neighboring lights

这代表了我们的训练数据和我们的测试数据之间的显着偏差，因此，如果我们天真地对高度规则的最近邻居集进行训练，我们应该会在测试时预期泛化很差。
This represents a significant deviation between our training data and our test data, and as such we should expect poor generalization at test time if we were to naively train on highly-regular sets of nearest neighbors.

对于每次训练迭代，我们首先在查询灯附近确定一组更大的 $m$ 个最近邻居（在这种情况下与舞台中的一个真实灯光相同），并在其中随机选择仅 $k < m$ 个邻居用于 活动集（实际上，我们使用 $m = 16$ 和 $k = 8$）。
For each training iteration, we first identify a larger set of $m$ nearest neighbors near the query light (which in this case is identical to one of the real lights in the stage), and among them randomly select only $k < m$ neighbors to use in the active set (in practice, we use $m = 16$ and $k = 8$).

### Alias-Free Pooling

我们的激活集是对输入图像的混叠观察，平均或最大汇聚允许这种混叠持续存在。
Our active set is an aliased observation of the input images, and average- or max-pooling allows this aliasing to persist.

我们使用加权平均作为我们的汇聚算子，其中激活集中每个项目的权重是查询光方向的连续函数，并且每个项目的权重在进入或离开激活集的那一刻保证为零。
We use a weighted average as our pooling operator where the weight of each item in our active set is a continuous function of the query light direction, and where the weight of each item is guaranteed to be zero at the moment it enters or leaves the active set.

$W(\ell,\ell_i)$: 权重函数
$$
\tilde{W}(\ell,\ell_i)=\max\left(0,\exp(s(\ell\cdot\ell_i-1))-\min_{j\in A(\ell)}\exp(s(\ell\cdot\ell_j-1))\right)\\
W(\ell,\ell_i)=\frac{\tilde{W}(\ell,\ell_i)}{\sum_{j}\tilde{W}(\ell,\ell_j)}
$$
$s$: 可学习的变量，根据距离调整权重的衰减，每个 $\ell$ 是 3D 空间中的归一化向量。

### Network Architecture

#### 编码器

$3\times3$ 卷积神经网络，步长为 2，每一步都做 group normalization，激活函数 PReLU。

一开始 32 通道，逐渐翻倍到 512 通道。

输入是 8 张 RGB 图片，图片上的每一个像素都和输入方向的 $x,y,z$ 值叠加了，形成了 8 张 6 通道的输入图片，形成一个 batch。

每次过卷积层后都汇聚成一张 n 通道的图片，用于解码器中。

#### 解码器

解码器从全连接层开始，对于 $3$ 个节点的输入方向，先全连接到 $32$ 个节点，再 $128$ 个节点，再 $512$ 个节点，每一步做 instance normalization，激活函数 PReLU。然后上采样到 $4\times4\times512$。

然后是 $3\times3$ 转置卷积神经网络，步长为 2，每一步都做 group normalization，激活函数 PReLU。

每个卷积层的输入是来自前一个解码器的上采样激活层的串联，来自编码器的汇聚激活层已从相同的空间尺度“跳过”连接。
The input to each layer’s conv block is a concatenation of the upsampled activations from the previous decoder level, with the pooled activations from the encoder that have been “skip” connected from the same spatial scale.

最后的输出层的激活函数是 sigmoid。

### Loss Functions and Training Strategy

#### Loss

L1 loss
$$
\mathcal L_d=\sum_i\|M\odot(I_i-I(\ell_i))\|_1
$$
$I_i$: ground truth

$I(\ell_i)$: 输出

$M$: 用于去掉背景的 mask

#### 训练策略

随机选取一个方向，Adam 优化器，batch size 为 1，learing rate $10^{-3}$。

### Progressive Traning



## Evaluation









