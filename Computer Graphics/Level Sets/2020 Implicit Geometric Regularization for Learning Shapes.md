# Implicit Geometric Regularization for Learning Shapes

## 0 Abstract

将形状表示为神经网络的水平集是使用以下任一方法计算的：（i）预先计算的隐式形状表示； 或 (ii) 在神经水平集上明确定义的损失函数。
Representing shapes as level sets of neural net works were computed using either: (i) pre-computed implicit shape representations; or (ii) loss functions explicitly defined over the neural level sets.

我们提供了一种新的范式，用于直接从原始数据（即点云，有或没有法向信息）计算高保真隐式神经表示。
we offer a new paradigm for computing high fidelity implicit neural representations directly from raw data (i.e., point clouds, with or without normal information).

我们观察到一个相当简单的损失函数，鼓励神经网络在输入点云上消失并具有单位范数梯度，具有隐式几何正则化特性，有利于平滑和自然的零水平集表面，避免糟糕的零损失解决方案。
We observe that a rather simple loss function, encouraging the neural network to vanish on the input point cloud and to have a unit norm gradient, possesses an implicit geometric regularization property that favors smooth and natural zero level set surfaces, avoiding bad zero-loss solutions.

## 1 Introduction

$$
\mathcal{M}=\{\mathbf{x}\in\mathbb{R}^3\mid f(\mathbf{x};\theta)=0, f:\mathbb{R}^3\times\mathbb{R}^m\mapsto\mathbb{R}\}
$$

神经隐式表示的好处是将网络的自由度（即参数）直接与形状相关联，而不是与环境 3D 空间的固定离散化相关联。
Neural implicit representations have the benefit of relating the degrees of freedom of the network (i.e., parameters) directly to the shape rather than to the fixed discretization of the ambient 3D space.

给定一个输入点云 $\mathcal{X}=\{\mathbf{x}_i\}\subset\mathbb{R}^3$，有或没有法向数据 $\mathcal{N}=\{\mathbf{ n}_i\}\subset\mathbb{R}^3$，我们的目标是计算 $\theta$ 使得 $f(\mathbf{x};\theta)$ 近似于一个可信的、由点数据 $\mathcal{X}$ 和法线 $\mathcal{N}$ 定义的有符号距离函数 $\mathcal{M}$ 的表面。
Given an input point cloud $\mathcal{X}=\{\mathbf{x}_i\}\subset\mathbb{R}^3$, with or without normal data $\mathcal{N}=\{\mathbf{n}_i\}\subset\mathbb{R}^3$, our goal is to compute $\theta$ such that $f(\mathbf{x};\theta)$ is approximately the signed distance function to a plausible surface $\mathcal{M}$ defined by the point data $\mathcal{X}$ and normals $\mathcal{N}$.

首先，从原始数据中找到隐含的表面表示是出了名的困难；其次，将重建与学习阶段脱钩会阻碍集体学习和形状的重建。例如，来自一种形状的信息不会用于改进对不同但相似形状的重建；也不会产生一致的重建。
First, finding implicit surface representation from raw data is a notoriously hard; second, decoupling the reconstruction from the learning stage would hinder collective learning and reconstruction of shapes. For example, information from one shape will not be used to improve reconstruction of a different, yet a similar shape; nor consistent reconstructions will be produced.

请注意，尽管有无数的解决方案使用神经水平集对输入数据进行插值，但优化达到了提供自然和直观重建的解决方案。
Note that although there is an infinit number of solutions with neural level sets interpolating the input data, the optimization reaches solutions that provide natural and intuitive reconstructions.

优化过程找到的优选局部最小值可以看作是神经网络优化中众所周知的隐式正则化现象的几何版本。
The preferable local minima found by the optimization procedure could be seen as a geometric version of the well known implicit regularization phenomenon in neural network optimization.

我们证明，如果从超平面 $\mathcal{H}$ 对点云 $\mathcal{X}$ 进行采样（偏差很小），并且线性模型的初始参数是随机的，那么，梯度下降收敛到超平面 $\mathcal{H}$ 的（近似）有符号距离函数概率为 1，避免了不好的临界解。
We prove that if the point cloud $\mathcal{X}$ is sampled (with small deviations) from a hyper-plane $\mathcal{H}$ and the initial parameters of the linear model are randomized, then, with probability one, gradient descent converges to the (approximated) signed distance function of the hyperplane $\mathcal{H}$, avoiding bad critical solutions.

## 2 Method

我们考虑损失函数的形式
We consider a loss of the form
$$
\ell(\theta)=\ell_{\mathcal{X}}(\theta)+\lambda\mathbb{E}_{\mathbf{x}}\bigl(\|\nabla_{\mathbf{x}}f(\mathbf{x};\theta)\|_2-1\bigr)^2
$$
where $\lambda>0$.

$$
\ell_{\mathcal{X}}(\theta)=\frac{1}{|I|}\sum_{i\in{I}}\bigl(|f(\mathbf{x}_i;\theta)|+\tau\|\nabla_{\mathbf{x}}f(\mathbf{x}_{i};\theta)-\mathbf{n}_i\|\bigr)
$$
鼓励 $f$ 在 $\mathcal{X}$ 上消失，并且，如果存在正常数据（即 $\tau=1$），则 $\nabla_{\mathbf{x}}f$ 接近提供的法向数据 $\mathcal{N}$。
encourages $f$ to vanish on $\mathcal{X}$ and, if normal data exists (i.e., $\tau=1$), that $\nabla_{\mathbf{x}}f$ is close to the supplied normals $\mathcal{N}$.

Eikonal 项鼓励梯度 $\nabla_{\mathbf{x}}f$ 的 2 范数为 1。期望是针对 $\mathbb{R}^3$ 中的某个概率分布 $\mathbf{x}\sim\mathcal{D}$ 进行的。
Eikonal term encourages the gradients $\nabla_{\mathbf{x}}f$ to be unit 2-norm. The expectation is taken with respect to some probability distribution $\mathbf{x}\sim\mathcal{D}$ in $\mathbb{R}^3$.

首先，为什么优化算法找到的临界点 $\theta^*$ 会导致 $f(\mathbf{x};\theta^*)$ 成为有符号距离函数？通常，添加具有有限权重的二次惩罚不能保证提供可行的关键解决方案。
First, why a critical point $\theta^*$ that is found by the optimization algorithm leads $f(\mathbf{x};\theta^*)$ to be a signed distance function? Usually, adding a quadratic penalty with a finite weight is not guaranteed to provide feasible critical solutions.

其次，即使找到的关键解决方案是有符号距离函数，为什么它会是一个合理的解决方案？因为在具有任意法线方向 $\mathcal{N}$ 的任意离散点集合 $\mathcal{X}$ 上存在无限数量的有符号距离函数。
Second, even if the critical solution found is a signed distance function, why would it be a plausible one? There is an infinite number of signed distance functions vanishing on arbitrary discrete sets of points $\mathcal{X}$ with arbitrary normal directions $\mathcal{N}$.

值得注意的是，使用随机梯度下降（或其变体）优化损失函数会产生接近有符号距离函数的解，具有平滑且令人惊讶的合理零水平集。
Remarkably, optimizing loss function using stochastic gradient descent (or a variant thereof) results in solutions that are close to a signed distance function with a smooth and surprisingly plausible zero level set.

MLP 的每一层 $f$ 的形式为 $\mathbf{y}^{l+1}=\sigma(\mathbf{W}\mathbf{y}^l+\mathbf{b})$，其中 $\sigma:\mathbb{R}\mapsto\mathbb{R}$ 是一个非线性可微激活函数（我们使用 softplus），并且 $\mathbf{W}$ 和 $\mathbf{b}$ 是每一层的可学习参数。因此，通过链式法则，梯度满足
Every layer of the MLP $f$ has the form $\mathbf{y}^{l+1}=\sigma(\mathbf{W}\mathbf{y}^l+\mathbf{b})$, where $\sigma:\mathbb{R}\mapsto\mathbb{R}$ is a non-linear differentiable activation (we use softplus) applied enrty-wise, and $\mathbf{W}$ and $\mathbf{b}$ are the layer's learnable parameters. Hence, by the chain-rule the gradients satisfy
$$
\nabla_{\mathbf{x}}\mathbf{y}^{l+1}=\mathrm{diag}(\sigma'(\mathbf{W}\mathbf{y}^l+\mathbf{b}))\mathbf{W}\cdot\nabla_{\mathbf{x}}\mathbf{y}^l
$$

其中 $\mathrm{diag}(\mathbf{z})$ 将其输入向量 $\mathbf{z}\in\mathbb{R}^k$ 排列在方阵 $\mathbb{R}^{k\times k}$ 的对角线上，$\sigma'$ 是 $\sigma$ 的导数。
where $\mathrm{diag}(\mathbf{z})$ is arranging its input vector $\mathbf{z}\in\mathbb{R}^k$ on the diagonal of a square matrix $\mathbb{R}^{k\times k}$ and $\sigma'$ is the derivative of $\sigma$.

![[../../Note Image/IGR network.png]]


## 3 Previous work and discussion

### 3.1 Deep learning of 3D shapes

### 3.2 Solving PDEs with neural networks

## 4 Analysis of the linear model and plane reproduction

我们考虑一个线性模型 $f(\mathbf{x};\mathbf{w})=\mathbf{w}^T\mathbf{x}$，其中损失采用以下形式：
We consider a linear model $f(\mathbf{x};\mathbf{w})=\mathbf{w}^T\mathbf{x}$ where the loss takes the form:
$$
\ell(\mathbf{w})=\sum_{i}(\mathbf{w}^T\mathbf{x}_i)^2+\lambda\bigl(\|\mathbf{w}\|^2-1\bigr)^2
$$
为简单起见，我们使用平方误差并删除了涉及法向的项；我们在 $\mathbb{R}^{d}$ 而不是 $\mathbb{R}^{3}$ 进行分析。
where for simplicity we used squared error and removed the term involving normal data; we present the analysis in $\mathbb{R}^{d}$ rather than $\mathbb{R}^{3}$.

假设我们的点云 $\mathcal{X}$ 是从平面 H 近似采样的，那么简化的损失函数中损失的梯度下降会收敛到 $\mathcal{H}$ 的近似有符号距离函数。
Assuming our point cloud $\mathcal{X}$ is sampled approximately from a plane H, then gradient descent of the simpliftied loss converges to the approximate signed distance function to $\mathcal{H}$.









