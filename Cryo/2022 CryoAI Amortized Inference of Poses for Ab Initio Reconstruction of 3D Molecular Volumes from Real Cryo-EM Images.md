# CryoAI: Amortized Inference of Poses for Ab Initio Reconstruction of 3D Molecular Volumes from Real Cryo-EM Images

## 0 Abstract

我们介绍了 cryoAI，一种用于均匀构象的从头算重建算法，它使用基于梯度的直接优化粒子位姿和来自单粒子 cryo-EM 的电子散射势。
We introduce cryoAI, an ab initio reconstruction algorithm for homogeneous conformations that uses direct gradient-based optimization of particle poses and the electron scattering potential from single-particle cryo-EM data.

CryoAI 将预测每个粒子图像位姿的学习编码器与基于物理的解码器相结合，以将每个粒子图像聚合成散射势体像的隐式表示。
CryoAI combines a learned encoder that predicts the poses of each particle image with a physics based decoder to aggregate each particle image into an implicit representation of the scattering potential volume.

该体像存储在傅立叶域中以提高计算效率，并利用现代坐标网络架构来提高内存效率。结合对称损失函数，该框架在模拟和实验数据方面达到了与最先进的 cryo-EM 相当的质量结果，大型数据集的速度提高了一个数量级，并且内存需求显着低于现有的方法。
This volume is stored in the Fourier domain for computational efficiency and leverages a modern coordinate network architecture for memory efficiency. Combined with a symmetrized loss function, this framework achieves results of a quality on par with state-of-the-art cryo-EM solvers for both simulated and experimental data, one order of magnitude faster for large datasets and with significantly lower memory requirements than existing methods.

## 1 Introduction

样品制备、仪器和数据收集能力的最新进展导致每次 cryo-EM 实验都记录了大量数据。
Recent advances in sample preparation, instrumentation, and data collection capabilities have resulted in very large amounts of data being recorded for each cryo-EM experiment.

尽管使用了智能姿态搜索策略和优化计划，但方向匹配步骤是现有 cryo-EM 重建算法的主要瓶颈，需要数小时来估计单个体像，并且随着数据集大小的增加而性能不佳。
In spite of using smart pose search strategies and optimization schedules, the orientation matching step is the primary bottleneck of existing cryo-EM reconstruction algorithms, requiring hours to estimate a single volume and scaling poorly with increasing dataset sizes.

我们的方法使用编码器-解码器流水线以无监督的方式在一组图像上运行。
Our method operates in an unsupervised manner over a set of images with an encoder–decoder pipeline.

编码器学习一个判别模型，将每个粒子图像与一个位姿 $\phi\in\mathrm{SO}(3)\times\mathbb{R}^2$ 相关联，而解码器是一个基于生成物理的流水线，它使用预测的位姿和体像描述来预测图像。
The encoder learns a discriminative model that associates each particle image with a pose and the decoder is a generative physics-based pipeline that uses the predicted pose and a description of the volume to predict an image.

体像由解码器中的隐式（即神经网络参数化）的表示维持，并且成像模型在傅里叶空间中模拟，从而避免通过傅里叶切片定理进行积分近似。
The volume is maintained by an implicit, i.e., neural network–parameterized, representation in the decoder, and the image formation model is simulated in Fourier space, thereby avoiding the approximation of integrals via the Fourier-slice theorem.

通过学习从图像到位姿的映射，cryoAI 避免了计算昂贵的方向匹配步骤，该步骤限制了现有的 cryo-EM 重建方法。
By learning a mapping from images to poses, cryoAI avoids the computationally expensive step of orientation matching that limits existing cryo-EM reconstruction methods.

具体来说，我们的贡献包括
Specifically, our contributions include

- 一个将图像映射到粒子位姿的学习框架，同时为均匀的 cryo-EM 单粒子重建电子散射势；
  a framework that learns to map images to particle poses while reconstructing an electron scattering potential for homogeneous single-particle cryo-EM;
- 演示在数据集大小上摊销的重建时间和内存消耗，与大型数据集上的现有算法相比几乎有一个数量级的改进；
  demonstration of reconstruction times and memory consumption that amortize over the size of the dataset, with nearly an order of magnitude improvement over existing algorithms on large datasets;
- 对称损失函数和隐式傅里叶域体像表示的公式，使我们展示的高质量重建成为可能。
  formulations of a symmetrized loss function and an implicit Fourier-domain volume representation that enable the high-quality reconstructions we show.

## 2 Related Work

## 3 Methods

### 3.1 Image Formation Model and Fourier-slice Theorem

在 cryo-EM 样品中，每个分子及其周围环境携带的电荷会产生一个静电势，该静电势会散射探测电子，我们将其称为电子散射“体像”，并将其视为映射：
In a cryo-EM sample, the charges carried by each molecule and their surrounding environment create an electrostatic potential that scatters probing electrons, which we refer to as the electron scattering “volume,” and consider as a mapping.
$$
V:\mathbb{R}^3\to\mathbb{R}
$$

在样本中，每个分子 $i$ 的方向是 $R_i\in\mathrm{SO}(3)\subset\mathbb{R}^{3\times3}$ 中未知的一个。探测电子束与静电势相互作用，其投影 $Q_i$ 被认为是从 $\mathbb{R}^2$ 到 $\mathbb{R}$ 的映射。
In the sample, each molecule $i$ is in an unknown orientation $R_i\in\mathrm{SO}(3)\subset\mathbb{R}^{3\times3}$. The probing electron beam interacts with the electrostatic potential and its projections $Q_i$ are considered mappings from $\mathbb{R}^2$ to $\mathbb{R}$.
$$
Q_i:(x,y)\mapsto\int_{z}V\left(R_i\begin{bmatrix}x\\y\\z\end{bmatrix}\right)\mathrm{d}z
$$

然后，光束与以点扩散函数 (PSF) $P_{i}$ 为特征的透镜系统相互作用，并从完整的显微照片中裁剪出单个粒子。获得的图像可能不完全以分子为中心，小平移由 $\mathbf{t}_{i}\in\mathbb{R}^{2}$ 建模。
The beam then interacts with the lens system characterized by the Point Spread Function (PSF) $P_{i}$ and individual particles are cropped from the full micrograph. The obtained images may not be perfectly centered on the molecule and small translations are modeled by $\mathbf{t}_{i}\in\mathbb{R}^{2}$.

最后，考虑到分子嵌入的玻璃冰产生的信号以及透镜和探测器的非理想性，每个图像 $Y_{i}$ 通常建模为：
Finally, taking into account signal arising from the vitreous ice into which the molecules are embedded as well as the non-idealities of the lens and the detector, each image $Y_{i}$ is generally modeled as:
$$
Y_{i}=T_{\mathbf{t}_{i}}*P_{i}*Q_{i}+\eta_{i}
$$

其中 $*$ 是卷积算子，$T_{\mathbf{t}}$ 是 $\mathbf{t}$-平移核，$\eta_{i}$ 是 $\mathbb{R}^2$ 上的高斯白噪声。
where $*$ is the convolution operator, $T_{\mathbf{t}}$ the $\mathbf{t}$-translation kernel and $\eta_{i}$ white Gaussian noise on $\mathbb{R}^2$.

避免积分和卷积的一种方法是使用傅里叶切片定理，该定理指出对于任何体像 $V$ 和任何方向 $R_{i}$，
A way to avoid integral and convolution is to use the Fourier-slice Theorem, which states that for any volume $V$ and any orientation $R_{i}$,
$$
\mathcal{F}_{2\mathrm{D}}[Q_{i}]=\mathcal{S}_{i}[\mathcal{F}_{3\mathrm{D}}[V]]
$$
其中 $\mathcal{F}_{2\mathrm{D}}$ 和 $\mathcal{F}_{3\mathrm{D}}$ 是 2D 和 3D 傅里叶变换算子，$\mathcal{S}_ {i}$ 切片运算符定义为对于任何 $\hat{V}:\mathbb{R}^3\mapsto\mathbb{C}$，
where $\mathcal{F}_{2\mathrm{D}}$ and $\mathcal{F}_{3\mathrm{D}}$ are the 2D and 3D Fourier transform operators and $\mathcal{S}_{i}$ the slice operator defined such that for any $\hat{V}:\mathbb{R}^3\mapsto\mathbb{C}$,
$$
\mathcal{S}_{i}[\hat{V}]:(k_x,k_y)\mapsto\hat{V}\left(R_i\begin{bmatrix}k_x\\k_y\\0\end{bmatrix}\right)
$$
简而言之
In a nutshell:
$$
\begin{align*}
\mathcal{F}[Y_{i}]&=\mathcal{F}[T_{\mathbf{t}_{i}}*P_{i}*Q_{i}+\eta_{i}]\\
&=\hat{T_{\mathbf{t}_{i}}}\odot\hat{P}_{i}\odot\hat{Q}_{i}+\hat{\eta}_{i}\\&=\hat{T_{\mathbf{t}_{i}}}\odot C_{i}\odot \mathcal{S}_{i}[\hat{V}]+\hat{\eta}_{i}\\
\end{align*}
$$
$C_i=\mathcal{F}_{2\mathrm{D}}$ 是对比度传递函数 (CTF)。
$C_i=\mathcal{F}_{2\mathrm{D}}$ is the Contrast Transfer Function (CTF).



$V:\mathbb{R}^3\to\mathbb{R}$ 体像，空间中某一点的电势能。
$Q_i:(x,y)\mapsto\int_{z}V\left(R_i\begin{bmatrix}x\\y\\z\end{bmatrix}\right)\mathrm{d}z$ 将体像旋转 $R_i$ 后 $z$ 方向上的累计值。
$\mathbf{t}_{i}\in\mathbb{R}^{2}$: 体像中心的平移量。
$Y_i$: 重建得到的图像。
$\hat{}$: Fourier 变换后的信号。
$\mathcal{S}_{i}[\hat{V}]:(k_x,k_y)\mapsto\hat{V}\left(R_i\begin{bmatrix}k_x\\k_y\\0\end{bmatrix}\right)$: 切片操作。

### 3.2 Overview of CryoAI

![](..\Note Image\cryo-AI pipeline.png)

编码器将图像 $Y_i$ 作为输入并输出预测方向 $R_i$ 以及预测平移 $\mathbf{t}_i$。$R_i$ 用于旋转大小为 $L^2$ 3D 坐标 $[k_x,k_y,0]\in\mathbb{R}^3$ 的 2D 网格，然后将其输入到神经网络 $\hat{V }_{\theta}$。
The encoder takes an image $Y_i$ as input and outputs a predicted orientation $R_i$ along with a predicted translation $\mathbf{t}_i$. $R_i$ is used to rotate a 2D grid of $L^2$ 3D-coordinate $[k_x,k_y,0]\in\mathbb{R}^3$ which are then fed into the neural network $\hat{V}_{\theta}$.

这个神经网络是当前估计的体像 $\hat{V}$（在傅里叶空间中）的隐式表示，这个查询操作对应于“切片”。
This neural network is an implicit representation of the current estimate of the volume $\hat{V}$ (in Fourier space), and this query operation corresponds to the “slicing”.

基于估计的平移 $\mathbf{t}_i$ 和给定的 CTF 参数 $C_i$，模拟成像模型的其余部分以获得 $\hat{X}_i$，即无噪声估计的 $\hat{Y}_i$。
Based on the estimated translation $\mathbf{t}_i$ and given CTF parameters $C_i$, the rest of the image formation model is simulated to obtain $\hat{X}_i$, a noise-free estimation of $\hat{Y}_i$.

这些图像使用损失进行比较，梯度在整个可微模型中反向传播，以优化编码器和神经表示。
These images are compared using a loss and gradients are back-propagated throughout the differentiable model in order to optimize both the encoder and the neural representation.

### 3.3 Pose Estimation

CryoAI 使用卷积神经网络从给定图像中预测参数 $R_i$ 和 $\mathbf{t}_i$，从而避免昂贵的方向匹配计算。
CryoAI uses a Convolutional Neural Network to predict the parameters $R_i$ and $\mathbf{t}_i$ from a given image, thereby avoiding expensive orientation matching computations.

该编码器的架构具有三层。
The architecture of this encoder has three layers.

- 低通滤波：$Y_i\in\mathbb{R}^{L\times L}$ 被送入一组高斯低通滤波器。
  Low-pass filtering: $Y_i\in\mathbb{R}^{L\times L}$ is fed into a bank of Gaussian low-pass filters.
- 特征提取：过滤后的图像按通道堆叠并输入 CNN，其架构的灵感来自 VGG16 的第一层，众所周知，它在图像分类任务中表现良好。
  Feature extraction: the filtered images are stacked channel-wise and fed into a CNN whose architecture is inspired by the first layers of VGG16, which is known to perform well on image classification tasks.
- 位姿估计：这个特征向量最终成为两个独立的全连接神经网络的输入。第一个输出 $S^2\times S^2$ 的维数为 6 的向量（$\mathbb{R}^3$ 中单位球面上的两个向量）并转换成矩阵 $R_i\in\mathbb {R}^{3\times 3}$。第二个输出一个维度为 2 的向量，直接解释为平移向量 $\mathbf{t}_i\in\mathbb{R}^2$。
  Pose estimation: this feature vector finally becomes the input of two separate fully-connected neural networks. The first one outputs a vector of dimension of 6 of $S^2\times S^2$ (two vectors on the unitary sphere in $\mathbb{R}^3$) and converted into a matrix $R_i\in\mathbb{R}^{3\times 3}$. The second one outputs a vector of dimension 2, directly interpreted as a translation vector $\mathbf{t}_i\in\mathbb{R}^2$.

### 3.4 Neural Representation in Fourier Space (FourierNet)

我们使用神经表示来维持当前的体积估计。该表示由 $\theta$ 参数化，可以看作是一个映射 $\hat{V}_{\theta}:\mathbb{R}^3\mapsto\mathbb{C}$。
We maintain the current estimate of the volume using a neural representation. This representation is parameterized by $\theta$ and can be see seen as a mapping $\hat{V}_{\theta}:\mathbb{R}^3\mapsto\mathbb{C}$.

由于这个电势能是空间坐标的平滑函数，其傅里叶系数 $\hat{V}(\mathbf{k})$ 的幅度预计会随着 $|\mathbf{k}|$ 而减小，遵循幂律。在实践中，这意味着 $|\hat{V}|$ 可以在几个数量级上变化，并且已知 SIREN 很难近似这类函数。
Since this potential is a smooth function of the spatial coordinates, the amplitude of its Fourier coefficients $\hat{V}(\mathbf{k})$ is expected to decrease with $|\mathbf{k}|$, following a power law. In practice, this implies that $|\hat{V}|$ can vary over several orders of magnitude and SIRENs are known to poorly approximate these types of functions.

我们的想法是让 SIREN 通过以指数函数提高其输出来表示 HDR 信号。换句话说，SIREN 仅表示与近似函数成对数比例的信号。
Our idea is to allow a SIREN to represent a signal with a high dynamic range by raising its output in an exponential function. Said differently, the SIREN only represents a signal that scales logarithmically with the approximated function.

由于傅立叶系数是在复平面上定义的，因此我们在隐式表示中使用第二个网络来解释相位变化。
Since Fourier coefficients are defined on the complex plane, we use a second network in our implicit representation to account for the phase variations.

由于 $\hat{V}_{\theta}$ 必须表示真实信号的傅里叶变换，我们规定
Since $\hat{V}_{\theta}$ must represent the Fourier transform of real signals, we enforce this property by defining
$$
\hat{V}_{\theta}(\mathbf{k})=\hat{V}_{\theta}^{*}(-\mathbf{k})\quad k_x<0
$$





