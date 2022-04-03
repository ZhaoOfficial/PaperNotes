# DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation

## 0 Abstract

DeepSDF，一类形状的学习连续符号距离函数 (SDF) 表示，可以实现高质量地表示形状、插值和从部分和嘈杂的 3D 输入数据中补全。
DeepSDF, a learned continuous Signed Distance Function (SDF) representation of a class of shapes that enables high quality shape representation, interpolation and completion from partial and noisy 3D input data.

DeepSDF 与其经典对应物一样，通过连续体积场表示形状的表面：场中点的大小表示到表面边界的距离，符号表示该区域是在内部 (-) 还是外部 (+) 形状，因此我们的表示隐含地将形状的边界编码为学习函数的零势集，同时明确地将空间分类表示为形状内部的一部分。
DeepSDF, like its classical counterpart, represents a shape's surface by a continuous volumetric field: the magnitude of a point in the field represents the distance to the surface boundary and the sign indicates whether the region is inside (-) or outside (+) of the shape, hence our representation implicitly encodes a shape's boundary as the zero-level-set of the learned function while explicitly representing the classification of space as being part of the shapes' interior or not.

## 1 Introduction

所提出的连续表示可以直观地理解为一个学习的形状条件分类器，其决策边界是形状本身的表面。
The proposed continuous representation may be intuitively understood as a learned shape-conditioned classifier for which the decision boundary is the surface of the shape itself.

我们的贡献包括：（i）具有连续隐式表面的生成形状条件 3D 建模的公式，（ii）基于概率自动解码器的 3D 形状学习方法，以及（iii）该公式的塑造造型和完成的演示和应用。
Our contributions include: (i) the formulation of generative shape-conditioned 3D modelling with a continuous implicit surface, (ii) a learning method for 3D shapes based on a probabilistic auto-decoder, and (iii) the demonstration and application of this formulation to shape modelling and completion.

## 2 Related Work

### 2.1 Representations for 3D Shape Learning

#### Point-based

### 2.2 Representation Learning Techniques

### 2.3 Shape Completion

## 3 Modelling SDFs with Neural Networks

SDF 是一个连续函数，对于给定的空间点，输出该点到最近表面的距离，其符号由该点是在表面的内部（负）还是外部（正）决定。表面由 $\mathrm{SDF}(\mathbf x)=0$ 的等值面隐式表示。
A signed distance function is a continuous function that, for a given spatial point, outputs the point's distance to the closest surface, whose sign encodes whether the point is inside (negative) or outside (positive) of the watertight surface. The underlying surface is implicitly represented by the iso-surface of $\mathrm{SDF}(\mathbf x)=0$.
$$
\mathrm{SDF}:\mathbb R^3\mapsto\mathbb R,\mathrm{SDF}(\mathbf x)=s
$$
我们的关键思想是使用深度神经网络直接从点样本中回归连续 SDF。由此产生的训练网络能够预测给定查询位置的 SDF 值，我们可以通过评估空间样本提取零势集表面。
Our key idea is to directly regress the continuous SDF from point samples using deep neural networks. The resulting trained network is able to predict the SDF value of a given query position, from which we can extract the zero level-set surface by evaluating spatial samples.
$$
f_{\theta}(\mathbf x)\approx\mathrm{SDF}(\mathbf x),\forall\mathbf x\in\Omega\\
\mathcal L(f_{\theta}(\mathbf x),s)=|\mathrm{clamp}(f_{\theta}(\mathbf x),\delta)-\mathrm{clamp}(s,\delta)|
$$
参数 $\delta$​ 控制与我们期望保持度量 SDF 的表面的距离。较大的 $\delta$​ 值允许快速光线追踪，因为每个样本都提供了安全步长的信息。较小的 $\delta$​ 可用于将网络容量集中在地表附近的细节上。我们在实践中使用 $\delta=0.1$​。
The parameter $\delta$​ controls the distance from the surface over which we expect to maintain a metric SDF. Larger values of $\delta$​ allow for fast ray-tracing since each sample gives information of safe step size. Smaller $\delta$​ can be used to concentrate network capacity on details near the surface. We uses $\delta=0.1$​ in practice.

解码器是一个前馈网络，由八个全连接层组成，每个层都应用了 dropout。所有内部层都是 512 维的并且用了 ReLU。tanh 输出非线性的回归 SDF 值。我们发现使用批量归一化进行训练是不稳定的，因此改用了权重归一化技术。对于训练，我们使用 Adam 优化器。
Decoder is a feed-forward network composed of eight fully connected layers, each of them applied with dropouts. All internal layers are 512-dimensional and have ReLU non-linearities. The output non-linearity regressing the SDF value is tanh. We found training with batch-normalization to be unstable and applied the weight-normalization technique instead. For training, we use the Adam optimizer.

## 4 Learning the Latent Space of Shapes

我们想要一个可以表示各种形状、发现它们的共同属性并将它们嵌入到低维潜在空间中的模型。
We want a model that can represent a wide variety of shapes, discover their common properties, and embed them in a low dimensional latent space.
$$
f_{\theta}(\mathbf z_i,\mathbf x)\approx\mathrm{SDF}_i(\mathbf x)
$$
for some shape indexed by $i$.

通过在潜在向量上调节网络输出，该公式允许使用单个神经网络对多个 SDF 进行建模。
By conditioning the network output on a latent vector, this formulation allows modelling multiple SDFs with a single neural network.

### 4.1 Motivating Encoder-less Learning

由于经过训练的编码器在测试时未使用，因此尚不清楚 (1) 训练编码器是否是对计算资源的有效利用，以及 (2) 研究人员是否有必要为各种 3D 输入表示（例如点、网格、八叉树）设计编码器，等等）。
Since the trained encoder is unused at test time, it is unclear whether (1) training encoder is an effective use of computational resources and (2) it is necessary for researchers to design encoders for various 3D input representations (e.g. points, meshes, octrees, etc).

### 4.2 Auto-decoder-based DeepSDF Formulation

给定一个数据集，如果 $N$ 个形状用符号距离函数 $\mathrm{SDF}_1, \dots,\mathrm{SDF}_N$ 表示，我们准备一组 $K$ 个点样本及其符号距离值：
Given a dataset if $N$ shapes represented with signed distance function $\mathrm{SDF}_1, \dots,\mathrm{SDF}_N$, we prepare a set of $K$ point samples and their signed distance values:
$$
X_i=\{(\mathbf x_j,s_j):s_j=\mathrm{SDF}_i(\mathbf x_j)\}
$$
对于自动解码器，由于没有编码器，每个潜在代码 $\mathbf z_i$ 与训练形状 $X_i$ 配对。
For an auto-decoder, as there is no encoder, each latent code $\mathbf z_i$ is paired with training shape $X_i$.
$$
P_{\theta}(\mathbf z_i|X_i)=\frac{P(\mathbf z_i)P_{\theta}(X_i|\mathbf z_i)}{P_{\theta}(X_i)}=P(\mathbf z_i)\prod_{(\mathbf x_j,s_j)\in X_i}P_{\theta}(s_j|\mathbf z_j,\mathbf x_j)
$$
$\theta$ 参数化了 SDF 的似然。$P(\mathbf z_i)\sim\sigma^2I$。

在基于自动解码器的 DeepSDF 公式中，我们通过深度前馈网络 $f_{\theta}(\mathbf z_i,\mathbf x_i)$ 表达 SDF 的似然，并且不失一般性，假设似然的形式为：
In the auto-decoder-based DeepSDF formulation we express the SDF likelihood via a deep feed-forward network $f_{\theta}(\mathbf z_i,\mathbf x_i)$ and, without loss of generality, assume that the likelihood takes the form:
$$
P_{\theta}(s_j|\mathbf z_j,\mathbf x_j)=\exp(-\mathcal L(f_{\theta}(\mathbf z_j,\mathbf x_j),s_j))
$$
损失函数是前面描述的那个。

在训练时，我们最大化所有训练形状上的联合对数后验对于单个形状码 $\{\mathbf z_i\}_{i=1}^{N}$ 和网络参数 $\theta$：
At training time we maximize the joint log posterior over all training shapes with respect to the individual shape codes $\{\mathbf z_i\}_{i=1}^{N}$ and the network parameters $\theta$:
$$
\mathop{\arg\max}_\limits{\theta,\{\mathbf z_i\}_{i=1}^{N}}\sum_{i=1}^{N}\left(\sum_{j=1}^{K}\mathcal L(f_{\theta}(\mathbf z_i,\mathbf x_j), s_j)+\frac{1}{\sigma^2}\|\mathbf z_i\|_2^2\right)
$$
在推理时，在训练和固定 $\theta$ 之后，形状 $X_i$ 的形状码 $\mathbf z_i$ 可以通过最大后验估计 (MAP) 得到：
At inference time, after training and fixing $\theta$, a shape code $\mathbf z_i$ for shape $X_i$​ can be estimated via Maximum a Posterior (MAP) estimation as:
$$
\hat{\mathbf z}=\mathop{\arg\min}_{\mathbf z}\sum_{(\mathbf x_j,s_j)\in X}\mathcal L(f_{\theta}(\mathbf z_,\mathbf x_j), s_j)+\frac{1}{\sigma^2}\|\mathbf z_i\|_2^2
$$
