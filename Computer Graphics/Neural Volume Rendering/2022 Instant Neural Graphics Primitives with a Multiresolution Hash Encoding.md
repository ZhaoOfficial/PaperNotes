# Instant Neural Graphics Primitives with a Multiresolution Hash Encoding

## 0 Abstract

我们通过一种通用的新输入编码来降低成本，该编码允许在不牺牲质量的情况下使用更小的网络，从而显著减少浮点和内存访问操作的数量：一个小型神经网络通过可训练特征向量的多分辨率哈希表进行扩充，网络的值通过随机梯度下降优化。
We reduce this cost with a versatile new input encoding that permits the use of a smaller network without sacrificing quality, thus significantly reducing the number of floating point and memory access operations: a small neural network is augmented by a multiresolution hash table of trainable feature vectors whose values are optimized through stochastic gradient descent.

多分辨率结构允许网络消除哈希冲突的歧义，从而形成一个简单的架构，使其可以在现代 GPU 上并行化。
The multiresolution structure allows the network to disambiguate hash collisions, making for a simple architecture that is trivial to parallelize on modern GPUs.

## 1 Introduction

我们希望能在捕捉高频、局部细节的同时保持快速和紧凑的表示。由多层感知器（MLPs）表示的函数，作为**神经图形基元**，已经被证明符合这些标准。
We desire representations that remain fast and compact while capturing high-frequency, local detail. Functions represented by multi-layer perceptrons (MLPs), used as **neural graphics primitives**, have been shown to match these criteria.

上述方法的重要共同点是将神经网络输入映射到更高维空间的编码，这是从紧凑模型中提取高近似质量的关键。这些编码中最成功的是可训练的、特定于任务的数据结构，它们承担了大部分学习任务。
The important commonality of the aforementioned approaches is an encoding that maps neural network inputs to a higher-dimensional space, which is key for extracting high approximation quality from compact models. Most successful among these encodings are trainable, task-specific data structures that take on a large portion of the learning task.

但是，此类数据结构依赖于启发式和结构修改（例如修剪、拆分或合并），这可能会使训练过程复杂化，将方法限制在特定任务，或因为在 GPU 上，控制流和指针追踪成本高昂，而限制 GPU 的性能。
However, such data structures rely on heuristics and structural modifications (such as pruning, splitting, or merging) that may complicate the training process, limit the method to a specific task, or limit performance on GPUs where control flow and pointer chasing is expensive.

我们通过我们的*多分辨率哈希编码*来解决这些问题，这种编码具有**自适应性**和**高效性**，**独立于任务**。它仅由两个值配置 -- 参数数量 $T$ 和所需的最佳分辨率 $N_{\mathrm{max}}$ -- 只需要几秒钟的训练就可以在各种任务上产生最先进的质量。
We address these concerns with our *multiresolution hash encoding*, which is **adaptive** and **efficient**, **independent of the task**. It is configured by just two values -- the number of parameters $T$ and the desired finest resolution $N_{\mathrm{max}}$ -- yielding state-of-the-art quality on a variety of tasks after a few seconds of training.

**Task-independent Adaptivity**

我们将级联的网格映射到相应的固定大小的特征向量数组。在低分辨率下，网格点和数组元素之间存在 1:1 映射。在高分辨率下，该数组被视为一个哈希表并使用空间哈希函数进行索引，其中多个网格点对应一个数组元素。
We map a cascade of grids to corresponding fixed-size arrays of feature vectors. At coarse resolutions, there is a 1:1 mapping from grid points to array entries. At fine resolutions, the array is treated as a hash table and indexed using a spatial hash function, where multiple grid points alias each array entry.

这种哈希碰撞导致碰撞的训练梯度平均化，这意味着最大的梯度 —— 那些与损失函数最相关的梯度 —— 将占主导地位。
Such hash collisions cause the colliding training gradients to average, meaning that the largest gradients -- those most relevant to the loss function -- will dominate.

因此，哈希表会自动优先考虑具有最重要的精细细节的稀疏区域。与之前的工作不同，在训练期间的任何时候都不需要对数据结构进行结构更新。
The hash tables thus automatically prioritize the sparse areas with the most important fine scale detail. Unlike prior work, no structural updates to the data structure are needed at any point during training.

**Efficiency**

我们的哈希表查找是 $O(1)$ 的，并且不需要控制流。这可以很好地映射到现代 GPU，避免了树搜中固有的分支预测和串行指针追逐。可以并行地查询所有分辨率的哈希表。
Our hash table lookups are $O(1)$ and do not require control flow. This maps well to modern GPUs, avoiding execution divergence and serial pointer-chasing inherent in tree traversals. The hash tables for all resolutions may be queried in parallel.

Gigapixel image: $(x,y)\to(\mathrm{RGB})$.

Neural signed distance function: $(x,y,z)\to d$.

Neural radiance caching: $()$.

Neural radiance and density field: $(x,y,z,\theta,\phi)\to(\mathrm{RGB},\sigma)$

## 2 Background and Related Work

将机器学习模型的输入编码到更高维空间的早期示例包括 one-hot 编码和内核技巧，通过这些技巧可以使复杂的数据排列成为*线性可分*的。
Early examples of encoding the inputs of a machine learning model into a higher-dimensional space include the one-hot encoding and the kernel trick by which complex arrangements of data can be made *linearly separable*.

对于神经网络来说，输入编码已被证明在递归结构的注意力部分是有用的。随后，转化器，它们帮助神经网络识别它目前正在处理的位置。我们将标量位置 $x\in\mathbb R$ 编码为 $L\in\mathbb N$ 个正弦和余弦函数的多分辨率序列。
For neural networks, input encodings have proven useful in the attention components of recurrent architectures and, subsequently, transformers where they help the neural network to identify the location it is currently processing. We encode scalar positions $x\in\mathbb R$ as a multiresolution sequence of $L\in\mathbb N$ sine and cosine functions:
$$
\mathrm{enc}(x)=(\sin(2^0x),\dots,\sin(2^{L-1}x),\cos(2^0x),\dots,\cos(2^{L-1}x))
$$
我们将把这个系列的编码称为*频率编码*。
We will refer to this family of encodings as *frequency encodings*.

基于光栅化内核的独热编码的连续变体，即独球编码，在有界域中可以达到比频率编码更精确的结果，但代价是尺度单一。
A continuous variant of the one-hot encoding based on rasterizing a kernel, the one-blob encoding, which can achieve more accurate results than frequency encodings in bounded domains at the cost of being single-scale.

### Parametric encoding

这个想法是在辅助数据结构（例如网格或树）中安排额外的没有权重和偏差的可训练参数，并根据输入向量查找和插值（可选）这些参数 $\mathbf{x} \in\mathbb{R}^d$。
The idea is to arrange additional trainable parameters (beyond weights and biases) in an auxiliary data structure, such as a grid or a tree and to look-up and (optionally) interpolate these parameters depending on the input vector $\mathbf{x}\in\mathbb{R}^d$.

这种安排以更大的内存占用换取更小的计算成本：而对于通过网络向后传播的每个梯度，必须更新完全连接的 MLP 网络中的每个权重；对于可训练的输入编码参数（“特征向量”），只有极少数受到影响。例如，对于特征向量的三线性插值 3D 网格，对于每个反向传播到编码的样本，只需要更新 8 个这样的网格点。这样，尽管参数编码的参数总数远高于固定输入编码，但训练期间更新所需的 FLOP 和内存访问的数量并没有显着增加。
This arrangement trades a larger memory footprint for a smaller computational cost: whereas for each gradient propagated backwards through the network, every weight in the fully connected MLP network must be updated, for the trainable input encoding parameters ("feature vectors"), only a very small number are affected. For example, with a trilinearly interpolated 3D grid of feature vectors, only 8 such grid points need to be updated for each sample back-propagated to the encoding. In this way, although the total number of parameters is much higher for a parametric encoding than a fixed input encoding, the number of FLOPs and memory accesses required for the update during training is not increased significantly.

### Sparse parametric encoding

完全没有任何输入编码（a），网络只能学习相当平滑的位置函数，导致光场的逼近很差。
Without any input encoding at all (a), the network is only able to learn a fairly smooth function of position, resulting in a poor approximation of the light field.

频率编码 (b) 允许相同大小的网络（8 个隐藏层，每个 256 宽）更准确地表示场景。
The frequency encoding (b) allows the same moderately sized network (8 hidden layers, each 256 wide) to represent the scene much more accurately.

中间图像 (c) 将较小的网络与 $128^3$ 三线性插值的 16 维特征向量的密集网格配对，总共有 3355 万个可训练参数。
The middle image (c) pairs a smaller network with a dense grid of $128^3$ trilinearly interpolated, 16-dimensional feature vectors, for a total of 33.6 million trainable parameters.

然而，密集的网格在两个方面是浪费的。首先，它分配给空白区域的特征与分配给表面附近的区域一样多。参数的数量增长为 $O(N^3)$，而感兴趣的可见表面的表面积仅增长为 $O(N^2)$。在此示例中，网格的分辨率为 $128^3$，但只有 53807 (2.57%) 个单元格接触可见表面。
However, the dense grid is wasteful in two ways. First, it allocates as many features to areas of empty space as it does to those areas near the surface. The number of parameters grows as $O(N^3)$, while the visible surface of interest has surface area that grows only as $O(N^2)$. In this example, the grid has resolution $128^3$, but only 53807 (2.57%) of its cells touch the visible surface.

使用一种编码，其中插值特征存储在八个同位网格中，分辨率从 $16^3$ 到 $173^3$，每个网格包含二维特征向量。这些被连接起来形成网络的 16 维（与 (c) 相同）输入。尽管参数数量少于 (c) 的一半，但重建质量是相似的。
Using an encoding in which interpolated features are stored in eight co-located grids with resolutions from $16^3$ to $173^3$ each containing 2-dimensional feature vectors. These are concatenated to form a 16-dimensional (same as (c)) input to the network. Despite having less than half the number of parameters as (c), the reconstruction quality is similar.

## 3 Multiresolution Hash Encoding

给定一个全连接的神经网络 $m(\mathbf y;\Phi)$，我们对其输入的编码 $\mathbf y = \mathrm{enc}(x;\theta)$ 感兴趣，该编码可以广泛应用于*提高近似质量和训练速度，而不产生明显的性能开销。*
Given a fully connected neural network $m(\mathbf y;\Phi)$, we are interested in an encoding of its inputs $y = \mathrm{enc}(x;\theta)$ that *improves the approximation quality and training speed across a wide range of applications without incurring a notable performance overhead.*

我们的神经网络不仅有可训练的权重参数 $\Phi$，还有可训练的编码参数 $\theta$。这些参数被排列成 $L$ 个层次，每个层次最多包含 $T$ 个维度的特征向量 $F$。
Our neural network not only has trainable weight parameters $\Phi$, but also trainable encoding parameters $\theta$. These are arranged into $L$ levels, each containing up to $T$ feature vectors with dimensionality $F$.

| Parameter                               | Symbol     | Value             |
| --------------------------------------- | ---------- | ----------------- |
| Number of levels                        | $L$        | $16$              |
| Max entries per level (hash table size) | $T$        | $[2^{14},2^{24}]$ |
| Number of feature dimensions per entry  | $F$        | $2$               |
| Coarsest resolution                     | $N_{\min}$ | $16$              |
| Finest resolution                       | $N_{\max}$ | $[2^9,2^{19}]$    |

每个层次都是独立的，并且概念上是在**网格的顶点存储特征向量**，其分辨率 $N_l$ 选择为最粗和最细分辨率  $[N_{\min}, N_{\max}]$ 之间的几何级数：
Each level is independent and conceptually **stores feature vectors at the vertices of a grid**, the resolution of which is chosen to be a geometric progression between the coarsest and finest resolutions $[N_{\min}, N_{\max}]$:
$$
N_l\gets\lfloor N_\min\cdot b^l\rfloor\\
b\gets\exp\left(\frac{\ln N_{\max}-\ln N_{\min}}{L-1}\right)
$$
$N_{\max}$ 的选择是为了匹配训练数据中最细微的细节。由于大量的层次 $L$，增长因子通常很小。我们的例子里 $b\in[1.26, 2]$。
$N_{\max}$ is chosen to match the finest detail in the training data. Due to the large number of levels $L$, the growth factor is usually small. Our use cases have $b \in [1.26, 2]$.

[   9,   10,   11,   12,   13,   14,   15,   16,   17,   18,   19]
[1.26, 1.32, 1.38, 1.45, 1.52, 1.59, 1.66, 1.74, 1.82, 1.91, 2.00]

考虑一个单层 $l$。输入的坐标 $\mathbf x\in\mathbb R^d$ 在向下和向上取整之前，被该层的网格分辨率 $N_l$ 所缩放 $\lfloor\mathbf x_l\rfloor\gets\lfloor\mathbf x\cdot N_{l}\rfloor,\lceil\mathbf x_l\rceil\gets\lceil\mathbf x\cdot N_l\rceil$。
Consider a single level $l$. The input coordinate $\mathbf x\in\mathbb R^d$ is scaled by that level's grid resolution $N_l$ before rounding down and up $\lfloor\mathbf x_l\rfloor\gets\lfloor\mathbf x\cdot N_{l}\rfloor,\lceil\mathbf x_l\rceil\gets\lceil\mathbf x\cdot N_l\rceil$.

对于粗略层次，密集网格少于 $T$ 个参数的，即 $(N_l+1)^d\le T$，这种映射是 $1:1$ 的。在更精细的层次上，我们使用一个哈希函数 $h:\mathbb Z^d\mapsto\mathbb Z_T$ 来索引到数组中，有效地把它当作一个哈希表，尽管没有明确的碰撞处理。
For coarse levels where a dense grid requires fewer than $T$ parameters, i.e. $(N_l+1)^d\le T$, this mapping is $1:1$. At finer levels, we use a hash function $h:\mathbb Z^d\mapsto\mathbb Z_T$ to index into the array, effectively treating it as a hash table, although there is no explicit collision handling.

相反，我们依靠基于梯度的优化来在阵列中存储适当的稀疏细节，以及随后的神经网络 $m(\mathbf y;\Phi)$ 进行碰撞解决。因此，可训练的编码参数 $\Theta$ 的数量是 $O(T)$，并以 $T\cdot L\cdot F$ 为上界。
We rely instead on the gradient-based optimization to store appropriate sparse detail in the array, and the subsequent neural network $m(y;\Phi)$ for collision resolution. The number of trainable encoding parameters $\theta$ is therefore $O(T)$ and bounded by $T\cdot L\cdot F$.

我们使用以下形式的空间哈希函数，其中 $\oplus$ 表示各位异或操作，$\pi_i$ 是独特的大素数。实际上，这个公式将每个维度的线性同余（伪随机）排列的结果进行异或，去相关了维度对散列值的影响。

##### We use a spatial hash function of the form below where $\oplus$ denotes the bit-wise XOR operation and $\pi_i$ are unique, large prime numbers. Effectively, this formula XORs the results of a per-dimension linear congruential (pseudo-random) permutation, decorrelating the effect of the dimensions on the hashed value.

$$
h(\mathbf x)=\left(\bigoplus_{i=1}^{d}x_i\pi_i\right)
$$
值得注意的是，为了实现（伪）独立性，只有 $d-1$ 个维度必须被置换，因此我们选择 $\pi_1\gets 1$ 以获得更好的缓存一致性。$\pi_2=2654435761,\pi_3=805459861$.
Notably, to achieve (pseudo-)independence, only $d − 1$ of the $d$ dimensions must be permuted, so we choose $\pi_1 \gets 1$ for better cache coherence. $\pi_2=2654435761,\pi_3=805459861$。

最后，每个角的特征向量根据其在超立方体内的相对位置进行 $d$-线性插值，插值权重为 $\mathbf w_l\gets\mathbf x_l-\lfloor\mathbf x_l\rfloor$。
Lastly, the feature vectors at each corner are $d$-linearly interpolated according to the relative position of within its hypercube, i.e. the interpolation weight is $\mathbf w_l\gets\mathbf x_l-\lfloor\mathbf x_l\rfloor$.

回想一下，这个过程对于每个 $L$ 层级都是独立进行的。每一层的插值特征向量，以及辅助输入 $\xi\in\mathbb R^E$（例如神经辐射缓存中的编码视图方向和纹理），连接起来产生 $\mathbf y\in\mathbb R^{LF+E}$，它是 MLP $m(\mathbf y;\Phi)$ 的编码输入 $\mathrm{enc}(\mathbf x;\theta)$。
Recall that this process takes place independently for each of the $L$ levels. The interpolated feature vectors of each level, as well as auxiliary inputs $\xi\in\mathbb R^E$ (such as the encoded view direction and textures in neural radiance caching), are concatenated to produce $\mathbf y\in\mathbb R^{LF+E}$, which is the encoded input $\mathrm{enc}(\mathbf x;\theta)$ to the MLP $m(\mathbf y;\Phi)$.

### Performance vs. quality

选择散列表大小 $T$ 提供了性能、内存和质量之间的折衷。较高的 $T$ 值会导致较高的质量和较低的性能。内存占用随 $T$ 线性增长，而质量和性能往往呈次线性增长。
Choosing the hash table size $T$ provides a trade-off between performance, memory and quality. Higher values of $T$ result in higher quality and lower performance. The memory footprint is linear in $T$, whereas quality and performance tend to scale sub-linearly.

超参数 $L$ 和 $F$ 还权衡了质量和性能。
The hyperparameters $L$ and $F$ also trade off quality and performance.

### Implicit hash collision resolution

这种编码能够在存在哈希冲突的情况下忠实地重建场景，这似乎违反直觉。其成功的关键在于不同的分辨率级别具有相辅相成的不同优势。
It may appear counter-intuitive that this encoding is able to reconstruct scenes faithfully in the presence of hash collisions. Key to its success is that the differen resolution levels have different strengths that complement each other.

较粗的级别，因此作为一个整体的编码，是单射的 -- 也就是说，它们根本没有冲突。但是，它们只能表示场景的低分辨率版本，因为它们提供的特征是从宽间隔的点网格中线性插值的。
The coarser levels, and thus the encoding as a whole, are injective -- that is, they suffer from no collisions at all. However, they can only represent a low-resolution version of the scene, since they offer features which are linearly interpolated from a widely spaced grid of points.

相反，精细级别由于其精细的网格分辨率可以捕获小特征，但会遭受许多冲突 -- 即散列到同一个表条目的不同点。幸运的是，这种碰撞是伪随机分散在空间中的，并且从统计上讲，对于单个给定点同时发生在每个级别上的可能性很小。
Conversely, fine levels can capture small features due to their fine grid resolution, but will suffer from many collisions -- that is, disparate points which hash to the same table entry. Luckily, such collisions are pseudo-randomly scattered across space, and statistically very unlikely to occur at every level for simultaneously a single given point.

当训练样本以这种方式碰撞时，它们的梯度会平均。考虑到对此类样本的最终重建的重要性很少相等。
When training samples collide in this way, their gradients average. Consider that the importance to the final reconstruction of such samples is rarely equal.

例如，辐射场可见表面上的一个点将对重建图像有很大的贡献（具有高可见性和高密度，这两个项都会成倍地影响反向传播梯度的大小）导致其表条目发生较大变化，而一个碰巧在空白空间中引用相同条目的点，权重会小得多。结果，更重要的样本的梯度支配了碰撞平均值，并且有问题的别名表条目自然会以反映更高权重点的需求的方式进行优化；然后，不太重要的点将由多分辨率层次结构中的其他级别校正其最终输出。
For example, a point on a visible surface of a radiance field will contribute strongly to the reconstructed image (having high visibility and high density, both terms which multiplicatively effect the magnitude of backpropagated gradients) causing large changes to its table entries, while a point in empty space that happens to refer to the same entry will have a much smaller weight. As a result, the gradients of the more important samples dominate the collision average and the aliased table entry in question will naturally be optimized in such a way that it reflects the needs of the higher-weighted point; the less important point will then have its final output corrected by other levels in the multiresolution hierarchy.

### Online adaptivity

请注意，如果输入 $\mathbf x$ 的分布在训练期间随时间而变化，例如，如果它们集中在一个小区域，那么更精细的网格级别将经历更少的碰撞，并且可以学习更准确的函数。
Note that if the distribution of inputs $\mathbf x$ changes over time during training, for example if they become concentrated in a small region, then finer grid levels will experience fewer collisions and a more accurate function can be learned.

换句话说，多分辨率哈希编码自动适应训练数据分布，继承了基于树的编码的优点，而无需维护可能导致训练期间离散跳跃的任务特定数据结构。
In other words, the multiresolution hash encoding automatically adapts to the train data distribution, inheriting the benefits of tree-based encodings without task-specific data structure maintenance that might cause discrete jumps during training.

### $d$-linear interpolation

对查询的哈希表条目进行插值，并通过链式法则与神经网络复合 $m(\mathrm{enc}(x;\theta ); \Phi)$，可确保编码 $\mathrm{enc}(\mathbf x;\theta)$ 是连续的。
Interpolating the queried hash table entries ensures that the encoding $\mathrm{enc}(\mathbf x;\theta)$, and by the chain rule its composition with the neural network $m(\mathrm{enc}(x;\theta); \Phi)$, are continuous.

如果没有插值，网络输出中将出现网格对齐的不连续性，这将导致不希望的块状外观。人们可能需要更高阶的平滑度，例如在逼近偏微分方程时。
Without interpolation, grid-aligned discontinuities would be present in the network output, which would result in an undesirable blocky appearance. One may desire higher-order smoothness, for example when approximating partial differential equations.

## 4 Implementation

### Performance considerations

为了优化推理和反向传播性能，我们以半精度（每个条目 2 个字节）存储哈希表条目。我们还以全精度维护参数的主副本，以实现稳定的混合精度参数更新。
In order to optimize inference and backpropagation performance, we store hash table entries at half precision (2 bytes per entry). We additionally maintain a master copy of the parameters in full precision for stable mixed-precision parameter updates.

为了优化使用 GPU 的缓存，我们逐级评估哈希表：在处理一批输入位置时，我们安排计算查找所有输入的多分辨率哈希编码的第一级，然后是输入的第二级，等等。因此，在任何给定时间，只有少量连续的哈希表必须驻留在缓存中，具体取决于 GPU 上可用的并行度。
To optimally use the GPU’s caches, we evaluate the hash tableslevel by level: when processing a batch of input positions, we schedule the computation to look up the first level of the multiresolution hash encoding for all inputs, followed by the second level for inputs, and so on. Thus, only a small number of consecutive hash tables have to reside in caches at any given time, depending on how much parallelism is available on the GPU. Thus, only a small number of consecutive hash tables have to reside in caches at any given time, depending on how much parallelism is available on the GPU.

在我们的硬件上，只要哈希表大小保持在 $T\le2^{19}$ 以下，编码的性能就会大致保持不变。超过此阈值，性能开始显着下降。
On our hardware, the performance of the encoding remains roughly constant as long as the hash table size stays below $T\le2^{19}$. Beyond this threshold, performance starts to drop significantly.

每次查找的最佳特征维度 $F$ 数量取决于 GPU 架构。一方面，在前面提到的流方法中，一小部分有利于缓存局部性，但另一方面，较大的 $F$ 通过允许 $F$ 宽的向量加载指令来支持内存一致性。
The optimal number of feature dimension $F$ per lookup depends on the GPU architecture. On one hand, a small number favors cache locality in the aforementioned streaming approach, but on the other hand, a large $F$ favors memory coherence by allowing for $F$-wide vector load instructions.

### MLP architecture

在所有任务中，除了我们将在后面描述的 NeRF 之外，我们使用具有两个隐藏层的 MLP，这些隐藏层的宽度为 64 个神经元和 ReLU 激活函数。
In all tasks, except for NeRF which we will describe later, we use an MLP with two hidden layers that have a width of 64 neurons and rectified linear unit (ReLU) activation functions.

### Initialization

我们初始化神经网络权重，以在整个神经网络层中提供合理的激活及其梯度的缩放。
We initialize neural network weights to provide a reasonable scaling of activations and their gradients throughout the layers of the neural network.

我们使用均匀分布 $\mathcal U(-10^{-4}, 10^{-4})$ 初始化哈希表条目，以提供少量随机性，同时鼓励接近于零的初始预测。
We initialize the hash table entries using the uniform distribution $\mathcal U(-10^{-4}, 10^{-4})$ to provide a small amount of randomness while encouraging initial predictions close to zero.

我们还尝试了各种不同的分布，包括零初始化，所有这些都导致初始收敛速度稍差。哈希表似乎对初始化方案是健壮的。
We also tried a variety of different distributions, including zero-initialization, which all resulted in a very slightly worse initial convergence speed. The hash table appears to be robust to the initialization scheme.

### Training

我们联合训练神经网络权重和哈希表条目。
We jointly train the neural network weights and the hash table entries.

Adam: $\beta_1=0.9,\beta_2=0.99,\epsilon=10^{-15}$.

为了防止在长时间训练后出现发散，我们对神经网络权重应用弱 $L2$ 正则化（因子 $10^{-6}$），而不是哈希表条目。
To prevent divergence after long training periods, we apply a weak $L2$ regularization (factor $10^{-6}$) to the neural network weights, but not to the hash table entries.

最后，对于梯度正好为 0 的哈希表条目，我们跳过 Adam 步骤。这在梯度稀疏时节省了约 10% 的性能，这在 T $\gg$ BatchSize 中很常见。尽管这种启发式违反了 Adam 背后的一些假设，但我们观察到收敛性没有下降。
Lastly, we skip Adam steps for hash table entries whose gradient is exactly 0. This saves ～10% performance when gradients are sparse, which is a common occurrence with T $\gg$ BatchSize. Even though this heuristic violates some of the assumptions behind Adam, we observe no degradation in convergence.

### Non-spatial input dimensions $\xi\in\mathbb R^E$

将辅助维度 $\xi\in\mathbb R^E$ 输入到神经网络中通常很有用，例如在学习光场时的观察方向和材料参数。在这种情况下，可以使用已建立的技术对辅助维度进行编码，其成本不会随维度超线性缩放。
It is frequently useful to input auxiliary dimensions $\xi\in\mathbb R^E$ to the neural network, such as the view direction and material parameters when learning a light field. In such cases, the auxiliary dimensions can be encoded with established techniques whose cost does not scale superlinearly with dimensionality.

## 5 Experiments

### 5.1 Gigapixel Image Approximation

**ACORN: Adaptive Coordinate Networks for Neural Representation.**

| ACORN    | NGP     |
| -------- | ------- |
| 38.59 dB | 41.9 dB |
| 36.9 h   | 4 min   |

很难直接将我们的编码性能与 ACORN 进行比较；$10$ 倍的一个因素源于我们使用由 tiny-cuda-nn 框架提供的完全融合的 CUDA 内核。输入编码允许使用比 ACORN 小得多的 MLP，这在剩余的 $10\sim100$ 倍加速中占了很大一部分。
It is difficult to directly compare the performance of our encoding to ACORN; a factor of $\sim10$ stems from our use of fully fused CUDA kernels, provided by the tiny-cuda-nn framework. The input encoding allows for the use of a much smaller MLP than with ACORN, which accounts for much of the remaining $10\times\sim100\times$ speedup.

### 5.2 Signed Distance Functions

**Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Shapes.**

通过使用针对参考形状量身定制的数据结构，NGLOD 实现了最高的视觉重建质量。然而，即使没有这种专用的数据结构，我们的编码在 g-IoU 度量方面也接近于 NGLOD 的保真度，具有相似的性能和内存成本。
By using a data structure tailored to the reference shape, NGLOD achieves the highest visual reconstruction quality. However, even without such a dedicated data structure, our encoding approaches a similar fidelity to NGLOD in terms of the g-IoU metric, with similar performance and memory cost.

我们的哈希编码在“联合交集”（IoU）方面表现出相似的数字质量，并且可以在场景边界框中的任何位置进行评估。然而，它也表现出视觉上不受欢迎的粗糙表面，我们将其归因于随机分布的哈希冲突。
Our hash encoding exhibits similar numeric quality in terms of "intersection over union" (IoU) and can be evaluated anywhere in the scene’s bounding box. However, it also exhibits visually undesirable surface roughness that we attribute to randomly distributed hash collisions.

| Shape     | NGLOD  | Frequency | NGP    |
| --------- | ------ | --------- | ------ |
| Cow       | 0.9761 | 0.6509    | 0.9811 |
| Cow       | 1: 43  | 3: 18     | 1: 06  |
| Sculpture | 0.9998 | 0.9824    | 0.9998 |
| Sculpture | 1: 38  | 5: 27     | 1: 46  |
| Alien     | 0.9906 | 0.7389    | 0.9862 |
| Alien     | 1: 24  | 3: 30     | 1: 11  |
| Machine   | 0.9723 | 0.2325    | 0.9646 |
| Machine   | 1: 37  | 3: 04     | 0: 58  |

### 5.3 Neural Radiance Caching

**Real-time Neural Radiance Caching for Path Tracing.**

### 5.4 Neural Radiance and Density Fields (NeRF)

**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.**

密度 MLP 将哈希编码位置 $\mathbf y=\mathrm{enc}(\mathbf x;\theta)$ 映射到 16 个输出值，我们将第一个值视为对数空间密度。颜色 MLP 添加了与视角相关的颜色变化。它的输入是

- 密度 MLP 的 16 个输出值，以及
- 投影到球谐函数基础的前 16 个系数上的视图方向（即最多 4 级）。这是对单位向量的自然频率编码。

The density MLP maps the hash encoded position $\mathbf y=\mathrm{enc}(\mathbf x;\theta)$ to 16 output values, the first of which we treat as log-space density. The color MLP adds view-dependent color variation. Its input is the concatenation of

- the 16 output values of the density MLP, and
- the view direction projected onto the first 16 coefficients of the spherical harmonics basis (i.e. up to degree 4). This is a natural frequency encoding over unit vectors.

它的输出是一个 RGB 颜色三元组，当训练数据具有低动态范围 (sRGB) 时，我们使用 sigmoid 激活或当它具有高动态范围（线性 HDR）时使用指数激活。由于与物理光传输更相似，我们更喜欢 HDR 训练数据。
Its output is an RGB color triplet, for which we use either a sigmoid activation when the training data has low dynamic-range (sRGB) or an exponential activation when it has high dynamic range (linear HDR). We prefer HDR training data due to the closer resemblance to physical light transport.

一个单隐藏层的 density MLP 和一个双隐藏层的 color MLP，每层都是 64 个神经元个数。
A 1-hidden-layer density MLP and a 2-hidden-layer color MLP, both 64 neurons wide.

#### Accelerated ray marching

当沿着射线进行训练和渲染时，我们希望放置样本，使得它们对图像的贡献有点均匀，从而最大限度地减少计算浪费。因此，我们通过维护一个粗略地标记空与非空空间的占用网格来将样本集中在表面附近。在大场景中，我们额外级联占用网格并以指数方式分布样本，而不是沿射线均匀分布。
When marching along rays for both training and rendering, we would like to place samples such that they contribute somewhat uniformly to the image, minimizing wasted computation. Thus, we concentrate samples near surfaces by maintaining an occupancy grid that coarsely marks empty vs. nonempty space. In large scenes, we additionally cascade the occupancy grid and distribute samples exponentially rather than uniformly along the ray.

#### Comparison with direct voxel lookups

MLP 允许解析镜面反射细节并减少由哈希冲突引起的背景噪声量。由于 MLP 的小尺寸和高效实施，它的成本仅高出 15% -- 非常值得显着提高质量。
The MLP allows for resolving specular details and reduces the amount of background noise caused by hash collisions. Due to the small size and efficient implementation of the MLP, it is only $15\%$ more expensive -- well worth the significantly improved quality.

#### Comparison with high-quality offline NeRF

一方面，我们的方法在具有高几何细节的场景（例如 Ficus 和 Lego）上表现最佳，在所有方法中实现了最佳 PSNR。另一方面，MIPNeRF 和 NSVF 在具有复杂的、依赖于视角的反射（例如材质）的场景上优于我们的方法；我们将此归因于我们必须采用的更小的 MLP，以使我们的加速比这些竞争的实现提高几个数量级。
On one hand, our method performs best on scenes with high geometric detail, such as Ficus and Lego, achieving the best PSNR of all methods. On the other hand, MIPNeRF and NSVF outperform our method on scenes with complex, view-dependent reflections such as Materials; we attribute this to the much smaller MLP that we necessarily employ to obtain our speedup of several orders of magnitude over these competing implementations.

## 6 Discussion and Future Work

#### Concatenation vs. reduction

我们更喜欢连接有两个原因。首先，它允许对每个分辨率进行独立、完全并行的处理。其次，将编码结果 $\mathbf y$ 从 $LF$ 到 $F$ 的维数减少可能太小而无法编码有用的信息。虽然 $F$ 可以按比例增加，但它会使编码变得更加昂贵。
We prefer concatenation for two reasons. First, it allows for independent, fully parallel processing of each resolution. Second, a reduction of the dimensionality of the encoded result $\mathbf y$ from $LF$ to $F$ may be too small to encode useful information. While $F$ could be increased proportionally, it would make the encoding much more expensive.

#### Microstructure due to hash collisions

颗粒是 MLP 无法完全补偿的哈希冲突的结果。我们相信，使用我们的编码在 SDF 上实现最先进质量的关键是找到克服这种微观结构的方法，例如通过过滤哈希表查找或在损失之前施加额外的平滑度。
The graininess is a result of hash collisions that the MLP is unable to fully compensate for. We believe that the key to achieving state-of-the-art quality on SDFs with our encoding will be to find a way to overcome this microstructure, for example by filtering hash table lookups or by imposing an additional smoothness prior on the loss.

#### Learning the hash function

两种可能的途径是 (i) 开发适合分析微分的连续哈希公式或 (ii) 应用可以有效探索离散函数空间的进化优化算法。
Two possible avenues are (i) developing a continuous formulation of hashing that is amenable to analytic differentiation or (ii) applying an evolutionary optimization algorithm that can efficiently explore the discrete function space.

#### Generative setting

#### Other applications

不均匀的体积密度场，例如存储在 VDB 数据结构中的云和烟雾，通常包括外部的空白空间、内部的实心核心以及体积表面上的稀疏细节。这使它们非常适合我们的编码。
Heterogenous volumetric density fields, such as cloud and smoke stored in a VDB data structure, often include empty space on the outside, a solid core on the inside, and sparse detail on the volumetric surface. This makes them a good fit for our encoding.

## 7 Conclusion
