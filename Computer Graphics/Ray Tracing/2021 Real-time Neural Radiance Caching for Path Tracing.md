# Realtime Neural Radiance Caching for Path Tracing

## 0 Abstract

我们的系统旨在处理完全动态的场景，并且不对光照、几何形状和材质做任何假设。
Our system is designed to handle fully dynamic scenes, and makes no assumptions about the lighting, geometry, and materials. 

我们的数据驱动方法回避了缓存算法的许多困难，例如定位、插值和更新缓存点。
The data-driven nature of our approach sidesteps many difficulties of caching algorithms, such as locating, interpolating, and updating cache points.

由于预训练神经网络以处理新颖的动态场景是一项艰巨的泛化挑战，因此我们取消了预训练，而是通过适应实现泛化，即我们选择在渲染时训练辐射缓存。
Since pretraining neural networks to handle novel, dynamic scenes is a formidable generalization challenge, we do away with pretraining and instead achieve generalization via adaptation, i.e. we opt for training the radiance cache while rendering.

我们采用自我训练来提供低噪声训练目标，并通过仅迭代少量反弹训练更新来模拟无限反弹传输。
We employ self-training to provide low-noise training targets and simulate infinite-bounce transport by merely iterating few-bounce training updates.

## 1 Introduction

幸运的是，辐射量具有显着的空间、方向和时间相关性，可以通过各种方式利用这些相关性来加速渲染。
Fortunately, radiative quantities feature significant spatial, directional, and temporal correlations, which can be exploited in various ways to accelerate rendering.

一种特别吸引人的方法是缓存辐射样本以供以后重复使用。这可以在预计算步骤中完成，也可以在渲染时完成。
One particularly appealing approach is to cache radiance samples for later reuse. This can be done in a precomputation step, or while rendering.

我们建议通过使用神经辐射缓存来缓解这些困难，因为众所周知，神经网络特别适合替换复杂的启发式算法。
We propose to alleviate these difficulties through the use of a neural radiance cache, as neural networks are known to be particularly apt at replacing complex heuristics.

- **动态内容。**要处理完全交互的内容，系统必须支持相机、照明、几何和材质的任意动态。我们努力寻求不需要预先计算的解决方案。
**Dynamic content.** To handle fully interactive content, the system must support arbitrary dynamics of the camera, lighting, geometry, and materials. We strive for a solution that does not require precomputation.
- **稳健性。**特定案例的处理最终会导致复杂、脆弱的系统。因此，缓存应该与材质和场景几何无关。
**Robustness.** Case-specific handling eventually leads to complex, brittle systems. Hence, the cache should be agnostic of materials and scene geometry.
- **可预测的性能和资源消耗。**工作负载和内存使用的波动会导致帧率不稳定。我们寻求一种具有稳定运行时开销和内存占用的解决方案，这两者都应该独立于场景复杂性。渲染成本必须在最坏的情况下与像素数成线性关系。
**Predictable performance and resource consumption.** Fluctuations in work load and memory usage lead to unstable frame rates. We seek a solution with stable runtime overhead and memory footprint, both of which should be independent of scene complexity. The rendering cost must scale at worst linearly with the number of pixels.

前两个原则 - 动态内容和鲁棒性 - 对预训练的网络提出了一个重大挑战：训练后的模型必须泛化到新的配置，更糟糕的是，内容可能以前从未观察到。
The first two principles - dynamic content and robustness - present a major challenge for pre-trained networks: the trained model must generalize to novel configurations and, worse, content possibly never observed before.

在实时应用程序的紧密渲染循环中进行优化和推理是一项有待解决的重要任务。
Fitting the optimization and inference inside the tight rendering loop of real-time applications is a nontrivial task that remains to be tackled.

我们提出了两个关键贡献，它们可以通过实时适应进行泛化。首先，我们描述了一种使用（相对）便宜的辐射估计优化网络的有效机制。该机制的核心是神经网络根据其自身在路径的后续顶点的预测进行自我训练，以跟踪单条光线或非常短的路径为代价提供多反射照明。
We present two key contributions that enable generalization via adaptation in real time. First, we describe an efficient mechanism for optimizing the network using (relatively) inexpensive radiance estimates. The core of this mechanism is self-training of the neural network from its own prediction at a later vertex of the path, providing multi-bounce illumination at the cost of tracing single rays or very short paths.

其次，我们提出了一种简化的网络架构，旨在在渲染完全动态的场景时最大限度地权衡质量成本。
Second, we propose a streamlined network architecture designed to maximize the quality-cost tradeoff when rendering fully dynamic scenes.

## 2 Related Work

### 2.1 Radiance Caching

用于模拟漫反射的现代技术遵循相同的假设，即辐照度倾向于在整个场景中平滑变化，并且可以使用反照率调制来恢复纹理细节。
Modern techniques for modeling diffuse inter-reflections follow the same assumption that irradiance tends to vary smoothly across the scene, and texture detail can be recovered using albedo modulation. 

各种缓存记录的插值和定位是这些技术中的一个关键挑战，尤其是当上述关于平滑度的假设不成立时。
The interpolation and location of the various cache records is a key challenge in these techniques, especially when the aforementioned assumptions on smoothness do not hold.

### 2.2 Precomputation-based Techniques

假设场景照明和几何形状都是固定的，则可以计算辐照度，然后使用光照贴图将其存储在纹理空间中，并使用光照探针将其存储在世界空间中。
Assuming both the scene lighting and geometry are fixed, irradiance can be computed and then stored in texture space using lightmaps and in world space using light probes.

### 2.3 Fully Dynamic Techniques

动态实时全局照明方法建立在有效的渲染算法之上，这些算法在像素之间重用着色和可见性计算，例如光子映射、多光渲染和辐射映射，通过各种近似值进一步提高效率。
Dynamic real-time global illumination methods build upon efficient rendering algorithms that reuse shading and visibility computation across pixels, such as photon mapping, many-light rendering and radiosity maps, extracting further efficiency through various approximations.

除了近似值固有的精度限制（例如模糊、缺少交互或关于材料模型的假设）之外，许多这些技术的一个关键限制是依赖于必须不断刷新的场景的双重表示。
Aside of accuracy limitations inherent to the approximations, such as blurring, missing interactions, or assumptions about the material model, a key limitation of many of these techniques is the reliance on a dual representation of the scene which must be continuously refreshed.

### 2.4 Path Guiding

### 2.5 Neural Techniques

A 提议训练一组局部神经辐射缓存，以单点光源的位置为条件。虽然可以动态改变光照并且可以使用一组点光照来近似区域光照，但代价是多次缓存查询，但由于训练过程的成本，几何体和材质必须保持静态。我们的技术在两个重要方面有所不同：（i）我们使用单个神经辐射缓存，利用最近提出的编码来适应局部场景变化，以及（ii）我们在线训练我们的模型，它允许完全动态的场景并在单个查询容易考虑所有场景中的照明。
A propose to train a set of local neural radiance caches, conditioned on the position of a single point light source. While lighting can be changed dynamically and area lighting can be approximated using a set of point lights at the cost of multiple cache queries, geometry and materials have to remain static as a consequence of the cost of the training procedure. Our technique differs on two important aspects: (i) we use a single neural radiance cache leveraging recently proposed encodings to adapt to local scene variations, and (ii) we train our model online which allows for fully dynamic scenes and readily accounts for all lighting in the scene in a single query.

## 3 Neural Radiance Caching

我们的目标是使用单个神经网络来缓存辐射度，该神经网络将空间方向坐标映射到辐射度值，并进行实时训练以支持动态场景。我们选择近似散射辐射，因为它是渲染方程中计算成本最高的部分。
Our goal is to cache radiance using one single neural network that maps spatio-directional coordinates to radiance values and is trained in real-time to support dynamic scenes. We opt for approximating the scattered radiance as it is the most computationally expensive part of the rendering equation.
$$
L_s(\mathbf{x},\omega)=\int_{\mathcal{S}^2}f_s(\mathbf{x},\omega,\omega_i)L_i(\mathbf{x},\omega_i)|\cos\theta_i|\mathrm{d}\omega_i
$$

我们的神经网络通过缓存的辐射 $\hat{L}_s$ 来近似 $L_s$。
Our neural network approximates $L_s$ by the cached radiance $\hat{L}_s$.

### 3.1 Algorithm Overview

单帧渲染包括计算像素颜色和更新神经辐射缓存。
Rendering a single frame consists of computing pixel colors and updating the neural radiance cache.

首先，我们跟踪短渲染路径，每个像素一个，并在辐射缓存提供的近似值被认为足够准确时立即终止它们。
First, we trace short rendering paths, one for each pixel, and terminate them as soon as the approximation provided by the radiance cache is deemed sufficiently accurate.

我们使用最初在光子密度估计的上下文中开发的启发式算法，仅在路径的扩展足够大以模糊缓存的小不准确时才查询缓存。
We use the heuristic that was originally developed in the context of photon density estimation, to only query the cache once the spread of the path is sufficiently large to blur small inaccuracies of the cache.

在每个中间顶点，我们使用下一个事件估计来整合来自发光体的光。为此，我们在第一个顶点使用屏幕空间 ReSTIR，在后续顶点使用 LightBVH，通过多重重要性采样与 BSDF 结合。在终端顶点 $\mathbf{x}_k$ 处截断路径，我们评估神经辐射缓存以近似 $L_s(\mathbf{x}_k,\omega_k)$。
At each intermediate vertex, we use next-event estimation to integrate light from emitters. To this end, we use screen-space ReSTIR at the primary vertex and a LightBVH, combined with the BSDF via multiple importance sampling, at the subsequent vertices. Truncating the path at the terminal vertex $\mathbf{x}_k$, we evaluate the neural radiance cache to approximate $L_s(\mathbf{x}_k,\omega_k)$.

其次，为了训练辐射缓存，我们将一小部分（通常低于 3%）的短渲染路径扩展了几个顶点 - 一个训练后缀。
Second, to train the radiance cache, we extend a fraction (typically under 3%) of the short rendering paths by a few vertices - a training suffix.

和以前一样，一旦它们后缀的区域扩展足够大，我们就会终止这些较长的训练路径；为此，我们将查询顶点 $\mathbf{x}_k$ 视为主要顶点。在大多数情况下，后缀由一个顶点组成。沿较长训练路径的所有顶点收集的辐射估计值用作训练辐射缓存的参考值。
As before, we terminate these longer training paths once the area spread of their suffix is sufficiently large; for that purpose we consider the query vertex $\mathbf{x}_k$ as a primary vertex. In the majority of cases, the suffix consists of one vertex. The radiance estimates collected along all vertices of the longer training paths are used as reference values for training the radiance cache.

