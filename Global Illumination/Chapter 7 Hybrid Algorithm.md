# Chapter 7 Hybrid Algorithm

## 7.5 Irradiance Caching

辐照度缓存，是一种有效的技术，可以加速漫反射场景中的间接光照的计算。
Irradiance caching, is an effective technique for accelerating the computation of indirect illumination in diffuse scenes.

辐照度缓存利用了漫反射表面的辐照度，虽然计算成本很高，但在大多数场景中是平滑变化的。
Irradiance caching exploits the insight that the irradiance at diffuse surfaces, while expensive to compute, varies smoothly in most scenes.

在这种技术中，辐照度被缓存在一个数据结构中，在可能的情况下，这些缓存的数值被内插以接近附近表面的辐照度。
In this technique, irradiance is cached in a data structure, and when possible, these cached values are interpolated to approximate irradiance at nearby surfaces.

### Interpolation

一种基于分割球模型的误差估计，被用来确定哪些样本可以用于插值而不（希望）引入可见的伪影。使用这个模型，位置 $P_i$ 处的缓存样本 $i$ 与点 $P$ 的误差给出如下：
An error estimate, based on the split-sphere model, is used to determine which samples can be used for interpolation without (hopefully) introducing visible artifacts. Using this model, the error at a point $P$ due to a cached sample $i$ at location $P_i$ is given as:
$$
\epsilon_i(P)=\frac{\|P-P_i\|}{R_i}+\sqrt{1-N_P\cdot N_{P_i}}
$$
其中 $R_i$ 为从缓存样本 $i$ 可见物体的平均谐波距离，$N_P$ 和 $N_{P_i}$ 分别为 $P$ 处的法线和 $P_i$ 处的样本。
where $R_i$ is the mean harmonic distance of objects visible from the cached sample $i$, and $N_P$ and $N_{P_i}$ are the normals at $P$ and the sample at $P_i$, respectively.

$P$ 点的辐照度是使用附近样本的缓存辐照度值插值的，其中第 $i$ 个样本的权重 $w_i$ 是：
Irradiance at the point $P$ is interpolated using the cached irradiance values of nearby samples using the weight $w_i$ for the $i$-th sample:
$$
w_i(P)=\frac{1}{\epsilon_i(P)}
$$
And the irradiance is:
$$
E(P)=\frac{\sum_{i=1}^Nw_i(P)E_i(P)}{\sum_{i=1}^Nw_i(P)}
$$

### The Irradiance Cache

缓存的样本被存储在一个在场景上构建的八叉树中。这种数据结构允许将几何图形与光照值解耦。
The cached samples are stored in an octree constructed over the scene. This data structure permits the decoupling of geometry from illumination values.

当必须计算一个点的辐照度时，八叉树被搜索以找到 "附近的 "缓存样本，这些样本足够精确，可用于近似辐照度；用户指定的权重截止值 $a$ 指定了搜索样本的半径。如果找到了这样的样本，它们将被用于使用上述的加权算法插值辐照度。如果不存在这样的样本，则为当前点计算一个样本。如果可能的话，这个样本会被储存在辐照度缓存中，以便以后再用于插值。
When the irradiance at a point must be computed, the octree is searched to find “nearby” cached samples that are accurate enough to be used to approximate irradiance; the user-specified weight cutoff $a$ specifies a radius over which the samples are searched. If such samples are found, they are used to interpolate irradiance using the weighting algorithm described above. If such samples do not exist, a sample is computed for the current point. This sample is then stored in the irradiance cache to be reused for interpolation later, if possible.

## 7.6 Photon Mapping

#### Tracing Photons: Pass 1

在第一遍中，“光子” 从光源发射到场景中。这些携带光通量信息的光子被缓存在一个数据结构中，称为光子图。在第二道工序中，使用存储在光子图中的信息来渲染图像。
In the first pass, "photons" are traced from the light sources into the scene. These photons, which carry flux information, are cached in a data structure, called the photon map. In the second pass, an image is rendered using the information stored in the photon map.

这方面最好的例子是焦散图，它被设计用来捕捉在到达漫反射表面之前与一个或多个镜面交互的光子。这些光路会导致焦散现象。传统的蒙特卡洛采样在正确产生良好的焦散现象方面可能非常缓慢。通过在焦散图中明确地捕捉焦散路径，光子映射技术可以有效地找到焦散。
The best example of this is the caustic map, which is designed to capture photons that interact with one or more specular surfaces before reaching a diffuse surface. These light paths cause caustics. Traditional Monte Carlo sampling can be very slow at correctly producing good caustics. By explicitly capturing caustic paths in a caustic map, the photon mapping technique can find caustics efficiently.

需要注意的一点是，光子映射是有偏的。
One point to note is that photon mapping is a biased technique. 

在第一遍中，光子从光源中追踪出来，并在场景中传播，就像光线追踪中的光线一样。
In the first pass, photons are traced from the light sources and propagated through the scene just as rays are in ray tracing.

为了促进对光子的有效搜索，用一个平衡的 kd 树来实现光子映射。
To facilitate efficient searches for photons, a balanced kd-tree is used to implement photon map.

#### Reflected Radiance using Photon Maps

The photon map represents incoming flux at each point in the scene; therefore, the photon density at a point estimates the irradiance at that point. The reflected radiance at a point can then be computed by multiplying the irradiance by the surface BRDF.

