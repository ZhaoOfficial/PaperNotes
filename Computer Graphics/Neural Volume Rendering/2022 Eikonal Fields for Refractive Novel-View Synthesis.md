# Eikonal Fields for Refractive Novel-View Synthesis

## 0 Abstract

我们针对 3D 变化的折射率 (IoR) 场进行优化，并根据光 eikonal 传输定律跟踪通过该场的光线朝向所述折射率的空间梯度弯曲。
We optimize for a field of 3D-varying index of refraction (IoR) and trace light through it that bends toward the spatial gradients of said IoR according to the laws of light eikonal transport.

## 1 Introduction

然而，由于神经辐射场不考虑折射，我们不能直接将其用于折射新视图合成（NVS）。
However, since neural radiance fields do not consider refraction, we can not use it out of the box for refractive novel-view synthesis (NVS).

用于学习 3D 场捕捉场景几何的现有解决方案是基于沿直线路径的不透明或透明光传输。然而，在存在透明物体的情况下，光线会弯曲，即它会改变其方向。光路弯曲的精确方式取决于在 IoR 场的空间梯度上运行的某个 eikonal 方程，我们展示了该方程可以通过适当的公式在实践中求解 - 并在学习中微分。
Existing solutions to learn 3D fields capturing scene geometry are based on opaque or transparent light transport along straight paths. In the presence of transparent objects, however, light bends, i.e., it changes its direction. The precise way in which light paths are curved depends on a certain eikonal equation operating on spatial gradients of the IoR field which we show can be solved - and differentiated over in learning - in practice with the appropriate formulation.

## 2 Related Work

### 2.3 Eikonal Rendering

## 3 Light Transport ODE Zoo

完整的模型处理折射和非折射场景，但仅适用于文献中的合成场景。发射吸收模型可用于逆渲染，但不包括折射。我们的 eikonal 模型与发射吸收模型相结合，可以在实际的逆渲染中处理折射透明度。
The complete one handles refractive and non-refractive scenes, but was only applied to synthetic scenes in the literature. The emission-absorption one can be used for inverse rendering, but excludes refraction. Our eikonal one, in combination with the emission-absorption one, can handle refractive transparency in practical inverse rendering.

|      Symbol       |               Description                |
| :---------------: | :--------------------------------------: |
| $s\in[0,+\infty[$ | the distance along a (curved) light path |
|      $n(s)$       |      the index of refraction (IoR)       |
|     $\sigma$      |          extinction coefficient          |
|    $q/\sigma$     |             source function              |

### 3.1 Complete Model

Radiative transfer equation:
$$
n(s)^2\frac{\mathrm{d}(L/n^2(s))}{\mathrm{d}s}=-\sigma(s)L(s)+q(s)\tag{1}
$$

对于空间变化的 $n$，根据 eikonal 光传输定律，由于折射，光也会改变其位置 $\mathbf{p}$ 和方向 $\mathbf{v}$。
For a spatially varying $n$, light also changes its position $\mathbf{p}$ and direction $\mathbf{v}$ due to refraction according to the laws of eikonal light transport.
$$
\frac{\mathrm{d}\mathbf{p}}{\mathrm{d}s}=\frac{\mathbf{v}(s)}{n(s)}\quad\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}s}=\nabla{n}(s)\tag{2}
$$
其中 $\mathbf{v}$ 不是单位长度，而是由 $n$ 归一化的。
where $\mathbf{v}$ is not unit length but normalized by $n$.

### 3.2 Emission-Absorption-Only Model

Radiative transfer equation:
$$
\frac{\mathrm{d}L}{\mathrm{d}s}=-\sigma(s)L(s)+q(s)\tag{3}
$$
但沿恒定方向 $\mathbf{v}$ 行进，方向变化假定为零。
but travels along a constant direction $\mathbf{v}$​ and the change of direction is assumed zero.
$$
\frac{\mathrm{d}\mathbf{p}}{\mathrm{d}s}=\mathbf{v}(s)\quad\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}s}=\mathbf{0}\tag{4}
$$

### 3.3 Eikonal-Only Model

Radiative transfer equation:
$$
\frac{\mathrm{d}L}{\mathrm{d}s}=0\tag{5}
$$
但根据 eikonal 光传输改变方向：
but changes direction as per eikonal light transport:
$$
\frac{\mathrm{d}\mathbf{p}}{\mathrm{d}s}=\frac{\mathbf{v}(s)}{n(s)}\quad\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}s}=\nabla{n}(s)\tag{6}
$$

### 3.4 Solving

简而言之，所有三个变体都可以表示为位置-运动-辐射的状态向量及其导数：
Concisely, all three variants can be formulated as position-motion-radiance state vector and its derivative:
$$
\mathbf{z}(s)=(\mathbf{p},\mathbf{v},L)\tag{7}
$$

$$
\mathbf{z}(s_1)=\mathbf{z}(s_0)+\int_{s_0}^{s_1}\frac{\mathrm{d}\mathbf{z}}{\mathrm{d}s}\mathrm{d}s=\mathrm{odeSolve}(s_0,s_1,\mathbf{z}(s),\mathrm{d}\mathbf{z}/\mathrm{d}s)\tag{8}
$$

向后工作，计算发射或吸收的梯度是通过前向欧拉求解器的自动微分完成的。不幸的是，这需要按求解器执行的步数顺序存储。当还通过许多小步骤来考虑 IoR 时，这很快就会变得令人望而却步。相反，我们使用来自 Neural ODE 的伴随公式，该公式也在向后模式下使用常量内存来执行 odeSolve。
Working backwards, to compute gradients of the emission or absorption is done by automatic differentiation of forward Euler solvers. Unfortunately, this requires memory in the order of the number of steps a solver takes. When also accounting for IoR with many small steps, this can quickly become prohibitive. Instead, we use the adjoint formulation from Neural ODE that uses constant memory also in backward mode to perform odeSolve.

## 4 Our Approach

我们首先训练一个对世界的非折射 3D 解释，该解释输入到第二次训练中，该训练在一个固定的非折射世界内对光线进行 3D 弯曲，以便可以解释 2D 输入图像。
We first train a non-refractive 3D explanation of the world which is input to a second training that 3D-bends rays inside a fixed nonrefractive world so that 2D input images can be explained.

### 4.1 Non-Eikonal Step

在这一步中，我们训练一个发射 ($\bar{q}$) 和吸收 ($\bar{\sigma}$) 的 NeRF 模型，假设光线是直的。这用于表示背景并找到模型未解释的 3D 区域。
In this step, we train a NeRF model of emission ($\bar{q}$) and absorption ($\bar{\sigma}$), assuming straight rays. This is used to represent the background and to find the 3D region not explained by the model.

#### Registration

第一步，我们计算矩阵以使用 COLMAP 将每个输入图像的相机空间转换为一个参考视图。
In a first step, we compute matrices to transform the camera space of each input image into one reference view using COLMAP.

#### Diffuse-Opaque Initialization

鉴于此信息，一个现成的 NeRF 将发射和吸收描述为两个 MLP 拟合的、将位置和方向映射到 RGB 颜色或标量不透明度的连续函数 $\bar{q}(\mathbf{p},\omega)\in\mathbb{R}^3$ 和 $\bar{\sigma}(\mathbf{p})\in\mathbb{R}^3\times\Omega\mapsto\mathbb{R}$。
Given this information, an off-the-shelf NeRF is learned that describes emission and absorption as two MLPs that fit continuous functions $\bar{q}(\mathbf{p},\omega)\in\mathbb{R}^3\times\Omega\mapsto\mathbb{R}^3$ and $\bar{\sigma}(\mathbf{p})\in\mathbb{R}^3\times\Omega\mapsto\mathbb{R}$ mapping position and direction to RGB color or scalar opacity.

#### Masking

上述 $\bar{q}$ 和 $\bar{\sigma}$ 的模型对于折射物体来说是不可靠的。因此，我们希望消除 3D 空间的这些部分，并通过我们的 eikonal 方法对其进行解释。非折射部分将被输入到此步骤。
The model above of $\bar{q}$ and $\bar{\sigma}$ will not be reliable for refractive objects. Hence, we would like to eliminate these parts of 3D space, and explain them by our eikonal approach. The parts that are non-refractive, will be input to this step.

我们假设场景的折射部分可以由一个仅包含折射对象的 3D 框 $\Pi\in\mathbb{R}^{3\times2}$ 界定。这产生了一个掩盖的发射模型 $q$，关于一个掩盖的 $\sigma$：
We assume the refractive part of the scene can be bounded by a 3D box $\Pi\in\mathbb{R}^{3\times2}$ that exclusively contains refractive objects. This results in a masked emission model $q$, respectively a masked $\sigma$:
$$
q(\mathbf{p},\omega)\text{ resp. }\sigma(\mathbf{p})=\begin{cases}
0&\mathbf{p}\in\Pi\\
\bar{q}(\mathbf{p},\omega)\text{ resp. }\bar{\sigma}(\mathbf{p})&\mathbf{p}\notin\Pi\\
\end{cases}\tag{9}
$$
我们通过向用户提供 10% 的均匀分布在屈光物体周围的训练图像来找到框 $\Pi$。用户在图像中折射对象的水平和垂直范围上选择几个点。一旦我们从图像中收集了这些 2D 点，我们就使用从 NeRF 模型计算的深度图来找到它们对应的 3D 位置。然后，我们在每个空间维度上取所有点的 0.02 和 0.98 个百分位数，并将它们乘以 1.2 的常数值，以确保框包含整个对象。$\Pi$ 的参数由点的最小和最大坐标值给出。
We find the box $\Pi$ by providing a user with 10 percent of the training images uniformly distributed around the refractive object. The user selects a few points on the horizontal and vertical extent of the refractive object in the image. Once we have collected these 2D points from the images, we use the depth map computed from the NeRF model to find their corresponding 3D locations. We then take 0.02 and 0.98 percentiles of all points along each spatial dimension and multiply them by a constant value of 1.2 to make sure the box encompasses the entire object. The parameters of $\Pi$ are given by the minimum and maximum coordinate values of the points.

#### Progressive Grids

问题是当光线弯曲很多时，很难找到输入图像和背景之间的对应关系。此外，弯曲取决于 IoR 的空间梯度，而不是直接依赖于 IoR，这是一种已知需要优化数值的操作。为应对这一挑战，我们将改为使用不同的逐渐精细的发射和吸收模型来学习 eikonal 传输。
The problem is that when rays bend a lot it becomes harder to find correspondences between input images and background. Moreover, the bending depends on the spatial gradient of the IoR rather than the IoR directly, which is an operation known to be numerically demanding to optimize over. Addressing this challenge, we will instead learn eikonal transport using different progressively finer versions of the emission and absorption models.



### 4.2 Eikonal Step



### 4.3 Implementation Details



## 5 Results



## 6 Conclusion
