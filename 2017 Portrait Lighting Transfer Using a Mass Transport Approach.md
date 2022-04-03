# Portrait Lighting Transfer Using a Mass Transport Approach

## Abstract

我们从标准颜色直方图匹配开始，它通过将参考的颜色分布转移到输入肖像上来捕获全局颜色和照明色调。 这种方法忽略了着色取决于面部几何形状的事实，并且不出所料地产生了低于标准的结果。
We start with standard color histogram matching, which captures the global color and tone of lighting by transferring the color distribution of the reference onto the input portrait. This approach ignores the fact that shading depends on face geometry and unsurprisingly produces subpar results.

## Introduction

1. 一种新的面部重新照明方法，它使用质量传输公式在图像之间传输照明。
  A novel approach to face relighting that uses a mass transport formulation to transfer illumination between images.
2. 一种正则化方案，使该技术对面部外观和几何形状的变化具有普适性。
  A regularization scheme that makes the technique robust to variations in face appearance and geometry.
3. 一个完整的流水线，可提高面部重新照明和合成的最新技术水平
  A complete pipeline that improves the state of the art in face relighting and compositing.

## Mass-Transport Formulation

给定输入图像 $I$ 和参考图像 $R$，我们使用参考照明和输入照片的姿势、身份和表情创建重新照明输出图像 $O$。 
Given an input image $I$ and a reference image $R$, we create a relight output image $O$ with the lighting of the reference and the pose, identity, and expression of the input photograph.

我们将 3D 模型的 3D 位置和表面法线投影到图像平面上以获得每个像素的位置和法线。这为我们在输入和参考图像的每个像素处提供了一个颜色-位置-法线向量 $(\mathbf c, \mathbf p, \mathbf n)$。
We project the 3D positions and surface normals of the 3D model onto the image plane to get per-pixel positions and normals. This gives us a color–position–normal vector $(\mathbf c, \mathbf p, \mathbf n)$ at every pixel of the input and reference images.

我们通过匹配这些高维向量在联合空间 $\text{colors}\times\text{positions}\times\text{normals}$ 中的分布，将照明从参考转移到输入图像。
We transfer lighting from the reference to the input image by matching the distributions of these high-dimensional vectors in the joint space $\text{colors}\times\text{positions}\times\text{normals}$.

### Reformulating Color Histogram Transfer

直观上，输入和参考直方图可以被看作是沙堆，人们试图移动沙子以将输入堆转换为参考堆，同时最大限度地减少工作量。
Intuitively, the input and reference histograms can be seen as sand heaps, and one seeks to move the sand to transform the input heap into the reference heap while minimizing the amount of work.

当两个直方图不同时，质量传输通常会生成不规则的映射。添加新样本近似到高斯卷积会生成更平滑的映射，即对相邻的样本进行类似变换。
When two histograms differ, the mass-transport solution often generates an irregular mapping. Adding new samples to approximate Gaussian convolution generates a smoother mapping, that is, nearby samples are transformed similarly.

$H_I$ 和 $H_R$ 分别表示输入和参考图像的归一化颜色直方图

$\mathbf c_i$ 和 $\mathbf c_j$ 表示输入和参考颜色。

$T_{ij}$ 表示 $\mathbf c_i$ 映射到 $\mathbf c_j$ 的比例。
$$
\arg\min_{T_{ij}}\sum_{i}\sum_{j}\|\mathbf c_i-\mathbf c_j\|^2T_{ij}\\
T_{ij}\ge0\quad\sum_{j}T_{ij}=H_I(\mathbf c_i)\quad\sum_{i}T_{ij}=H_R(\mathbf c_j)
$$
这里所求的是这个映射的代价最小时候的映射。

我们寻求一种映射的解决方案，即所有具有相同颜色的像素都关联到相同的参考颜色。
we seek a solution that is a mapping, that is, all the pixels with the same color are associated to the same reference color.

$\mathbf c_{ji}$ 表示输入 $\mathbf c_i$ 所映射到的唯一的颜色，$f$ 就是这个映射，$f(\mathbf c_i)=\mathbf c_{ji}$
$$
\arg\min_{f}\sum_{i}\sum_{j}\|\mathbf c_i-f(\mathbf c_i)\|^2H_I(\mathbf c_i)\\
H_{f(I)}=H_R
$$

### Incorporating Positions and Normals

$\mathbf s=(\mathbf c,\mathbf p,\mathbf n)$ 表示颜色，位置，法向。

$\hat{H}_I$, $\hat{H}_R$ 是在颜色，位置，法向的积空间内归一化的图像。
$$
\arg\min_{f}\sum_{i}\sum_{j}\left(w_c\|\mathbf c_i-\hat{f}_c(\mathbf s_i)\|^2+w_p\|\mathbf p_i-\hat{f}_p(\mathbf s_i)\|^2+w_n\|\mathbf n_i-\hat{f}_n(\mathbf s_i)\|^2\right)H_I(\mathbf c_i)\\
\hat{H}_{\hat{f}(I)}=\hat{H}_R
$$
因为这种传输是在积空间中定义的，所以除了颜色之外，它还会修改输入像素位置和法线，也就是说，它会改变颜色并扭曲面部几何形状。
Because this transport is defined in the product space, it modifies the input pixel positions and normals in addition to their colors, that is, it alters the colors and warps the face geometry.

我们将传输的效果限制在颜色空间，即我们只将 $\hat{f}_c$ 应用于输入图像 $I$。
We restrict the effect of the transport to the color space, that is, we only apply $\hat{f}_c$ to the input image $I$.

### Regularization via Stochastic Sampling

我们将每个样本 $\mathbf s=(\mathbf c,\mathbf p,\mathbf n)$ 替换为 $n_s$ 个具有相同位置 $\mathbf p$ 和 $\mathbf n$ 但颜色 $\mathbf c+ \mathbf v$ 的随机样本，其中随机向量 $\mathbf v$ 遵循高斯分布 $G_{\sigma}$。直观地说，额外的样本提供了新的对应选项，允许求解器找到更平滑的映射。
We replace each sample $\mathbf s=(\mathbf c,\mathbf p,\mathbf n)$ by $n_s$ stochastic samples with the same position $\mathbf p$ and $\mathbf n$ but color $\mathbf c+\mathbf v$ where the random vector $\mathbf v$ follows a Gaussian distribution $G_{\sigma}$. Intuitively, the additional samples offer new correspondence options that allow the solver to find a smoother mapping.

### Solving the Mass-Transport Problem

### Discussion

虽然我们的质量传输公式在全局范围内匹配输入和参考分布，但它还捕获了局部照明变化。
While our mass transport formulation globally matches the input and reference distributions, it also captures localized lighting variations.

$w_p=w_n=w_c=1$ 是比较好的参数。