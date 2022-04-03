# TensoRF: Tensorial Radiance Fields

## 0 Abstract

与纯粹使用 MLP 的 NeRF 不同，我们将场景的辐射场建模为 4D 张量。 3D 体素网格的每个体素都具有多通道的特征。
Unlike NeRF that purely uses MLPs, we model the radiance field of a scene as a 4D tensor, which represents a 3D voxel grid with per-voxel multi-channel features.

我们的中心思想是**将 4D 场景张量分解为多个紧凑的低秩张量分量**。
Our central idea is to **factorize the 4D scene tensor into multiple compact low-rank tensor components**.

我们引入了一种新的向量矩阵（VM）分解，它放宽了张量的两种模式的低秩约束，并将张量分解为紧凑向量和矩阵因子。
We introduce a novel vector-matrix (VM) decomposition that relaxes the low-rank constraints for two modes of a tensor and factorizes tensors into compact vector and matrix factors.

## 1 Introduction

我们追求一种新颖的方法，既能节省训练时间，又能减少内存占用，同时实现最先进的渲染质量。
We pursue a novel approach that is both efficient in training time and compact in memory footprint, and at the same time achieves state-of-the-art rendering quality.

我们建议将辐射场的张量分解为**多个低阶张量分量**，从而获得准确而紧凑的场景表示。
We propose to factorize the tensor of radiance fields into multiple **low rank tensor components**, leading to an accurate and compact scene representation. 

虽然与 CP 中基于纯矢量的分解相比，这增加了模型大小，但我们使每个组件都能够表达更高等级的更复杂的张量数据，从而显着减少了辐射场建模所需的组件数量，并带来更好的渲染质量。
While this increases the model size compared to pure vector-based factorization in CP, we enable each component to express more complex tensor data of higher ranks, thus significantly reducing the required number of components in radiance field modeling and leading to better rendering quality.

与之前直接重建体素的工作相比，我们的张量分解将空间复杂度从 $O(n^3)$ 降低到 $O(n)$（使用 CP）或 $O(n^2)$（使用 VM），显着降低内存占用。
In contrast to previous works that directly reconstruct voxels, our tensor factorization reduces space complexity from $O(n^3)$ to $O(n)$ (with CP) or $O(n^2)$ (with VM), significantly lowering memory footprint.

## 2 Related Work

### 2.1 Tensor Decomposition

Tucker decomposition, CP decomposition, singular value decomposition (SVD).

### 2.2 Scene representations and radiance fields

## 3 CP and VM Decomposition

### 3.1 CP Decomposition

给定一个 3D 张量 $\mathcal T\in\mathbb R^{I\times J\times K}$，CP 分解将其分解为向量外积的总和
Given a 3D tensor $\mathcal T\in\mathbb R^{I\times J\times K}$, CP decomposition factorizes it into a sum of outer products of vectors
$$
\mathcal T=\sum_{r=1}^{R}\mathbf v_r^1\circ\mathbf v_r^2\circ\mathbf v_r^3
$$
其中 $\mathbf v_r^1\circ\mathbf v_r^2\circ\mathbf v_r^3$ 对应一个**秩一张量**分量，$\mathbf v_r^1\in\mathbb R^I$, $\mathbf v_r^2\in\mathbb R^J$ 和 $\mathbf v_r^3\in\mathbb R^K$ 是 $r^{th}$ 分量的三种模式的分解向量。
where $\mathbf v_r^1\circ\mathbf v_r^2\circ\mathbf v_r^3$ corresponds to a **rank-one tensor** component, and $\mathbf v_r^1\in\mathbb R^I$, $\mathbf v_r^2\in\mathbb R^J$, and $\mathbf v_r^3\in\mathbb R^K$ are factorized vectors of the three modes for the $r^{th}$ components.
$$
\mathcal T_{ijk}=\sum_{r=1}^R\mathbf v_{r,i}^1\circ\mathbf v_{r,j}^2\circ\mathbf v_{r,k}^3
$$
然而，由于过于紧凑，CP分解可能需要许多组件来模拟复杂的场景，导致辐射场重建的计算成本很高。
However, because of too high compactness, CP decomposition can require many components to model complex scenes, leading to high computational costs in radiance field reconstruction.

### 3.2 Vector-Matrix Decomposition

$$
\mathcal T=\sum_{r=1}^{R_1}\mathbf v_r^1\circ\mathbf M_r^{2,3}+\sum_{r=1}^{R_2}\mathbf v_r^2\circ\mathbf M_r^{1,3}+\sum_{r=1}^{R_3}\mathbf v_r^3\circ\mathbf M_r^{1,2}
$$

$\mathbf M_r^{2,3}\in\mathbb R^{J\times K},\mathbf M_r^{1,3}\in\mathbb R^{I\times K},\mathbf M_r^{1,2}\in\mathbb R^{I\times J}$

对于每个分量，我们将其两个模式的秩放宽为任意大，同时将第三个模式限制为秩一。
For each component, we relax its two mode ranks to be arbitrarily large, while restricting the third mode to be rank-one.

请注意，我们的每个分量张量都比 CP 分解中的一个分量具有更多参数。虽然这会导致较低的紧凑性，但 VM 分量张量可以表示比 CP 分量更复杂的高维数据，从而在建模相同复杂的函数时减少所需的分量数量。 另一方面，VM 分解仍然具有非常高的紧凑性，与密集网格表示相比，将内存复杂度从 $O(N^3)$ 降低到 $O(N^2)$。
Note that, each of our component tensors has more parameters than a component in CP decomposition. While this leads to lower compactness, a VM component tensor can express more complex high-dimensional data than a CP component, thus reducing the required number of components when modeling the same complex function. On the other hand, VM decomposition is still of very high compactness, reducing memory complexity from $O(N^3)$ to $O(N^2)$ compared to dense grid representations.

### 3.3 Tensor for Scene Modeling

$$
\mathcal T=\sum_{r=1}^{R}\left(\mathbf v_r^X\circ\mathbf M_r^{Y,Z}+\mathbf v_r^Y\circ\mathbf M_r^{X,Z}+\mathbf v_r^Z\circ\mathbf M_r^{X,Y}\right)\\
\mathcal T_{ijk}=\sum_{r=1}^R\sum_{\star}A^{\star}_{r,ijk}\quad\star\in\{X,Y,Z\}
$$

## 4 Tensorial Radiance Field Representation

### 4.1 Feature Grids and Radiance Field

我们将常规 3D 网格 $\mathcal G$（按特征通道）拆分为几何网格 $\mathcal G_{\sigma}$ 和外观网格 $\mathcal G_c$，分别建模体积密度 $\sigma$ 和视角相关的颜色$c$。
We split a regular 3D grid $\mathcal G$ (by feature channels) into a geometry grid $\mathcal G_{\sigma}$ and an appearance grid $\mathcal G_c$, separately modeling the volume density $\sigma$ and view-dependent color $c$.

我们的方法支持 $\mathcal G_c$ 中的各种类型的外观特征，这取决于将外观特征向量和观察方向 $d$ 转换为颜色 $c$ 的预选函数 $S$。$S$ 可以是一个小的 MLP 或球谐函数（SH），其中 $\mathcal G_c$ 分别包含神经特征和 SH 系数。
Our approach supports various types of appearance features in $\mathcal G_c$, depending on a pre-selected function $S$ that coverts an appearance feature vector and a viewing direction $d$ to color $c$. $S$ can be a small MLP or spherical harmonics (SH) functions, where $\mathcal G_c$ contains neural features and SH coefficient respectively.

我们考虑一个单通道网格 $\mathcal G_{\sigma}$，其值直接表示体积密度，不需要额外的转换函数。
We consider a single-channel grid $\mathcal G_{\sigma}$​ whose values represent volume density directly, without requiring an extra converting function.
$$
\sigma,c=\mathcal G_{\sigma}(\mathbf x),S(\mathcal G_c(\mathbf x),d)
$$
其中 $\mathcal G_{\sigma}(\mathbf x)$ 和 $\mathcal G_c(\mathbf x)$ 由 $\mathbf x$ 所在格子三线性插值得来。

$\mathcal G_{\sigma}\in\mathbb R^{I\times J\times K}$, $\mathcal G_c\in\mathbb R^{I\times J\times K\times P}$, $I,J,K$ 是沿着 $X,Y,Z$ 轴对应的分辨率，$P$ 是特征向量的通道数。

### 4.2 Factorizing Radiance Fields

$$
\mathcal G_{\sigma}=\sum_{r=1}^{R_{\sigma}}\left(\mathbf v_{\sigma,r}^X\circ\mathbf M_{\sigma,r}^{Y,Z}+\mathbf v_{\sigma,r}^Y\circ\mathbf M_{\sigma,r}^{X,Z}+\mathbf v_{\sigma,r}^Z\circ\mathbf M_{\sigma,r}^{X,Y}\right)=\sum_{r=1}^{R_{\sigma}}\sum_{\star}A_{\sigma,r}^{\star}\quad\star\in\{X,Y,Z\}
$$

外观张量 $\mathcal G_c$ 具有对应于特征通道维度的附加模式。请注意，与 $X,Y,Z$ 模式相比，此模式通常具有较低的维度，导致秩较低。因此，我们没有将这种模式与矩阵因子中的其他模式结合起来，而是只使用向量，在分解中用 $\mathbf b_r$ 表示这种模式。
The appearance tensor $\mathcal G_c$ has an additional mode corresponding to the feature channel dimension. Note that, compared to the $X,Y,Z$ modes, this mode is often of lower dimension, leading to a lower rank. Therefore, we do not combine this mode with other modes in matrix factors and instead only use vectors, denoted by $\mathbf b_r$ for this mode in the factorization.
$$
\begin{align*}
\mathcal G_c&=\sum_{r=1}^{R_{c}}\left(\mathbf v_{c,r}^X\circ\mathbf M_{c,r}^{Y,Z}\circ\mathbf b_{3r-2}+\mathbf v_{c,r}^Y\circ\mathbf M_{c,r}^{X,Z}\circ\mathbf b_{3r-1}+\mathbf v_{c,r}^Z\circ\mathbf M_{c,r}^{X,Y}\circ\mathbf b_{3r}\right)\\
&=\sum_{r=1}^{R_{c}}\left(A_{c,r}^X\circ\mathbf b_{3r-2}+A_{c,r}^Y\circ\mathbf b_{3r-1}+A_{c,r}^Z\circ\mathbf b_{3r}\right)\\
\end{align*}
$$
请注意，我们有 $3R_c$ 个向量 $\mathbf b_r$ 来匹配分量的总数。
Note that we have $3R_c$ vectors $\mathbf b_r$ to match the total number of components.

![pipeline](../../Note Image/tensorf.png)

一般来说，我们采用 $R_{\sigma}\ll I,J,K$ 和 $R_{c}\ll I,J,K$ 导致可以编码高分辨率密集网格的高度紧凑的表示。
In general, we adopt $R_{\sigma}\ll I,J,K$ and $R_{c}\ll I,J,K$ leading to a highly compact representation that can encode a high-resolution dense grid.

外观特征模式向量 $\mathbf b_r$ 表示全局外观相关性。通过将所有 $\mathbf b_r$ 作为列堆叠在一起，我们有一个 $P\times3R_c$ 矩阵 $B$； 这个矩阵 $B$ 也可以看作是一个全局外观字典，它抽象了整个场景的外观共性。
The appearance feature-mode vectors $\mathbf b_r$ express the global appearance correlations. By stacking all $\mathbf b_r$ as columns together, we have a $P\times3R_c$ matrix $B$; this matrix $B$ can also be seen as a global appearance dictionary that abstracts the appearance commonalities across the entire scene.

### 4.3 Efficient Feature Evaluation

#### Direct evaluation

$$
\mathcal G_{\sigma,ijk}=\sum_{r=1}^{R_{\sigma}}\sum_{\star}A_{\sigma,r,ijk}^{\star}\quad\star\in\{X,Y,Z\}
$$

$$
\mathcal G_{c,ijk}=\sum_{r=1}^{R_{c}}\left(A_{c,r,ijk}^X\circ\mathbf b_{3r-2}+A_{c,r,ijk}^Y\circ\mathbf b_{3r-1}+A_{c,r,ijk}^Z\circ\mathbf b_{3r}\right)
$$

在这里，特征模式没有额外的索引，因为我们计算了一个完整的向量。
Here, there’s no additional indexing for the feature mode, since we compute a full vector.







