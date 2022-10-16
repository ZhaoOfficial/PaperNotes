# SLIC Superpixels Compared to State-of-the-art Superpixel Methods

## 0 Abstract

简单线性迭代聚类（SLIC）调整了 k-means 聚类方法来有效地生成超级像素。
Simple Linear Iterative Clustering (SLIC) adapts a k-means clustering approach to efficiently generate superpixels.

## 1 Introduction

- 超级像素应该很好地贴近图像边界。
    Superpixels should adhere well to image boundaries.
- 当作为预处理步骤用于降低计算的复杂性时，超级像素应该是计算快速、内存高效、使用简单。
    When used to reduce computational complexity as a preprocessing step, superpixels should be fast to compute, memory efficient, and simple to use.
- 当用于分割目的时，超级像素应同时提高速度和改善结果的质量。
    When used for segmentation purposes, superpixels should both increase the speed and improve the quality of the results.

## 2 Existing Superpixel Methods

### A Graph-based Algorithm

### B Gradient-ascent-based Algorithm

## 3 SLIC Superpixels

- 通过将搜索空间限制在与超级像素大小成比例的区域内，优化过程中的距离计算数量大大减少。这将复杂性降低到与像素数 $N$ 成线性关系，并且与超级像素数 $k$ 无关。
    The number of distance calculations in the optimization is dramatically reduced by limiting the search space to a region proportional to the superpixel size. This reduces the complexity to be linear in the number of pixels $N$ and independent of the number of superpixels $k$.
- 加权的距离测量结合了颜色和空间的接近性，同时对超级像素的大小和紧凑性进行了控制。
    A weighted distance measure combines color and spatial proximity, while simultaneously providing control over the size and compactness of the superpixels.

### A Algorithm

对于 CIELAB 颜色空间中的彩色图像，聚类过程从初始化步骤开始，其中 $k$ 个初始聚类中心 $C_i=\begin{bmatrix}l_i&a_i&b_i&x_i&y_i\end{bmatrix}^T$ 在间隔为 $S$ 像素的均匀网格上采样。为了产生大致相同大小的超像素，网格间隔为 $S=\sqrt{N/k}$。
For color images in the CIELAB color space, the clustering procedure begins with an initialization step where $k$ initial cluster centers $C_i=\begin{bmatrix}l_i&a_i&b_i&x_i&y_i\end{bmatrix}^T$ are sampled on a regular grid spaced $S$ pixels apart. To produce roughly equally sized superpixels, the grid interval is $S=\sqrt{N/k}$.

这些中心被移动到与 $3\times3$ 邻域中的最低梯度位置相对应的种子位置。这样做是为了避免超像素在边缘上居中，并减少用噪声像素播种超像素的机会。
The centers are moved to seed locations corresponding to the lowest gradient position in a $3\times3$ neighborhood. This is done to avoid centering a superpixel on an edge, and to reduce the chance of seeding a superpixel with a noisy pixel.

接下来，在分配步骤中，每个像素 $i$ 与最近的聚类中心相关联，该聚类中心的搜索区域与其位置重叠。
Next, in the assignment step, each pixel $i$ is associated with the nearest cluster center whose search region overlaps its location.

这是加速我们算法的关键，因为限制搜索区域的大小显着减少了距离计算的数量，并且与传统的 k-means 聚类相比具有显着的速度优势，传统的 k-means 聚类必须将每个像素与所有聚类中心进行比较。
This is the key to speeding up our algorithm because limiting the size of the search region significantly reduces the number of distance calculations, and results in a significant speed advantage over conventional k-means clustering where each pixel must be compared with all cluster centers.

由于超像素的预期空间范围是近似大小为$S\times S$ 的区域，因此在超像素中心周围的 $2S\times2S$ 区域中搜索相似像素。
Since the expected spatial extent of a superpixel is a region of approximate size $S\times S$, the search for similar pixels is done in a region $2S\times2S$ around the superpixel center.

```python
def SimpleLinearIterativeCluster(k: int):
    S = np.sqrt(N / k)
    cluster_center = SamplePixelAtRegularGridWithStep(S)
    cluster_center = MoveToLowestGradientPosition(cluster_center)

    while True:
        for c in cluster_center:
            for i in RegionAround(c, 2 * S):
                KMeans(i, c)
        if KMeansError < threshold:
            break
```

### B Distance Measure

要将这两个距离组合成一个度量，有必要通过它们各自在集群内的最大距离 $N_s$ 和 $N_c$ 来归一化颜色接近度和空间接近度。
To combine the two distances into a single measure, it is necessary to normalize color proximity and spatial proximity by their respective maximum distances within a cluster, $N_s$ and $N_c$.
$$
\begin{align*}
d_c&=\sqrt{(l_j-l_j)^2+(a_j-a_j)^2+(b_j-b_j)^2}\\
d_s&=\sqrt{(x_j-x_j)^2+(y_j-y_j)^2}\\
D&=\sqrt{\left(\frac{d_c}{N_c}\right)^2+\left(\frac{d_s}{N_s}\right)^2}\\
N_s&=\sqrt{\frac{N}{S}}\\
N_c&\in[1,40]
\end{align*}
$$
当 $N_c$ 很大时，空间接近度更重要，并且生成的超像素更紧凑（即它们具有较低的面积与周长比）。当$N_c$ 较小时，生成的超像素更紧密地附着在图像边界上，但尺寸和形状不太规则。
When $N_c$ is large, spatial proximity is more important and the resulting superpixels are more compact (i.e. they have a lower area to perimeter ratio). When $N_c$ is small, the resulting superpixels adhere more tightly to image boundaries, but have less regular size and shape.

灰度图：$d_c=\sqrt{(l_j-l_j)^2+(a_j-a_j)^2+(b_j-b_j)^2}$。

### C Post-processing

### D Complexity

$O(N)$ where traditional is $O(NkI)$.