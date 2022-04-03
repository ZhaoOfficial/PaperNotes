# Bridging the Gap between Deep Learning and Sparse Matrix Format Selection

## Abstract

这项工作对深度学习的前景和特殊挑战，即稀疏矩阵格式选择的进行了系统的探索——确定矩阵的最佳存储格式以最大化稀疏矩阵向量乘法 (SpMV) 的性能的问题。
This work presents a systematic exploration on the promise and special challenges of deep learning for sparse matrix format selection—a problem of determining the best storage format for a matrix to maximize the performance of Sparse Matrix Vector Multiplication (SpMV).

## Introduction

稀疏矩阵向量乘法 (SpMV) 是许多科学应用中最重要、最广泛使用的内核之一。人们观察到的 SpMV 性能的最重要因素之一是选择合适的格式来表示内存中的稀疏矩阵。
Sparse matrix vector multiplication (SpMV) is one of the most important, widely used kernels in many scientific applications. One of the most important factors people have observed for the SpMV performance is the selection of the proper format to represent sparse matrices in memory.

### Inspiration

稀疏矩阵格式选择问题是 CNN 可解决的一个很好的候选。
Sparse matrix format selection problem is a good candidate for CNN to solve.

首先，稀疏矩阵的正确格式取决于其矩阵大小、非零元素分布、体系结构特征等。
First, the proper format of a sparse matrix depends on its matrix size, nonzero distribution, architecture characteristics, and so on.

其次，由于难以提取用于学习的矩阵的正确特征，以及 SpMV 性能与稀疏矩阵的正确格式之间的复杂关系。
Second, due to the difficulties in coming up with the right features of matrices for learning and the complex relations between SpMV performance and the proper format of a sparse matrix.

第三，该问题类似于 CNN 已证明有效的其他一些任务。它类似于图像分类。在这两个问题中，正确的决策主要取决于输入中元素的空间模式。对于图像分类，模式是像素，对于稀疏矩阵格式选择，它们是非零元素。
Third, the problem resembles some other tasks that CNN has proved effective. It is akin to image classification. In both problems, the right decisions are primarily determined by the spatial patterns of the elements in an input. For image classification, the patterns are of pixels, and for sparse matrix format selection, they are of non-zero elements.

### Problem

另一方面，这个问题对CNN提出了一些特殊的挑战。 其中三个尤为突出。
On the other hand, the problem poses some special challenges to CNN. Three of them are especially prominent.

第一个是输入表示。 CNN 通常要求所有输入数据的大小相同。对于图像，可以裁剪、缩放或采样。但是稀疏矩阵格式选择对矩阵的一些微妙特征很敏感，这些特征可能会被那些传统的图像变化所迷惑。缩放会将非对角矩阵变换为非对角矩阵；由于对角线严重影响 SpMV 的格式选择，因此缩放图像会误导 CNN 学习和预测。
The first is input representation. CNN typically requires all input data to be of the same size. For images, the transformation could be cropping, scaling or sampling. But sparse matrix format selection is sensitive to some subtle features of a matrix, which can get lost by those traditional transformations. Scaling creates some diagonals for a non-diagonal matrix; as having diagonals critically affects format selection for SpMV, the scaled images mislead CNN learning and prediction.

第二个特殊挑战是设计合适的 CNN 结构。
The second special challenge is the design of the suitable CNN structures.

第三个挑战是稀疏矩阵格式选择的架构依赖性。机器的许多因素可能会影响 SpMV 在特定矩阵格式和给定矩阵的最佳格式上的性能。如何有效地跨系统迁移模型对 HPC 特别重要的问题。
The third challenge is the architectural dependence of sparse matrix format selection. Many factors of a machine could affect the performance of SpMV on a particular matrix format and the best format for a given matrix. How to efficiently migrating a model across systems is a problem specifically important for HPC problems.

## Background

### Sparse Matrix Storage Format

COO, CSR, DIA

### Convolutional Neural Networks

## Overview

构建过程包括四个步骤。它假设已经有大量的稀疏矩阵 $S$ 可以用作构造的训练输入，以及一个目标计算平台 $P$（SpMV 的执行发生在那里）。
The construction process consists of four steps. It assumes that there are already a large set of sparse matrices $S$ that the construction can use as its training inputs, and a target computing platform $P$ (where the executions of SpMV happen).

第一步收集训练标签。它多次为 $P$ 上的所有矩阵运行 SpMV，每次使用不同的矩阵格式。通过测量每个矩阵的执行时间，它找出 SpMV 运行最快的格式，并用该格式的 ID 标记该矩阵。
The first step collects training labels. It runs SpMVs for all matrices on $P$ multiple times, using a different matrix format each time. By measuring the execution times, for each matrix, it finds out the format that SpMV runs fastest and labels that matrix with the ID of that format.

第二步将每个矩阵归一化为固定大小，以便它们可以输入 CNN 的输入层。
The second step normalizes each of the matrices into a fixed size such that they can be fed into the input layers of CNN.

第三步设计CNN的结构。所设计的 CNN 的参数被初始化为一些随机值。输出层由 $K$ 个输出节点组成，每个节点对应于 $K$ 个矩阵格式之一以供选择。
The third step designs the structure of CNN. The parameters of the designed CNN are initialized to some random values. The output layer is composed of K output nodes with each corresponding to one of the K matrix formats to choose from.

第四步，在收集到的标签和归一化矩阵上运行标准的CNN训练算法，最终确定CNN各个参数的取值，结束构建过程。
The fourth step runs the standard CNN training algorithm on the collected labels and the normalized matrices to finally determine the value of each parameter of the CNN, and concludes the construction process.

此外，生成的 CNN 特定于训练平台 $P$，因为它使用在 $P$ 上收集的标签。如何将学习到的 CNN 快速迁移到另一个平台是另一个研究挑战。
Additionally, the resulting CNN is specific to the training platform $P$ as it uses the labels collected on $P$. How to quickly migrate the learned CNN to another platform is another research challenge.

## Input Representations

矩阵归一化：各种矩阵大小到一个单一大小。
Matrix normalization: Various matrix sizes to one single size.

### Representation from Traditional Methods

在基于 CNN 的图像处理中，输入归一化是通过图像缩放来实现的，它对大图像进行降采样或对小图像进行插值。
In CNN-based image processing, input normalization is through image scaling, which down-samples large images or interpolates smal images.

因为非零元素的空间模式与 SpMV 性能相关，而不是它们的确切值，所以归一化矩阵的元素值设置为二进制。如果原始块包含全零，则新元素设置为零，否则设置为 1。它产生一个二进制矩阵。
Because the spatial patterns of non-zero elements rather than their exact values are relevant to SpMV performance, the values of the elements of the normalized matrix are set to binary. The new element is set to zero if the original block contains all zeros, and 1 otherwise. It results in a binary matrix.

### Augmentation with Density Representation

在密度表示中，它不是为原始矩阵的每个块生成 0 或 1，而是生成 0 到 1 之间的十进制值，等于块中非零条目的数量除以块大小。
In density representation, instead of producing zero or one for each block of the original matrix, it produces a decimal value between 0 and 1, equaling the number of nonzero entries in a block divided by the block size.

### Distance Histogram Representation

直方图表示通过直方图存储矩阵中非零元素的空间分布。
Histogram representations stores the spatial distribution of non-zero elements in a matrix through histograms.

它由两个矩阵组成，一个存储原始矩阵行的直方图，另一个存储列的直方图。
It consists of two matrices, with one storing the histograms for the rows of the original matrix, and the other for the columns.

直方图基于元素与原始矩阵的主对角线之间的距离。
The histogram is based on the distance between an element and the principal diagonal of the original matrix.

```python
def histNorm(A: np.ndarray, r: int, BINS: int) -> np.ndarray:
    R = np.zeros((r, BINS))
    scale_ratio = A.shape[0] / r
    max_dim = max(A.shape)
    for i, row in enumerate(A):
        for j, e in enumerate(row):
            if e != 0.0:
                R_row = i / scale_ratio
                R_bin = BINS * np.abs(i - j) / max_dim
                R[R_row, R_bin] += 1
    return R
```

然后通过除以每个矩阵中的最大值，将两个矩阵中的值归一化到 $[0,1]$ 的范围内。
The values in both matrices are then normalized to the range of $[0,1]$ by dividing the largest value in each.

使用数值和利用距离而不是直接的空间位置——往往会捕获关于矩阵中非零元素分布的更丰富的信息。同时，它们的尺寸可以更灵活地调整。
Using numerical values and leveraging distances rather than direct spatial locations—tend to capture richer information about the distribution of non-zero elements in the matrix. Meanwhile, their sizes are more flexible to adjust.

## CNN Structure Designs

### Insufficiency of Common Structures

### Late-merging Structure

## Cross-Architecture Adaptations

标签是输入矩阵的最佳格式，从一台机器到另一台机器可能会有很大差异。
The labels are the best formats for the input matrices, which could differ significantly from one machine to another.

Transfer learning.

### Concept and Two-fold Contributions

### Options Explored

#### Continuous evolvement

该方法将现有的 CNN 视为新 CNN 的中间状态，在此基础上，将新机器上收集的新训练数据集馈送至 CNN，继续训练 CNN，直至收敛。
This method treats the existing CNN as an intermediate state of the new CNN, based on which, it feeds the new set of training data collected on the new machine to the CNN to continue training the CNN until a convergence.

#### Top evolvement

第二种方法继承了之前CNN的所有部分，并保持结构和参数不变，除了顶部全连接层的参数。
This second method inherits all parts of the previous CNN and keeps both the structure and the parameters unchanged, except the parameters of the top fully connected layer.

#### Qualitative Comparisons

定性地说，与第一个选项相比，顶级演化涉及的参数更改集要小得多。
Qualitatively, the top evolvement involves a much smaller set of parameter changes than the first option does.

## Evaluations







