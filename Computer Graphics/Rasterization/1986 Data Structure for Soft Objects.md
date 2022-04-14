# Data Structure for Soft Objects

## 0 Abstract

软物是一类物体，包括织物、靠垫、活体、泥土和水。
Soft objects is class of objects includes fabrics, cushions, living forms, mud and water.

## 1 Introduction

柔软物体的形状会因周围环境施加在其上的力而不断变化。
The shape of the soft object varies constantly because of the forces imposed on it by its surroundings.

物体被认为占据了函数值大于某个阈值的空间，因此物体的表面是场函数的等值面。
The object is considered to occupy the space over which the function has a value greater than some threshold so the surface of the object is an iso-surface of the field function.

通过适当选择的场函数，我们可以方便地代表各种各样的形状，原则上是以某种方式的任何形状。在本文中，我们专注于基于接近给定数据点的简单函数。我们通过指定这些关键点的运动来实现动画，而不会改变我们的函数。
By suitable choice of field function, we can represent a wide variety of shapes conveniently and in principle, any shape somehow. For this paper we concentrate on simple functions based on proximity to given data points. We achieve animation by specifying the motion of these key points, without otherwise altering our function.

## 2 Space Function or Field

我们想要构造一个函数，使我们能够在绘制等值面时表示任意形状。因此，该功能将取决于一组给定的关键点。我们假设关键点是独立的。
We want to construct a function which will enable us to represent arbitrary shapes when we plot the iso-surfaces. The function is therefore going to depend on a set of given key points. We assume that the key points are independent.

我们使用不受超出一定距离的任何点影响的函数。这个距离被称为影响半径，$R$。
We use a function which is not influenced by any point beyond a certain distance away. This distance is known as the radius of influence, $R$.

根据定义，任何超出其影响半径的关键点对场的贡献 $R$ 为零。点本身位置的贡献将是某个最大值（我们使用 1.0），我们希望场在 $R$ 处平滑下降到零。
By definition, the contribution to the field made by any key point beyond its radius of influence, $R$, is zero. The contribution at the position of the point itself will be some maximum value (We use 1.0) and we would like to arrange that the field drops smoothly to zero at $R$.
$$
\begin{align*}
C(0)=1&&C'(0)=0\\
C(R)=0&&C'(R)=0
\end{align*}
$$
假设有个 $C_1(r)$ 函数
$$
C_1(r)=2\frac{r^3}{R^3}-3\frac{r^2}{R^2}+1
$$
$r$ 是空间中任意一点到选取的关键点的距离，因此计算 3 次方需要开根号。我们用另一个函数：
$$
C_2(r)=-\frac{4r^6}{9R^6}+\frac{17r^3}{9R^3}-\frac{22r^3}{9R^3}+1
$$
作为替代，这样只需要 3 次加法 5 次乘法。

## 3 Defining the Iso-surface

绘制连接场值等于 $m$ 的所有点的等值面。
Plot the iso-surface connecting all points whose field value equals $m$.

提供两个液滴合并时的合理效果。
Provide a reasonable effect when two droplets merge.

## 4 Producing the Surface







