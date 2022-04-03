# Marching Cubes A High Resolution 3D Surface Construction Algorithm

## 0 Abstract

Marching cube 从 3D 医疗数据创建恒定密度表面的三角形模型。使用分治方法来生成切片间的连通性，我们创建了一个定义三角形拓扑的案例表。
Marching cube creates triangle models of constant density surfaces from 3D medical data. Using a divide-and-conquer approach to generate inter-slice connectivity, we create a case table that defines triangle topology.

该算法以扫描线顺序处理 3D 医学数据，并使用线性插值计算三角形顶点。
The algorithm processes the 3D medical data in scan-line order and calculates triangle vertices using linear interpolation.

从生成的表面模型生成的图像中的细节是保持原始 3D 数据中存在的切片间连通性、表面数据和梯度信息的结果。
The detail in images produced from the generated surface models is the result of maintaining the inter-slice connectivity, surface data, and gradient information present in the original 3D data.

## 1 Introduction

Marching cube 从 3D 数据数组创建恒定密度表面的多边形表示。可以使用以软件或硬件实现的传统图形渲染算法来显示生成的模型。
Marching cubes creates a polygonal representation of constant density surfaces from a 3D array of data. The resulting model can be displayed with conventional graphics-rendering algorithms implemented in software or hardware.

## 2 Information Flow for 3D Medical Algorithm

*Data acquisition*

*Image processing*

*Surface construction*

*Display*

## 4 Marching Cubes Algorithm

我们处理表面构造问题的方法有两个主要步骤。首先，我们定位对应于用户指定值的表面并创建三角形。然后，为了确保表面的质量图像，我们计算每个三角形的每个顶点的表面法线。
There are two primary step in our approach to the surface construction problem. First, we locate the surface corresponding to a user-specified value and create triangles. Then, to ensure a quality image of the surface, we calculate the normals to the surface at each vertex if each triangle.

Marching cube 使用分治方法将表面定位在由八个像素创建的逻辑立方体中：两个相邻切片中的每四个像素。
Marching cube uses a divide-and-conquer approach to locate the surface in a logical cube created from eight pixels: four each from two adjacent slices.

该算法确定表面如何与这个立方体相交，然后前进到下一个立方体。为了找到立方体中的表面交点，如果该顶点处的数据值超过（或等于）我们正在构建的表面的值，我们将一个 1 分配给立方体的顶点，这些顶点在表面外部。具有低于曲面的值的立方体顶点得到一个 1，这些点位于曲面内部（或表面上）。如果一个顶点位于曲面外部（1），另一个顶点位于曲面内部（0），则曲面与立方体边相交。
The algorithm determines how the surface intersects this cube, then marches to the next cube. To find the surface intersection in a cube, we assign a one to a cube's vertex if the data value at that vertex exceeds (or equals) the value of the surface we are constructing. These vertices are outside the surface. Cube vertices with values below the surface receive a zero and are inside (or on) the surface. The surface intersects those cube edges where one vertex is outside the surface (one) and the other is inside the surface (zero).

由于每个立方体中有 8 个顶点和 2 种状态，内部和外部，因此表面可以通过 256 种方式与立方体相交。
Since there are 8 vertices in each cube and 2 state, inside and outside, there are 256 ways a surface can intersect the cube.

首先，如果曲面值与立方体的关系颠倒，则三角曲面的拓扑不会改变。互补情况，其中大于表面值的顶点与小于该值的顶点互换，是等效的。因此，只需要考虑大于表面值 0 到 4 个顶点的情况，将情况减少到 128 个。使用第二个对称属性，旋转对称，我们通过检查将问题减少到 14 个模式。
First, the topology of the triangulated surface is unchanged if the relationship of the surface value to the cubes is reversed. Complementary cases, where vertices greater than the surface value are interchanged with those less than the value, are equivalent. Thus, only cases with 0 to 4 vertices greater than the surface value need be considered, reducing the number of cases to 128. Using the second symmetry property, rotational symmetry, we reduced the problem to 14 patterns by inspection.

我们为每种情况创建了一个索引，基于顶点的状态，八位索引每个顶点一位。
We created an index for each cases, based on the state of the vertex, the eight bit index contains one bit for each vertex.

使用索引来判断曲面与哪条边相交，我们可以沿边对曲面相交进行插值。我们使用线性插值，但尝试了更高阶的插值。
Using the index to tell which edge the surface intersects, we can interpolate the surface intersection along the edge. We use linear interpolation, but have experimented with higher degree interpolations.

Marching cube 的最后一步计算是每个三角形顶点的单位法线。
The final step in marching cubes calculates a unit normal for each triangle vertex.

恒定密度的表面沿表面切线方向具有零梯度分量；因此，梯度向量 $\vec{g}$ 的方向垂直于表面。
A surface of constant density has a zero gradient component along the surface tangential direction; consequently, the direction of the gradient vector, $\vec{g}$, is normal to the surface.
$$
\vec{g}(x,y,z)=\nabla \vec{f}(x,y,z)
$$
为了估计表面的梯度向量，我们首先估计立方体顶点的梯度向量，并在交点处线性插值梯度。立方体顶点 $(i,j,k)$ 处的梯度是使用沿三个坐标轴的中心差估计的：
To estimate the gradient vector at the surface of interest, we first estimate the gradient vectors at the cube vertices and linearly interpolate the gradient at the point of intersection. The gradient at cube vertex $(i,j,k)$, is estimated using central differences along the three coordinate axes by:
$$
G_x(i,j,k)=\frac{D(i+1,j,k)-D(i-1,j,k)}{\Delta x}\\
G_y(i,j,k)=\frac{D(i,j+1,k)-D(i,j-1,k)}{\Delta y}\\
G_z(i,j,k)=\frac{D(i,j,k+1)-D(i,j,k-1)}{\Delta z}\\
$$
其中 $D(i,j,k)$ 是切片 $k$ 中像素 $(i,j)$ 处的密度，$\Delta x,\Delta y,\Delta z$ 是立方体边的长度。
where $D(i,j,k)$ is the density at pixel $(i,j)$ in slice $k$ and $\Delta x,\Delta y,\Delta z$ are the lengths of the cube edges.

## 5 Enhancements to the Basic Algorithm

### 5.1 Efficiency Enhancements


对于原始数据限制内部的多维数据集，每个多维数据集只需要插入三个新边。
For cubes interior to the original data limits, only three new edges need to be interpolated for each cube.







