# Modeling and Rendering Architecture from Photographs: A hybrid geometry- and image-based approach

从一组稀疏的静止照片中建模和渲染现有的建筑场景。**结合基于几何和基于图像的技术。**
Modeling and rendering existing architectural scenes from a sparse set of still photographs. **Combining both geometry-based and image-based techniques.**

第一个组件是摄影测量建模方法，它有助于恢复拍摄场景的基本几何形状。第二个组件是基于模型的立体算法，它揭示了真实场景如何偏离基本模型。
The first component is a photogrammetric modeling method which facilitates the recovery of the basic geometry of the photographed scene. The second component is a model-based stereo algorithm, which recovers how the real scene deviates from the basic model.

通过使用该模型，我们的立体技术可以从宽间距的图像对中稳健地恢复准确的深度。
By making use of the model, our stereo technique robustly recovers accurate depth from widely-spaced image pairs.

为了生成渲染，我们提出了依赖于视角的纹理映射，这是一种合成场景的多个视图的方法，可以更好地模拟基本模型上的几何细节。
For producing renderings, we present view-dependent texture mapping, a method of compositing multiple views of a scene that better simulates geometric detail on basic models.

## 1 Introduction

#### Geometry-based:

首先，该过程非常耗费人力，通常涉及勘测场地、定位和数字化建筑平面图（如果可用）或转换现有 CAD 数据（再次，如果可用）。其次，很难验证生成的模型是否准确。然而，最令人失望的是最终模型的渲染明显是由计算机生成的。即使是那些使用自由纹理映射的人，通常也无法与真实照片相似。
First, the process is extremely labor-intensive, typically involving surveying the site, locating and digitizing architectural plans (if available), or converting existing CAD data (again, if avail able). Second, it is difficult to verify whether the resulting model is accurate. Most disappointing, though, is that the renderings of the resulting models are noticeably computer-generated; even those that employ liberal texture-mapping generally fail to resemble real photographs.

#### Image-based

其中一些最有前途的系统依赖于计算立体视觉的计算机视觉技术，从可用的多张照片中自动确定场景的结构。然而，因此，这些系统仅与底层立体算法一样强大。
Some of the most promising of these systems rely on the computer vision technique of computational stereopsis to automatically determine the structure of the scene from the multiple photographs available. As a consequence, however, these systems are only as strong as the underlying stereo algorithms.

在此框架中，为可真实渲染的模型捕获数据将需要不切实际的密集照片数量，并且从照片中获取深度可能需要不切实际的用户输入量。这些对立体算法弱点的让步对于从照片创建大规模、可自由导航的虚拟环境来说是个坏兆头。
In this framework, capturing the data for a realistically renderable model would require an impractical number of closely spaced photographs, and deriving the depth from the photographs could require an impractical amount of user input. These concessions to the weakness of stereo algorithms bode poorly for creating large-scale, freely navigable virtual environments from photographs.

![](..\..\Note Image\geometry-based and image-based.png)

结果是我们的建模和渲染架构方法只需要一组稀疏的照片，并且可以从任意视点生成逼真的渲染。 在我们的方法中，架构的基本几何模型通过易于使用的摄影测量建模系统交互地恢复，使用视图相关的纹理映射创建新视图，并且可以通过立体对应自动恢复额外的几何细节。 可以使用当前基于图像的渲染技术来渲染最终图像。
The result is that our approach to modeling and rendering architecture requires only a sparse set of photographs and can produce realistic renderings from arbitrary viewpoints. In our approach, a basic geometric model of the architecture is recovered interactively with an easy-to-use photogrammetric modeling system, novel views are created using view-dependent texture mapping, and additional geometric detail can berecovered automatically through stereo correspondence. The finaimages can be rendered with current image-based rendering techniques.























