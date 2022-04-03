# 最先进的神经渲染 State of the Art on Neural Rendering

## Abstract

自动生成形状，材质，光照和其他视角是图形学现存的几个挑战。
The automatic generation of shape, materials, lighting, and other aspects of scenes remains a challenging problem.

## Introduction

神经渲染结合了物理和机器学习的方法来生成图片。
Neural rendering combines physical knowledge, with learned components to yield new and powerful algorithms for controllable image generation.

## Neural Rendering

### Control

What do we want to control and how do we condition the rendering on the control signal?

Novel view synthesis, Relighting under novel lighting, Animating faces and bodies under novel expressions and poses.

One strategy is to directly pass the scene parameters as input to the first or an intermediate network layer.

Another approach is to rely on the spatial structure of images and employ an image-to-image translation network to map from a "guide image" or "conditioning image" to the output.

### Computer Graphics Modules

Which computer graphics modules are used and how are they integrated into a neural rendering pipeline?

One simple way to integrate graphics knowledge is a non-differentiable computer graphics module. Such a module can for example be used to render an image of the scene and pass it as dense conditioning input to the network. This transforms the problem to an image-to-image translation task.

A deeper integration of graphics knowledge into the network is possible based on differentiable graphics modules. Such a differentiable module can for example implement a complete computer graphics renderer, a 3D rotation or an illumination model.

### Explicit vs. Implicit Control

Does the method give explicit control over the parameters or is it done implicitly by showing an example of what we expect to get as output?

Explicit control: a user can edit the scene parameters manually in a semantically meaningful manner, more supervision. Camera viewpoint, scene illumination, facial pose and expression.

Implicit control: Less supervision. Transfer human head motion from a reference video to a target person, retarget full-body motion.

### Multi-modal Synthesis

Is the method trained to output multiple optional outputs, given a specific input?

For parameter visulization.

### Generality

Is the rendering approach generalized over multiple scenes/objects?

For an instance to a class.

## 6.2 Novel View Synthesis for Objects and Scenes

Novel view synthesis methods deal with image and video synthesis conditioned on camera pose.

### Neural Image-based Rendering

Neural Image-based Rendering (N-IBR) is a hybrid between classical image-based rendering and deep neural networks that replaces hand-crafted heuristics with learned components. Old IBR gives accurate results only for diffuse objects with precise geometry reconstructed with a sufficien number of captured views.

### Neural Rerendering

Neural Rerendering combines classical 3D representation and renderer with deep neural networks that rerender the classical render into a more complete and realistic views.

### Novel View Synthesis with Multiplane Images

### Neural Scene Representation and Rendering

### Voxel-based Novel View Synthesis Methods

## Lecture

2017 Pix2pix.
