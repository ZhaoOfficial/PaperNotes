# Total Relighting Learning to Relight Portraits for Background Replacement

## Abstract

## Introduction

一个完整的系统——从数据生成到野外推理——用于人像重新照明和背景替换。
A complete system – from data generation to in-the-wild inference – for portrait relighting and background replacement.

基于深度学习的重新照明框架中的一种新颖的每像素照明表示，可产生最先进的人像重新照明结果。
A novel per-pixel lighting representation within a deep learning based relighting framework, which produces state-of-the art portrait relighting results.

野外肖像的真实感重新照明和合成结果，证明了所提出方法的有效性和概括性以及高质量数据的重要性。
Photorealistic relighting and compositing results for in-the-wild portraits, demonstrating the effectiveness and generalization of the proposed approach and the importance of high quality ground truth data.

## Framework

抠图模块根据给定的 RGB 图像估计 alpha 遮罩和前景。然后将估计的前景和目标 HDR 照明环境提供给重打光模块，该模块推断表面几何形状和反照率，并使用每像素照明表示对渲染外观的漫反射和镜面反射分量进行显式建模。重打光的结果、alpha 遮罩和新背景最终合成在一起，生成具有新背景的重打光人像，其中人像的照明条件与新背景的照明条件相匹配。
A matting module estimates the alpha matte and foreground from a given RGB image. The estimated foreground and a target HDR lighting environment are then provided to a relighting module, which infers surface geometry and albedo and uses a per-pixel lighting representation to explicitly model the diffuse and specular reflection components of rendered appearance. The relighted result, the alpha matte, and the new background are finally composited together, producing a relighted portrait with a new background, where the lighting conditions of the portrait match that of the novel background.