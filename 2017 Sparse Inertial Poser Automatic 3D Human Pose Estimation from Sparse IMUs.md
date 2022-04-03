# Sparse Inertial Poser: Automatic 3D Human Pose Estimation from Sparse IMUs

## Abstract

通过使用连接到身体的一小组惯性传感器，使野外的人体动作捕捉更加实用。利用包含人体测量约束的现实统计身体模型；使用联合优化框架使模型适合多帧的方向和加速度测量。
Making human motion capture in the wild more practical by using a small set of inertial sensors attached to the body. Making use of a realistic statistical body model that includes anthropometric constraints; using a joint optimization framework to fit the model to orientation and acceleration measurements over multiple frames.

## Introduction

基于标记的系统具有侵入性，并将运动限制在受控实验室空间。从图像中估计 3D 人体姿态通常涉及多相机校准系统，这再次限制了适用性。
Marker-based systems are intrusive and restrict motions to controlled laboratory spaces. The estimation of 3D human pose from images typically involves multi-camera calibrated systems, which again limit applicability.

惯性测量单元（IMUs）
Inertial Measurement Units (IMUs)

手腕、小腿、腰部和头部安装了六个传感器，用于测量方向和加速度，从而形成了一种捕获人类活动的侵入性最低的解决方案。
Six sensors, measuring orientation and acceleration are attached to the wrists, lower legs, waist and head, resulting in a minimally intrusive solution to capture human activities.

