# A Survey of Octree Volume Rendering Methods

## Abstract

## Introduction

传统的八叉树能保证规则的，不重叠的节点间隔，因此非常适合作为直线标量场数据的容器。
A conventional octree guarantees regular, non-overlapping node spacing, thus is well-suited as a container for rectilinear scalar field data.

通常来说，八叉树允许以自适应级别的分辨率存储数据，并通过二进制散列函数快速访问。
For general purposes, octrees allow data to be stored with adaptive levels of resolution, and accessed quickly via a binary hash function.

## Octree Structures and Hash Schemes

### Pointer Octree and Hashing

#### Point Location

#### Neighbor Finding

#### Region Finding

### Octree Varieties

#### Full Octree

#### Linear Octree

#### Branch-on-need Octree

## Octrees in Volume Rendering

### Extraction

一般来说，八叉树的目的是提供一个内存占用小的结构，用来封装一个体积的单元，从中可以提取网格。
In general, the purpose of the octree was to provide a structure with a small memory footprint that could encapsulate cells of a volume, and from which a mesh could be extracted.











