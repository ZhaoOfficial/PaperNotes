# Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations

## 0 Abstract

物理信息神经网络--经过训练以解决监督学习任务的神经网络，同时遵循一般非线性偏微分方程描述的任何给定物理定律。
Physics-informed neural networks – neural networks that are trained to solve supervised learning tasks while respecting any given laws of physics described by general nonlinear partial differential equations.

我们在解决两类主要问题的背景下介绍了我们的发展：偏微分方程的数据驱动的解决方案和数据驱动的发现。
We present our developments in the context of solving two main classes of problems: data-driven solution and data-driven discovery of partial differential equations.

根据可用数据的性质和排布，我们设计了两种不同类型的算法，即连续时间模型和离散时间模型。第一类的模型形成了一个新的数据高效时空函数逼近器系列，而后一类允许使用任意精度的、具有无限阶的隐式 Runge-Kutta 时间步长方案。

Depending on the nature and arrangement of the available data, we devise two distinct types of algorithms, namely continuous time and discrete time models. The first type of models forms a new family of data-efficient spatio-temporal function approximators, while the latter type allows the use of arbitrarily accurate implicit Runge–Kutta time stepping schemes with unlimited number of stages.

## 1 Introduction

在分析复杂的物理、生物或工程系统的过程中，数据获取的成本高得令人望而却步，我们不可避免地面临在部分信息下得出结论和决策的挑战。在这种小数据体系中，绝大多数最先进的机器学习技术（例如，深度/卷积/递归神经网络）缺乏健壮性，并且无法提供任何收敛保证。
In the course of analyzing complex physical, biological or engineering systems, the cost of data acquisition is prohibitive, and we are inevitably faced with the challenge of drawing conclusions and making decisions under partial information. In this small data regime, the vast majority of state-of-the-art machine learning techniques (e.g., deep/convolutional/recurrent neural networks) are lacking robustness and fail to provide any guarantees of convergence.

拯救我们的是，对于许多与物理和生物系统建模有关的案例，存在大量当前尚未在现代机器学习实践中使用的先验知识。假设它是支配系统的时间相关动力学的原则性物理定律，或者一些经验验证的规则或其他领域专业知识，这些先验信息可以作为一个正则化项，将可接受解决方案的空间限制在可管理的大小。
Coming to our rescue, for many cases pertaining to the modeling of physical and biological systems, there exists a vast amount of prior knowledge that is currently not being utilized in modern machine learning practice. Let it be the principled physical laws that govern the time-dependent dynamics of a system, or some empirically validated rules or other domain expertise, this prior information can act as a regularization agent that constrains the space of admissible solutions to a manageable size.

作为回报，将此类结构化信息编码到学习算法中会放大算法所看到的数据的信息内容，使其能够快速引导自己走向正确的解决方案。即使只有少数训练示例可用，也能很好地泛化。
In return, encoding such structured information into a learning algorithm results in amplifying the information content of the data that the algorithm sees, enabling it to quickly steer itself towards the right solution and generalize well even when only a few training examples are available.

尽管高斯过程在编码先验信息方面具有灵活性和数学上的优雅，但非线性问题的处理引入了两个重要的限制。 首先，作者必须及时对任何非线性项进行局部线性化，从而限制了所提出方法对离散时间域的适用性，并损害了它们在强非线性状态下的预测准确性。其次，高斯过程回归的贝叶斯性质需要某些先验假设，这可能会限制模型的表示能力并引起鲁棒性/脆弱性问题，尤其是对于非线性问题。
Despite the flexibility and mathematical elegance of Gaussian processes in encoding prior information, the treatment of nonlinear problems introduces two important limitations. First, the authors had to locally linearize any nonlinear terms in time, thus limiting the applicability of the proposed methods to discrete-time domains and compromising the accuracy of their predictions in strongly nonlinear regimes. Secondly, the Bayesian nature of Gaussian process regression requires certain prior assumptions that may limit the representation capacity of the model and give rise to robustness/brittleness issues, especially for nonlinear problems.

## 2 Problem setup

通过使用深度神经网络并利用它们作为通用函数逼近器的能力，我们可以直接解决非线性问题，而无需承诺任何先前的假设、线性化或本地时间步长。我们利用最新发展的自动微分来微分神经网络的输入坐标和模型参数，以获得*物理信息神经网络*。此类神经网络受限于源自支配观测数据的物理定律的任何对称性、不变性或守恒原理，如一般的时间相关和非线性偏微分方程建模。
By employing deep neural networks and leverage their well known capability as universal function approximators, we can directly tackle nonlinear problems without the need for committing to any prior assumptions, linearizion, or local time-stepping. We exploit recent developments in automatic differentiation to differentiate neural networks with respect to their input coordinates and model parameters to obtain *physics-informed neural networks*. Such neural networks are constrained to respect any symmetries, invariance, or conservation principles originating from the physical laws that govern the observed data, as modeled by general time-dependent and nonlinear partial differential equations.

在这项工作中，我们考虑一般形式的参数化和非线性偏微分方程：
In this work, we consider parametrized and nonlinear partial differential equations of the general form:
$$
u_t+\mathcal{N}[u;\lambda]=0,x\in\Omega,t\in[0,T]\tag{1}
$$

|       Symbols        |           Descriptions            |
| :------------------: | :-------------------------------: |
|       $u(t,x)$       |             Solution              |
| $\mathcal{N}[\cdot]$ | A nonlinear differential operator |
|       $\Omega$       |    A subset of $\mathbb{R}^D$     |

给定系统的噪声测量，我们对两个不同问题的解决方案感兴趣。第一个问题是偏微分方程的推理、过滤和平滑，或数据驱动的解，即给定固定模型参数 $\lambda$ ，可以从未知隐藏状态 $u(t ,x)$ 的系统得到什么？ 第二个问题是偏微分方程的学习、系统识别或数据驱动发现，即最准确描述观察到的数据的参数 $\lambda$ 是什么？
Given noisy measurements of the system, we are interested in the solution of two distinct problems. The first problem is that of inference, filtering and smoothing, or data-driven solutions of partial differential equations which states: given fixed model parameters $\lambda$ what can be said about the unknown hidden state $u(t,x)$ of the system? The second problem is that of learning, system identification, or data-driven discovery of partial differential equations stating: what are the parameters $\lambda$ that best describe the observed data?

## 3 Data-driven solutions of partial differential equations

$$
u_t+\mathcal{N}[u]=0,x\in\Omega,t\in[0,T]\tag{2}
$$

$u(t,x)$ us the latent solution, $\mathcal{N}[\cdot]$ is a nonlinear differential operator, $\Omega$ is a subset of $\mathbb{R}^D$.

### 3.1 Continuous time models

let 
$$
f\gets u_t+\mathcal{N}[u\tag{3}]
$$
并通过深度神经网络逼近 $u(t,x)$。
and proceed by approximating $u(t,x)$ by a deep neural network.

该网络（$f$）可以通过应用链式法则来导出，使用自动微分来微分复合函数，并且具有与表示 $u(t,x)$ 的网络相同的参数，尽管由于微分算子的作用而具有不同的激活函数 $\mathcal{N}$。神经网络 $u(t,x)$ 和 $f(t,x)$ 之间的共享参数可以通过最小化均方误差损失来学习：
This network ($f$) can be derived by applying the chain rule for differentiating compositions of functions using automatic differentiation, and has the same parameters as the network representing $u(t,x)$, albeit with different activation functions due to the action of the differential operator $\mathcal{N}$. The shared parameters between the neural networks $u(t,x)$ and $f(t,x)$ can be learned by minimizing the mean squared error loss:
$$
\begin{align*}
\mathrm{MSE}&=\mathrm{MSE}_u+\mathrm{MSE}_f\\
\mathrm{MSE}_u&=\frac{1}{N_u}\sum_{i=1}^{N_u}\left|u(t^i_u,x^i_u)-u^i\right|^2\\
\mathrm{MSE}_f&=\frac{1}{N_f}\sum_{i=1}^{N_f}\left|f(t^i_f,x^i_f)\right|^2\\
\end{align*}\tag{4}
$$

|              Symbols              |                         Descriptions                         |
| :-------------------------------: | :----------------------------------------------------------: |
| $\{t^i_u,x^i_u,u^i\}^{N_u}_{i=1}$ |      The initial and boundary training data on $u(t,x)$      |
|   $\{t^i_f,x^i_f\}^{N_f}_{i=1}$   |             The collocations points of $f(t,x)$              |
|         $\mathrm{MSE}_u$          |            Loss of the initial and boundary data             |
|         $\mathrm{MSE}_f$          | Enforces the structure imposed by equation $(2)$ at a finite set of collocation point |

拟议的工作旨在通过重新构建为基础微分算子量身定制的“自定义”激活和损失函数更进一步。这使我们能够通过理解和欣赏深度学习领域中自动微分所发挥的关键作用来打开黑盒。
The proposed work aims to go one step further by revisiting the construction of “custom” activation and loss functions that are tailored to the underlying differential operator. This allows us to open the black-box by understanding and appreciating the key role played by automatic differentiation within the deep learning field.

在所有与数据驱动的偏微分方程求解有关的情况下，训练数据的总数 $N_u$ 相对较少（几百到几千点），我们选择使用 L-BFGS，伪牛顿，基于全批次梯度的优化算法优化所有损失函数。对于较大的数据集，例如所讨论的数据驱动模型发现的例子，可以使用随机梯度下降及其现代变体轻松采用计算效率更高的小批量设置。
In all cases pertaining to data-driven solution of partial differential equations, the total number of training data $N_u$ is relatively small (a few hundred up to a few thousand points), and we chose to optimize all loss functions using L-BFGS, a quasi-Newton, full-batch gradient-based optimization algorithm. For larger data-sets, such as the data-driven model discovery examples discussed, a more computationally efficient mini-batch setting can be readily employed using stochastic gradient descent and its modern variants.

#### 3.1.1 Example (Schroedinger equation)

$$
ih_t+0.5h_{xx}+|h^2|h=0,x\in[-5,5],t\in[0,\pi/2]\\
\begin{cases}
h(0,x)=2\mathrm{sech}(x)\\
h(t,-5)=h(t,5)\\
h_x(t,-5)=h_x(t,5)\\
\end{cases}\tag{5}
$$

|            Symbols            |                   Descriptions                   |
| :---------------------------: | :----------------------------------------------: |
|           $h(t,x)$            |  Complex-value solution, represented by an MLP   |
|      $u(t,x)$, $v(t,x)$       |     The real and imaginary part of $h(t,x)$      |
| $\{x^i_0,h^i_0\}^{N_0}_{i=1}$ |            The initial data, $N_0=50$            |
|    $\{t^b_i\}^{N_b}_{i=1}$    | The collocation points on the boundary, $N_b=50$ |
| $\{t^i_f,x^i_f\}^{N_f}_{i=1}$ | The collocations points of $f(t,x)$, $N_f=20000$ |

The mean squared error loss:
$$
\begin{align*}
\mathrm{MSE}&=\mathrm{MSE}_0+\mathrm{MSE}_b+\mathrm{MSE}_f\\
\mathrm{MSE}_0&=\frac{1}{N_0}\sum_{i=1}^{N_0}\left|h(0,x^i_0)-h^i_0\right|^2\\
\mathrm{MSE}_b&=\frac{1}{N_b}\sum_{i=1}^{N_b}\left(\left|h^i(t^i_b,-5)-h^i(t^i_b,5)\right|^2+\left|h^i_x(t^i_b,-5)-h^i_x(t^i_b,5)\right|^2\right)\\
\mathrm{MSE}_f&=\frac{1}{N_f}\sum_{i=1}^{N_f}\left|f(t^i_f,x^i_f)\right|^2\\
\end{align*}\tag{6}
$$

我们选择使用每层 100 个神经元的 5 层深度神经网络和双曲正切激活函数联合表示潜在函数 $h(t,x) = [u(t,x)\ v(t,x)]$。
We chose to jointly represent the latent function $h(t,x) = [u(t,x)\ v(t,x)]$ using a 5-layer deep neural network with 100 neurons per layer and a hyperbolic tangent activation function.

通过采用贝叶斯方法并监控预测后验分布的方差，可以采用一种可行的方法来评估预测解决方案的准确性。
One viable path towards assessing the accuracy of the predicted solution could come by adopting a Bayesian approach and monitoring the variance of the predictive posterior distribution.

迄今为止考虑的连续时间神经网络模型的一个潜在限制源于需要使用大量内点 $N_f$，以便在整个时空域中强制执行基于物理的约束。虽然这对一维或二维空间的问题没有造成重大问题，但它可能会在高维问题中引入严重的瓶颈，因为全局强制执行基于物理的约束所需的点总数（在我们的例子中是偏微分方程）将呈指数增长。
One potential limitation of the continuous time neural network models considered so far stems from the need to use a large number of collocation points $N_f$ in order to enforce physics-informed constraints in the entire spatio-temporal domain. Although this poses no significant issues for problems in one or two spatial dimensions, it may introduce a severe bottleneck in higher dimensional problems, as the total number of collocation points needed to globally enforce a physics-informed constrain (i.e., in our case a partial differential equation) will increase exponentially.

### 3.2 Discrete time models

Let us apply the general form of Runge–Kutta methods with $q$ stages to equation $(2)$ and obtain:
$$
\begin{align*}
u^n&=u^{n+c_i}+\Delta{t}\sum_{j=1}^qa_{ij}\mathcal{N}[u^{n+c_j}],i=1,\cdots,q\\
u^n&=u^{n+1}+\Delta{t}\sum_{j=1}^qb_{j}\mathcal{N}[u^{n+c_j}]\\
u^{n+c_j}(x)&=u\bigl(t^n+c_j\Delta{t},x\bigr),j=1,\cdots,q\\
\end{align*}\tag{7}
$$
This general form encapsulates both implicit and explicit time-stepping schemes, depending on the choice of the parameters $\{a_{ij},b_j,c_j\}$. Equations $(7)$ can be equivalently expressed as:
$$
u^n=u^n_i\quad i=1,\dots,q\\
u^n=u^n_{q+1}\tag{8}
$$
where:
$$
u^n_i=u^{n+c_i}+\Delta{t}\sum_{j=1}^qa_{ij}\mathcal{N}[u^{n+c_j}],i=1,\cdots,q\\
u^n_{q+1}=u^{n+1}+\Delta{t}\sum_{j=1}^qb_{j}\mathcal{N}[u^{n+c_j}]\tag{9}
$$
We proceed by placing a multi-output neural network prior on:
$$
[u^{n+c_1}(x),\dots,u^{n+c_q}(x), u^{n+1}(x)]\tag{10}
$$
which is:
$$
[u^n_1(x),\dots,u^n_q(x), u^n_{q+1}(x)]\tag{11}
$$

#### 3.2.1 Example (Allen–Cahn equation)

$$
u_t+0.0001u_{xx}+5u^3-5u=0,x\in[-1,1],t\in[0,1]\\
\begin{cases}
u(0,x)=x^2\cos(\pi x)\\
h(t,-1)=h(t,1)\\
h_x(t,-1)=h_x(t,1)\\
\end{cases}\tag{12}
$$

For the Allen–Cahn equation, the nonlinear operator in equation (9) is given by:
$$
\mathcal{N}[u^{n+c_j}]+0.0001u^{n+c_j}_{xx}+5(u^{n+c_j})^3-5u^{n+c_j}=0
$$
The sum of squared error loss:
$$
\begin{align*}
\mathrm{SSE}&=\mathrm{SSE}_n+\mathrm{SSE}_b\\
\mathrm{SSE}_n&=\sum_{i=1}^{q+1}\sum_{i=1}^{N_n}\left|u^n_j(x^{n,i})-u^{n,i}\right|^2\\
\mathrm{SSE}_b&=\sum_{i=1}^{q}\left|u^{n+c_i}(-1)-u^{n+c_i}(1)\right|^2+\left|u^{n+1}(-1)-u^{n+1}(1)\right|^2\\
&+\sum_{i=1}^{q}\left|u^{n+c_i}_x(-1)-u^{n+c_i}_x(1)\right|^2+\left|u^{n+1}_x(-1)-u^{n+1}_x(1)\right|^2\\
\end{align*}\tag{13}
$$
在经典数值分析中，由于显式方案的稳定性约束或隐式公式的计算复杂性约束，这些时间步长通常被限制在很小的范围内。
In classical numerical analysis, these time-steps are usually confined to be small due to stability constraints for explicit schemes or computational complexity constrains for implicit formulations.

随着 Runge-Kutta 阶段总数 $q$ 的增加，这些约束变得更加严格，并且对于大多数实际感兴趣的问题，需要采取数千到数百万个这样的步骤，直到解决方案达到所需的最终时间。
These constraints become more severe as the total number of Runge–Kutta stages $q$ is increased, and, for most problems of practical interest, one needs to take thousands to millions of such steps until the solution is resolved up to a desired final time.

与经典方法形成鲜明对比的是，在这里我们可以采用具有任意大量阶段的隐式龙格-库塔方案，而实际上额外成本非常低。这使我们能够在保持稳定性和高预测准确性的同时采取非常大的时间步长，从而使我们能够在一个步骤中解决整个时空解决方案。
In sharp contrast to classical methods, here we can employ implicit Runge–Kutta schemes with an arbitrarily large number of stages at effectively very little extra cost. This enables us to take very large time steps while retaining stability and high predictive accuracy, therefore allowing us to resolve the entire spatio-temporal solution in a single step.

我们采用离散时间物理信息神经网络，每层有 4 个隐藏层和 200 个神经元，而输出层预测 101 个感兴趣的数量，对应于 $q = 100$ Runge–Kutta 阶段 $u^{n+c_i}(x)$，以及最后时刻的解 $u^{n+1}(x)$。
We employ a discrete time physics-informed neural network with 4 hidden layers and 200 neurons per layer, while the output layer predicts 101 quantities of interest corresponding to the $q = 100$ Runge–Kutta stages $u^{n+c_i}(x)$, and the solution at final time $u^{n+1}(x)$.  

低阶方法无法保持其对大时间步长的预测准确性，因此需要一种具有多个小规模时间步长的解决方案策略。 另一方面，将 Runge-Kutta 阶段的数量推到 32 甚至更高的能力允许我们采取非常大的时间步长，并在不牺牲我们预测的准确性的情况下在一个步骤中有效地解决解决方案。
Low-order methods cannot retain their predictive accuracy for large time-steps, thus mandating a solution strategy with multiple time-steps of small size. On the other hand, the ability to push the number of Runge–Kutta stages to 32 and even higher allows us to take very large time steps, and effectively resolve the solution in a single step without sacrificing the accuracy of our predictions.

## 4 Data-driven discovery of partial differential equations

### 4.1 Continuous time models

#### 4.1.1 Example (Navier–Stokes equation)

$$
\frac{\partial\mathbf{u}}{\partial{t}}+\lambda_1\mathbf{u}\cdot\nabla\mathbf{u}=-\nabla\mathbf{p}+\lambda\nabla^2\mathbf{u}
$$

化为坐标形式：
$$
u_t+\lambda_1(uu_x+vu_y)=-p_x+\lambda_2(u_{xx}+u_{yy})\\
v_t+\lambda_1(uv_x+vv_y)=-p_v+\lambda_2(v_{xx}+v_{yy})\tag{15}
$$

其中 $u(t,x,y)$ 是速度场的 $x$ 分量，$v(t,x,y)$ 是速度场的 $y$ 分量。

Navier-Stokes 方程的解在无散度函数集合中搜索；即：
Solutions to the Navier–Stokes equations are searched in the set of divergence-free functions; i.e.,
$$
u_x+v_y=0\tag{16}
$$
这个额外的方程是不可压缩流体的连续性方程，它描述了流体的质量守恒。
This extra equation is the continuity equation for incompressible fluids that describes the conservation of mass of the fluid.
$$
u=\psi_y,v=-\psi_x\tag{17}
$$
对于一些潜在的势函数 $\psi(t,x,y)$。
for some latent potential function $\psi(t,x,y)$.

We define $f(t,x,y)$ and $g(t,x,y)$ to be:
$$
f:=u_t+\lambda_1(uu_x+vu_y)+p_x-\lambda_2(u_{xx}+u_{yy})\\
g:=v_t+\lambda_1(uv_x+vv_y)+p_v-\lambda_2(v_{xx}+v_{yy})\tag{18}
$$
The mean squared error loss:
$$
\begin{align*}
\mathrm{MSE}&=\mathrm{MSE}_{uv}+\mathrm{MSE}_{fg}\\
\mathrm{MSE}_{uv}&=\frac{1}{N}\sum_{i=1}^{N_b}\left(\left|u(t^i,x^i,y^i)-u^i\right|^2+\left|v(t^i,x^i,y^i)-v^i\right|^2\right)\\
\mathrm{MSE}_{fg}&=\frac{1}{N}\sum_{i=1}^{N_f}\left(\left|f(t^i,x^i,y^i)\right|^2+\left|g(t^i,x^i,y^i)\right|^2\right)
\end{align*}\tag{19}
$$
在这里，我们考虑通过圆柱体的不可压缩流的原型问题；一个已知的问题，在雷诺数 $Re=u_{\infty}D/\nu$ 的不同状态下表现出丰富的动态行为和转换。
Here we consider the prototype problem of incompressible flow past a circular cylinder; a problem known to exhibit rich dynamic behavior and transitions for different regimes of the Reynolds number $Re=u_{\infty}D/\nu=1\times1\times0.01$.

这里使用的神经网络架构由 9 层组成，每层有 20 个神经元。
The neural network architecture used here consists of 9 layers with 20 neurons in each layer.

一个更有趣的结果源于网络在没有关于压力本身的任何训练数据的情况下提供对整个压力场 $p(t,x,y)$ 的定性准确预测的能力。
A more intriguing result stems from the network’s ability to provide a qualitatively accurate prediction of the entire pressure field $p(t,x,y)$ in the absence of any training data on the pressure itself.

### 4.2 Discrete time models

$$
\begin{align*}
u^n&=u^{n+c_i}+\Delta{t}\sum_{j=1}^qa_{ij}\mathcal{N}[u^{n+c_j};\lambda],i=1,\cdots,q\\
u^n&=u^{n+1}+\Delta{t}\sum_{j=1}^qb_{j}\mathcal{N}[u^{n+c_j};\lambda]\\
\end{align*}\tag{20}
$$

$u^{n+c_j}(x)=u\bigl(t^n+c_j\Delta{t},x\bigr),j=1,\cdots,q$ is the hidden state.

This general form encapsulates both implicit and explicit time-stepping schemes, depending on the choice of the parameters $\{a_{ij},b_j,c_j\}$. Equations $(20)$​ can be equivalently expressed as:
$$
u^n=u^n_i\quad i=1,\dots,q\\
u^{n+1}=u^{n+1}_i\quad i=1,\dots,q\tag{21}
$$
where:
$$
u^n_i=u^{n+c_i}+\Delta{t}\sum_{j=1}^qa_{ij}\mathcal{N}[u^{n+c_j};\lambda],i=1,\cdots,q\\
u^{n+1}_i=u^{n+c_i}+\Delta{t}\sum_{j=1}^q(a_{ij}-b_{j})\mathcal{N}[u^{n+c_j};\lambda]\tag{22}
$$
We proceed by placing a multi-output neural network prior on:
$$
[u^{n+c_1}(x),\dots,u^{n+c_q}(x)]\tag{23}
$$
This prior assumption along with equations $(22)$ result in two physics-informed neural networks:
$$
[u^n_1(x),\dots,u^n_q(x), u^n_{q+1}(x)]\tag{24}
$$

and:
$$
[u^{n+1}_1(x),\dots,u^{n+1}_q(x), u^{n+1}_{q+1}(x)]\tag{25}
$$
The sum of squared errors:
$$
\begin{align*}
\mathrm{SSE}&=\mathrm{SSE}_n+\mathrm{SSE}_{n+1}\\
\mathrm{SSE}_n&=\sum_{i=1}^{q}\sum_{i=1}^{N_n}\left|u^n_j(x^{n,i})-u^{n,i}\right|^2\\
\mathrm{SSE}_{n+1}&=\sum_{i=1}^{q}\sum_{i=1}^{N_{n+1}}\left|u^{n+1}_j(x^{{n+1},i})-u^{{n+1},i}\right|^2
\end{align*}\tag{26}
$$

#### 4.2.1 Example (Korteweg-de Vries equation)

$$
u_t+\lambda_1uu_x+\lambda_2u_{xxx}=0\tag{27}
$$

## 5 Conclusion

我们引入了基于物理的神经网络，这是一类新的通用函数逼近器，它能够编码任何支配给定数据集的基本物理定律，并且可以用偏微分方程来描述。
We have introduced physics-informed neural networks, a new class of universal function approximators that is capable of encoding any underlying physical laws that govern a given data-set, and can be described by partial differential equations.

