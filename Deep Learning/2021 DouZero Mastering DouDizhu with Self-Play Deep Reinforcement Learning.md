# DouZero Mastering DouDizhu with Self-Play Deep Reinforcement Learning

## 0 Abstract

在这项工作中，我们提出了一个概念上简单但有效的斗地主 AI 系统，即 DouZero，它通过深度神经网络、动作编码和并行参与者增强了传统的蒙特卡洛方法。
In this work, we propose a conceptually simple yet effective DouDizhu AI system, namely DouZero, which enhances traditional Monte-Carlo methods with deep neural networks, action encoding, and parallel actors.

## 1 Introduction

首先，斗地主的玩家需要在一个部分可观察的环境中与他人竞争和合作，沟通有限。
First, the players in DouDizhu need to both compete and cooperate with others in a partially observable environment with limited communication.

其次，斗地主拥有大量平均规模非常大的信息集，并且由于卡片的组合而具有高达 $10^4$ 的可能动作的非常复杂和大的动作空间。
Second, DouDizhu has a large number of information sets with a very large average size and has a very complex and large action space of up to $10^4$ possible actions due to combinations of cards.

组合 Q 网络 (CQN) 提出通过将动作解耦为分解选择和最终移动选择来减少动作空间。
Combination Q-Network (CQN) proposes to reduce the action space by decoupling the actions into decomposition selection and final move selection.

它通过使用贝叶斯方法来推断隐藏信息并根据他们自己的策略网络对其他玩家的行为进行采样，从而实现了类似于 AlphaZero 的算法。为了抽象动作空间，DeltaDou 基于启发式规则预训练了一个 kicker 网络。
It enables an AlphaZero-like algorithm by using Bayesian methods to infer hidden information and sampling the other players's actions based on their own policy networks. To abstract the action space, DeltaDou pre-trains a kicker network based on heuristic rules.

此外，贝叶斯推理和搜索在计算上是昂贵的。即使使用启发式的监督回归初始化网络，训练 DeltaDou 也需要两个多月的时间。因此，现有的斗地主 AI 程序在计算上非常昂贵，并且可能不是最优的，因为它们高度依赖于人类知识的抽象。
Moreover, the Bayesian inference and the search are computationally expensive. It takes more than two months to train DeltaDou even when initializing the networks with supervised regression to heuristics. Therefore, the existing DouDizhu AI programs are computationally expensive and could be sub-optimal since they highly rely on abstractions with human knowledge.

在这项工作中，我们展示了 DouZero，这是一个概念上简单但有效的用于 DouDizhu 的 AI 系统，没有状态/动作空间或任何人类知识的抽象。
In this work, we present DouZero, a conceptually simple yet effective AI system for DouDizhu without the abstraction of the state/action space or any human knowledge.

首先，与 DQN 不同，它不易受到高估偏差的影响。其次，通过将动作编码成卡片矩阵，它可以自然地泛化在整个训练过程中不常见的动作。
First, unlike DQN, it is not susceptible to overestimation bias. Second, by encoding the actions into card matrices, it can naturally generalize over the actions that are not frequently seen throughout the training process.

通过构建系统，我们证明 DouZero 经典蒙特卡洛方法可以在需要在巨大状态和动作空间上进行竞争和合作推理的大型复杂纸牌游戏中提供强大的结果。
Through building system, we demonstrate that DouZero classical Monte-Carlo methods can be made to deliver strong results in large-scale and complex card games that need to reason about both competing and cooperation over huge state and action spaces.

## 2 Background of DouDizhu

首先，农民需要合作对抗地主。
First, the Peasants need to cooperate in fighting against the Landlord.

其次，斗地主由于卡牌的组合，动作空间复杂而大。
Second, DouDizhu has a complex and large action space due to the combination of cards.

## 3 Deep Monte-Carlo

### 3.1 Monte-Carlo Methods with Deep Neural Networks

Monte-Carlo (MC) 方法是基于平均样本回报的传统强化学习算法。
Monte-Carlo (MC) methods are traditional reinforcement learning algorithms based on averaging sample returns.

MC 方法是为回合任务设计的，其中经历可以分为回合，所有回合最终都会终止。为了优化策略，每次访问 MC 可用于通过迭代执行以下过程来估计 Q 表 $Q(s;a)$：
MC methods are designed for episodic tasks, where experiences can be divided into episodes and all the episodes eventually terminate. To optimize a policy, every-visit MC can be used to estimate Q-table $Q(s; a)$ by iteratively executing the following procedure:

1. 使用 $\pi$ 生成回合。
    Generating an episode using $\pi$.
2. 对于回合中出现的每个 $s,a$，计算并更新 $Q(s,a)$，并使用与 $s,a$ 相关的所有样本的平均回报。
    For each $s,a$ appeared in the episode, calculate and update $Q(s,a)$ with the return averaged over all the samples concerning $s,a$.
3. 对于回合中的每个 $s$，$\pi(s)\gets\arg\max_{a}Q(s,a)$。
    For each $s$ in the episode, $\pi(s)\gets\arg\max_{a}Q(s,a)$.

具体来说，我们可以用神经网络替换 Q 表，并在步骤 2 中使用均方误差 (MSE) 来更新 Q 网络。
Specifically, we can replace the Q-table with a neural network and use mean square error (MSE) to update the Q-network in Step 2.

### 3.2 Comparison with Policy Gradient Methods

### 3.3 Comparison with Deep Q-Learning

## 4 DouZero System

### 4.1 Card Representation and Neural Architecture

我们用一个独热的 $4\times15$ 矩阵对每个卡片组合进行编码。由于花色在斗地主中无关紧要，我们使用每一行来表示牌或大小王的数量。
We encode each card combination with a one-hot $4\times15$ matrix. Since suits are irrelevant in DouDizhu, we use each row to represent the number of cards of a specific rank or joker.

对于状态，我们提取了几个卡片矩阵来表示手牌，其他玩家手牌的并集和最近的移动，以及一些独热向量来表示其他玩家到此为止的卡片数量和炸弹数量玩。同样，我们使用一个卡片矩阵来编码动作。
For the state, we extract several card matrices to represent the hand cards, the union of the other players' hand cards and the most recent moves, and some one-hot vectors to represent the number of cards of the other players and the number of bombs played so far. Similarly, we use one card matrix to encode the action.

对于神经架构，LSTM 用于编码历史动作，输出与其他状态/动作特征连接。最后，我们使用隐藏大小为 512 的六层 MLP 来生成 Q 值。
For the neural architecture, LSTM is used to encode historical moves, and the output is concatenated with the other state/action features. Finally, we use six layers of MLP with a hidden size of 512 to produce Q-values.

### 4.2 Parallel Actors

我们将地主表示为 L，将在地主之前移动的玩家表示为 U，将在地主之后移动的玩家表示为 D。
We denote Landlord as L, the player that moves before the Landlord as U, and the player that moves after the Landlord as D.

我们将 DMC 与多个参与者进程和一个学习者进程并行化，算法如下。
We parallelize DMC with multiple actor processes and one learner process, below are algorithm.

> Actor Process
>
> **输入**：有 $B$ 个位置，每个位置大小为 $S$ 的共享缓存 $B_L,B_U,B_D$；$\epsilon$，$\gamma$。
>
> 首先初始化 Q-网络 $Q_L,Q_U,Q_D$ 和临时缓存 $D_L,D_U,D_D$。
>
> 开始循环
>
> - $Q_L,Q_U,Q_D$ 和 Learner Process 进行同步。
> - 生成一个回合。在 $T$ 次出牌内：
>     - 根据当前出牌者选择从 $Q_L,Q_U,Q_D$ 选择一个网络 $Q$。
>     - 进行 epsilon greedy：随机选一个行动，或者 $\arg\max_aQ(s_t,a)$，得到这步行动 $a_t$。
>     - 进行这步行动，得到下一步的状态 $s_{t+1}$ 和奖励 $r_t$。
>     - 把对应的 $\{s_t,a_t,r_t\}$ 存储到对应的临时缓存 $D_L,D_U,D_D$。
> - 回溯。获得这个回合累积的奖励，从获得的奖励里面，倒着：
>     - 用 $r_t\gets r_t+\gamma r_{t+1}$ 更新在临时缓存 $D_L,D_U,D_D$ 的 $r_t$。
> - 对于某个属于 $L,U,D$ 的 $p$ （多线程优化）：
>     - 如果 $D_p$ 的长度超过 $L$：
>         - 等待一个 $B_p$ 里面空的位置。
>         - 把这 $L$ 个 $\{s_t,a_t,r_t\}$ 从 $D_p$ 移到 $B_p$。

> Learner Process
>
> **输入**：有 $B$ 个位置，每个位置大小为 $S$ 的共享缓存 $B_L,B_U,B_D$；batch 大小 $M$；学习率 $\psi$。
>
> 首先初始化全局 Q-网络 $Q_L^g,Q_U^g,Q_D^g$。
>
> 迭代直到收敛：
>
> - 对于某个属于 $L,U,D$ 的 $p$ （多线程优化）：
>     - 如果 $B_p$ 里面满的位置超过了 $M$ 个：
>         - 提取一个 $M\times S$ 的 batch，清空对应位置。
>         - 用 MSE loss 训练 $Q_p^g$ ，更新学习率 $\psi$。

学习器为三个位置维护三个全局 Q 网络，并根据参与者进程提供的数据更新具有 MSE 损失的网络以逼近目标值。
The learner maintains three global Q-networks for the three positions and updates the networks with MSE loss to approximate the target values based on the data provided by the actor processes. 

每个参与者维护三个局部 Q 网络，它们定期与全局网络同步。参与者将重复从游戏引擎中采样轨迹，并计算每个状态——动作对的累积奖励。
Each actor maintains three local Q-networks, which are synchronized with the global networks periodically. The actor will repeatedly sample trajectories from the game engine and calculate cumulative reward for each state-action pair.

## 5 Experiments

### 5.1 Experiments Setup

WP（Winning Percentage）：A获胜的局数除以总局数。
WP (Winning Percentage): The number of the games won by A divided by the total number of games.

ADP（Average Difference in Points）：A和B每场比赛得分的平均差值，以 1 为基础分。每出一个炸弹，得分翻倍。
ADP (Average Difference in Points): The average difference of points scored per game between A and B. The base point is 1. Each bomb will double the score.

如果使用 ADP 作为奖励，agent 在玩炸弹时往往会非常谨慎，因为玩炸弹是有风险的，可能会导致更大的 ADP 损失。相反，以 WP 为目标，即使会输，代理也倾向于积极地使用炸弹，因为炸弹不会影响 WP。
If using ADP as reward, the agent tends to be very cautious about playing bombs since playing a bomb is risky and may lead to larger ADP loss. In contrast, with WP as objective, the agent tends to aggressively play bombs even if it will lose because a bomb will not affect WP.

## C Additional Details of Feature Representation and Neural Architecture

### C.1 Action and State Representation

神经网络的输入是状态和动作的连接表示。对于每个 $4\times15$ 卡片矩阵，我们首先将矩阵展平为大小为 60 的一维向量。然后我们删除六个始终为零的条目，因为只有一个大小王。
The input of the neural network is the concatenated representation of state and action. For each $4\times15$ card matrix, we firs flatten the matrix into a 1-dimensional vector of size 60. Then we remove six entries that are always zero since there is only one black or red joker.

除了卡片矩阵，我们进一步使用一个独热向量来表示其他两个玩家当前的手牌。对于农民，我们使用大小为 17 的向量，其中每个条目对应于当前状态下的手牌数量。对于地主，向量的大小是 20。同样，我们使用一个 15 维向量来表示当前状态下的炸弹数量。
In addition to card matrices, we further use a one-hot vector to represent the other two players' current hand cards. For Peasants, we use a vector of size 17, where each entry corresponds to the number of hand cards in the current state. For the Landlord, the vector's size is 20. Similarly, we use a 15-dimension vector to represent the number of bombs in the current state.

对于历史记录，我们考虑最近的 15 步，并连接每三个连续步的表示；也就是说，历史记录被编码成一个 $5\times162$ 矩阵。历史记录被输入 LSTM，我们使用最后一个单元格中的隐藏表示来表示历史记录。如果历史上少于 15 个动作，我们使用零矩阵来表示缺失的动作。
For historical moves, we consider the most recent 15 moves and concatenate the representations of every three consecutive moves; that is, the historical moves are encoded into a $5\times162$ matrix. The historical moves are fed into an LSTM, and we use the hidden representation in the last cell to represent the historical moves. If there are less than 15 moves historically, we use zero matrices for the missing moves.

### C.2 Data Collection and Neural Architecture of Supervised Learning

经过过滤，我们获得了 226,230 个人类专家比赛。我们将每个动作视为一个实例，并使用监督损失来训练网络。该问题可以表述为分类问题，我们的目标是根据给定状态预测动作，共有 27,472 个类别。
After filtering, we obtain 226,230 human expert matches. We treat each move as an instance and use a supervised loss to train the networks. The problem can be formulated as a classification problem, where we aim at predicting the action based on a given state, with a total of 27,472 classes.

然而，我们在实践中发现大多数操作都是非法的，并且迭代所有类的成本很高。受 Q-network 设计的启发，我们将问题转化为二元分类任务。
However, we find in practice that most of the actions are illegal, and it is expensive to iterate over all the classes. Motivated by Q-network's design, we transform the problem into a binary classification task.

我们将用户数据转换为正例，并根据未选择的合法移动生成负例。最终，训练数据包含 49,990,075 个实例。我们进一步发现数据是不平衡的，其中负实例的数量远大于正实例的数量。因此，我们采用基于正负实例分布的重新加权交叉熵损失。
We transform the user data into positive instances and generate negative instances based on the legal moves that are not selected. Eventually, the training data consists of 49,990,075 instances. We further find that the data is imbalanced, where the number of negative instances is much larger than that of positive instances. Thus, we adopt a re-weighted cross-entropy loss based on the distribution of positive and negative instances.

### C.3 Neural Architecture and Training Details of Bidding Network

抢地主阶段的目标是根据手牌的强度来决定玩家是否应该成为地主。这个决定比打牌简单得多，因为玩家只需要考虑手牌和其他玩家的决定，我们只需要进行二元预测。
The bidding phase's goal is to determine whether a player should become the landlord based on the strengths of the hand cards. This decision is much simpler than card-playing since the agent only needs to consider the hand cards and the other players' decisions, and we only need to make a binary prediction.

我们提取了 128 个特征来表示手牌和玩家的移动。对于网络架构，我们使用 (512, 256, 128, 64, 32, 16) MLP。与有监督的纸牌代理一样，我们在输出中添加了一个 Sigmoid 函数，并使用二元交叉熵损失来训练网络。
We extract 128 features to represent hand cards and the players' moves. For the network architecture, we use a (512, 256, 128, 64, 32, 16) MLP. Like the supervised card playing agent, we add a Sigmoid function to the output and train the network with binary cross-entropy loss.
