# 鲁棒性  Robustness

通过暴露和修复漏洞来确保AI系统的安全性和可靠性

1. 识别并防御新的攻击
2. 设计新的对抗性训练方法来增强对攻击的抵御能力
3. 开发新的度量来评估稳健性。

## 相关文献

### 高优先级

#### 对抗样本设计与抵御

[Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey](https://arxiv.org/pdf/1801.00553.pdf)

总结了目前主流的对抗性样本攻击及抵御方法

[EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES](https://arxiv.org/pdf/1412.6572.pdf)

Goodfellow 等人对对抗性样本的解释以及控制方法

[Synthesizing Robust Adversarial Examples](https://arxiv.org/pdf/1707.07397.pdf)

设计强大的对抗样本

[DELVING INTO TRANSFERABLE ADVERSARIAL EXAMPLES AND BLACK-BOX ATTACKS](https://arxiv.org/pdf/1611.02770.pdf)

对抗样本通常不特定于某个模型或架构，针对某个神经网络架构生成的对抗样本可以很好地转换到另一个架构中。这意味着有可能对一个完全的黑箱模型创建一个对抗样本。伯克利的一个小组使用这种方法在商业性的人工智能分类系统中发起了一次成功的攻击

[Exploring the Hyperparameter Landscape of Adversarial Robustness](https://arxiv.org/pdf/1905.03837.pdf)

探讨了对抗性训练的一些实际挑战，提出了一种实用的方法，利用超参数优化技术来调整对抗性训练，以最大限度地提高稳健性。

[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083.pdf)

介绍了提高显著提高对抗性攻击抵御能力的方法

[Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images](https://link.zhihu.com/?target=http%3A//arxiv.org/pdf/1412.1897)

介绍了如何用对抗样本欺骗神经网络做出错误的判断



#### 鲁棒性评估

[Ensemble Adversarial Training Attacks and Defenses](https://arxiv.org/pdf/1705.07204.pdf)

Goodfellow 等人阐述了如何评判一个模型针对对抗扰动的抵抗性，以及同时进行白盒攻击和黑盒攻击的重要性。

[CERTIFIED DEFENSES AGAINST ADVERSARIAL EXAMPLES](https://arxiv.org/pdf/1801.09344.pdf)

评估神经网络的对抗鲁棒性

[CNN-Cert: An Efficient Framework for Certifying Robustness of Convolutional Neural Networks](https://arxiv.org/pdf/1811.12395.pdf)

提出一个通用且有效的框架：CNN-Cert，它能够证明一般卷积神经网络的鲁棒性。

[Evaluating the Robustness of Neural Networks: An Extreme Value Theory Approach](https://arxiv.org/pdf/1801.10578.pdf)

提供了将鲁棒性分析转换为局部Lipschitz常数估计问题的理论证明，并提出使用极值理论进行有效评估。我们的分析产生了一种新的鲁棒性度量标准，称为CLEVER，CLEVER是第一个可以应用于任何神经网络分类器的独立于攻击(attack-independent) 的稳健性度量。



#### 其他鲁棒性研究

[Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks](https://www.researchgate.net/profile/Guy_Katz/publication/313394663_Reluplex_An_Efficient_SMT_Solver_for_Verifying_Deep_Neural_Networks/links/58ab336692851cf0e3ca4659/Reluplex-An-Efficient-SMT-Solver-for-Verifying-Deep-Neural-Networks.pdf)

深度神经网络的可验证性,，提出了一种用于神经网络错误检测的新算法 Reluplex

[PROVEN: Verifying Robustness of Neural Networks with a Probabilistic Approach](https://arxiv.org/pdf/1812.08329.pdf)

提出了一种新的概率框架，可以通过统计保证 **(statistical guarantees)** 对神经网络进行概率论验证

[Efficient Neural Network Robustness Certification with General Activation Functions](http://papers.nips.cc/paper/7742-efficient-neural-network-robustness-certification-with-general-activation-functions.pdf)

介绍了CROWN，这是一个根据激活函数来验证神经网络鲁棒性的通用框架。



### 次要优先级

[Defensive Quantization: When Efficiency Meets Robustness](https://arxiv.org/pdf/1904.08444.pdf)

旨在提高人们对量化模型安全性的认识，并设计了一种新的量化方法，共同优化深度学习量化模型的效率和鲁棒性

[Kernel-Based Reinforcement Learning in Robust Markov Decision Processes](https://papers.nips.cc/paper/5183-reinforcement-learning-in-robust-markov-decision-processes.pdf)

设计了一种适用于潜在对抗行为的算法来确保马尔可夫决策过程在意外或对抗系统行为方面的稳健性

[Analyzing Federated Learning through an Adversarial Lens](https://arxiv.org/pdf/1811.12470.pdf)
探讨了联合学习领域的一些恶意攻击的策略从而突出联合学习的脆弱性以及制定有效防御策略的必要性

[L2 - Nonexpansive Neural Networks](https://arxiv.org/pdf/1802.07896.pdf)

优化了控制Lipschitz常数的方法，以实现其最大化鲁棒性的全部潜力，提出的分类器在针对白盒L2限制对抗性攻击的鲁棒性方面超过了现有技术水平

[Structured Adversarial Attack: Towards General Implementation and Better Interpretability](https://arxiv.org/pdf/1808.01664.pdf)

提出了 StrAttack 模型来探索对抗性扰动中的群体稀疏性

[Query-Efficient Hard-label Black-box Attack: An Optimization-based Approach](https://arxiv.org/pdf/1807.04457.pdf)

研究了在硬标签黑盒设置中攻击机器学习模型的问题

[AutoZOOM: Autoencoder-based Zeroth Order Optimization Method for Attacking Blackbox Neural Networks](https://arxiv.org/pdf/1805.11770.pdf)

提出了一种可以高效查询的黑盒攻击通用框架 AutoZOOM

[Anytime Best+Depth-First Search for Bounding Marginal MAP](https://pdfs.semanticscholar.org/4a09/10dce5c57e2b7a1b4454057ecbcee3eeb030.pdf)

引入了新的随时搜索算法，这些算法将最佳优先和深度优先搜索结合到图形模型中的边际MAP推理的混合方案中

[Attacking Visual Language Grounding with Adversarial Examples: A Case Study on Neural Image Captioning](https://arxiv.org/pdf/1712.02051.pdf)

为了研究语言基础对机器视觉和感知中的对抗性扰动的鲁棒性，提出了Show-and-Fool，一种用于制作神经图像字幕中的对抗性示例的新算法。

[BlockDrop: Dynamic Inference Paths in Residual Networks](https://arxiv.org/pdf/1711.08393.pdf)

介绍了 BlockDrop，动态的选择使用深层网络中的哪些层，从而在不降低预测准确率的情况下最佳的减少总计算量

[Exploiting Rich Syntactic Information for Semantic Parsing with Graph-to-Sequence Model](https://arxiv.org/pdf/1808.07624.pdf)

采用图形到序列模型来编码句法图并解码逻辑形式。 通过编码更多的句法信息，也可以提高模型的鲁棒性。

[Adversarial Phenomenon from the Eyes of Bayesian Deep Learning](https://arxiv.org/pdf/1711.08244.pdf)

考虑使用贝叶斯神经网络来检测对抗性实例

[Protecting Intellectual Property of Deep Neural Networks with Watermarking](https://gzs715.github.io/pubs/WATERMARK_ASIACCS18.pdf)

提出了一种水印植入方法，将水印注入深度学习模型，并设计了一种远程验证机制来确定模型所有权，用水印技术保护神经网络的知识产权。

[Unravelling Robustness of Deep Learning based Face Recognition Against Adversarial Attacks](https://arxiv.org/pdf/1803.00401.pdf)

通过利用网络中隐藏层的响应适当地设计分类器，能够以非常高的精度检测攻击。最后，我们提出了几种有效的对策来减轻对抗性攻击的影响，并提高基于DNN的人脸识别的整体稳健性。

[EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples](https://arxiv.org/pdf/1709.04114.pdf)

我们通过对抗性的例子来描述攻击DNN的过程，作为弹性网络正则化优化问题。

# 公平性 Fairness

设计检测并且减消除偏见的方法来确保 AI 不会被人们的偏见影响，也不会激化人们的偏见

1. 对数据集进行预处理，消除数据集中的偏见
2. 消除模型带来的偏见
3. 对模型进行公平性评估

## 相关文献

### 高优先级

[Automated Test Generation to Detect Individual Discrimination in AI Models](https://arxiv.org/pdf/1809.03260.pdf)

解决了检测模型是否具有个体歧视的问题

[Design AI so that it's fair](https://www.nature.com/magazine-assets/d41586-018-05707-8/d41586-018-05707-8.pdf)

寻找和消除神经网络带来的偏差

[Fairness GAN: Generating Datasets with Fairness Properties using a Generative Adversarial Network](http://krvarshney.github.io/pubs/SattigeriHCV_safeml2019.pdf)

[Fairness Gan](https://arxiv.org/pdf/1805.09910.pdf)

使用公平性的生成对抗网络生成数据集，产生公平合理的图像

[AI Fairness 360: An Extensible Toolkit for Detecting, Understanding, and Mitigating Unwanted Algorithmic Bias](https://arxiv.org/pdf/1810.01943.pdf)

介绍了一个新的开源python库：AIF360，为公平研究人员提供共享和评估算法的通用框架

[Towards Composable Bias Rating of AI Systems](https://arxiv.org/pdf/1808.00089.pdf)

设想建立独立于API生产者和消费者的第三方机构来对AI系统的公平性进行评估

[An End-To-End Machine Learning Pipeline That Ensures Fairness Policies](https://arxiv.org/pdf/1710.06876.pdf)

手动理解策略并确保不透明ML系统的公平性是耗时且容易出错的，因此需要端到端系统来确保数据所有者和用户始终遵守公平政策。该系统可以：1）理解用自然语言编写的策略，2）警告用户违反策略，3）记录执行的每个活动，以便后续证明策略合规性。



### 次要优先级

[Scalable Fair Clustering](https://arxiv.org/pdf/1902.03519.pdf)

提出了一种线性时间的聚类算法，能更精细的控制聚类的平衡

[Scalable Fair Clustering](https://arxiv.org/pdf/1902.03519.pdf)

研究人脸识别神经网络的公平性，提供了人类可解释的面部特征的定量测量，推动创建更公平和准确的人脸识别系统

[Data Pre-Processing for Discrimination Prevention: Information-Theoretic Optimization and Analysis](http://krvarshney.github.io/pubs/CalmonWVRV_jstsp2018.pdf)

[Optimized Pre-Processing for Discrimination Prevention](https://arxiv.org/pdf/1704.03354.pdf)

介绍了一种新的概率预处理方法，用于减少歧视

[Analyze, Detect and Remove Gender Stereotyping from Bollywood Movies](http://proceedings.mlr.press/v81/madaan18a/madaan18a.pdf)

分析电影或者海报中的性别偏见

[Modeling Epistemological Principles for Bias Mitigation in AI Systems: An Illustration in Hiring Decisions](https://arxiv.org/pdf/1711.07111.pdf)

本文提出了一种结构化方法，以减轻人工智能系统偏见造成的歧视和不公平。研究AI对招聘简历的分析。

[Fairness in Deceased Organ Matching](https://www.cse.unsw.edu.au/~tw/mswaies2018.pdf)

研究如何公平地决定如何将已故捐献者捐赠的器官与患者相匹配

# 可解释性  Explainability

了解 AI 输出结果的依据是可信的关键要素，尤其是对企业级 AI 而言。为了提高透明度：

1. 研究模型及输出的局部可解释性和全局可解释性
2. 训练可解释模型并且将模型内的信息流可视化

## 相关文献

### 高优先级

[Understanding black-box predictions via influence functions](https://arxiv.org/pdf/1703.04730.pdf)

描述神经网络的可解释性

[Seq2Seq-Vis: A Visual Debugging Tool for Sequence-to-Sequence Models](https://arxiv.org/pdf/1804.09299.pdf)

设计了一款可用于 Seq2Seq 翻译模型 debug 的可视化工具

[Teaching Meaningful Explanations](https://arxiv.org/pdf/1805.11648.pdf)

提出了一种可解释的方法，让训练数据除了包含特征和标签之外，还包含用户给出的解释，然后使用联合模型进行学习，针对输入特征输出标签和解释。

[Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives](https://arxiv.org/pdf/1802.07623.pdf)

提出了一种对黑盒分类器提供对比解释的新方法，来证明分类是否合理



### 次要优先级

[Why Interpretability in Machine Learning? An Answer Using Distributed Detection and Data Fusion Theory](https://arxiv.org/pdf/1806.09710.pdf)

使用分布式检测理论来表征系统的性能，并证明具有可解释性的分类系统优于黑盒系统

[Collaborative Human-AI (CHAI): Evidence-Based Interpretable Melanoma Classification in Dermoscopic Images](https://arxiv.org/pdf/1805.12234.pdf)

提出了一种基于证据的皮肤图像分类方法

[Interpretable to Whom? A Role-based Model for Analyzing Interpretable Machine Learning Systems](https://arxiv.org/ftp/arxiv/papers/1806/1806.07552.pdf)

识别代理在机器学习系统中实现的不同角色以及如何影响其目标，并且定义可解释性的含义。

[Improving Simple Models with Confidence Profiles](https://arxiv.org/pdf/1807.07506.pdf)

提出了 ProfWeight 方法将信息从具有高测试精度的预训练深度神经网络传递到更简单的可解释模型或低复杂度和先验低测试精度的非常浅的网络

# 可追溯性 Lineage

确保 AI 系统所有的部件和事件都是可追溯的

1. 设计事件生成记录模块
2. 设计可扩展的事件提取和管理模块
3. 设计高效的可追溯查询模块来管理 AI 系统的完整生命周期

## 相关文献

[FactSheets: Increasing Trust in AI Services through Supplier's Declarations of Conformity](https://arxiv.org/pdf/1808.07261.pdf)

提出供应商的AI服务符合性声明（SDoC），以描述产品的沿袭以及它经历的安全性和性能测试，帮助增加对AI服务的信任。 我们设想用于人工智能服务的SDoC包含目的，性能，安全性，安全性和出处信息，由AI服务提供商完成并自愿发布，供消费者检查。 重要的是，它传达了产品级而不是组件级的功能测试。 



# 名词解释

对抗性样本（Adversarial Sample）：

对输入样本故意添加一些人无法察觉的细微的干扰，导致模型以高置信度给出一个错误的输出。

误差放大效应（error amplification effect）：

由于神经网络的结构复杂，而且会经过多次叠加，即使很小扰动，累加起来也很可观。



