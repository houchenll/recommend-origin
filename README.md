# recommend

## 概述
推荐系统是信息过滤系统的一个分支，可以自动地挖掘用户和物品之间的联系，旨在为 user (用户) 推荐一个（或一系列）未观测的 item (物品，电影，新闻等)。  
推荐系统中有用户集合U和物品集合I，系统中的数据包括用户和物品的交互记录、用户信息、物品信息，这些数据统称为场景信息C(Context)。其他信息还有社会关系、知识图谱等。推荐系统数据结构非常适合转化为图结构，并且GNN在图数据的表示学习上具有非常强大的学习能力。  
推荐系统基于用户本身的多维度属性数据（如年龄、地域、性别等）以及行为数据的反馈（如点击、收藏、点赞、购买等），结合物品自身属性数据（如标题、标签、类别、正文等），以预测用户对待推荐物品的评分或偏好。  
任务：在推荐系统中建立一个推荐模型f(U,I,C)，预测用户对所有物品的兴趣评分，再根据兴趣评分对物品进行排序，选取前K个推荐物品构成推荐列表。  
分为一般的推荐和序列化推荐。一般的推荐视用户为静态的，序列化推荐认为用户偏好随时间动态变化。  

### 诞生背景 & 解决的痛点
1、信息过载，互联网信息爆炸式增长，用户无法轻松获取感兴趣的优质内容  
2、长尾问题，少部分服务或商品受到了大多数流量的关注，而很大一部分服务或商品面临无人问津的状况  

### 价值
用户角度：基于用户个人的兴趣偏好进行千人千面的自动推荐，帮助用户挑选内容，智能推荐，节省搜索的时间，缓解信息过载问题。  
物品角度：其自身属性及对应的交互行为差异，通过各种推荐方式是可以触达到对其更感兴趣的用户群体中，将较少用户关注的物品推荐给合适的用户，使得更多优质内容可以提供给用户，缓解了曝光不足带来的长尾问题。  
企业角度：带来了更好的产品交互方式，达到了沉浸式体验的效果，从而进一步提升了用户的黏性，并最终大幅度提升了转化收益。  

### 任务
Top-K：给出每个用户可能感兴趣的K个物品的推荐列表


## 经典推荐算法

### 1.1 协同过滤，Collaborative Filtering, CF
基于用户和物品历史交互数据进行推荐，利用由用户和物品的交互记录所构成的评分矩阵，来预测用户对没有交互过的物品的喜好程度。在工业界广泛应用。  
思想：在行为上类似的用户会喜欢类似的物品，或类似的物品适合推荐给同一类用户。  
任务：设计用户之间或物品之间的类似程度的度量方式  
优势：不用考虑用户和物品特征，仅基于用户与物品历史交互信息就可以推荐，无需领域知识，简单有效。可应用在广泛场景，适用性强。  
缺陷：数据稀疏、冷启动、可解释性差。数据稀疏：在系统启动初期，用户的历史交互记录稀疏时，效果欠佳。冷启动：当一个新物品加入系统时，往往无法将该物品推荐出去。  
优化方案：使用辅助信息缓解问题，但传统方法没有足够的容量来复用这些知识，传统的推荐技术没有复用语义信息、关键字信息和层次结构知识，同样会导致模型性能的下降。  
分类：基于用户、基于物品、基于模型  

#### 1.1.1 基于用户的协同过滤
思想：人以群分，爱好相似的人喜欢相似的物品。考虑用户之间交互行为的相似程度，往往浏览过的物品集合越相似的用户之间的相似度越高，给用户推荐与其相似的用户喜欢的物品。  
挑战：如何合理计算用户之间的相似度  

#### 1.1.2 基于物品的协同过滤
思想：利用交互矩阵来计算物品与物品的相似度，推荐与用户喜欢的物品相似的物品。  
挑战：如何合理得到物品之间的类似程度  

#### 1.1.3 基于模型的协同过滤
思想：将用户和物品投影到低维向量上，用向量来表示用户和物品，设计交互计算方式来计算用户对目标物品的偏好度，模型计算出的分数越高，代表用户更有可能与该物品交互  

### 1.2 基于内容，Content-Based Filtering, CBF
思想：根据物品内容之间的类似程度计算，给用户推荐与其交互过的物品在内容上更为类似的物  
缺陷：信息茧房：总是推荐给用户内容上类似的物品，存在推荐结果新颖性差的问题。  

**文章：**
- [基于物品的协同过滤和基于内容的推荐有什么区别][1]

### 矩阵分解 / 分解因子机

### 深度学习
利用深度神经网络逐层非线性处理信息的优势，可以很好的提取出数据的特征。  
深度学习一方面可以帮助提取物品的图片和文字之类的信息特征，可以用对应的深度学习技术来获取这些信息的深层特征  
另外深度学习也可以对用户和物品特征之间的交互进行建模，提取这些特征交互的深层次信息。  
深度学习可以对用户和物品之间的复杂交互模式进行模拟，另外深度学习可以帮助获得一些文本或图片的特征向量。  
基于深度学习的推荐算法可以从用户和物品的id以及一些辅助信息中提取出特征，最后预测用户对目标物品的偏好程度。  
思路：输入用户与物品的交互数据以及用户和物品的属性信息，通过神经网络学习用户与物品间复杂的交互模式，得到用户和物品的特征向量，再对特征向量进行交叉组合，得到最终的评分预测值。  

### 基于知识图谱
知识图谱嵌入（Knowledge Graph Embedding, KGE）可以解决知识图谱作为复杂图结构难以直接利用的问题，KGE可以学习KG中实体关系的低维向量表示。  

#### 基于嵌入的方法
利用知识图谱来增强实体的表示，并用来增强推荐算法中物品的表示  
但存在知识图谱嵌入学习的目标和推荐算法的目标不完全一致，导致影响推荐性能的问题  

#### 基于路径的方法
利用知识图谱中的路径模式来指导推荐算法  
需要人工设计路径模式，需要一定的领域知识，并且一个领域的路径不能直接迁移到其他领域  

#### 基于传播
通过聚合实体在知识图谱中的多跳邻居来获取实体的更好的表示，然后利用用户和物品的表示来预测用户对物品的偏好程度  
问题：对用户的历史记录在知识图谱中进行传播时，一方面没有对物品粒度的信息进行挖掘，另一方面没有区分用户的历史记录中的每个物品的重要程序；其次现有的基于知识图谱的传播没有考虑到用户和物品实体集合内部实体之间的交互，即没能捕捉到集合内部的实体之间的相关性信息，另外也没有考虑用户实体集合和物品实体集合之间的交互信息  

### 图神经网络
基于知识图谱的推荐算法大都基于图神经网络(GNN)构建。  
利用GNN在知识图谱上进行图嵌入，从而得到节点的嵌入向量，再将该向量作为节点的特征表示来预测用户对物品的评分，最后完成推荐任务。  
现有算法通过为目标节点的每个邻居节点设置不同的注意力系数，来区分不同邻居节点传递到目标节点的信息的重要程度。  

### 现在算法问题：
1、现有算法没有很好地利用图谱中物品和用户交互的时序信息。而在实际场景中，用户的兴趣偏好是随时间变化的。用户的某些长期偏好可以从用户的所有历史交互行为得到，而用户近期偏好与其近期的行为高度相关。  
2、现在算法预测过程中，直接计算向量内积，将该内积作为预测评分。这种方法难以获取到更多的组合特征和非线性特征。  

**经典算法：**
- KGCN: `1 Knowledge Graph Convolutional Networks for Recommender Systems.pdf`
- KGAT: `2 KGAT- Knowledge Graph Attention Network for Recommendation`
- AKGE: `3 Hierarchical Attentive Knowledge Graph Embedding for Personalized Recommendation`
- RippleNet: `4 RippleNet- Propagating User Preferences on the Knowledge Graph for Recommender Systems`, https://github.com/hwwang55/RippleNet, https://github.com/qibinc/RippleNet-PyTorch
- KDD: `5 Meta-Graph Based Recommendation Fusion over Heterogeneous Information Networks`

**文章：**
- [知识图谱嵌入(KGE)主流模型简介][2]
- [论文(KGCN)：知识图谱 + 图卷积神经网络的推荐系统][3]
- [论文(AKGE)：基于注意力知识图谱嵌入的个性化推荐系统][6]
- [论文(KGAT)：融合知识图谱的 CKG 表示 + 图注意力机制的推荐系统][7]
- [论文(KDD)：港科大 KDD 2017 录用论文作者详解：基于异构信息网络元结构融合的推荐系统][4]
- [论文(RippleNet)：RippleNet : 知识图谱+用户偏好传播的推荐系统][5]

**github:**
- [xiangwang1223/knowledge_graph_attention_network][8]

### 其他推荐算法

#### 基于人口统计过滤，Demographic Filtering, DF
思想：基于具有某些共同个人属性（性别、年龄、国家、地区）的用户具有共同偏好进行推荐，根据人口统计属性对用户进行分类从而生成相应推荐的结果  
优点：不需要用户对基于内容和协同过滤方法所必需的物品进行评分或交互操作；当项目信息量有限时，这些方法特别有用。  
缺点：由于涉及安全和隐私问题，收集用户完整信息不切实际且不合法规。强制系统向相关群体的用户推荐相同的商品。  

### 上下文感知，Context Aware-Based Filtering, CABF
常用上下文信息有：时间、地点、几何位置信息、或其他相关人（朋友、女朋友/男朋友、亲戚、同事）的交互信息  

### 辅助信息
属性信息、知识图谱、用户评论  
地点信息用来做兴趣点的推荐  
知识图谱可以表达实体与实体间的联系  
用户对物品的评论中包含了用户对物品的兴趣  
挑战：合适的提取辅助信息的特征  

##  相似度计算

### 内积


## 资料

### github
- [推荐系统的技术栈][9]

### 教程
- [推荐系统入门教程][10]

### blog
1. [知识图谱增强下的智能推荐系统与应用-于敬](https://www.jiqizhixin.com/articles/2022-11-15)
2. [一文概览知识图谱在推荐系统的发展现状](https://zhuanlan.zhihu.com/p/160799232)
3. [基于知识图谱和Transformer的专利推荐方法](https://www.patentguru.com/cn/CN110737778A)
4. [融合知识图谱与注意力机制的推荐算法](http://cea.ceaj.org/CN/10.3778/j.issn.1002-8331.2109-0126)
5. [github - awesome-knowledge-graph](https://github.com/husthuke/awesome-knowledge-graph)
6. [如何将知识图谱引入推荐系统](https://www.infoq.cn/article/xbfzm40gevl1fiyd3dsu)
7. [知识图谱的推荐系统综述-常亮](http://html.rhhz.net/tis/html/201805001.htm)
8. [一种基于知识图谱与内容的推荐算法 - 张雅歌](https://image.hanspub.org/Html/18-1542448_49521.htm)
9. [Movie Recommendations powered by Knowledge Graphs and Neo4j](https://towardsdatascience.com/movie-recommendations-powered-by-knowledge-graphs-and-neo4j-33603a212ad0)
10. [Hands on Explainable Recommender Systems with Knowledge Graphs](https://explainablerecsys.github.io/recsys2022/)
11. [Knowledge Graph with Job Recommendation](https://blogs.sap.com/2022/11/21/knowledge-graph-with-job-recommendation/)
12. [图谱推荐综述：Knowledge Graph-Based Recommender Systems](https://zhuanlan.zhihu.com/p/157636795)
13. [Power recommendation and search using an IMDb knowledge graph](https://aws.amazon.com/cn/blogs/machine-learning/part-1-power-recommendation-and-search-using-an-imdb-knowledge-graph/)

### video
[Graph Representation Learning From Knowledge Graphs to Recommender Systems / Hongwei Wang (UIUC)](https://www.youtube.com/watch?v=fC8HfepCDgE)

### 论文
* [Reading List on RecSys](https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/readingList.md)
* [Paperlist-for-Recommender-Systems](https://github.com/mengfeizhang820/Paperlist-for-Recommender-Systems)
* [awesome-graph-self-supervised-learning-based-recommendation](https://github.com/juyongjiang/awesome-graph-self-supervised-learning-based-recommendation)

#### 综述
1. A Survey on Knowledge Graph-Based Recommender Systems  
   该综述对基于知识图谱的推荐系统进行了全面的概述和分类，包括数据集、评估指标、方法和应用等方面，并对各个方面的研究进行了详细的介绍和分析。  
2. Knowledge Graph-Based Recommender Systems: A Comprehensive Survey  
   该综述对基于知识图谱的推荐系统的研究进展进行了全面的介绍，包括知识图谱构建、用户兴趣模型、推荐算法、评估指标等方面，并从数据集、技术和应用三个方面分别进行了分析。  
3. Knowledge Graph and Recommendation: A Survey and Future Perspectives  
   该综述对知识图谱和推荐系统的研究进展进行了综合的介绍，包括知识图谱建模、用户兴趣模型、推荐算法、实现方法和应用等方面，并对未来研究的方向进行了展望。  
4. A Survey on Knowledge Graph Embedding: Approaches, Applications and Benchmarks  
   该综述主要介绍了知识图谱嵌入技术，包括知识图谱表示学习的背景、嵌入方法、应用和评估指标等方面，并对该领域未来的研究方向进行了展望。  
5. A Comprehensive Survey of Graph Embedding: Problems, Techniques, and Applications  
   该综述主要介绍了图嵌入技术，包括嵌入方法、应用和评估指标等方面，并对未来的研究方向进行了展望。  

[1]: https://www.zhihu.com/question/19971859
[2]: https://zhuanlan.zhihu.com/p/114470141
[3]: https://zhuanlan.zhihu.com/p/364005929
[4]: https://www.leiphone.com/category/ai/kefvQPk1FMybCleb.html
[5]: https://blog.csdn.net/m0_46522688/article/details/113758212
[6]: https://zhuanlan.zhihu.com/p/364006795
[7]: https://zhuanlan.zhihu.com/p/364002920
[8]: https://github.com/xiangwang1223/knowledge_graph_attention_network
[9]: https://github.com/datawhalechina/fun-rec/blob/master/docs/ch01/ch1.3.md
[10]: https://datawhalechina.github.io/fun-rec/#/