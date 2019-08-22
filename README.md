# Network_Embedding :dart:
#### For generating MetaPAths with M-D-G-D-M patterns. This file could be ran on windows with cmd or powershell.
> - where M means miRNA, D means Disease, G means Gene.
#### 网络结构图绘制
> - [NN SVG](http://alexlenail.me/NN-SVG/index.html)
## DBSI
> - disease based similarity inference
## Link2Vec
> - Start Now !
#### Tips
> - 网络节点编码时，应当处理为不同的id。比如: disease节点从0开始编码，miRNA也从0开始编码。这样做可能不够合理，因为所利用的嵌入算法是对网络节点id进行随机游走，从而形成语料库。若出现不同类型的节点拥有相同的id，可能会造成训练词向量时的误差。
