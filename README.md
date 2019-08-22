# 链接预测
> - 基于规则
> - 网络嵌入
## 基于规则的链接预测常规方法
> - DBSI
> - MBSI
### DBSI
> - disease based similarity inference
```python
def DBSI(disease_similaritity, data_train):
    """
    :param disease_similaritity:        disease相似度矩阵
    :param data_train:                  miRNA-disease邻接矩阵
    :return:                            疾病-miRNA预测分值——【相似的疾病可能存在相同的连接】
    """
    DBSI_result = np.ones((data_train.shape[1], data_train.shape[0]))*0
    for i in range(data_train.shape[1]):
        for j in range(data_train.shape[0]):
            val = 0.00
            sum_up = 0.00
            sum_low = 0.00
            for ix in range(data_train.shape[1]):
                if ix != i:
                    sum_up += (disease_similaritity[i, ix] * data_train[j, ix])
                    sum_low += (disease_similaritity[i, ix])
            val = sum_up/sum_low
            DBSI_result[i, j] = val
    return DBSI_result
```
## Network_Embedding :dart:
#### For generating MetaPAths with M-D-G-D-M patterns. This file could be ran on windows with cmd or powershell.
> - where M means miRNA, D means Disease, G means Gene.
#### 网络结构图绘制
> - [NN SVG](http://alexlenail.me/NN-SVG/index.html)
#### Link2Vec
> - Start Now !
#### Tips
> - 网络节点编码时，应当处理为不同的id。比如: disease节点从0开始编码，miRNA也从0开始编码。这样做可能不够合理，因为所利用的嵌入算法是对网络节点id进行随机游走，从而形成语料库。若出现不同类型的节点拥有相同的id，可能会造成训练词向量时的误差。
