# text-summarization
text-summarization

以新闻数据为例，实践文本摘要的框架思路

数据见data文件夹下

训练过程中打印以下两段文本的摘要预测，作为模型输出的参考

```
sentence1 = '新华社受权于18日全文播发修改后的《中华人民共和国立法法》，修改后的立法法分为“总则”“法律”“行政法规”“地方性法规、自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。'
summary1 = '修改后的立法法全文公布'
sentence2 = '一辆小轿车，一名女司机，竟造成9死24伤。日前，深圳市交警局对事故进行通报：从目前证据看，事故系司机超速行驶且操作不当导致。目前24名伤员已有6名治愈出院，其余正接受治疗，预计事故赔偿费或超一千万元。'
summary2 = '深圳机场9死24伤续：司机全责赔偿或超千万'
```



实验记录

1.1

​	首先是简单的seq2seq，

​	encoder为双层双向GRU

​	decoder为双层单向GRU

​	解码加入attention，采用transformer的Multi-head Attention机制

​	推理阶段采用topk=3的beam search。

​	训练效果：  （Acc没太大意义，于是就马赛克了）

​	![image-20200610234239555](C:\Users\daish\AppData\Roaming\Typora\typora-user-images\image-20200610234239555.png)

train_loss 持续降低到 了3.6388，	

val_loss在4.5附近就降不动了，最低为4.5011，此时的预测摘要为：

![image-20200610234542823](C:\Users\daish\AppData\Roaming\Typora\typora-user-images\image-20200610234542823.png)	

长度上并没有填满max_length 32，但生成的摘要与原文关联较小，重复字较多



1.2

​	增加layer normalization，希望可以加速收敛

![image-20200611094714624](C:\Users\daish\AppData\Roaming\Typora\typora-user-images\image-20200611094714624.png)

train_loss 持续降低到 了1.8386，	

val_loss在4.0附近就降不动了，最低为3.8669，此时的预测摘要为：

![image-20200611112420572](C:\Users\daish\AppData\Roaming\Typora\typora-user-images\image-20200611112420572.png)

loss有大幅下降，生成的内容个别字词与原文有了一定相关性，仍然无法有效代表原文，且存在重复字

1.3

​	加入源文本的词概率分布，希望可以使得摘要尽量多考虑原文中的字

![image-20200613091803898](C:\Users\daish\AppData\Roaming\Typora\typora-user-images\image-20200613091803898.png)

这里明显较少epoch就到达低点了是因为换了显卡，batch_size放大了，由128提升到512了

train_loss 持续降低到 了2.8425，	

val_loss在3.8附近就降不动了，最低为3.8159，此时的预测摘要为：

![image-20200613091920214](C:\Users\daish\AppData\Roaming\Typora\typora-user-images\image-20200613091920214.png)

loss有一定下降，生成的内容与原文相关性有一定，更多字是从原文中来，但仍然无法有效代表原文，初步猜测主要是由于Encoder，双向GRU的mask处理不够好。可以考虑优化GRU的训练过程，每一步都精准mask，或者换transformer采用self-attention。

1.4

​	合并encoder，decoder的embedding层，给encoder的双向GRU的reverse加入mask，更精准



1.4

​	加入coverage，降低已经生成过的字的权重，避免出现大量的重复字



1.5

​	将encoder替换成transformer的encoder，即引入self-attention

