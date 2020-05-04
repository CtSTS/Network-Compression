# Network-Compression
## pytorch卷积神经网络流程压缩框架-本科生毕业论文

## 使用技术清单：
<br>1.Network Slimming
<br>2.DDPG
<br>3.Knowledge Distillation
<br>4.K-means

## 主要包含文件说明：
<br>1.main.py     -个人在毕设中使用的一个命令行入口，对于输入参数进行明确定义，通过该函数可以实现端到端的网络训练、剪枝、再训练、压缩过程
<br>2.model.py    -用于定义可以作为框架使用的网络结构，目前包含VGG\ResNet\MobileNet三种
<br>3.train.py    -定义网络的训练（包括正常训练、加入通道稀疏项训练、知识蒸馏训练）、测试部分代码
<br>4.pruning_action_list_env.py  -用于定义DDPG算法进行剪枝率搜索时用到的搜索空间
<br>5.DDPG-agent.py -定义DDPG算法的核心内容，包括Actor、Critric网络的训练、动作的生成等
<br>6.DDPG-memory.py-定义DDPG算法使用的存储空间函数
<br>7.pruning.py  -定义真正对网络进行剪枝的过程，包括整体剪枝与分层剪枝两种
<br>8.kmeans.py   -定义后端压缩使用的kmeans算法，同时包括后端压缩恢复的算法
<br>9.utils.py    -定义一些网络参数测量函数、参数类型转换函数等
<br>10.GradCAMplus.py -定义对网络attention map进行绘制的函数类
