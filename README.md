# Siamese-Network-Image-Recognition
本算法使用基于孪生网络 (Siamese Network) 的模型，在训练阶段, 对具有两组相同卷积神经网络的孪生网络进行训练, 本质为训练一个二分类器, 若样本标签为 1, 则说明两张图像属于同一类; 若标签为 0 则相反, 以此来学习两张输入图像彼此的相似度. 在测试阶段, 该模型会将待测试图像分别与任务中训练集里每个类别的图像单独输入孪生网络中, 并判断待测试图像为相似度最高的输出结果所对应的类别。那么在本实验的模型中，两组相同的卷积神经网络结构为两层结构——第一层，将三层卷积层改进为六层卷积，第二层使用三层线性层。

模型中有两个姊妹网络，它们是相同的神经网络，具有完全相同的权重。映像对中的每个映像都馈送到这些网络之一。选择使用对比损失函数优化网络，而 siamese 架构的目标不是对输入图像进行分类，而是区分它们。因此，分类损失函数（如交叉熵）不是最佳拟合。相反，此体系结构更适合使用对比函数。直观地说，这个函数只是评估网络区分给定图像对的程度。
对比损失函数如下：

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7baa6003-b6c7-47eb-b748-8bf65b483031/Untitled.png)

其中 Dw 定义为姊妹暹罗网络输出之间的欧氏距离，，X1和X2是输入数据对，Y值为1或0，如果模型预测输入是相似的，那么Y的值为0，否则Y为1，max（）是表示0和m-Dw之间较大值的函数，m是大于0的边际价值（margin value），有一个边际价值表示超出该边际价值的不同对不会造成损失。从数学上讲，欧氏距离为：

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4cfe14ee-5197-4e56-bc57-3d4ec9c2ed8f/Untitled.png)

其中 Gw 是其中一个姊妹网络的输出。X1 和 X2 是输入数据对。

数据集的构成包括 Taining Set ,Support Set 和 Query 的，其中 Taining Set 选择了 CASIA 中 40 个人的 8603 张人脸图片用于训练，Support Set 选择了五位明星的 5 张照片，五位分别是关晓彤、杨幂、杨紫、刘昊然、张子枫，Query 选择以上五位明星的 1-2 张小时候照片进行测试。

改进前使用原始数据集，数据集中有只有成人照片，卷积层为三层，训练周期 100，学习效果较差，图像对比结果出现了很大的问题。正确率极低，仅有少部分的结果正确，且图像距离较小。
将周期改为 150，其次，训练集数据集更换为刚才介绍的数据集，并由于缺少各年龄段的特征学习，加入了少量的儿童时期的图像。卷积层改成六层，emmm再加电脑跑不动了，最效果计算的图像距离减小了很多，并且正确概率也得到显著提升。

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b563a4c4-a91f-4c9e-9f00-b2f77b63ae03/Untitled.png)

改进前

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9591757a-8921-4357-a40f-2f7d24f46c97/Untitled.png)

改进后

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c8fd8f49-2a6a-4c4c-a386-232b00e64d51/Untitled.png)
