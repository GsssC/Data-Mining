天池比赛文档整理
===========================
方便查阅资料

竞赛相关
-------
-  [盐城汽车上牌量预测](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.1fefe14csBETgs&raceId=231641&_is_login_redirect=true&accounttraceid=c1ad14c1-3d7b-46e5-a340-fe100ce4459e)

-  [成绩](https://tianchi.aliyun.com/competition/myScore.htm?spm=5176.11165268.5678.7.2fd0feebVCSzhk&raceId=231641)

论坛例子
-------
-  [寒武纪の盐城车牌预测数据初探 之 三](https://tianchi.aliyun.com/forum/new_articleDetail.html?spm=5176.8366600.0.0.71271025sF3zRk&raceId=231641&postsId=3809)
-  [盐城汽车上牌量预测Baseline](https://tianchi.aliyun.com/forum/new_articleDetail.html?spm=5176.8366600.0.0.3a3de22fpzg09j&raceId=231641&postsId=3918)

模型列表
-------
-  [Boosting和梯度Boosting](http://blog.csdn.net/yueyedeai/article/details/15205165)
-  [梯度提升树GBDT(Gradient Boosting Decision Tree)](http://blog.csdn.net/a819825294/article/details/51188740) 
-  [XGBoost(eXtreme Gradient Boosting)](http://blog.csdn.net/totoro1745/article/details/53328725?utm_source=itdadao&utm_medium=referral)
-  [lightGBM(Gradient Boosting Machine)](http://blog.csdn.net/niaolianjiulin/article/details/76584785)

__`Boosting`是一种思想，意思是用一些弱分类器的组合来构造一个强分类器。__ 对于分类问题而言，给定一个训练样本集，求比较粗糙的分类规则（弱分类器）要比求精确的分类规则（强分类器）容易得多，`Boosting`思想就是从弱学习算法出发，反复学习，得到一系列弱分类器，然后组合这些弱分类器--（《统计学习方法》第8章）。和这个理念相对应的是一次性构造一个强分类器，像支持向量机，逻辑回归等。通常，我们通过相加来组合这些弱分类器，形式如下:

`F=a0·f0+a1·f1+...+an·fn`

这里F为最终构造出来的强分类器，fn为各个弱分类器，an为这些分类器所占权重。
由此，有一个问题需要回答：

__每一轮训练如何改变训练数据的权值和概率分布。__ 解决这个问题的方法叫做`AdaBoosting`,《统计学习方法》中含有许多数学推导，没细看，占坑。

将`AdaBoosting`方法和`决策树`结合便构成了提升树。当损失函数是平方损失和指数损失时，此方法的每一步优化都十分简单，但对于一般的损失函数而言，往往每一步优化并不那么容易。 

__针对这个问题，Freidman提出了梯度提升`GBDT`，利用最速下降法的近似方法，即损失函数的负梯度在当前模型的值，作为回归问题提升树中残差的近似值，拟合一个回归树。__ 有许多数学推导，没细看，占坑。

__`XGBoost`和`lightGBM`是上述算法的实现，表现为可以被Python调用的包。__ 如何使用还没细看，占坑。

python 模块API
-------
各个模块API的PDF文档下载见github库内文件
-  Numpy
-  Pandas
-  Matplotlib
-  XGBoost
-  LightGBM


