天池比赛文档整理
===========================
方便查阅资料
竞赛相关
-------
-  `盐城汽车上牌量预测 <https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.1fefe14csBETgs&raceId=231641&_is_login_redirect=true&accounttraceid=c1ad14c1-3d7b-46e5-a340-fe100ce4459e>`__

-  `成绩<https://tianchi.aliyun.com/competition/myScore.htm?spm=5176.11165268.5678.7.2fd0feebVCSzhk&raceId=231641>`

论坛例子
-------
-  `寒武纪の盐城车牌预测数据初探 之 三<https://tianchi.aliyun.com/forum/new_articleDetail.html?spm=5176.8366600.0.0.71271025sF3zRk&raceId=231641&postsId=3809>`

模型列表
-------
-  `梯度提升树GBDT(Gradient Boosting Decision Tree) <http://blog.csdn.net/a819825294>`__
-  `XGBoost(eXtreme Gradient Boosting) <http://blog.csdn.net/totoro1745/article/details/53328725?utm_source=itdadao&utm_medium=referral>`__
-  `lightGBM(Gradient Boosting Machine)<http://blog.csdn.net/niaolianjiulin/article/details/76584785>`__

python 模块API
-------
-  `numpy`
-  `pandas`
-  `matplotlib`




-  LightGBM binary file（LightGBM 二进制文件）

加载后的数据存在 ``Dataset`` 对象中.

**要加载 ligsvm 文本文件或 LightGBM 二进制文件到 Dataset 中:**

.. code:: python

    train_data = lgb.Dataset('train.svm.bin')

**要加载 numpy 数组到 Dataset 中:**

.. code:: python

    data = np.random.rand(500, 10)  # 500 个样本, 每一个包含 10 个特征
    label = np.random.randint(2, size=500)  # 二元目标变量,  0 和 1
    train_data = lgb.Dataset(data, label=label)

**要加载 scpiy.sparse.csr\_matrix 数组到 Dataset 中:**

.. code:: python

    csr = scipy.sparse.csr_matrix((dat, (row, col)))
    train_data = lgb.Dataset(csr)

**保存 Dataset 到 LightGBM 二进制文件将会使得加载更快速:**

.. code:: python

    train_data = lgb.Dataset('train.svm.txt')
    train_data.save_binary('train.bin')

**创建验证数据:**

.. code:: python

    test_data = train_data.create_valid('test.svm')

or

.. code:: python

    test_data = lgb.Dataset('test.svm', reference=train_data)

在 LightGBM 中, 验证数据应该与训练数据一致（格式一致）.

**指定 feature names（特征名称）和 categorical features（分类特征）:**

.. code:: python

    train_data = lgb.Dataset(data, label=label, feature_name=['c1', 'c2', 'c3'], categorical_feature=['c3'])

LightGBM 可以直接使用 categorical features（分类特征）作为 input（输入）.
它不需要被转换成 one-hot coding（毒热编码）, 并且它比 one-hot coding（毒热编码）更快（约快上 8 倍）

**注意**: 在你构造 ``Dataset`` 之前, 你应该将分类特征转换为 ``int`` 类型的值.

**当需要时可以设置权重:**

.. code:: python

    w = np.random.rand(500, )
    train_data = lgb.Dataset(data, label=label, weight=w)

或者

.. code:: python

    train_data = lgb.Dataset(data, label=label)
    w = np.random.rand(500, )
    train_data.set_weight(w)

并且你也可以使用 ``Dataset.set_init_score()`` 来初始化 score（分数）, 以及使用 ``Dataset.set_group()`` ；来设置 group/query 数据以用于 ranking（排序）任务.

**内存的高使用:**

LightGBM 中的 ``Dataset`` 对象由于只需要保存 discrete bins（离散的数据块）, 因此它具有很好的内存效率.
然而, Numpy/Array/Pandas 对象的内存开销较大.
如果你关心你的内存消耗. 您可以根据以下方式来节省内存: 

1. 在构造 ``Dataset`` 时设置 ``free_raw_data=True`` （默认为 ``True``）

2. 在 ``Dataset`` 被构造完之后手动设置 ``raw_data=None`` 

3. 调用 ``gc``

设置参数
------------------

LightGBM 可以使用一个 pairs 的 list 或一个字典来设置 `参数 <./Parameters.rst>`__.
例如:

-  Booster（提升器）参数:

   .. code:: python

       param = {'num_leaves':31, 'num_trees':100, 'objective':'binary'}
       param['metric'] = 'auc'

-  您还可以指定多个 eval 指标:

   .. code:: python

       param['metric'] = ['auc', 'binary_logloss']

训练
--------

训练一个模型时, 需要一个 parameter list（参数列表）和 data set（数据集）:

.. code:: python

    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])

在训练完成后, 可以使用如下方式来存储模型:

.. code:: python

    bst.save_model('model.txt')

训练后的模型也可以转存为 JSON 的格式:

.. code:: python

    json_model = bst.dump_model()

以保存模型也可以使用如下的方式来加载.

.. code:: python

    bst = lgb.Booster(model_file='model.txt')  #init model

交叉验证
--------

使用 5-折 方式的交叉验证来进行训练（4 个训练集, 1 个测试集）:

.. code:: python

    num_round = 10
    lgb.cv(param, train_data, num_round, nfold=5)

提前停止
--------------

如果您有一个验证集, 你可以使用提前停止找到最佳数量的 boosting rounds（梯度次数）.
提前停止需要在 ``valid_sets`` 中至少有一个集合.
如果有多个，它们都会被使用:

.. code:: python

    bst = lgb.train(param, train_data, num_round, valid_sets=valid_sets, early_stopping_rounds=10)
    bst.save_model('model.txt', num_iteration=bst.best_iteration)

该模型将开始训练, 直到验证得分停止提高为止.
验证错误需要至少每个 `early_stopping_rounds` 减少以继续训练.

如果提前停止, 模型将有 1 个额外的字段: `bst.best_iteration`.
请注意 `train()` 将从最后一次迭代中返回一个模型, 而不是最好的一个.

This works with both metrics to minimize (L2, log loss, etc.) and to maximize (NDCG, AUC).
Note that if you specify more than one evaluation metric, all of them will be used for early stopping.

这与两个度量标准一起使用以达到最小化（L2, 对数损失, 等等）和最大化（NDCG, AUC）.
请注意, 如果您指定多个评估指标, 则它们都会用于提前停止.

预测
----------

已经训练或加载的模型都可以对数据集进行预测:

.. code:: python

    # 7 个样本, 每一个包含 10 个特征
    data = np.random.rand(7, 10)
    ypred = bst.predict(data)

如果在训练过程中启用了提前停止, 可以用 `bst.best_iteration` 从最佳迭代中获得预测结果:

.. code:: python

    ypred = bst.predict(data, num_iteration=bst.best_iteration)

.. _Python-package: https://github.com/Microsoft/LightGBM/tree/master/python-package
