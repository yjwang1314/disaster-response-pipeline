# 灾难应对管道项目

### 项目概览
在本项目中，你将应用数据工程技术分析来自 Figure8 的灾害消息，构建一个分类灾害消息的模型并用于 APP。

项目使用一个包含灾害发生时发出的真实消息的数据集，创建一个机器学习管道对这些事件进行分类，使得以将消息发送到合适的灾害应对机构。

此项目会分为一下三部分：

1. ETL管道（数据清洗管道）：
	- 加载 messages 和 categories 数据集
	- 将两个数据集进行合并 (merge)
	- 清洗数据
	- 将其存储到 SQLite 数据库中
	
2. 机器学习管道：
	- 从 SQLite 数据库中加载数据
	- 将数据集分成训练和测试集
	- 搭建文本处理和机器学习管道
	- 使用 GridSearchCV 对模型进行训练和微调
	- 输出测试集的结果
	- 将最终的模型输出为 pickle 文件
	
3. Flask网络应用程序：
	- 实时展示模型分类结果

### 如何运行：
1. 在项目的跟目录运行以下命令建立你的数据库和模型。
	- 运行ETL管道，清洗数据并保存到数据库
		'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
	- 运行机器学习管道训练模型并保存到pkl文件
		'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
		
2. 在app目录下运行以下命令运行网络应用程序
	'python run.py'
	
3. 网络应用程序地址在http://0.0.0.0:3001/