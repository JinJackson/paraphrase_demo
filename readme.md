# Paraphrase Identification Demo

本Demo基于BERT搭建起一个简单的匹配模型，输入两个文本，输出结果是否互为复述。

目录结构：

+ /Model
  + MatchModel.py 模型文件，包含了输入输出的运算过程（其中包含了其他预训练模型的匹配模型，本demo只使用了BERT，其余可以忽略）
+ /utils
  + 很多乱七八糟的文件，基本只有评测指标有用，其他都没啥用
+ all_dataset.py  读入数据文件，将数据转换成pytorch的dataset格式
+ parser1.py  参数设置文件，可以通过命令行传入
+ Train_baseline.py  训练代码，通过train_lcqmc.sh调用
+ train_lcqmc.sh  启动训练脚本（入口，传入参数，调整参数）
+ requirements.txt  运行环境所需包  （pip3 install -r requirements.txt）
+ readme.md  本文件，说明

