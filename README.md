# 中草药识别代码使用说明
## 一、文件结构：
```
competition_frame/
      │
      └── classification/
               ├── data：分类数据集
               │     ├── data1: 数据集
               │     │     ├── images:中草药图片
               │     │     └── train.csv:存放有images相对路径及其labels

               ├── out：结果保存文件
               └── src：脚本文件
                     ├── dataset.py: 数据读取脚本
                     ├── model_utils.py: 模型配置脚本
                     ├── test.py: 测试脚本
                     └── train.py: 训练脚本
     


- ***train.py***：
  - 训练代码

- ***dataset.py***：
  - 数据集预处理代码

- ***model_utils.py***:
  - 模型代码

- ***test.py***:
  - 测试代码
