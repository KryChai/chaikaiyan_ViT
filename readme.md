 # SETR-应用于冠状血管分割的模型
 
> 2024年北京邮电大学计算机学院9组夏令营 柴铠琰

参考CVPR论文 [SETR](https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_Rethinking_Semantic_Segmentation_From_a_Sequence-to-Sequence_Perspective_With_Transformers_CVPR_2021_paper.html)。

数据集采用XCAD心脏血管造影数据集。

模型训练在autoDL算力云平台的英伟达RTX4090进行训练

## 目录结构
````
|-- ViT
    |-- readme.md
    |-- Model
    |   |-- Dataset.py
    |   |-- SETR_Model.py
    |   |-- Trian.py
    |   |-- __init__.py
    |   |-- d2l
    |   |   |-- mxnet.py
    |   |   |-- tensorflow.py
    |   |   |-- torch.py
    |   |   |-- __init__.py
    |   |-- d2l-0.15.1
    |   |   |-- d2l-0.15.1
    |   |       |-- PKG-INFO
    |   |       |-- README.md
    |   |       |-- setup.cfg
    |   |       |-- setup.py
    |   |       |-- d2l.egg-info
    |   |           |-- dependency_links.txt
    |   |           |-- PKG-INFO
    |   |           |-- requires.txt
    |   |           |-- SOURCES.txt
    |   |           |-- top_level.txt
    |   |           |-- zip-safe
    |   |-- __pycache__
    |-- Result
    |-- ViT
    |   |-- Result
    |       |-- Model_Struct_See.py
    |       |-- test.pth
    |       |-- savefile
    |           |-- test.xlsx
    |-- XCAD
        |-- test
        |   |-- images
        |   |-- masks
        |-- train
            |-- images
            |-- masks
````

## 项目文件说明
### Model文件夹
在Model文件夹下\
Dataset.py导入数据\
SETR_Model.py定义了模型结构，其中包括ViT和SETR的实现\
Trian.py定义了模型训练过程，可以在此修改训练参数

### ViT文件夹
在Result \ savefile文件夹下的.xlsx文件保存了训练的模型评估参数\
Model_Struct_See.py文件用于可视化模型结构\
test.pth文件存储了模型的最优参数，可用于新数据的预测

# chaikaiyan_ViT
