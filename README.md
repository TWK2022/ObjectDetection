## pytorch目标检测训练框架
### 1，环境
>torch: https://pytorch.org/get-started/previous-versions/
>```
>pip install tqdm wandb opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
>```
### 2，数据格式
>(标准YOLO格式)  
>├── 数据集路径：data_path  
>&emsp; &emsp; └── image：存放所有图片  
>&emsp; &emsp; └── label：存放所有图片的标签，名称:图片名.txt，内容:类别号 x_center y_center w h(x,y,w,h为相对图片的比例值)  
>&emsp; &emsp; └── train.txt：训练图片的绝对路径(或相对data_path下路径)  
>&emsp; &emsp; └── val.txt：验证图片的绝对路径(或相对data_path下路径)  
>&emsp; &emsp; └── class.txt：所有的类别名称  
### 3，run.py
>模型训练，argparse中有每个参数的说明
### 4，predict.py
>模型预测
### 5，export_onnx.py
>onnx模型导出
### 6，predict_onnx.py
>onnx模型预测
### 其他
>github链接: https://github.com/TWK2022/ObjectDetection  
>学习笔记: https://github.com/TWK2022/notebook  
>邮箱: 1024565378@qq.com
