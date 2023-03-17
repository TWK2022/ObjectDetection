## pytorch图片分类训练代码
>!目前只实现了基本训练和预测功能!  
>代码兼容性较强，使用的是一些基本的库、基础的函数  
>在argparse中可以选择使用wandb，能在wandb网站中生成可视化的训练过程  
>输入模型的图片要为固定大小、RGB通道(如batch,640,640,3)，图片四周的填充值为(127,127,127)  
>归一化、减均值、除以方差、维度变换会在模型中会自动完成  
>计算指标使用的是nms前/后的输出框，两种计算方式  
>github链接:https://github.com/TWK2022/ObjectDetection
### 数据格式如下  
>(标准YOLO格式)  
>├── 数据集路径：data_path  
>&emsp; &emsp; └── image：存放所有图片  
>&emsp; &emsp; └── label：存放所有图片的标签，名称:图片名.txt，内容:类别号 x_center y_center w h(x,y,w,h为相对图片的比例值)  
>&emsp; &emsp; └── train.txt：训练图片的绝对路径(或相对data_path下路径)  
>&emsp; &emsp; └── val.txt：验证图片的绝对路径(或相对data_path下路径)  
>&emsp; &emsp; └── class.txt：所有的类别名称  
### 1，run.py
>模型训练时运行该文件，argparse中有对每个参数的说明
### 2，test_pt.py
>使用训练好的pt模型预测
### 3，export_onnx.py
>将pt模型导出为onnx模型