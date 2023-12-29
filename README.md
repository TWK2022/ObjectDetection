## pytorch目标检测训练框架
>代码兼容性较强，使用的是一些基本的库、基础的函数  
>在argparse中可以选择使用wandb，能在wandb网站中生成可视化的训练过程  
>测试时输入模型的图片会填充为固定大小、RGB通道(如batch,640,640,3)，图片四周的填充值为(128,128,128)  
### 1，环境
>torch：https://pytorch.org/get-started/previous-versions/
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
>模型训练时运行该文件，argparse中有对每个参数的说明
### 4，test_pt.py
>使用训练好的pt模型预测
### 5，export_onnx.py
>将pt模型导出为onnx模型
### 6，flask_start.py
>用flask将程序包装成一个服务，并在服务器上启动
### 7，flask_request.py
>以post请求传输数据调用服务
### 8，gunicorn_config.py
>用gunicorn多进程启动flask服务：gunicorn -c gunicorn_config.py flask_start:app
### 其他
>github链接：https://github.com/TWK2022/ObjectDetection  
>学习笔记：https://github.com/TWK2022/notebook  
>邮箱：1024565378@qq.com  