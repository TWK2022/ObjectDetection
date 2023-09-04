# 使用flask在服务器上创建一个可以调用的服务
import cv2
import json
import flask
import base64
import argparse
import numpy as np
from inference import detection_class

parser = argparse.ArgumentParser('|在服务器上启动flask服务|')
# ocr
parser.add_argument('--det_model', default='PPOCRv3_det', type=str, help='paddle模型文件夹位置，包含.pdmodel、.pdiparams')
parser.add_argument('--cls_model', default='PPOCRv3_cls', type=str, help='paddle模型文件夹位置，包含.pdmodel、.pdiparams')
parser.add_argument('--rec_model', default='PPOCRv3_rec', type=str, help='paddle模型文件夹位置，包含.pdmodel、.pdiparams')
parser.add_argument('--inference', default='trt', type=str)
parser.add_argument('--rec_label', default='rec_label_zhengjian.txt', type=str, help='识别模型的标签')
# classification
parser.add_argument('--model_path', default='classification.onnx', type=str, help='|onnx模型位置|')
parser.add_argument('--input_size', default=480, type=int, help='|模型输入图片大小，要与导出的模型对应|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量，要与导出的模型对应|')
# 设备
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--float16', default=True, type=bool)
args = parser.parse_args()
app = flask.Flask(__name__)  # 创建一个服务框架
model = detection_class(args)  # 初始化检测函数


def image_decode(image_json):
    image_base64 = json.loads(image_json).encode()  # json->base64
    image_byte = base64.b64decode(image_base64)  # base64->字节类型
    array = np.frombuffer(image_byte, dtype=np.uint8)  # 字节类型->一行数组
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)  # 一行数组->BGR图片
    return image


@app.route('/test/', methods=['POST'])  # 每当调用服务时会执行一次flask_app函数
def flask_app():
    image_json = flask.request.get_data()
    image = image_decode(image_json)
    result = model.detection(image)
    return result


if __name__ == '__main__':
    print('| 使用flask启动服务 |')
    app.run(host='0.0.0.0', port=9999, debug=False)  # 启动服务
