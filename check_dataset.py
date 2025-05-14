import os
import cv2
import numpy as np

# -------------------------------------------------------------------------------------------------------------------- #
# 检查数据集
# -------------------------------------------------------------------------------------------------------------------- #
dataset_path = 'dataset'


# -------------------------------------------------------------------------------------------------------------------- #
class check_dataset_class:
    def __init__(self):
        image_dir = f'{dataset_path}/image'
        label_dir = f'{dataset_path}/label'
        self.image_path = [f'{image_dir}/{_}' for _ in os.listdir(image_dir)]
        self.label_path = [f'{label_dir}/{os.path.splitext(os.path.basename(_))[0]}.txt' for _ in self.image_path]

    def check_dataset(self):
        self.draw_image()

    def draw_image(self):
        index = 0
        image = cv2.imdecode(np.fromfile(self.image_path[index], dtype=np.uint8), cv2.IMREAD_COLOR)
        with open(self.label_path[index], 'r', encoding='utf-8') as f:
            line_all = [list(map(float, _.strip().split())) for _ in f.readlines()]
        image_h, image_w, _ = image.shape
        for line in line_all:
            classification, x, y, w, h = line
            point1 = (int((x - w / 2) * image_w), int((y - h / 2) * image_h))
            point2 = (int((x + w / 2) * image_w), int((y + h / 2) * image_h))
            image = cv2.rectangle(image, point1, point2, color=(0, 255, 0), thickness=2)
            image = cv2.putText(image, str(int(classification)), point1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite('check_dataset.jpg', image)


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = check_dataset_class()
    model.check_dataset()
