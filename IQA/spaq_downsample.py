import os
import torchvision.transforms as T
from PIL import Image

SPAQ = "/media/hdd1/hzh/iqa-dataset/SPAQ1/TestImage"  # 原始图片文件夹路径
output_dir = "/media/hdd1/fubohan/Dataset/IQA/SPAQ"  # 新的保存路径
os.makedirs(output_dir, exist_ok=True)

preprocess = T.Compose([T.Resize((512, 384))])
inspaqfolder = os.listdir(SPAQ)
for item in inspaqfolder:
    imgpath = os.path.join(SPAQ, item)
    img = Image.open(imgpath)
    resized_img = preprocess(img)
    # 保存到新路径，文件名不变
    save_path = os.path.join(output_dir, item)
    resized_img.save(save_path)
    print(save_path)
