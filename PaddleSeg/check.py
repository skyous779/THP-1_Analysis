import os
from PIL import Image

def find_max_pixel_value(folder_path):
    max_value = 0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否为PNG格式
        if filename.endswith(".png"):

            # 打开图像文件
            image = Image.open(file_path)

            # 获取图像的最大像素值
            max_pixel_value = max(image.getdata())
            print(type(image.getdata()))
            # # 更新最大值
            # if max_pixel_value > max_value:
            #     max_value = max_pixel_value
            # if max_pixel_value!=255:
            print(max_pixel_value)

            # 关闭图像文件
            image.close()

    return max_value

# 指定文件夹路径
folder_path = "data/EM_seg/mask"

# 调用函数并打印最大值
max_value = find_max_pixel_value(folder_path)
# print("最大像素值:", max_value)