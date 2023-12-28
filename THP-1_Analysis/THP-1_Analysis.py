import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import argparse

parser = argparse.ArgumentParser(description='A simple example of argparse')
parser.add_argument('--THP_1_path', type=str, help='Path to the green file')
parser.add_argument('--mask_path', type=str, help='Path to the mask file')
parser.add_argument('--save_xlsv_path', type=str, help='save_csv_pathe')





# 从中心进行缩放
def resize_mask_with_center(mask, scale):
    # 计算掩膜中label的重心坐标
    label_indices = np.argwhere(mask == 1)
    center_x = np.mean(label_indices[:, 0])
    center_y = np.mean(label_indices[:, 1])

    # 创建新的掩膜并进行缩小
    new_mask = np.zeros_like(mask)
    for i, j in label_indices:
        # 将像素点的坐标减去重心坐标并乘以缩小倍数
        new_i = int((i - center_x) * scale + center_x)
        new_j = int((j - center_y) * scale + center_y)

        # 设置新掩膜对应位置处的像素值为1
        new_mask[new_i, new_j] = 1

    return new_mask


# 求最大的椭圆长短径, 把count_ellipse嵌入mask_123label函数！！！
def count_ellipse(binary_mask):
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化最大椭圆面积和对应的椭圆参数
    max_area = 0
    max_ellipse = None

    # 对每个轮廓进行椭圆拟合
    for contour in contours:
        if len(contour) >= 5:  # 至少需要5个点才能拟合椭圆
            ellipse = cv2.fitEllipse(contour)
            # 计算椭圆面积
            area = np.pi * ellipse[1][0] * ellipse[1][1]
            # 如果当前椭圆面积大于最大椭圆面积，则更新最大椭圆参数
            if area > max_area:
                max_area = area
                max_ellipse = ellipse
    return max_ellipse


def mask_123label(mask_path):
    """
    把一份掩膜分成3份,scale为1，0.7，0.3。
    """
    original_mask = cv2.imread(mask_path, 0)
    ellipse = []


    if original_mask.max() == 255:
        original_mask[original_mask == 255] = 1

    new_mask1 = np.copy(original_mask) # 需要使用deepcopy，否则new_mask与original_mask会同时变

    # 储存 长短径
    ellipse.append(count_ellipse(new_mask1)[1])

    scale = 0.67  # 缩小倍数
    new_mask2 = resize_mask_with_center(original_mask, scale)
    new_mask1[new_mask2==1] = 2
    ellipse.append(count_ellipse(new_mask2)[1])
  
    scale = 0.33  # 缩小倍数
    new_mask3 = resize_mask_with_center(original_mask, scale)
    new_mask1[new_mask3==1] = 3
    ellipse.append(count_ellipse(new_mask3)[1])

    return new_mask1, ellipse    

def count_intensity(image_path, mask_path):
    """
    mask_path为1024*1024的shape,
    image_path也一样；
    """
    # 灰度图读取image --> 使用绿色通道进行读取image
    image = cv2.imread(image_path, 1)
    image = image[:, :, 1]  # 绿色通道在OpenCV中的索引为1
    mask_result, ellipse = mask_123label(mask_path)

    # plt.imshow(image, cmap='viridis')  # cmap参数用于指定灰度图像的颜色映射，默认为'viridis'
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()

    intensity_list = []
    for i in range(3):
        label = i + 1
        mask_ = np.zeros_like(mask_result)
        mask_[mask_result==label] = 1  # 获取环

        # plt.imshow(mask_ , cmap='viridis')  # cmap参数用于指定灰度图像的颜色映射，默认为'viridis'
        # plt.axis('off')  # 关闭坐标轴
        # plt.colorbar()
        # plt.show()

        intensity = np.sum(mask_ * image) / np.sum(mask_)

        # plt.imshow(mask_ * image, cmap='viridis')  # cmap参数用于指定灰度图像的颜色映射，默认为'viridis'
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()

        print(intensity, "maks_sum:", np.sum(mask_), "mask_max:", np.max(mask_))
        intensity_list.append(intensity)
    
    return ellipse, intensity_list

def count_intensity_for_mask(image_path, mask_path):
    # 用于计算每一份case的平均荧光强度，不分内中外三个环

    # 灰度图读取image --> 使用绿色通道进行读取image
    image = cv2.imread(image_path, 1)
    image = image[:, :, 1]  # 绿色通道在OpenCV中的索引为1
    original_mask = cv2.imread(mask_path, 0)

    print(np.sum(original_mask), np.max(original_mask), np.min(original_mask))
    print(np.sum(image), np.max(image), np.min(image))
    intensity = np.sum(original_mask * image) / np.sum(original_mask)

    # plt.imshow(original_mask * image, cmap='viridis')  # cmap参数用于指定灰度图像的颜色映射，默认为'viridis'
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()

    # print(intensity)
    return intensity


if __name__ == '__main__':

    args = parser.parse_args()


    # 批量进行计算
    # 绿色荧光文件路径
    green_img_path = args.THP_1_path

    # 对应掩膜路径
    mask_img_path = args.mask_path

    # 表格保存路径
    save_xlsv_path = args.save_xlsv_path

    green_img_files = glob.glob(os.path.join(green_img_path, '*.png'))
    green_img_files.sort()

    case_name = []
    big1, big2 = [], []
    middle1, middle2 = [], []
    small1, small2 = [], []
    big_I, middle_I, small_I = [], [], []

    # 荧光占比
    big_I_p, middle_I_p, small_I_p = [], [], []

    # 平均荧光强度
    mean_I = []


    for green_img_file in green_img_files:
        green_img_name = green_img_file.split("/")[-1]
        mask_img_file = os.path.join(mask_img_path, green_img_name.replace("G.png", "B.png"))
        print(green_img_file, mask_img_file)
        ellipse, intensity_list = count_intensity(green_img_file,
                                            mask_img_file)
        mean_I.append(count_intensity_for_mask(green_img_file,
                                            mask_img_file))
        case_name.append(green_img_name)

        #判断大小
        if ellipse[0][0] >= ellipse[0][1]:
            big1.append(ellipse[0][0])
            big2.append(ellipse[0][1])
        else:
            big2.append(ellipse[0][0])
            big1.append(ellipse[0][1])

        if ellipse[1][0] >= ellipse[1][1]:
            middle1.append(ellipse[1][0])
            middle2.append(ellipse[1][1])
        else:
            middle2.append(ellipse[1][0])
            middle1.append(ellipse[1][1])   

        if ellipse[2][0] >= ellipse[2][1]:
            small1.append(ellipse[2][0])
            small2.append(ellipse[2][1])
        else:
            small2.append(ellipse[2][0])
            small1.append(ellipse[2][1])        


        big_I.append(intensity_list[0])
        middle_I.append(intensity_list[1])
        small_I.append(intensity_list[2])

        sum_I = intensity_list[0] + intensity_list[1] + intensity_list[2]
        big_I_p.append(intensity_list[0]/sum_I)
        middle_I_p.append(intensity_list[1]/sum_I)
        small_I_p.append(intensity_list[2]/sum_I)

    """把数据储存成字典，再保存为csv格式"""




    #create DataFrame
    df = pd.DataFrame({'case':case_name,
                        '外长径': big1,
                    '外短径': big2,
                    '中长径': middle1,
                    '中短径': middle2,
                    '内长径': small1,
                    '内短径': small2,
                    '外强度': big_I,
                    '中强度': middle_I,
                    '内强度': small_I,
                    '外强度占比': big_I_p,
                    '中强度占比': middle_I_p,
                    '内强度占比': small_I_p,
                    "平均荧光强度": mean_I,
                    })

    df.to_excel(save_xlsv_path, index=False)