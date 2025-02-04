import SimpleITK as sitk
import numpy as np
from skimage import measure, morphology
import h5py
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation
import copy
import cv2
from skimage.morphology import binary_closing
from scipy.stats import pearsonr,binned_statistic
from sklearn.metrics import mean_absolute_error,r2_score,mean_absolute_percentage_error
from natsort import natsorted
import openpyxl
from sklearn.metrics import confusion_matrix
import pandas as pd
from scipy.ndimage import label, generate_binary_structure
from batchgenerators.utilities.file_and_folder_operations import *
import os
import shutil
from scipy.signal import butter,lfilter,savgol_filter
from scipy.signal import find_peaks
from collections import OrderedDict
import logging
logging.basicConfig(level=logging.DEBUG)
from collections import Counter
import math

# 设置打印选项，使得所有数组都以小数形式输出，且设置小数点后保留的位数
np.set_printoptions(suppress=True, precision=8)  # suppress=True 禁用科学记数法，precision设置小数点后的位数

##remain the  Connected region whichs more than 1000 voxel
def remove_small_volums(mask):
    mask1 = np.where(mask > 0, 1, 0)
    segmentation_sitk = sitk.GetImageFromArray(mask1)
    # 计算标签图像的连通分量
    connected_components_filter = sitk.ConnectedComponentImageFilter()
    labeled_image = connected_components_filter.Execute(segmentation_sitk)
    # 获取每个连通分量的大小
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(labeled_image)
    # 初始化一个空的数组来存储处理后的mask
    cleaned_segmentation = np.zeros_like(mask1)
    # 遍历每个连通分量，保留体积大于min_volume的连通区域
    for i in range(1, label_shape_filter.GetNumberOfLabels() + 1):
        if label_shape_filter.GetNumberOfPixels(i) >= 14000:
            binary_mask = sitk.Equal(labeled_image, i)
            binary_mask_array = sitk.GetArrayFromImage(binary_mask)
            cleaned_segmentation[binary_mask_array == 1] = 1
    # 返回处理后的mask
    cleaned_segmentation = cleaned_segmentation * mask
    return cleaned_segmentation.astype(np.int16)

def to_windowdata(image, WC, WW):
    # image = (image + 1) * 0.5 * 4095
    # image[image == 0] = -2000
    # image=image-1024
    center = WC  # 40 400//60 300
    width = WW  # 200
    try:
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5
    except:
        # print(WC[0])
        # print(WW[0])
        center = WC[0]  # 40 400//60 300
        width = WW[0]  # 200
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5
    dFactor = 255.0 / (win_max - win_min)
    image = image - win_min
    image = np.trunc(image * dFactor)
    image[image > 255] = 255
    image[image < 0] = 0
    image = image / 255  # np.uint8(image)
    # image = (image - 0.5)/0.5
    return image

def caw(mask):
    """
    使用矢量化操作重新赋值3D掩模中的像素值。

    :param mask: 一个三维numpy数组,代表3D掩模。
    :return: 重新赋值后的3D掩模。
    """
    reassigned_mask = np.zeros_like(mask)
    # 数值区间及其对应的赋值
    reassigned_mask[(mask >= 130) & (mask <= 199)] = 1
    reassigned_mask[(mask >= 200) & (mask <= 299)] = 2
    reassigned_mask[(mask >= 300) & (mask <= 399)] = 3
    reassigned_mask[mask >= 400] = 4
    return reassigned_mask

def calcium_Severity(se0output,se2output):
    img = sitk.ReadImage(se0output, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
    img_array = sitk.GetArrayFromImage(img)
    # mask_path1=mask_path.replace("data/yml/data/p2_nii","use/p2")
    mask_read = sitk.ReadImage(se2output, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
    mask_array = sitk.GetArrayFromImage(mask_read)
    mask_array = remove_small_volums(mask_array)
    mask = copy.deepcopy(mask_array)

    # process
    mask[mask > 0] = 1
    img_array1 = img_array * mask  # 勾画区域
    reassigned_Ca = np.where(img_array1 > 130, 1, 0)
    seg_ncct = img_array * reassigned_Ca
    calciuum_array = caw(seg_ncct)
    # ca_mea = np.sum(calciuum_array)
    ca_mea = np.sum(calciuum_array)*0.67*0.67*1.25/3#0.67*0.67*1.25/3=0.187
    if ca_mea >400:#400 101
        cal_mea = 5
    else:
        cal_mea = 1
    return cal_mea

def calcium_Severity2(img_array,mask_arrayy):
    mask_array=copy.deepcopy(mask_arrayy)
    mask_array[mask_array > 0] = 1
    img_array1 = img_array * mask_array  # 勾画区域
    reassigned_Ca = np.where(img_array1 > 130, 1, 0)
    seg_ncct = img_array * reassigned_Ca
    calciuum_array = caw(seg_ncct)
    # ca_mea = np.sum(calciuum_array)
    ca_mea = np.sum(calciuum_array)*0.67*0.67*1.25/3#0.67*0.67*1.25/3=0.187
    if ca_mea > 400:#400 101
        cal_mea = 5
    else:
        cal_mea = 1
    return cal_mea

#statis1：先判别动脉瘤
def compute_confusion_matrix_unique1(ref_mask, pred_mask, img_array):
    # threshold1 = [14000, 14000, 14000, 14000]  # nor dml jc xz
    # threshold2 = [14000,48000,200000,14000]

    # threshold1 = [14000,14000,14000,14000]  # nor jc dml xz
    # threshold2 = [14000,48000,64000,8000]
    threshold1 = [14000,14000,14000,14000]  # nor jc dml xz
    threshold2 = [14000,48000,64000,14000]
    confusion_matrix = np.zeros((5, 5), dtype=int)
    confusion_matrixx=copy.deepcopy(confusion_matrix)
    re_ca=calcium_Severity2(img_array, ref_mask)
    pred_ca=calcium_Severity2(img_array, ref_mask)
    confusion_matrixx[re_ca-1,pred_ca-1]=1 #记录钙化分级
    unique_classes_ref = np.unique(ref_mask)
    unique_classes_ref = unique_classes_ref[unique_classes_ref > 0]  # 去除背景类

    ref=[]
    for ref_class in unique_classes_ref:
        region_ref = ref_mask == ref_class
        ss=np.sum(region_ref)
        if ss >= threshold1[ref_class-1]:
            ref.append(ref_class)

    if len(ref)>2:
        ref=ref[:2]
        ref_mask[ref_mask > ref[1]] = 1

    unique_classes_pred = np.unique(pred_mask)
    unique_classes_pred = unique_classes_pred[unique_classes_pred > 0]  # 去除背景类
    pred = []
    for pred_class in unique_classes_pred:
        region_pred = pred_mask == pred_class
        if np.sum(region_pred) >= threshold2[pred_class-1]:
            pred.append(pred_class)
    if len(pred) > 2:
        pred = pred[:2]
        pred_mask[pred_mask > pred[1]] = 1
        # if pred[1] in ref:
        #     confusion_matrix[pred[1]-1,pred[1]-1]=1
    if len(ref)==1 and len(pred)== 2:
        confusion_matrix[ref[0]-1, pred[1] - 1]= 1
    elif len(ref) == 2 and len(pred) == 1:
        confusion_matrix[ref[1]-1,pred[0] - 1]= 1
    elif len(ref) == 2 and len(pred) == 2:
        confusion_matrix[ref[1]-1,pred[1] - 1]= 1

    if confusion_matrix[0,:].sum()!=0:
        col = np.nonzero(confusion_matrix[0,:])[0]
        confusion_matrix[0,:]=0
        confusion_matrix[re_ca-1,col]=1#dml,jc,xz
    elif confusion_matrix[:,0].sum()!=0:
        row = np.nonzero(confusion_matrix[:,0])[0]
        confusion_matrix[:,0] = 0
        confusion_matrix[row,pred_ca - 1] = 1  # dml,jc,xz
    else:
        confusion_matrix+=confusion_matrixx#calcium

    aa=copy.deepcopy(confusion_matrix)
    aa[0,0]=0 #判断其它位置是否有0
    if aa.sum()!=0:
        confusion_matrix[0,0]=0#保证有病变就不会是健康
    else:
        confusion_matrix[0, 0]=1

    bb=copy.deepcopy(confusion_matrix[:4,:4])
    if bb.sum()!=0:
        confusion_matrix[4,4]=0
    return confusion_matrix

#statis2：先判别夹层
def compute_confusion_matrix_unique2(ref_mask, pred_mask, img_array):
    ref_mask[ref_mask == 2] = -1  # 先将 2 替换为临时值 -1
    ref_mask[ref_mask == 3] = 2  # 将 3 替换为 2
    ref_mask[ref_mask == -1] = 3  # 最后将临时值 -1 替换为 3

    pred_mask[pred_mask == 2] = -1  # 先将 2 替换为临时值 -1
    pred_mask[pred_mask == 3] = 2  # 将 3 替换为 2
    pred_mask[pred_mask == -1] = 3  # 最后将临时值 -1 替换为 3
    # threshold1 = [14000,48000,48000,14000]  # nor jc dml xz
    # threshold2 = [14000,48000,48000,14000]
    threshold1 = [14000,14000,14000,14000]  # nor jc dml xz
    threshold2 = [14000,64000,48000,14000]
    confusion_matrix = np.zeros((5, 5), dtype=int)
    confusion_matrixx=copy.deepcopy(confusion_matrix)
    re_ca=calcium_Severity2(img_array, ref_mask)
    pred_ca=calcium_Severity2(img_array, ref_mask)
    confusion_matrixx[re_ca-1,pred_ca-1]=1 #记录钙化分级
    unique_classes_ref = np.unique(ref_mask)
    unique_classes_ref = unique_classes_ref[unique_classes_ref > 0]  # 去除背景类

    ref=[]
    for ref_class in unique_classes_ref:
        region_ref = ref_mask == ref_class
        ss=np.sum(region_ref)
        if ss >= threshold1[ref_class-1]:
            ref.append(ref_class)

    if len(ref)>2:
        ref=ref[:2]
        ref_mask[ref_mask > ref[1]] = 1

    unique_classes_pred = np.unique(pred_mask)
    unique_classes_pred = unique_classes_pred[unique_classes_pred > 0]  # 去除背景类
    pred = []
    for pred_class in unique_classes_pred:
        region_pred = pred_mask == pred_class
        if np.sum(region_pred) >= threshold2[pred_class-1]:
            pred.append(pred_class)
    if len(pred) > 2:
        pred = pred[:2]
        pred_mask[pred_mask > pred[1]] = 1
        # if pred[1] in ref:
        #     confusion_matrix[pred[1]-1,pred[1]-1]=1
    if len(ref)==1 and len(pred)== 2:
        confusion_matrix[ref[0]-1, pred[1] - 1]= 1
    elif len(ref) == 2 and len(pred) == 1:
        confusion_matrix[ref[1]-1,pred[0] - 1]= 1
    elif len(ref) == 2 and len(pred) == 2:
        confusion_matrix[ref[1]-1,pred[1] - 1]= 1

    if confusion_matrix[0,:].sum()!=0:
        col = np.nonzero(confusion_matrix[0])[0]
        confusion_matrix[0,:]=0
        confusion_matrix[re_ca-1,col]=1#dml,jc,xz
    else:
        confusion_matrix+=confusion_matrixx#calcium

    aa=copy.deepcopy(confusion_matrix)
    aa[0,0]=0 #判断其它位置是否有0
    if aa.sum()!=0:
        confusion_matrix[0,0]=0#保证有病变就不会是健康
    else:
        confusion_matrix[0, 0]=1

    bb=copy.deepcopy(confusion_matrix[:4,:4])
    if bb.sum()!=0:
        confusion_matrix[4,4]=0#保证有病变就不会是健康
    arr=confusion_matrix
    row_2 = arr[1, :].copy()  # 第 2 行（索引为 1）
    row_3 = arr[2, :].copy()  # 第 3 行（索引为 2）
    arr[1, :], arr[2, :] = row_3, row_2

    # 保存第 3 行和第 3 列的数据
    col_2 = arr[:, 1].copy()  # 第 2 列（索引为 1）
    col_3 = arr[:, 2].copy()  # 第 3 列（索引为 2）
    arr[:, 1], arr[:, 2] = col_3, col_2
    return arr

def remove_common_elements_except_1(list1, list2):

    # 将两个列表转换为集合，并找到交集（不包括 1）
    set1 = set(list1)
    set2 = set(list2)
    common_elements = set1.intersection(set2) - {1}

    # 使用列表推导式过滤掉相同的元素（除了 1）
    new_list1 = [x for x in list1 if x not in common_elements]
    new_list2 = [x for x in list2 if x not in common_elements]

    return new_list1, new_list2

#statis3：多病变判别
def compute_confusion_matrix_mutil(ref_mask, pred_mask, img_array):
    # threshold1 = [14000,14000,14000,14000]  # nor jc dml xz
    # threshold2 = [14000,48000,64000,14000]
    threshold1 = [4000,4000,4000,4000]  # nor jc dml xz
    threshold2 = [14000,48000,64000,14000]#0116
    # threshold2 = [14000,48000,64000,6000]#0117
    confusion_matrix = np.zeros((5, 5), dtype=int)
    re_ca=calcium_Severity2(img_array, ref_mask)
    pred_ca=calcium_Severity2(img_array, ref_mask)
    confusion_matrix[re_ca-1,pred_ca-1]=1
    unique_classes_ref = np.unique(ref_mask)
    unique_classes_ref = unique_classes_ref[unique_classes_ref > 0]  # 去除背景类
    ref=[]
    for ref_class in unique_classes_ref:
        region_ref = ref_mask == ref_class
        ss=np.sum(region_ref)
        if ss >= threshold1[ref_class-1]:
            ref.append(ref_class)

    unique_classes_pred = np.unique(pred_mask)
    unique_classes_pred = unique_classes_pred[unique_classes_pred > 0]  # 去除背景类
    pred = []
    for pred_class in unique_classes_pred:
        region_pred = pred_mask == pred_class
        ss=np.sum(region_pred)
        if ss >= threshold2[pred_class - 1]:
            pred.append(pred_class)
            if pred_class in ref and pred_class > 1:#不统计正常的，正常的最后统计
                confusion_matrix[pred_class - 1, pred_class - 1]= 1

    nref,npred=remove_common_elements_except_1(ref,pred)#剔除除1以外的相同类别，剔除1不好统计误诊和漏诊
    classes_setr = set(nref)
    new_refmask = np.where(np.isin(ref_mask, list(classes_setr)), ref_mask, 0)

    classes_setp = set(npred)
    new_predmask = np.where(np.isin(pred_mask, list(classes_setp)), pred_mask, 0)

    #统计漏诊
    for class_id in nref:
        region_mask1 = new_refmask == class_id
        mask2_in_region = new_predmask[region_mask1]
        filtered_mask2 = mask2_in_region[mask2_in_region != 0]
        unique_classes_mask2_in_region, counts = np.unique(filtered_mask2, return_counts=True)
        if len(counts)==0:
            continue

        max_overlap_class = unique_classes_mask2_in_region[np.argmax(counts)]# 找到最大重叠的类别
        # max_overlap_count = np.max(counts)
        confusion_matrix[class_id-1, max_overlap_class-1]= 1
        npred=[x for x in npred if x != max_overlap_class]#

    #统计误诊
    for pred_id in npred:
        region_mask2 = new_predmask == pred_id
        mask1_in_region = new_refmask[region_mask2]
        filtered_mask1 = mask1_in_region[mask1_in_region != 0]
        unique_classes_mask1_in_region, counts = np.unique(filtered_mask1, return_counts=True)
        if len(counts)==0:
            continue

        max_overlap_class = unique_classes_mask1_in_region[np.argmax(counts)]# 找到最大重叠的类别
        # max_overlap_count = np.max(counts)
        confusion_matrix[max_overlap_class-1,pred_id-1]= 1  # 误诊

    if re_ca>1:#严重钙化
        non_zero_positions1 = []
        rows, cols = confusion_matrix.shape
        for i in range(1):## 遍历矩阵，提取非对角且非零元素的位置坐标
            for j in range(cols-1):
                if i != j and confusion_matrix[i, j] != 0:# 检查是否是非对角且非零元素
                    non_zero_positions1.append((i, j))
        confusion_matrix[0,1:4]=0
        for _, j in non_zero_positions1:
            confusion_matrix1 = np.zeros((5, 5), dtype=int)
            confusion_matrix1[4,j]=1
            confusion_matrix[4,4]=0 #需要保证真实该严重钙化数量
            confusion_matrix+=confusion_matrix1

    if pred_ca>1:#严重钙化
        non_zero_positions2 = []
        rows, cols = confusion_matrix.shape
        for i in range(1):##列
            for j in range(rows-1):
                if j != i and confusion_matrix[j, i] != 0:# 检查是否是非对角且非零元素
                    non_zero_positions2.append((j, i))
        confusion_matrix[1:4,0]=0
        for j,_ in non_zero_positions2:
            confusion_matrix1 = np.zeros((5, 5), dtype=int)
            confusion_matrix1[j,4]=1
            confusion_matrix+=confusion_matrix1

    aa = copy.deepcopy(confusion_matrix)
    aa[0, 0] = 0  # 判断其它位置是否有0
    if aa.sum() != 0:
        confusion_matrix[0, 0] = 0  # 保证有病变就不会是健康
    else:
        confusion_matrix[0, 0] = 1
    return confusion_matrix

# statis1，直接对分割结果统计
def confusion_matrix1a():
    # models=["nnUNetTrainerMednext","nnUNetTrainerMaCNN2","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet","nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]
    #         "nnUNetTrainerAortaNet1","nnUNetTrainerMaCNN","nnUNetTrainerMaCNNC","nnUNetTrainerMaCNN3",

    # models=["nnUNetTrainerMednext","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet"]#"nnUNetTrainerMaCNN2",

    # models=["nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]
    models=["nnUNetTrainerMaCNN2"]
    #"nnUNetTrainerMaCNN",
    labels = ["hnnk", "lz", "cq"]
    # labels = ["lz"]
    # out = "./p3t1a_unique_confusion/"
    out = "./14000/p3t1a_mutil_confusion/"
    os.makedirs(out, exist_ok=True)
    for model in models:
        # model="nnUNetTrainerMaCNN"
        pp = "test1/p3t_1a/"+model
        output_file = out + model + "_p3confusion.txt"
        i = 0
        for label in labels:
            labelsTs = "/media/bit301/data/yml/data/p3/external/"+label  # hnnk lz cq
            # labelsTs = "/media/bit301/data/yml/data/p3/external/cq/dis/dml/PA248"
            # out_test=labelsTs.replace("hnnk","unet/hnnk")
            confusion_matrix = np.zeros((5, 5), dtype=int)#nor+dml+jc+xz
            labelsTs_list = []
            for root, dirs, files in os.walk(labelsTs, topdown=False):
                for k in range(len(files)):
                    path = os.path.join(root, files[k])
                    if "3.nii.gz" in path:
                        # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                        labelsTs_list.append(path)
            for path in labelsTs_list:
                read0 = sitk.ReadImage(path.replace("3.nii.gz","0.nii.gz"), sitk.sitkInt16)
                image = sitk.GetArrayFromImage(read0)  # real

                read = sitk.ReadImage(path, sitk.sitkInt16)
                mask1 = sitk.GetArrayFromImage(read)#real
                # mask1=copy.deepcopy(mask11)

                target_path = path.replace("p3", pp)  # test/MedNeXtx2
                target_path=target_path.replace("3.nii.gz","2.nii.gz")
                read = sitk.ReadImage(target_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                mask2 = sitk.GetArrayFromImage(read)#predict

                # confusion_matrix += compute_confusion_matrix_unique1(mask1, mask2,image)#先
                # confusion_matrix += compute_confusion_matrix_unique2(mask1, mask2,image)#>先夹层
                confusion_matrix += compute_confusion_matrix_mutil(mask1, mask2, image)  # >400 重度钙化

                i = i + 1
                if i % 10 == 0:
                    print('numbers:', i)
            # 输出混淆矩阵
            outname = model + "    " + label
            with open(output_file, 'a') as file:
                file.write(outname + "\n")
                file.write(f"{confusion_matrix}\n")
                file.write("\n")
        print("finished "+model)

#裁减之后病变分割的情况
def confusion_matrix1b():
    # models=["nnUNetTrainerMednext","nnUNetTrainerMaCNN2","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet","nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]
    # models=["nnUNetTrainerMaCNN2","nnUNetTrainerMednext","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet"]
    # models=["nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]
    models=["nnUNetTrainerMaCNN4"]
    #"nnUNetTrainerMaCNN",
    labels = ["hnnk", "lz", "cq"]
    # out = "./14000/p3t1b_unique_confusion/"
    out = "./14000/p3t1a_mutil_confusion/"
    os.makedirs(out, exist_ok=True)
    for model in models:
        # model="nnUNetTrainerMaCNN"
        pp = "test1/p3t_1b/"+model
        output_file = out + model + "_p3confusion.txt"
        i = 0
        for label in labels:
            labelsTs = "/media/bit301/data/yml/data/p3_crop_preseg/external/"+label  # hnnk lz cq
            # out_test=labelsTs.replace("hnnk","unet/hnnk")
            confusion_matrix = np.zeros((5, 5), dtype=int)#nor+dml+jc+xz
            labelsTs_list = []
            out_test_list=[]
            for root, dirs, files in os.walk(labelsTs, topdown=False):
                for k in range(len(files)):
                    path = os.path.join(root, files[k])
                    if "3.nii.gz" in path:
                        # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                        labelsTs_list.append(path)
            for path in labelsTs_list:
                read0 = sitk.ReadImage(path.replace("3.nii.gz","0.nii.gz"), sitk.sitkInt16)
                image = sitk.GetArrayFromImage(read0)  # real

                read = sitk.ReadImage(path, sitk.sitkInt16)
                mask1 = sitk.GetArrayFromImage(read)#real
                # mask1=copy.deepcopy(mask11)

                target_path = path.replace("p3_crop_preseg", pp)  # test/MedNeXtx2
                target_path=target_path.replace("3.nii.gz","2.nii.gz")
                read = sitk.ReadImage(target_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                mask2 = sitk.GetArrayFromImage(read)#predict

                # confusion_matrix += compute_confusion_matrix_unique1(mask1, mask2,image)#>400 重度钙化
                confusion_matrix += compute_confusion_matrix_mutil(mask1, mask2, image)  # >400 重度钙化

                i = i + 1
                if i % 10 == 0:
                    print('numbers:', i)
            # 输出混淆矩阵
            outname = model + "    " + label
            with open(output_file, 'a') as file:
                file.write(outname + "\n")
                file.write(f"{confusion_matrix}\n")
                file.write("\n")
        print("finished "+model)

def confusion_matrix2ad():
    # 第一级为p2a-nnUNetTrainerMaCNN2分割binary mask。
    # 第二级为p2ad-多个模型分割. ##动脉瘤，钙化，血栓，软斑块，内膜片

    # models=["nnUNetTrainerMednext","nnUNetTrainerMaCNN2","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet","nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]
    # models=["nnUNetTrainerMaCNN2","nnUNetTrainerMednext","nnUNetTrainerNnformer",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet"]
    # models=["nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerSwinUNETR","nnUNetTrainerSegResNet"]
    models=["nnUNetTrainerMaCNN2"]
    #"nnUNetTrainerMaCNN4",
    labels = ["hnnk", "lz", "cq"]
    out = "./p3t2ad_unieuqe_confusion/"
    os.makedirs(out, exist_ok=True)
    for model in models:
        # model="nnUNetTrainerMaCNN"
        pp = "test2ad/p3t/"+model
        output_file = out + model + "_p3t2adconfusion.txt"
        i = 0
        for label in labels:
            labelsTs = "/media/bit301/data/yml/data/p3_crop_preseg/external/"+label  # hnnk lz cq
            # out_test=labelsTs.replace("hnnk","unet/hnnk")
            confusion_matrix = np.zeros((5, 5), dtype=int)
            labelsTs_list = []
            out_test_list=[]
            for root, dirs, files in os.walk(labelsTs, topdown=False):
                for k in range(len(files)):
                    path = os.path.join(root, files[k])
                    if "3.nii.gz" in path:
                        # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                        labelsTs_list.append(path)
            for path in labelsTs_list:
                read0 = sitk.ReadImage(path.replace("3.nii.gz","0.nii.gz"), sitk.sitkInt16)
                image = sitk.GetArrayFromImage(read0)  # real

                read = sitk.ReadImage(path, sitk.sitkInt16)
                mask1 = sitk.GetArrayFromImage(read)#real

                # pp = "test2/p3t/nnUNetTrainerMaCNN"
                target_path = path.replace("p3_crop_preseg", pp)  # test/MedNeXtx2
                target_path=target_path.replace("3.nii.gz","2.nii.gz")
                read = sitk.ReadImage(target_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                mask2 = sitk.GetArrayFromImage(read)#predict
                confusion_matrix += compute_confusion_matrix_unique1(mask1, mask2,image)#>400 重度钙化
                # confusion_matrix += compute_confusion_matrix_mutil(mask1, mask2, image)  # >400 重度钙化

                i = i + 1
                if i % 10 == 0:
                    print('numbers:', i)
            # 输出混淆矩阵
            outname = model + "    " + label
            with open(output_file, 'a') as file:
                file.write(outname + "\n")
                file.write(f"{confusion_matrix}\n")
                file.write("\n")
        print("finished "+model)

###统计级联分割的判别情况
def confusion_matrix3():
    # 第一级为p2a-nnUNetTrainerMaCNN2分割binary mask。
    # 第二级为p2aa-多个模型分割 mutil-types mask. ##动脉瘤，钙化，血栓，软斑块，内膜片
    # 第三级为p3 nnUNetTrainerMaCNN2

    # models=["nnUNetTrainerMednext","nnUNetTrainerMaCNN2","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet","nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]
    # models=["nnUNetTrainerMaCNN2","nnUNetTrainerMednext","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet"]
    # models=["nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]
    models=["nnUNetTrainerMaCNN2"]
    #"nnUNetTrainerMaCNN",
    labels = ["hnnk", "lz", "cq"]
    out = "./p3t3_unieuqe_confusion/"
    os.makedirs(out, exist_ok=True)
    for model in models:
        # model="nnUNetTrainerMaCNN"
        pp = "test3/p3ta/"+model
        output_file = out + model + "_p3t3confusion.txt"
        i = 0
        for label in labels:
            labelsTs = "/media/bit301/data/yml/data/p3_crop_preseg/external/"+label  # hnnk lz cq
            # out_test=labelsTs.replace("hnnk","unet/hnnk")
            confusion_matrix = np.zeros((5, 5), dtype=int)
            labelsTs_list = []
            out_test_list=[]
            for root, dirs, files in os.walk(labelsTs, topdown=False):
                for k in range(len(files)):
                    path = os.path.join(root, files[k])
                    if "3.nii.gz" in path:
                        # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                        labelsTs_list.append(path)
            for path in labelsTs_list:
                read0 = sitk.ReadImage(path.replace("3.nii.gz","0.nii.gz"), sitk.sitkInt16)
                image = sitk.GetArrayFromImage(read0)  # real

                read = sitk.ReadImage(path, sitk.sitkInt16)
                mask1 = sitk.GetArrayFromImage(read)#real

                # pp = "test2/p3t/nnUNetTrainerMaCNN"
                target_path = path.replace("p3_crop_preseg", pp)  # test/MedNeXtx2
                target_path=target_path.replace("3.nii.gz","2.nii.gz")
                read = sitk.ReadImage(target_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                mask2 = sitk.GetArrayFromImage(read)#predict
                confusion_matrix += compute_confusion_matrix_unique1(mask1, mask2,image)#>400 重度钙化
                # confusion_matrix += compute_confusion_matrix_mutil(mask1, mask2, image)  # >400 重度钙化

                i = i + 1
                if i % 10 == 0:
                    print('numbers:', i)
            # 输出混淆矩阵
            outname = model + "    " + label
            with open(output_file, 'a') as file:
                file.write(outname + "\n")
                file.write(f"{confusion_matrix}\n")
                file.write("\n")
        print("finished "+model)

########计算血管直径用于判断########
def remain(mask):
    mask1=copy.deepcopy(mask)
    mask1=np.where(mask1>0,1,0)
    # mask2 = mask
    # mask2=np.where(mask2>1,1,0)
    segmentation_sitk = sitk.GetImageFromArray(mask1)
    # 计算标签图像的连通分量
    connected_components_filter = sitk.ConnectedComponentImageFilter()
    labeled_image = connected_components_filter.Execute(segmentation_sitk)

    # 获取每个连通分量的大小
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(labeled_image)

    # 找到最大的连通分量ID
    max_size = 0
    largest_label = 0
    for i in range(1, label_shape_filter.GetNumberOfLabels() + 1):  # Label index starts from 1
        if label_shape_filter.GetNumberOfPixels(i) > max_size:
            max_size = label_shape_filter.GetNumberOfPixels(i)
            largest_label = i

    # 仅保留最大连通分量
    binary_mask = sitk.Equal(labeled_image, largest_label)
    cleaned_segmentation = sitk.Cast(binary_mask, segmentation_sitk.GetPixelID())
    cleaned_segmentation = sitk.GetArrayFromImage(cleaned_segmentation)
    non_zero_positions = np.where(cleaned_segmentation != 0)
    min_coords1 = [np.min(pos) for pos in non_zero_positions]
    cleaned_segmentation2=mask-cleaned_segmentation*mask
    non_zero_positions = np.where(cleaned_segmentation2 != 0)
    min_coords2 = [np.min(pos) for pos in non_zero_positions]
    if min_coords1[0]<min_coords2[0]:#不是依靠体积区分，而是依靠位置区分
        cleaned_segmentation=cleaned_segmentation2
    # print(cleaned_segmentation.max())
    return cleaned_segmentation.astype(np.int16)

def remain_ascending(mask):
    v1=copy.deepcopy(mask)
    z_dims = mask.shape[0]
    mask[mask>0]=1
    flag=2
    cleaned_segmentation=0
    for slice_idx in range(z_dims):  # 如果存在升主动脉，去掉升主动脉
        slice_idx=z_dims-slice_idx-1 #因为计算机读取出来的数据，腹主动脉开始到主动脉弓
        slice_lume = mask[slice_idx, :, :]
        img_label, num = measure.label(slice_lume, connectivity=2, return_num=True)
        props = measure.regionprops(img_label)
        props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
        if len(props_sorted) > 1 and props_sorted[0].area > 300 and flag==0:#存在主动脉弓的情况
            if props_sorted[1].area>300:
                mask[slice_idx+1:, :, :]=0
                try:
                    v2=remain(mask)#找升主动脉
                    # aa=v2.sum()
                    if v2.sum()<6400:#设定升主动脉最小体积
                        continue
                except:
                    continue
                index = np.nonzero(v2)
                cleaned_segmentation = v1*v2#[np.min(index[0]):np.max(index[0])]
                break
        if (len(props_sorted) == 1 and props_sorted[0].area > 500) or (len(props_sorted) > 1
                                                                       and props_sorted[0].area > 500 and props_sorted[1].area < 64):
            flag = 0
        else:
            flag = 1
    return cleaned_segmentation

def ascending_type(mask):
    # lumen_mask = np.where(mask == 1, 1, 0)  # 管腔（正常血液流动区域）
    v1=copy.deepcopy(mask)
    z_dims = mask.shape[0]
    l1=0
    l2=0
    for slice_idx in range(z_dims):  # 如果存在升主动脉，去掉升主动脉
        slice_lume = mask[slice_idx, :, :]
        img_label, num = measure.label(slice_lume, connectivity=2, return_num=True)
        props = measure.regionprops(img_label)
        props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
        if len(props_sorted) == 1 and props_sorted[0].area > 300:
            x,y=props_sorted[0].centroid
            l1=-y,-x,slice_idx #bottom
            break
    x2=0
    y2=0
    flag=2
    for slice_idx in range(z_dims):  # 如果存在升主动脉，去掉升主动脉
        slice_idx=z_dims-slice_idx-1 #因为计算机读取出来的数据，腹主动脉开始到主动脉弓
        slice_lume = mask[slice_idx, :, :]
        img_label, num = measure.label(slice_lume, connectivity=2, return_num=True)
        props = measure.regionprops(img_label)
        props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
        if len(props_sorted) > 1 and props_sorted[0].area > 300 and flag==0:#存在主动脉弓的情况
            if props_sorted[1].area>300:
                v1[slice_idx+1:, :, :]=0
                try:
                    v1=remain_ascending(v1)#找升主动脉
                    if v1.sum()<6400:#设定升主动脉最小体积
                        continue
                except:
                    continue
                non_zero_positions = np.where(v1 != 0)
                min_coords = [np.min(pos) for pos in non_zero_positions]
                # max_coords = [np.max(pos) for pos in non_zero_positions]
                z=min_coords[0]
                slice_lume = v1[z, :, :]#不能计算生主动脉最下一层的x y坐标用于血管拉直
                if slice_lume.sum()<300:
                    z=min_coords[0]
                    slice_lume = mask[z, :, :]
                    slice_lume[slice_lume>0]=1
                img_label, num = measure.label(slice_lume, connectivity=2, return_num=True)
                props = measure.regionprops(img_label)
                props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
                x2, y2 = props_sorted[0].centroid
                l2 =-y2,-x2,z
                break
        if (len(props_sorted) ==1 and props_sorted[0].area > 500) or (len(props_sorted) >1
            and props_sorted[0].area > 500 and props_sorted[1].area < 64):
            flag=0
        else:
            flag=1
    if x2==0 or y2==0:
        for slice_idx in range(z_dims):  # 如果存在升主动脉，去掉升主动脉
            slice_idx = z_dims - slice_idx - 1  # 因为计算机读取出来的数据，腹主动脉开始到主动脉弓
            slice_lume = mask[slice_idx, :, :]
            img_label, num = measure.label(slice_lume, connectivity=2, return_num=True)
            props = measure.regionprops(img_label)
            props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
            if len(props_sorted) == 1 and props_sorted[0].area > 300:  # 存在主动脉弓的情况
                x2, y2 = props_sorted[0].centroid
                l2 = -y2, -x2, slice_idx
                break
    return l1,l2

def diameter_area(mask):
    # lumen_mask = np.where(mask == 1, 1, 0)  # 管腔（正常血液流动区域）
    z_dims = mask.shape[0]
    area_per_lumen = []
    diameter_in=[]
    diameter_ex = []
    for slice_idx in range(z_dims):  # 获取当前切片
        slice_lume = mask[slice_idx, :, :]
        if slice_lume.sum()<16:
            diameter_in.append(0)
            diameter_ex.append(0)
            area_per_lumen.append(0)  # 将面积添加到列表中
        else:
            img_label, num = measure.label(slice_lume, connectivity=2, return_num=True)
            props = measure.regionprops(img_label)
            props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
            max_label = props_sorted[0].label
            slice_lume = (img_label == max_label).astype(int)
            filled_slice_lume = binary_closing(slice_lume)
            gray_img = np.uint8(filled_slice_lume * 255)
            contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            dist_map = cv2.distanceTransform(gray_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            _, max_radius, _, max_center = cv2.minMaxLoc(dist_map)
            rin = 2 * max_radius*0.67  # 内切圆直径

            # largest_contour = max(contours, key=cv2.contourArea)
            # (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
            # rex = 2*radius*0.67 #1.25x0.67x0.67

            area = np.sum(slice_lume > 0)#计算填充前的面积
            diameter_in.append(rin)
            # diameter_ex.append(rex)
            area_per_lumen.append(area)
    return diameter_in, area_per_lumen #

def diameter_index():
    models=["nnUNetTrainerMaCNN2"]
    labels = ["hnnk", "lz", "cq"]
    out = "./p3t_unique_confusion/"
    os.makedirs(out, exist_ok=True)
    for model in models:
        # model="nnUNetTrainerMaCNN"
        pp = "test2a/p3t/" + model
        i = 0
        for label in labels:
            labelsTs = "/media/bit301/data/yml/data/p3/external/" + label  # hnnk lz cq
            labelsTs_list = []
            for root, dirs, files in os.walk(labelsTs, topdown=False):
                for k in range(len(files)):
                    path = os.path.join(root, files[k])
                    if "3.nii.gz" in path:
                        # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                        labelsTs_list.append(path)

            for path in labelsTs_list:
                read0 = sitk.ReadImage(path, sitk.sitkInt16)
                gd = sitk.GetArrayFromImage(read0)  # real

                target_path = path.replace("p3", pp)  # test/MedNeXtx2
                target_path = target_path.replace("3.nii.gz", "22.nii.gz")##22.nii.gz
                read = sitk.ReadImage(target_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                seg_mask = sitk.GetArrayFromImage(read)  # predict
                seg_mask = remove_small_volums(seg_mask)
                maskk = copy.deepcopy(seg_mask)
                maskk=remain_ascending(maskk)
                try:
                    index = np.nonzero(maskk)
                    mask = maskk[np.min(index[0]):np.max(index[0])]
                except:
                    mask=0

                if np.all(mask == 0):
                    per_index1=[0]
                    ex_diameter1a,area1a = [0], [0]
                    ex_diameter1b, area1b=[0],[0]
                else:
                    lumen_mask1a = copy.deepcopy(mask)  # 管腔
                    lumen_mask1a = np.where(lumen_mask1a > 0, 1, 0)
                    lumen_mask1a = remove_small_volums(lumen_mask1a)
                    ex_diameter1a, area1a = diameter_area(lumen_mask1a)#外接圆半径，狭窄指数

                lumen_mask2a = copy.deepcopy(seg_mask)-maskk  # 管腔
                lumen_mask2a = np.where(lumen_mask2a > 0, 1, 0)
                ex_diameter2a, area2a = diameter_area(lumen_mask2a)#外接圆半径，狭窄指数

                path_save=os.path.join(out,target_path.split("test2a/")[1].replace("p3t","p3t_index"))
                h5_path_save = path_save.replace(".nii.gz", ".h5")
                file_path=os.path.dirname(h5_path_save)
                if os.path.exists(file_path):#存在就删除然后再创建
                    shutil.rmtree(file_path)
                os.makedirs(file_path, exist_ok=True)
                with h5py.File(h5_path_save, 'w') as f:  # 创建一个dataset
                    f.create_dataset('ex_diameter1a', data=ex_diameter1a)  # 升主动脉
                    f.create_dataset('area1a', data=area1a)  # 整体

                    f.create_dataset('ex_diameter2a', data=ex_diameter2a)  # 拉直
                    f.create_dataset('area2a', data=area2a)  # 整体
                i = i + 1
                if i % 10 == 0:
                    print('numbers:', i)
        print("finished "+model)

def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs  # 尼奎斯特频率
    normal_cutoff = cutoff / nyquist  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

#直径+滑动窗判别，仅对判别为正常的数据金进一步处理
def caculate_by_index(ex_diameter):
    ex_diameter=copy.deepcopy(ex_diameter[12:-12])#去除首尾异常值
    index=np.where(ex_diameter>15)[0]
    st=index.min()
    ed=index.max()
    ex_diameter2a = ex_diameter[st:ed]
    # fs = 100  # 采样频率 (假设每秒采样 100 次)
    # cutoff_frequency = 1  # 截止频率 (单位：Hz)
    # ex_diametera = low_pass_filter(ex_diameter2a, cutoff_frequency, fs)
    # # per_index2 = low_pass_filter(per_index2, cutoff_frequency, fs)

    window_length = 51  # 窗口长度，必须为奇数
    polyorder = 4  # 多项式阶数
    try:
        ex_diameter2a = savgol_filter(ex_diameter2a, window_length=window_length, polyorder=polyorder)
    except:
        aa=1

    # max_len = len(ex_diameter2a)
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4, 9), sharey=True, layout='constrained')
    # line_width = 1
    # for ax in axs:
    #     ax.invert_yaxis()
    # line_color1 = 'r'
    # # line_color2 = 'b'
    # axs[0].plot(ex_diameter2a, range(max_len), color=line_color1, linewidth=line_width, label='True')
    # # axs[0].plot(ex_diameter2a[::-1], range(max_len), color=line_color2, linewidth=line_width)
    # # axs[0].set_title('Lumen Diameter')
    # # # max_len = len(per_index2)
    # # # axs[1].plot(per_index2[::-1], range(max_len), color=line_color2, linewidth=line_width)
    # axs[1].set_title('Vessel Diameter')
    # plt.show()

    # le=len(ex_diameter2a)
    threshold_diameter=45 #55 50
    thresold_stenosis_index=0.4
    sum_d = np.sum(ex_diameter2a > threshold_diameter)
    flag1=0
    flag2=0
    flag=0
    if sum_d>4:#避免异常值
        flag1=2

    # # 3. 构建滑动窗口并检测动脉瘤
    window_size = 45  # 窗口大小（可以根据需要调整）
    stride = 10  # 滑动步长（可以根据需要调整）
    for i in range(0, len(ex_diameter2a) - window_size + 1, stride):
        window = ex_diameter2a[i:i + window_size]

        # 计算窗口内的最小直径，作为该窗口的“正常血管直径”
        sorted_arr = np.sort(window)#从小达到
        normal_diameter=sorted_arr[1:int(window_size/8)*4].mean()
        hig_mean=sorted_arr[-4:].mean()
        # normal_diameter = np.min(window)
        threshold = 2 * normal_diameter
        # if hig_mean<30:
        #     threshold = 1.5 * normal_diameter  # 设定阈值为最小直径的 1.5 倍
        # else:
        #     threshold = 2 * normal_diameter  # 设定阈值为最小直径的 1.5 倍

        # 检查窗口内是否存在超过阈值的直径
        if np.any(window > threshold):
            flag2 = 2
    if flag1>0 and flag2>0:
        flag=2
    return flag

#融合判别
def fusion_confusion():
    #1-nor,2-dml,3-jc,4-xz,10-cal
    # models=["nnUNetTrainerMednext","nnUNetTrainerMaCNN2","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet","nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]

    # models=["nnUNetTrainerMednext","nnUNetTrainerMaCNN2","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet"]


    # models=["nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]
    models=["nnUNetTrainerMaCNN2"]
    # out = "./p3t1a_unique_fusion/"
    out = "./14000/p3t1a_mutil_fusion/"
    p_index = "p3t_index/nnUNetTrainerMaCNN2/external"
    pp="test1/p3t_1a/"
    # pp="test2ad/p3t/"
    # labels = ["hnnk", "lz", "cq"]
    labels = ["hnnk", "lz"]
    # labels = ["cq"]
    for model in models:
        output_file = out + model + "_p3confusion.txt"
        folder_path = os.path.dirname(output_file)
        os.makedirs(folder_path, exist_ok=True)
        i = 0
        for label in labels:
            labelsTs = "/media/bit301/data/yml/data/p3/external/" + label  # hnnk lz cq
            # labelsTs = "/media/bit301/data/yml/data/p3/external/lz/dis/xz/PA171"
            # labelsTs="/media/bit301/data/yml/data/p3/external/lz/nor/PA205"###升主动脉
            # labelsTs = "/media/bit301/data/yml/data/p3/external/cq/dis/dml/PA248"
            # labelsTs = "/media/bit301/data/yml/data/p3/external/cq/dis/dmzyyh/PA0"
            # labelsTs="/media/bit301/data/yml/data/p3/external/lz/dis/jc/PA134"
            # out_test=labelsTs.replace("hnnk","unet/hnnk")
            confusion_matrix = np.zeros((5, 5), dtype=int)  # nor+dml+jc+xz
            labelsTs_list = []
            for root, dirs, files in os.walk(labelsTs, topdown=False):
                for k in range(len(files)):
                    path = os.path.join(root, files[k])
                    if "3.nii.gz" in path:
                        # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                        labelsTs_list.append(path)
            for path in labelsTs_list:
                index_path=os.path.join(p_index,path.split("external/")[1]).replace("3.nii.gz","22.h5")
                with h5py.File(index_path, 'r') as f_mea:
                    ex_diameter1a = f_mea['ex_diameter1a'][:]#升主动脉
                    area1a=f_mea['area1a'][:]
                    ex_diameter2a = f_mea['ex_diameter2a'][:] #胸腹主动脉
                    area2a = f_mea['area2a'][:]
                # d1=np.sqrt(area1a/math.pi)*2*0.67
                # d2 = np.sqrt(area2a / math.pi) * 2*0.67
                # ex_diameter1a=np.where(ex_diameter1a<1.25*d1,ex_diameter1a,d1)
                # ex_diameter2a = np.where(ex_diameter2a < 1.25*d2, ex_diameter2a, d2)

                le = len(ex_diameter1a)
                if le > 50:
                    le = 50
                else:
                    le = le
                ex_diameter1aa = ex_diameter1a[::-1]#最大外接圆直径采用非拉直升主动脉+拉直胸腹主动脉
                # ex_diameter2a = ex_diameter1aa
                ex_diameter2a[-le:] = ex_diameter1aa[-le:]
                try:
                    flag=caculate_by_index(ex_diameter2a)
                except:
                    flag=0

                read0 = sitk.ReadImage(path.replace("3.nii.gz", "0.nii.gz"), sitk.sitkInt16)
                image = sitk.GetArrayFromImage(read0)  # real

                read = sitk.ReadImage(path, sitk.sitkInt16)
                mask1 = sitk.GetArrayFromImage(read)  # real
                # mask1=copy.deepcopy(mask11)

                target_path = path.replace("p3", pp+model)  # test/MedNeXtx2
                target_path = target_path.replace("3.nii.gz", "2.nii.gz")
                read = sitk.ReadImage(target_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                mask2 = sitk.GetArrayFromImage(read)  # predict
                # confusion_matrix2= compute_confusion_matrix_unique1(mask1, mask2, image)  # 动脉瘤
                # confusion_matrix2= compute_confusion_matrix_unique2(mask1, mask2, image)  # 夹层
                confusion_matrix2 = compute_confusion_matrix_mutil(mask1, mask2, image)  # 可能返回不止一个数值
                confusion_matrix1 = np.zeros((5, 5), dtype=int)  # nor+dml+jc+xz
                if flag>0:
                    x,y=np.nonzero(confusion_matrix2)
                    if y[0]==0 or y[0]==4:#对判别为正常或者严重钙化的进行修正
                        aa=confusion_matrix2[4,4]#缓存
                        confusion_matrix2[x,y[0]]=0
                        confusion_matrix2[x,1]=1
                        confusion_matrix2[4, 4]=aa
                        confusion_matrix += confusion_matrix2
                        # if x[0] != 1:
                        #     print(path)
                    else:
                        confusion_matrix += confusion_matrix2
                else:
                    confusion_matrix += confusion_matrix2

                # aa=confusion_matrix2[1:4,1:4]
                # diagonal_elements = np.diag(aa)
                # has_nonzero_diagonal = np.any(diagonal_elements != 0)
                # if has_nonzero_diagonal:
                #     with open(output_file_path, 'a') as file:
                #             file.write(path + '\n')  # 写入路径，并换行

                i = i + 1
                if i % 10 == 0:
                    print('numbers:', i)
            # 输出混淆矩阵
            outname = model + "    " + label
            with open(output_file, 'a') as file:
                file.write(outname + "\n")
                file.write(f"{confusion_matrix}\n")
                file.write("\n")
        print("finished " + model)

##效果不好
def fusion_confusionad():
    #1-nor,2-dml,3-jc,4-xz,10-cal
    # models=["nnUNetTrainerMednext","nnUNetTrainerMaCNN2","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet","nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]

    # models=["nnUNetTrainerMednext","nnUNetTrainerMaCNN2","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet"]

    # models=["nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]
    models=["nnUNetTrainerMaCNN4"]
    # out = "./p3t1a_unique_fusion/"
    out = "./14000/p3t1b_mutil_fusion/"
    p_index = "p3t_index/nnUNetTrainerMaCNN2/external"
    pp="test1/p3t_1b/"
    # pp="test2ad/p3t/"
    labels = ["hnnk", "lz", "cq"]
    # labels = ["cq"]
    for model in models:
        output_file = out + model + "_p3confusion.txt"
        folder_path = os.path.dirname(output_file)
        os.makedirs(folder_path, exist_ok=True)
        i = 0
        for label in labels:
            labelsTs = "/media/bit301/data/yml/data/p3_crop_preseg/external/"+label  # hnnk lz cq
            confusion_matrix = np.zeros((5, 5), dtype=int)  # nor+dml+jc+xz
            labelsTs_list = []
            for root, dirs, files in os.walk(labelsTs, topdown=False):
                for k in range(len(files)):
                    path = os.path.join(root, files[k])
                    if "3.nii.gz" in path:
                        # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                        labelsTs_list.append(path)
            for path in labelsTs_list:
                index_path=os.path.join(p_index,path.split("external/")[1]).replace("3.nii.gz","22.h5")
                with h5py.File(index_path, 'r') as f_mea:
                    ex_diameter1a = f_mea['ex_diameter1a'][:]#升主动脉
                    area1a=f_mea['area1a'][:]
                    ex_diameter2a = f_mea['ex_diameter2a'][:] #胸腹主动脉
                    area2a = f_mea['area2a'][:]
                # d1=np.sqrt(area1a/math.pi)*2*0.67
                # d2 = np.sqrt(area2a / math.pi) * 2*0.67
                # ex_diameter1a=np.where(ex_diameter1a<1.25*d1,ex_diameter1a,d1)
                # ex_diameter2a = np.where(ex_diameter2a < 1.25*d2, ex_diameter2a, d2)

                le = len(ex_diameter1a)
                if le > 50:
                    le = 50
                else:
                    le = le
                ex_diameter1aa = ex_diameter1a[::-1]#最大外接圆直径采用非拉直升主动脉+拉直胸腹主动脉
                # ex_diameter2a = ex_diameter1aa
                ex_diameter2a[-le:] = ex_diameter1aa[-le:]
                try:
                    flag=caculate_by_index(ex_diameter2a)
                except:
                    flag=0

                read0 = sitk.ReadImage(path.replace("3.nii.gz", "0.nii.gz"), sitk.sitkInt16)
                image = sitk.GetArrayFromImage(read0)  # real

                read = sitk.ReadImage(path, sitk.sitkInt16)
                mask1 = sitk.GetArrayFromImage(read)  # real
                # mask1=copy.deepcopy(mask11)

                target_path = path.replace("p3_crop_preseg", pp+model)  # test/MedNeXtx2
                target_path = target_path.replace("3.nii.gz", "2.nii.gz")
                read = sitk.ReadImage(target_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                mask2 = sitk.GetArrayFromImage(read)  # predict
                # confusion_matrix2= compute_confusion_matrix_unique1(mask1, mask2, image)  # 动脉瘤
                # confusion_matrix2= compute_confusion_matrix_unique2(mask1, mask2, image)  # 夹层
                confusion_matrix2 = compute_confusion_matrix_mutil(mask1, mask2, image)  # 可能返回不止一个数值
                confusion_matrix1 = np.zeros((5, 5), dtype=int)  # nor+dml+jc+xz
                if flag > 0:
                    x, y = np.nonzero(confusion_matrix2)
                    if y[0] == 0 or y[0] == 4:  # 对判别为正常或者严重钙化的进行修正
                        aa = confusion_matrix2[4, 4]  # 缓存
                        confusion_matrix2[x, y[0]] = 0
                        confusion_matrix2[x, 1] = 1
                        confusion_matrix2[4, 4] = aa
                        confusion_matrix += confusion_matrix2
                        # if x[0] != 1:
                        #     print(path)
                    else:
                        confusion_matrix += confusion_matrix2
                else:
                    confusion_matrix += confusion_matrix2
                i = i + 1
                if i % 10 == 0:
                    print('numbers:', i)
            # 输出混淆矩阵
            outname = model + "    " + label
            with open(output_file, 'a') as file:
                file.write(outname + "\n")
                file.write(f"{confusion_matrix}\n")
                file.write("\n")
        print("finished " + model)

def crop_mask(preseg):
    output_size = [1000, 300, 300]  # 统计mask范围得到my=262 mx=217
    # output_size = [1000, 360, 320]
    index = np.nonzero(preseg)
    # 检查是否有非零值
    if index[0].size == 0:
        print("No non-zero values in the mask.")
    else:
        # 计算非零值的范围
        # z_min = np.min(index[0])
        # z_max = np.max(index[0])
        # 统计mask范围得到my=262 mx=217
        y_min = np.min(index[1])
        y_max = np.max(index[1])
        x_min = np.min(index[2])
        x_max = np.max(index[2])
        # z_middle = int((z_min + z_max) / 2)
        y_middle = int((y_min + y_max) / 2)
        x_middle = int((x_min + x_max) / 2)
        crop_y_down = y_middle - int(output_size[1] / 2)
        crop_y_up = y_middle + int(output_size[1] / 2)
        if crop_y_down < 0:
            crop_y_down = 64
            crop_y_up = crop_y_down + output_size[1]
        elif crop_y_up > 512:
            crop_y_up = 448
            crop_y_down = crop_y_up - output_size[1]

        crop_x_down = x_middle - int(output_size[2] / 2)
        crop_x_up = x_middle + int(output_size[2] / 2)
        if crop_x_down < 0:
            crop_x_down = 64
            crop_x_up = crop_x_down + output_size[2]
        elif crop_x_up > 512:
            crop_x_up = 448
            crop_x_down = crop_x_up - output_size[2]
        # if crop_y_down > y_min or crop_y_up < y_max or crop_x_down > x_min or crop_x_up < x_max:

    preseg= preseg[:, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
    return preseg

#形态特征后处理的结果
def postprocess_mask():
    #1-nor,2-dml,3-jc,4-xz,10-cal
    # models=["nnUNetTrainerMednext","nnUNetTrainerMaCNN2","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet","nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]

    # models=["nnUNetTrainerMednext","nnUNetTrainerMaCNN2","nnUNetTrainerUNETR","nnUNetTrainerSwinUNETR",
    #         "nnUNetTrainerSegMamba","nnUNetTrainerUXnet"]

    # models=["nnUNetTrainerUxLSTMBot","nnUNetTrainer",
    #         "nnUNetTrainerUMambaBot","nnUNetTrainerNnformer","nnUNetTrainerSegResNet"]
    models=["nnUNetTrainerMaCNN4"]#nnUNetTrainerMaCNN2
    # out = "./p3t1a_unique_fusion/"
    # out = "./14000/p3t1b_mutil_fusion/"
    p_index = "p3t_index/nnUNetTrainerMaCNN2/external"
    pp="test1/p3t_1b/"
    ppp="test2a/p3t/nnUNetTrainerMaCNN2"
    labels = ["hnnk","lz", "cq"]#"hnnk",
    # labels = ["cq"]
    for model in models:
        i = 0
        for label in labels:
            labelsTs = "/media/bit301/data/yml/data/p3_crop_preseg/external/" + label  # hnnk lz cq
            confusion_matrix = np.zeros((5, 5), dtype=int)  # nor+dml+jc+xz
            labelsTs_list = []
            for root, dirs, files in os.walk(labelsTs, topdown=False):
                for k in range(len(files)):
                    path = os.path.join(root, files[k])
                    if "3.nii.gz" in path:
                        # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                        labelsTs_list.append(path)
            for path in labelsTs_list:
                index_path=os.path.join(p_index,path.split("external/")[1]).replace("3.nii.gz","22.h5")
                with h5py.File(index_path, 'r') as f_mea:
                    ex_diameter1a = f_mea['ex_diameter1a'][:]#升主动脉
                    area1a=f_mea['area1a'][:]
                    ex_diameter2a = f_mea['ex_diameter2a'][:] #胸腹主动脉
                    area2a = f_mea['area2a'][:]
                le = len(ex_diameter1a)
                if le > 50:
                    le = 50
                else:
                    le = le
                ex_diameter1aa = ex_diameter1a[::-1]#最大外接圆直径采用非拉直升主动脉+拉直胸腹主动脉
                # ex_diameter2a = ex_diameter1aa
                ex_diameter2a[-le:] = ex_diameter1aa[-le:]
                try:
                    flag=caculate_by_index(ex_diameter2a)
                except:
                    flag=0

                read0 = sitk.ReadImage(path.replace("3.nii.gz", "0.nii.gz"), sitk.sitkInt16)
                image = sitk.GetArrayFromImage(read0)  # real

                read = sitk.ReadImage(path, sitk.sitkInt16)
                mask1 = sitk.GetArrayFromImage(read)  # real
                # mask1=copy.deepcopy(mask11)

                target_path = path.replace("p3_crop_preseg", pp+model)  # test/MedNeXtx2
                target_path = target_path.replace("3.nii.gz", "2.nii.gz")
                read = sitk.ReadImage(target_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                mask2 = sitk.GetArrayFromImage(read)  # predict
                mask22=copy.deepcopy(mask2)
                confusion_matrix2 = compute_confusion_matrix_mutil(mask1, mask2, image)  # 可能返回不止一个数值
                if flag>0:
                    x,y=np.nonzero(confusion_matrix2)
                    if y[0]==0 or y[0]==4:#对判别为正常或者严重钙化的进行修正
                        mask0_path = path.replace("p3_crop_preseg", ppp)  # test/MedNeXtx2
                        mask0_path = mask0_path.replace("3.nii.gz", "22.nii.gz")
                        read = sitk.ReadImage(mask0_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
                        mask0 = sitk.GetArrayFromImage(read)  # predict
                        mask0=crop_mask(mask0)

                        mask22[mask22>0]=1
                        mask0[mask0>0]=1
                        mask0_only = np.logical_and(mask0, np.logical_not(mask22)).astype(int)
                        mask0_only[mask0_only>0]=1
                        mask0_only=remain(mask0_only)
                        mask2=mask2+mask0_only*2
                        out2 = sitk.GetImageFromArray(mask2)
                        out2_path = target_path.replace("/nnUNetTrainerMaCNN4/", "/nnUNetTrainerMaCNN4p/")  # "33.nii.gz"
                        folder_path = os.path.dirname(out2_path)
                        os.makedirs(folder_path, exist_ok=True)
                        sitk.WriteImage(out2, out2_path)
                    else:
                        aa=1
                else:
                    aa=1
                i = i + 1
                if i % 10 == 0:
                    print('numbers:', i)
        print("finished " + model)

################ 数据分布统计+依据混淆计算评测结果##############
def statis():
    labels = ["hnnk", "lz", "cq"]
    # labels = ["cq"]
    for label in labels:
        labelsTs = "/media/bit301/data/yml/data/p3/external/" + label  # hnnk lz cq
        # labelsTs = "/media/bit301/data/yml/data/p3/internal/"  # hnnk lz cq
        labelsTs_list = []
        for root, dirs, files in os.walk(labelsTs, topdown=False):
            for k in range(len(files)):
                path = os.path.join(root, files[k])
                if "3.nii.gz" in path:# and "hnnk" not in path:
                    # path="/media/bit301/data/yml/data/p2_nii/external/cq/dis/dmzyyh/PA57/2.nii.gz"
                    labelsTs_list.append(path)
        tref = []
        for path in labelsTs_list:
            read0 = sitk.ReadImage(path.replace("3.nii.gz", "0.nii.gz"), sitk.sitkInt16)
            img_array = sitk.GetArrayFromImage(read0)  # real
            read = sitk.ReadImage(path, sitk.sitkInt16)
            ref_mask = sitk.GetArrayFromImage(read)  # real
            threshold1 = [4000, 4000, 4000, 4000]
            ref = []
            re_ca = calcium_Severity2(img_array, ref_mask)
            unique_classes_ref = np.unique(ref_mask)
            unique_classes_ref = unique_classes_ref[unique_classes_ref > 1]  # 去除背景类
            for ref_class in unique_classes_ref:
                region_ref = ref_mask == ref_class
                ss = np.sum(region_ref)
                if ss >= threshold1[ref_class - 1]:
                    ref.append(ref_class)
            if re_ca == 5:
                ref.append(re_ca)  # 钙化
            if len(ref) == 0:  # 没有病变，则为正常
                ref.append(1)  # 如果没有病变，或者没有1
            tref += ref
        count_dict = dict(Counter(tref))
        for key in sorted(count_dict):
            print(f"{key}: {count_dict[key]}")

def parse_confusion_matrix(file_path, ss):
    """
    从 txt 文件中读取多个混淆矩阵，并返回一个字典，键为标签，值为混淆矩阵。
    参数:
    file_path (str): 包含混淆矩阵的 txt 文件路径
    返回:
    dict: 键为标签，值为混淆矩阵的字典
    """
    confusion_matrices = {}
    current_label = None
    current_matrix = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 如果是空行或注释，跳过
        if not line or line.startswith('#'):
            i += 1
            continue
        # 如果遇到新的标签行
        if line.startswith(ss):
            # 如果有未处理的矩阵，先保存
            if current_label is not None and current_matrix:
                confusion_matrices[current_label] = np.array(current_matrix, dtype=int)
                current_matrix = []
            # 提取标签
            parts = line.split()
            current_label = parts[1]
            i += 1
            continue
        # 如果遇到矩阵行
        if line.startswith('[') and line.endswith(']'):
            # 去掉方括号并分割成数字
            row = line.strip('[]').split()
            row = [int(x) for x in row]
            current_matrix.append(row)
            # 如果当前矩阵已经完整（5x5），保存并重置
            if len(current_matrix) == 5:
                confusion_matrices[current_label] = np.array(current_matrix, dtype=int)
                current_matrix = []
                current_label = None
        i += 1
    # 如果文件末尾还有未处理的矩阵，保存
    if current_label is not None and current_matrix:
        confusion_matrices[current_label] = np.array(current_matrix, dtype=int)

    return confusion_matrices

def conf_index(confusion_matrix):
    # 2-TP/TN/FP/FN的计算
    weight = confusion_matrix.sum(axis=0) / confusion_matrix.sum()  ## 求出每列元素的和
    FN = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)  # 所有对的 TP.sum=TP+TN
    TN = confusion_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    epsilon = 1e-40  # 非常小的偏移量
    TPR = np.where(TP + FN > 0, TP / (TP + FN + epsilon), 0)  # Sensitivity/ hit rate/ recall/ true positive rate
    TNR = np.where(TN + FP > 0, TN / (TN + FP + epsilon), 0)  # Specificity/ true negative rate
    PPV = np.where(TP + FP > 0, TP / (TP + FP + epsilon), 0)  # Precision/ positive predictive value
    NPV = np.where(TN + FN > 0, TN / (TN + FN + epsilon), 0)  # Negative predictive value
    FPR = np.where(TN + FP > 0, FP / (TN + FP + epsilon), 0)  # Fall out/ false positive rate
    FNR = np.where(TP + FN > 0, FN / (TP + FN + epsilon), 0)  # False negative rate
    FDR = np.where(TP + FP > 0, FP / (TP + FP + epsilon), 0)  # False discovery rate
    sub_ACC = np.where(TP+ TN+FP + FN > 0, (TP + TN) / (TP+ TN+FP + FN + epsilon), 0)  # accuracy of each class
    IOU=np.where(TP + FP+ FN > 0, TP / (TP + FP+ FN + epsilon), 0)  # False discovery rate

    average_acc = TP.sum() / (TP.sum() + FN.sum())
    # F1_Score = 2 * TPR * PPV / (PPV + TPR)
    F1_Score = np.where(PPV + TPR > 0, 2 * TPR * PPV / (PPV + TPR + epsilon), 0)
    Macro_F1 = F1_Score.mean()
    weight_F1 = (F1_Score * weight).sum()  # 应该把不同类别给与相同权重,不应该按照数量进行加权把？
    # print('acc:',average_acc)
    # print('Sensitivity:', TPR.mean())#Macro-average方法
    # print('Specificity:', TNR.mean())
    # print('Precision:', PPV.mean())
    # print('Macro_F1:',Macro_F1)
    # 创建一个字典来存储每个类别的评价指标
    metrics = {
        'average_acc': average_acc,
        'Macro_F1': Macro_F1,
        'Sensitivity':TPR.mean(),#模型预测为正类的样本中，实际为正类的比例。
        'Specificity': TNR.mean(),#模型预测为正类的样本中，实际为负类的比例。
        'Precision': PPV.mean(), #预测为正类的样本，模型预测正确的比例
        'IOU': IOU.mean(),
    }

    # 为每个类别创建一个字典，存储其具体的评价指标
    class_metrics = {}
    for i in range(len(TP)):
        class_metrics[f'Class_{i}'] = {
            'sub_ACC': sub_ACC[i],
            'F1_Score': F1_Score[i],
            'Sensitivity': TPR[i],
            'Specificity': TNR[i],
            'Precision': PPV[i],
            'IOU': IOU[i],
        }

    # return average_acc, TPR.mean(), TNR.mean(), PPV.mean(), Macro_F1
    return metrics, class_metrics

def confuse_plot(cm, save_path):
    save_path += ".tif"
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap='Blues')#Blues Oranges

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel('Value', rotation=-90, va="bottom", fontsize=11)  # 设置颜色条标题的字号

    # 调整颜色条的刻度标签字体大小
    cbar.ax.tick_params(labelsize=11)

    # 显示数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > (cm.max() / 2.) else "black",
                    fontsize=11)  # 设置数值的字号

    # 设置坐标轴标签
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(['Normal', 'Aneurysm', 'Dissection', 'Stenosis', 'Calcification'], fontsize=11)
    ax.set_yticklabels(['Normal', 'Aneurysm', 'Dissection', 'Stenosis', 'Calcification'], fontsize=11)
    # ax.set_xticklabels(['Type 1', 'Type 2', 'Type 3','Type 4','Type 5'], fontsize=11)  # 设置X轴标签字号
    # ax.set_yticklabels(['Type 1', 'Type 2', 'Type 3','Type 4','Type 5'], fontsize=11)  # 设置Y轴标签字号

    # 旋转顶部的标签,避免重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=11)  # 设置X轴刻度字号

    # 设定底部和右侧的边框不可见
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 设定底部和左侧的边框线宽
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 调整子图布局,防止坐标标签被截断
    plt.tight_layout()
    plt.savefig(save_path,dpi=300, format='tif')
    # plt.savefig(save_path, dpi=600, format='tif')
    # plt.show()
    plt.close(fig)

def confusion_matrics():
    # models = ["nnUNetTrainerMednext", "nnUNetTrainerMaCNN4", "nnUNetTrainerUNETR", "nnUNetTrainerSwinUNETR",
    #           "nnUNetTrainerSegMamba", "nnUNetTrainerUXnet", "nnUNetTrainerUxLSTMBot", "nnUNetTrainer",
    #           "nnUNetTrainerUMambaBot", "nnUNetTrainerNnformer", "nnUNetTrainerSegResNet",]
    models=["nnUNetTrainerMaCNN4"]
    path = "./14000/p3t1a_mutil_confusion/"
    # path = "./14000/p3t1a_mutil_fusion/"#
    # path = "./14000/p3t1b_mutil_fusion/"#
    out_put = path +"Confuse_disp"
    if not os.path.isdir(out_put):
        os.makedirs(out_put)
    for model in models:
        confusion_path = path + model + "_p3confusion.txt"
        file_path = path + "Metrics/"+ model + "metrics.txt"
        folder_path = os.path.dirname(file_path)
        os.makedirs(folder_path, exist_ok=True)
        matrices = parse_confusion_matrix(confusion_path, model)
        for label, matrix in matrices.items():
            save_path = os.path.join(out_put, model +label)  # mm="matrix1b_m"
            confuse_plot(matrix, save_path)
            metrics, class_metrics = conf_index(matrix)
            with open(file_path, 'a') as file:
                file.write(label + "\n")
                file.write("Overall Metrics:\n")
                for key, value in metrics.items():
                    file.write(f"{key}: {value}\n")
                file.write("\n")

                # 打印每个类别的指标
                file.write("\nClass Metrics:\n")
                for class_name, class_metric in class_metrics.items():
                    file.write(f"{class_name}:")
                    for key, value in class_metric.items():
                        file.write(f"  {key}: {value}\n")
                    file.write("\n")
        print("finished " + model)

def disp():
    f = open("/media/bit301/data/yml/project/python310/p3/process/disp.txt")  # hnnk test.txt
    path_list = []
    for line in f.readlines():#tile_step_size=0.75较好处理官腔错位问题
            path=line.split('\n')[0]
            path_list.append(path)
    # path_list=["/media/bit301/data/yml/data/p3/external/cq/dis/dml/PA1/3.nii.gz"]
    ij=0
    out = "./14000/p3t1a_mutil_fusion/"
    p_index = "p3t_index/nnUNetTrainerMaCNN2/external"
    for path in path_list:
        index_path = os.path.join(p_index, path.split("external/")[1]).replace("3.nii.gz", "22.h5")
        with h5py.File(index_path, 'r') as f_mea:
            ex_diameter1a = f_mea['ex_diameter1a'][:]  # 升主动脉
            area1a = f_mea['area1a'][:]
            ex_diameter2a = f_mea['ex_diameter2a'][:]  # 胸腹主动脉
            area2a = f_mea['area2a'][:]
        le = len(ex_diameter1a)
        if le > 50:
            le = 50
        else:
            le = le
        ex_diameter1aa = ex_diameter1a[::-1]  # 最大外接圆直径采用非拉直升主动脉+拉直胸腹主动脉
        # ex_diameter2a = ex_diameter1aa
        ex_diameter2a[-le:] = ex_diameter1aa[-le:]

        ex_diameter2a = copy.deepcopy(ex_diameter2a[12:-12])  # 去除首尾异常值
        # index = np.where(ex_diameter > 15)[0]
        # st = index.min()
        # ed = index.max()
        # ex_diameter2a = ex_diameter[st:ed]

        window_length = 51  # 窗口长度，必须为奇数
        polyorder = 4  # 多项式阶数
        ex_diameter2aa = savgol_filter(ex_diameter2a, window_length=window_length, polyorder=polyorder)

        max_len = len(ex_diameter2aa)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 8), sharey=True, layout='constrained')#(4, 12)
        # line_color1 = 'b'
        # line_color2 = 'r'
        line_color1 = "#F49568"
        line_color2 = "#82C61E"
        line_width = 1
        # q1_color = "deepskyblue"
        # q2_color = "r"
        axs.invert_yaxis()

        axs.plot(ex_diameter2a[::-1], range(max_len), color=line_color1, linewidth=line_width, label='Original')
        axs.plot(ex_diameter2aa[::-1], range(max_len), color=line_color2, linewidth=line_width, label='Smoothed')
        # axs.set_title('Diameter',fontsize=10)
        axs.set_xlabel('Diameter', fontsize=11)  # 添加横轴标题 pixel
        axs.set_ylabel('Slice Number', fontsize=11)  # 添加纵轴标题
        axs.legend(loc='best', fontsize=10)  # 添加图例，loc='best' 表示自动寻找最佳位置放置图例

        # plt.savefig("High resoltion.png", dpi=600)
        out_put = out+"disp"
        if not os.path.isdir(out_put):
            os.makedirs(out_put)
        file=path.split("external/")[1].split("/2")[0].replace("/","_")+".tif"
        save_path = os.path.join(out_put, file)
        # plt.savefig(save_path)#矢量图
        plt.savefig(save_path, dpi=300,format='tif')
        # plt.show()
        # 显式关闭当前figure
        plt.close(fig)

#管腔，血管，钙化指数 曲线
def disp1():
    f = open("/media/bit301/data/yml/project/python39/p2/Aorta_net/data/disp1.txt")  # hnnk test.txt
    path_list = []
    for line in f.readlines():#tile_step_size=0.75较好处理官腔错位问题
            path=line.split('\n')[0]
            path_list.append(path)
    ij=0
    for path in path_list:
        calcium_gd = 0
        calcium_mea = 0
        with h5py.File(path, 'r') as f_gd:#评估四分位数据
            diameter_per_lumen = f_gd['diameter_per_lumen'][:]
            diameter_per_total = f_gd['diameter_per_total'][:]
            calcium_per_index = f_gd['calcium_per_index'][:]
            total_calcium_index = f_gd['total_calcium_index'][:]

        path_mea=path.replace("2.h5","22.h5")
        with h5py.File(path_mea, 'r') as f_mea:#评估四分位数据
            diameter_per_lumen_mea = f_mea['diameter_per_lumen'][:]
            diameter_per_total_mea = f_mea['diameter_per_total'][:]
            calcium_per_index_mea = f_mea['calcium_per_index'][:]
            total_calcium_index_mea = f_mea['total_calcium_index'][:]

        max_len = len(diameter_per_lumen)
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(5, 10), sharey=True, layout='constrained')#(4, 12)
        line_color1 = 'b'
        line_color2 = 'r'
        line_width = 1
        # q1_color = "deepskyblue"
        # q2_color = "r"
        for ax in axs:
            ax.invert_yaxis()

        axs[0].plot(diameter_per_lumen[::-1], range(max_len), color=line_color1, linewidth=line_width, label='True')
        axs[0].plot(diameter_per_lumen_mea[::-1], range(max_len), color=line_color2, linewidth=line_width, label='Pred')
        axs[0].set_title('Lumen',fontsize=10)
        axs[0].set_xlabel('Diameter', fontsize=10)  # 添加横轴标题 pixel
        axs[0].set_ylabel('Slice Number', fontsize=10)  # 添加纵轴标题
        axs[0].legend(loc='best', fontsize=10)  # 添加图例，loc='best' 表示自动寻找最佳位置放置图例

        axs[1].plot(diameter_per_total[::-1], range(max_len), color=line_color1, linewidth=line_width, label='True')
        axs[1].plot(diameter_per_total_mea[::-1], range(max_len), color=line_color2, linewidth=line_width, label='Pred')
        axs[1].set_title('Vessel ',fontsize=10)
        axs[1].set_xlabel('Diameter', fontsize=10)  # 添加横轴标题
        axs[1].legend(loc='best', fontsize=10)  # 添加图例，loc='best' 表示自动寻找最佳位置放置图例

        axs[2].plot(calcium_per_index[::-1], range(max_len), color=line_color1, linewidth=line_width, label='True')
        axs[2].plot(calcium_per_index_mea[::-1], range(max_len), color=line_color2, linewidth=line_width, label='Pred')
        axs[2].set_title('Calcification',fontsize=10)
        axs[2].set_xlabel('Index', fontsize=10)  # 添加横轴标题
        axs[2].legend(loc='best', fontsize=10)  # 添加图例，loc='best' 表示自动寻找最佳位置放置图例
        # plt.savefig("High resoltion.png", dpi=600)
        out_put = "disp"
        if not os.path.isdir(out_put):
            os.makedirs(out_put)
        file=path.split("Aortic_index/")[1].split("/2")[0].replace("/","_")+".tif"
        save_path = os.path.join(out_put, file)
        plt.savefig(save_path)#矢量图
        # plt.savefig(save_path, dpi=600)
        # plt.show()
        # 显式关闭当前figure
        plt.close(fig)

#Bland-Altman
def dot_plot(data,save_path):
    gd="True "+save_path.split("_")[-1]
    mea="Predict "+save_path.split("_")[-1]+" from NCCT"
    save_path=save_path+".tif"
    data=np.array(data)
    # true_values = data[:, 0]  # 请替换为实际真实值数组
    # predicted_values = data[:, 1]
    # errors = np.abs(predicted_values - true_values)
    true_values = data[:, 0]  # 请替换为实际真实值数组
    predicted_values = data[:, 1]
    errors = data[:, 2]
    fig, ax = plt.subplots()
    scatter = ax.scatter(true_values, predicted_values, c=errors, cmap='viridis', s=20, alpha=0.8)
    cbar = fig.colorbar(scatter, ax=ax, label='Mean Absolute Percentage Error (%)')
    plt.plot([np.nanmin(true_values), np.nanmax(true_values)],
             [np.nanmin(true_values), np.nanmax(true_values)],
             'r--', label='Perfect reconstruction line')
    ax.set_xlabel(gd, fontsize=10)
    ax.set_ylabel(mea, fontsize=10)
    plt.legend()
    ax.set_title('Predict vs True with Mean Absolute Percentage Error', fontsize=10)
    # plt.savefig(save_path)  # 矢量图
    plt.savefig(save_path, dpi=600)
    # plt.show()
    plt.close(fig)# 显式关闭当前figure

if __name__ == '__main__':
    # statis()
    # confusion_matrix1a()#
    # fusion_confusion()

    # confusion_matrix1b()#
    # fusion_confusionad()
    # confusion_matrics()#

    # diameter_index()

    # confusion_matrix2ad()
    # confusion_matrix3()#tr a


    postprocess_mask()
    # disp()




