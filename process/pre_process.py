import numpy as np
import shutil
import random
import os
import SimpleITK as sitk
import cv2
from skimage import measure
import itk
from natsort import natsorted
import copy
from skimage.morphology import ball,binary_closing,disk,binary_dilation
from scipy.ndimage import binary_dilation as binary_dilationn
import datetime

# from p3.Aorta_net.data.data_process import crop_by_mask


############# 文件操作模块 #############
# 将各个子文件夹合并到一起
def mv_file():
    # path = "../../../data/diag_data/"  # CT_CTA disease
    catch = "../../../data/catch"
    if not os.path.isdir(catch):
        os.makedirs(catch)

    input = '../output/Cyc/1e34/'  ######
    output = '../output/diag1a_data/'  ######
    # output = '/media/yml/yml/data/make_choice/diag0a_data/'  ######
    if not os.path.isdir(input):
        os.makedirs(input)
    if not os.path.isdir(output):
        os.makedirs(output)
    path_list = []
    for root, dirs, files in os.walk(input, topdown=False):
        if "SE1" in root:
            path_list.append(root)
    path_list.sort()

    ii = 0
    for sub_path in path_list:
        input_files = os.listdir(sub_path)
        input_files.sort()
        input_files.sort(key=lambda x: (int(x.split('IM')[1])))
        sub_out = sub_path.replace(input, output)
        sub_out = sub_out.replace("SE1", "SE3")  #######
        if not os.path.isdir(sub_out):
            os.makedirs(sub_out)

        for j in range(len(input_files)):
            in_file_path = os.path.join(sub_path, input_files[j])
            out_file_path = os.path.join(sub_out, input_files[j])
            shutil.move(in_file_path, out_file_path)

        ii = ii + 1
        if ii % 10 == 0:
            print('numbers:', ii)

#批量移除文件
def remove_file():
    path = "../../../data/p2_nii/"  # CT_CTA disease
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        if "4.nii.gz" in files:
            path = os.path.join(root, '4.nii.gz')
            # path_list.append(path)
            os.remove(path)
    # path_list.sort()
    # for sub_path in path_list:
    #     aa = os.path.join(sub_path.split('SE0')[0], 'SE2')
    #     if os.path.isdir(aa):
    #         shutil.rmtree(aa)

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

# 剔除一些因为配准变形的数据
def remove_s_slices():
    path = "../../../data/p2_s/xy/xy1/dis/dml/PA35/SE0"
    # path="../../../data/pv_nii"
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            # if "0.nii.gz" in path and "aug" in path:
            if "0.nii.gz" in path:
                path_list.append(path)
    ii = 1
    for se0output in path_list:
        se1output = se0output.replace("0.nii.gz", "1.nii.gz")
        img = sitk.ReadImage(se1output, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        img_array = sitk.GetArrayFromImage(img)
        le = img_array.shape[0] - 1
        le1 = 2
        le2 = le
        # aa=img_array[0:1,:,:]
        for k in range(0, 12):
            img1 = img_array[k, :, :]
            img1 = np.where(img1 == 0, 1, 0)
            img_label, num = measure.label(img1, connectivity=2, return_num=True)  # 输出二值图像中所有的连通域
            props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等
            for i in range(0, len(props)):
                are = props[i].area
                if are > 500:
                    le1 = k + 1
                    continue
        for kk in range(le - 12, le):
            if le2 < le - 1:
                break
            img2 = img_array[kk, :, :]
            img2 = np.where(img2 == 0, 1, 0)
            img_label, num = measure.label(img2, connectivity=2, return_num=True)  # 输出二值图像中所有的连通域
            props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等
            for j in range(0, len(props)):
                are = props[j].area
                if are > 500:
                    le2 = kk + 1
                    break
            # le = img_array.shape[0] - 3
        # le1=80
        img_array = img_array[le1:le2, :, :]
        out = sitk.GetImageFromArray(img_array)
        os.remove(se1output)
        sitk.WriteImage(out, se1output)

        # os.remove(se0output)
        img = sitk.ReadImage(se0output, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        img_array = sitk.GetArrayFromImage(img)
        img_array = img_array[le1:le2, :, :]
        out = sitk.GetImageFromArray(img_array)
        os.remove(se0output)
        sitk.WriteImage(out, se0output)

        try:
            read = se0output.replace("0.nii.gz", "2.nii.gz")
            img = sitk.ReadImage(read, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
            img_array = sitk.GetArrayFromImage(img)
            img_array = img_array[le1:le2, :, :]
            out = sitk.GetImageFromArray(img_array)
            os.remove(read)
            sitk.WriteImage(out, read)
            # os.remove(read)
        except:
            continue

        try:
            read = se0output.replace("0.nii.gz", "3.nii.gz")
            img = sitk.ReadImage(read, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
            img_array = sitk.GetArrayFromImage(img)
            img_array = img_array[le1:le2, :, :]
            out = sitk.GetImageFromArray(img_array)
            os.remove(read)
            sitk.WriteImage(out, read)
        except:
            continue
        ii = ii + 1
        if ii % 10 == 0:
            print('numbers:', ii)

#剔除层面太少的数据
def check_slice_num():
    path = "../../../data/p3"
    # path="../../../data/pv_nii/xy/xy1/dis/jc/PA1"
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "0.nii.gz" in path:
                path_list.append(path)
    i = 1
    for se1output in path_list:
        # read = sitk.ReadImage(se0output, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        # img_array0 = sitk.GetArrayFromImage(read)
        #
        # se1output = se0output.replace("0.nii.gz", "1.nii.gz")
        read = sitk.ReadImage(se1output, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        img_array1 = sitk.GetArrayFromImage(read)
        le = img_array1.shape[0]
        if le < 150:
            print(le)
            print(se1output)
        i = i + 1
        if i % 10 == 0:
            print('numbers:', i)

def aid_mask():
    #辅助标注，在check数据发现有问题，使用该部分代码辅助处理
    #catch1
    path = "/media/bit301/data/yml/data/p3/external/lz/dis/jc/PA146/44.nii.gz" #
    read = sitk.ReadImage(path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
    img_array0 = sitk.GetArrayFromImage(read)
    img_array0 = np.where(img_array0 == 3, 1, 0)
    # img_array0 = np.where(img_array0 == 3, 1, img_array0)
    out = sitk.GetImageFromArray(img_array0.astype(np.int16))
    sitk.WriteImage(out, path)

    # catch2
    path = "/media/bit301/data/yml/data/p3/internal/xm/xm1/dis/jc/DICOM2/PA11/SE0/2.nii.gz" #
    read = sitk.ReadImage(path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
    img_array = sitk.GetArrayFromImage(read)
    img_array = np.where(img_array > 0, 1, 0)
    # img_array = np.where(img_array == 3, 1, 0)

    path = "/media/bit301/data/yml/data/p3/internal/xm/xm1/dis/jc/DICOM2/PA11/SE0/33.nii.gz" #
    read = sitk.ReadImage(path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
    img_array0 = sitk.GetArrayFromImage(read)
    img_array00=np.where(img_array0>0,1,0)
    selem = ball(5)  # 半径为1的球形结构元素
    # selem = cube(3)  # 如果需要立方体结构元素，可以使用cube
    img_array0 = binary_dilation(img_array00, selem)
    img_arrayy=img_array-img_array0#+img_array
    img_arrayy=np.where(img_arrayy==1,1,0)
    img_arrayy=binary_closing(img_arrayy)
    # img_array0=img_array0+img_arrayy
    out = sitk.GetImageFromArray(img_arrayy.astype(np.int16))
    sitk.WriteImage(out, path)

#将标注的数据进行处理(剪影)
def rmake_mask():
    #将初步标记的 管腔+钙化板块（2.nii.gz）与血栓整合成新的mask文件
    # 保留动脉血流范围的图像；mask分别为动脉血流、钙化斑块、血栓、软斑块；如果只有动脉血流则保持不变。
    path = "/media/bit301/data/yml/data/p33" #
    # path = "/media/bit301/data/yml/data/p33/internal/xm/xm1/dis" #
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            # if "2.nii.gz" in files and "1.nii.gz" not in files or "0.nii.gz" not in files:
            #     print(path)
            if "2.nii.gz" in path:
                path_list.append(path)
    i = 0
    for se2output in path_list:
        se0output = se2output.replace("2.nii.gz", "0.nii.gz")
        read = sitk.ReadImage(se0output, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        img_array0 = sitk.GetArrayFromImage(read)
        img_array_w0 = to_windowdata(img_array0, 130, 10)
        img_array_w0 = np.where(img_array_w0 > 0, 1, 0)  # 支架与钙化斑块
        # img_array_w00=1-img_array_w0

        se1output = se2output.replace("2.nii.gz", "1.nii.gz")
        read = sitk.ReadImage(se1output, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        img_array1 = sitk.GetArrayFromImage(read)

        read = sitk.ReadImage(se2output, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        img_array2 = sitk.GetArrayFromImage(read)
        if img_array2.all() == 1:  # mask数值都为1，表示为只有血流。
            print("it's ok")
        else:  # 否者交换mask序号
            # img_array2 = np.where(img_array2 == 1, 22, img_array2)
            # img_array2 = np.where(img_array2 == 2, 1, img_array2)
            # img_array2 = np.where(img_array2 == 22, 2, img_array2)
            img_array2 = np.where(img_array2 > 2, 0, img_array2)
        img_array22 = np.where(img_array2 > 0, 1, 0)  # 将mask合并
        # 裁剪有效范围
        su = np.sum(img_array2, axis=(1, 2))
        index = np.where(su > 16)[0]  # 避免噪声影响
        # index=np.nonzero(img_array2)[0]
        # st = min(index)
        # en = max(index) + 1

        se3output = se2output.replace("2.nii.gz", "33.nii.gz")  # 血液分割标签。有可能不存在
        try:
            img3 = sitk.ReadImage(se3output, sitk.sitkInt16)
            img_array3 = sitk.GetArrayFromImage(img3)
            img_array33 = np.where(img_array3 > 0, 1, 0)
            # print(img_array2.min())
            # print(img_array2.max())

            img_array33 = img_array33 - img_array22  # 血流、化斑块与血栓、软斑块不重叠
            img_array33 = np.where(img_array33 > 0, 1, 0)
            img_array333 = img_array33 + 2  # 血栓和软斑块的mask
            img_array333 = np.where(img_array333 < 3, 0, img_array333)  # 血栓为3，斑块为4
            img_array333 = img_array2 + img_array333  # mask整合

            mask1 = img_array_w0 - img_array_w0 * img_array22  # 支架
            mask2 = 1 - mask1  # 去除金属支架
            img_array333 = mask2 * img_array333
            out2 = sitk.GetImageFromArray(img_array333.astype(np.int16))
            os.remove(se2output)
            os.remove(se3output)
            sitk.WriteImage(out2, se2output)
        except:
            print("不存在血栓和软斑块", se3output)
            out2 = sitk.GetImageFromArray(img_array2.astype(np.int16))
            os.remove(se2output)
            sitk.WriteImage(out2, se2output)
            # continue

        os.remove(se0output)
        out0 = sitk.GetImageFromArray(img_array0.astype(np.int16))#[st:en, :, :]
        sitk.WriteImage(out0, se0output)
        out1 = sitk.GetImageFromArray(img_array1.astype(np.int16))
        sitk.WriteImage(out1, se1output)

        i = i + 1
        if i % 10 == 0:
            print('numbers:', i)

def dml_jc_xz_list():
    path = "/media/bit301/data/yml/data/p3"  #p33为备份数据
    # path = "/media/bit301/data/yml/data/p2_nii/internal/xm"
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "3.nii.gz" in path:
                # filee = root.split("/")[-1]
                path_list.append(path)
    file = "dml_jc_xz_list.txt"
    f1 = open(file, "w")  # 564
    i = 0
    today = datetime.datetime.now().date()
    path_list = natsorted(path_list)
    for se2output in path_list:
        # se2output = "/media/bit301/data/yml/data/p3catch/PA1/2.nii.gz"
        # f1.writelines(se2output + "\n")
        read = sitk.ReadImage(se2output, sitk.sitkInt16)  #
        img_array = sitk.GetArrayFromImage(read)
        if img_array.max()>3:
            f1.writelines(se2output + "\n")
        # 获取文件的修改时间
        # modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(se2output)).date()
        # if modification_time == today:
        #     print(se2output)
            # read = sitk.ReadImage(se2output, sitk.sitkInt16)  #
            # img_array = sitk.GetArrayFromImage(read)
            # img_array[img_array > 0] = 1
            # out2 = sitk.GetImageFromArray(img_array.astype(np.int16))
            # sitk.WriteImage(out2, se2output)

        i = i + 1
        if i % 10 == 0:
            print('numbers:', i)
    f1.close()
    print('finished:!')

def revert_z(path):
    # path = "/media/bit301/data/yml/data/p33/internal/xy/xy3/dis/jc/PA7"  #p33为备份数据
    # path = "/media/bit301/data/yml/data/p2_nii/internal/xm"
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if ".nii.gz" in path:
                # filee = root.split("/")[-1]
                path_list.append(path)
    ij = 0
    path_list = natsorted(path_list)
    for se2output in path_list:
        read = sitk.ReadImage(se2output, sitk.sitkInt16)  #
        img_arrayy = sitk.GetArrayFromImage(read)[::-1, :, :]

        out2 = sitk.GetImageFromArray(img_arrayy.astype(np.int16))
        sitk.WriteImage(out2, se2output)
        ij = ij + 1
        if ij % 10 == 0:
            print('numbers:', ij)
    print('finished:!')

#pp3数据中存在一些首尾颠倒情况，进行修正
def revert_list():
    path = "/media/bit301/data/yml/data/p33"  #p33为备份数据
    # path = "/media/bit301/data/yml/data/p2_nii/internal/xm"
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "2.nii.gz" in path:
                # filee = root.split("/")[-1]
                path_list.append(path)
    file = "revert_list.txt"
    f1 = open(file, "w")  # 564
    i = 0
    path_list = natsorted(path_list)
    for se2output in path_list:
        read = sitk.ReadImage(se2output, sitk.sitkInt16)
        img_array = sitk.GetArrayFromImage(read)
        img_array[img_array > 0] = 1  # 转换为布尔类型
        non_zero_positions = np.where(img_array != 0)
        min_coords1 = [np.min(pos) for pos in non_zero_positions]
        max_coords1 = [np.max(pos) for pos in non_zero_positions]
        data1=img_array[min_coords1[0]:min_coords1[0]+50,:,:].sum()
        data2 = img_array[max_coords1[0]-50:max_coords1[0], :, :].sum()
        if data1>data2:
            f1.writelines(se2output + "\n")
            path= os.path.dirname(se2output)
            revert_z(path)
            i = i + 1
            continue
        # img_label, num = measure.label(img_array[min_coords1[0]+50,:,:], connectivity=2, return_num=True)
        # props = measure.regionprops(img_label)
        # props_sorted1 = sorted(props, key=lambda x: x.area, reverse=True)
        # if len(props_sorted1) > 1 and props_sorted1[1].area > 300:
        #     path= os.path.dirname(se2output)
        #     # revert_z(path)
        #     f1.writelines(se2output + "\n")
        i = i + 1
        if i % 10 == 0:
            print('numbers:', i)
    f1.close()
    print('finished:!')

def copy_and_paste():
    # path = "/media/bit301/backup/use/p3"  #
    path = "/media/bit301/data/yml/data/p3/"  # p33为备份数据
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "2.nii.gz" in path:
                path_list.append(path)
    # out="/media/bit301/data/yml/data/p3"
    ii = 0
    for se2output in path_list:
        out2 = se2output.replace("p3", "p3_backup_mask2")
        target_directory = os.path.dirname(out2)
        os.makedirs(target_directory, exist_ok=True)
        shutil.copy(se2output, out2)
        se3output=se2output.replace("2.nii.gz", "3.nii.gz")

        # read3 = sitk.ReadImage(se3output, sitk.sitkInt16)  #
        # img_array3 = sitk.GetArrayFromImage(read3)  # 假腔
        # img_array3[img_array3 ==4] =2  # have some caclified。只区分假腔血液，不管假腔的血栓
        # img_array3[img_array3 == 5] = 4
        # out3 = sitk.GetImageFromArray(img_array3.astype(np.int16))
        # sitk.WriteImage(out3, se3output)

        out3=out2.replace("2.nii.gz", "3.nii.gz")
        shutil.copy(se3output, out3)

        # 打印进度
        ii += 1
        if ii % 10 == 0:
            print(f'Processed {ii} files.')
    print("All files processed.")

def copy_and_paste2():
    # path = "/media/bit301/backup/use/p3" #
    path = "/media/bit301/data/yml/data/p3/"  # p33为备份数据
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "0.nii.gz" in path:
                path_list.append(path)
    # out="/media/bit301/data/yml/data/p3"
    ii = 0
    for se2output in path_list:
        out2 = se2output.replace("p3", "p3s")
        target_directory = os.path.dirname(out2)
        os.makedirs(target_directory, exist_ok=True)
        shutil.copy(se2output, out2)
        se3output=se2output.replace("0.nii.gz", "3.nii.gz")
        out3=out2.replace("0.nii.gz", "3.nii.gz")
        shutil.copy(se3output, out3)

        # 打印进度
        ii += 1
        if ii % 10 == 0:
            print(f'Processed {ii} files.')
    print("All files processed.")

def remove_cta():
    path = "/media/bit301/data/yml/data/pp3"  # p33为备份数据
    # path = "/media/bit301/data/yml/data/p2_nii/internal/xm"
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "2.nii.gz" in path:
                path_list.append(path)
    i = 0
    for se2output in path_list:
        path = se2output.replace("2.nii.gz", "1.nii.gz")
        os.remove(path)

        # shutil.copy(path, path.replace("p33","p3"))
        i = i + 1
        if i % 10 == 0:
            print('numbers:', i)
    print("finished!")

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

#区分真假腔
def process_mask1():
    #0.nii.gz-NCCT;
    #1.nii.gz-CTA;
    #p3_orig
    # 2.nii.gz-Mask; 1-管腔，2-钙化，3-非钙化，假腔-4
    ## 3.nii.gz-lesion;1正常，2-动脉瘤，3-夹层(2+1)，动脉瘤+夹层-2（动脉瘤优先级），4-狭窄(4+1)，动脉瘤+狭窄，夹层+狭窄归为一类
    path = "/media/bit301/data/yml/data/p3_orig"  #p33为备份数据
    # path = "/media/bit301/data/yml/data/p33/external/cq/dis/xz/PA232"
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "2.nii.gz" in path:
                # filee = root.split("/")[-1]
                path_list.append(path)
    # file = "dml_jc_xz_list.txt"
    # with open(file, 'r') as f:
    #     for line in f:
    #         line=line.strip()
    #         # 去除每行的换行符，并添加到列表中
    #         path=line.replace("p3","p3_orig").replace("3.nii.gz","2.nii.gz")
    #         path_list.append(path)

    i = 0
    for se2output in path_list:
        # if i<310:
        #     i=i+1
        #     continue
        # se2output="/media/bit301/data/yml/data/p3/external/cq/dis/jc/PA20/2.nii.gz"
        read = sitk.ReadImage(se2output, sitk.sitkInt16)  #
        img_array2 = sitk.GetArrayFromImage(read)
        img_array=copy.deepcopy(img_array2)
        img_array[img_array > 0] = 1
        img_array = remove_small_volums(img_array)#去掉冗余  remove_small_volums
        img_array33 = copy.deepcopy(img_array)
        img_array2=img_array2*img_array
        img_array2[img_array2>3]=3#血栓和软斑块归纳为非钙化
        out2 = sitk.GetImageFromArray(img_array2.astype(np.int16))#必须要，要不然没有夹层，这里的数据不会被保存
        se2outputt = se2output.replace("p3_orig", "p3")
        sitk.WriteImage(out2, se2outputt)

        se3output = se2output.replace("2.nii.gz", "3.nii.gz")
        se4output = se2output.replace("2.nii.gz", "4.nii.gz")
        se5output = se2output.replace("2.nii.gz", "5.nii.gz")
        se6output = se2output.replace("2.nii.gz", "6.nii.gz")
        if os.path.exists(se3output):
            read3 = sitk.ReadImage(se3output, sitk.sitkInt16)  #
            img_array3 = sitk.GetArrayFromImage(read3)#假腔
            img_array3[img_array3>1]=0#have some caclified。只区分假腔血液，不管假腔的血栓
            img_array2=img_array2+img_array3*3*img_array#1+3=4
            img_array2[img_array2 > 4] = 4#
            out2 = sitk.GetImageFromArray(img_array2.astype(np.int16))
            # os.remove(se2output)
            se2outputt=se2output.replace("p3_orig","p3")
            sitk.WriteImage(out2, se2outputt)
            # os.remove(se3output)
        if i<1000:
            i=i+1
            if i % 10 == 0:
                print('numbers:', i)
            continue

        if os.path.exists(se4output):#dml
            read4 = sitk.ReadImage(se4output, sitk.sitkInt16)
            img_array4 = sitk.GetArrayFromImage(read4)*img_array
            img_array4[img_array4>0]=1
            img_array33=img_array33+img_array4
        if os.path.exists(se6output):  # jc    用于病变数据合成（对应于3.nii.gz）
            read6 = sitk.ReadImage(se6output, sitk.sitkInt16)
            img_array6 = sitk.GetArrayFromImage(read6) * img_array
            img_array6[img_array6 > 0] = 2
            img_array33 = img_array33 + img_array6
            # os.remove(se4output)
            img_array33[img_array33>3]=2 #将dml+jc归纳为dml
        if os.path.exists(se5output):#xz
            read5 = sitk.ReadImage(se5output, sitk.sitkInt16)
            img_array5 = sitk.GetArrayFromImage(read5)*img_array
            img_array5[img_array5>0]=3
            img_array33=img_array33+img_array5#
            # os.remove(se5output)
        img_array33[img_array33 > 4] = 4
        out3 = sitk.GetImageFromArray(img_array33.astype(np.int16))
        se3output=se3output.replace("p3_orig", "p3")
        sitk.WriteImage(out3, se3output)
        i = i + 1
        if i % 10 == 0:
            print('numbers:', i)

#不区分真假腔，按照成分区分，血液、钙化、非钙化（血栓+软斑块）、内膜片
def process_mask2():
    #0.nii.gz-NCCT;
    #1.nii.gz-CTA;
    #p3_orig
    # 2.nii.gz-Mask; 1-血液、2-钙化、3-非钙化（血栓+软斑块）、4-内膜片
    ## 3.nii.gz-lesion;1正常，2-动脉瘤，3-夹层(2+1)，动脉瘤+夹层-2（动脉瘤优先级），4-狭窄(4+1)，动脉瘤+狭窄，夹层+狭窄归为一类
    path = "/media/bit301/data/yml/data/p3_orig"  #p33为备份数据
    # path = "/media/bit301/data/yml/data/p3_orig/external/cq/dis/dml/PA1"
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "2.nii.gz" in path:
                # filee = root.split("/")[-1]
                path_list.append(path)
    # file = "dml_jc_xz_list.txt"
    # with open(file, 'r') as f:
    #     for line in f:
    #         line=line.strip()
    #         # 去除每行的换行符，并添加到列表中
    #         path=line.replace("p3","p3_orig").replace("3.nii.gz","2.nii.gz")
    #         path_list.append(path)

    i = 0
    for se2output in path_list:
        # if i<310:
        #     i=i+1
        #     continue
        # se2output="/media/bit301/data/yml/data/p3/external/cq/dis/jc/PA20/2.nii.gz"
        read = sitk.ReadImage(se2output, sitk.sitkInt16)  #
        img_array2 = sitk.GetArrayFromImage(read)
        img_array=copy.deepcopy(img_array2)
        img_array[img_array > 0] = 1
        img_array = remove_small_volums(img_array)#去掉冗余  remove_small_volums
        img_array33 = copy.deepcopy(img_array)
        img_array2=img_array2*img_array
        img_array22=copy.deepcopy(img_array2)
        # img_array2[img_array2>3]=3#血栓和软斑块归纳为非钙化
        out2 = sitk.GetImageFromArray(img_array2.astype(np.int16))#必须要，要不然没有夹层，这里的数据不会被保存
        se2outputt = se2output.replace("p3_orig", "p3")
        sitk.WriteImage(out2, se2outputt)

        se3output = se2output.replace("2.nii.gz", "3.nii.gz")
        se4output = se2output.replace("2.nii.gz", "4.nii.gz")
        se5output = se2output.replace("2.nii.gz", "5.nii.gz")
        se6output = se2output.replace("2.nii.gz", "6.nii.gz")
        if os.path.exists(se3output):
            read3 = sitk.ReadImage(se3output, sitk.sitkInt16)  #
            img_array3 = sitk.GetArrayFromImage(read3)#假腔
            img_array3[img_array3>0]=1#have some caclified。只区分假腔血液，不管假腔的血栓

            img_array22[img_array22 > 3] = 0
            img_array22[img_array22 > 0] = 1 #img_array22为总的主动脉
            aa=img_array22 - img_array3
            true_lumen_mask = np.where(aa> 0, 1, 0)#img_array3为假腔
            # true_lumen_mask = remove_small_volums(true_lumen_mask)
            true_lumen_dilated = dilate_mask(true_lumen_mask, 2)

            false_lumen_dilated=dilate_mask(img_array3,2)

            flip = true_lumen_dilated & false_lumen_dilated
            flip = flip.astype(np.uint8)
            # flip=img_array3-img_array22
            # flip=np.where(flip>0,1,0)
            img_array2=img_array2+flip*5
            img_array2[img_array2 >5]= 5#
            out2 = sitk.GetImageFromArray(img_array2.astype(np.int16))
            # os.remove(se2output)
            se2outputt=se2output.replace("p3_orig","p3")
            sitk.WriteImage(out2, se2outputt)
            # os.remove(se3output)
        if i<1000:
            i=i+1
            if i % 10 == 0:
                print('numbers:', i)
            continue

        if os.path.exists(se4output):#dml
            read4 = sitk.ReadImage(se4output, sitk.sitkInt16)
            img_array4 = sitk.GetArrayFromImage(read4)*img_array
            img_array4[img_array4>0]=1
            img_array33=img_array33+img_array4
        if os.path.exists(se6output):  # jc    用于病变数据合成（对应于3.nii.gz）
            read6 = sitk.ReadImage(se6output, sitk.sitkInt16)
            img_array6 = sitk.GetArrayFromImage(read6) * img_array
            img_array6[img_array6 > 0] = 2
            img_array33 = img_array33 + img_array6
            # os.remove(se4output)
            img_array33[img_array33>3]=2 #将dml+jc归纳为dml
        if os.path.exists(se5output):#xz
            read5 = sitk.ReadImage(se5output, sitk.sitkInt16)
            img_array5 = sitk.GetArrayFromImage(read5)*img_array
            img_array5[img_array5>0]=3
            img_array33=img_array33+img_array5#
            # os.remove(se5output)
        img_array33[img_array33 > 4] = 4
        out3 = sitk.GetImageFromArray(img_array33.astype(np.int16))
        se3output=se3output.replace("p3_orig", "p3")
        sitk.WriteImage(out3, se3output)
        i = i + 1
        if i % 10 == 0:
            print('numbers:', i)

def rmask_plaque():
    path = "/media/bit301/data/yml/data/p3" #
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "2.nii.gz" in  path:
                path_list.append(path)

    # file = "dml_jc_xz_list.txt"
    # with open(file, 'r') as f:
    #     for line in f:
    #         line=line.strip()
    #         # 去除每行的换行符，并添加到列表中
    #         path=line.replace("3.nii.gz","2.nii.gz")
    #         path_list.append(path)
    i = 0
    for se2output in path_list:
        # se2output="/media/bit301/data/yml/data/p3/internal/xy/xy2/dis/dml/PA63_zj/SE0/2.nii.gz"
        # if i<704:
        #     i=i+1
        #     continue
        # read = sitk.ReadImage(se2output, sitk.sitkInt16)  #
        # img_array2 = sitk.GetArrayFromImage(read)
        #
        # se0output = se2output.replace("2.nii.gz", "0.nii.gz")
        # read = sitk.ReadImage(se0output, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        # img_array0 = sitk.GetArrayFromImage(read)
        # img_array_w0 = to_windowdata(img_array0, 130, 10)
        # img_array_w0 = np.where(img_array_w0 > 0, 1, 0)  # 支架与钙化斑块

        se0output = se2output.replace("2.nii.gz", "0.nii.gz")
        read = sitk.ReadImage(se0output, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        img_array = sitk.GetArrayFromImage(read)

        mask_read = sitk.ReadImage(se2output, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
        mask_array = sitk.GetArrayFromImage(mask_read)
        mask2 = copy.deepcopy(mask_array)
        # process
        mask2[mask2 > 0] = 1
        # selem = ball(5)  # 半径为1的球形结构元素
        # mask2 = binary_dilation(mask2, selem)#有些可能被遗漏的也给找出来
        # mask2 = binary_closing(mask2)

        img_arrayy = img_array * mask2
        img_array2 = np.where(img_arrayy > 130, 1, 0)#plaque 可能包括了金属支架
        inverted_mask = 1 - img_array2
        mask_array[mask_array == 2] = 1  # 让钙化也先划分为管腔。注意4为
        mask_array = mask_array * inverted_mask  # 腾出钙化与金属支架区域
        binary_array = np.zeros_like(img_arrayy)
        binary_array[(img_arrayy >130) & (img_arrayy <= 2000)] = 1
        mask_array = mask_array + binary_array * 2  # 钙化为2
        mask_array[mask_array >4] = 4
        # 2.nii.gz-Mask; 1-管腔，2-钙化，3-非钙化，假腔-4
        # 3.nii.gz-lesion;1-动脉瘤，2-假腔，5-狭窄。动脉瘤+假腔-3，动脉瘤+狭窄-6，假腔+狭窄-7 #lesion
        if mask_array.max() > 4:
            print(se2output)
        label = sitk.GetImageFromArray(mask_array.astype(np.int16))
        sitk.WriteImage(label, se2output)
        i = i + 1
        if i % 10 == 0:
            print('numbers:', i)
    print('finished:!')

def reprocess2final():#将dml+jc、xz归为一类，dml+jc数量太少了，没必要单独归类。
    path = "/media/bit301/data/yml/data/p3/"
    # path = "/media/bit301/backup/use/pp3"
    # path = "/media/bit301/data/yml/data/p3a/mix"  #p33为备份数据
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "2.nii.gz" in path:#3.nii.gz
                # filee = root.split("/")[-1]
                path_list.append(path)
    i = 0
    for se3output in path_list:
        read3 = sitk.ReadImage(se3output, sitk.sitkInt16)  #
        img_array3 = sitk.GetArrayFromImage(read3)  # 假腔
        img_array3[img_array3 > 4] = 4  # have some caclified。只区分假腔血液，不管假腔的血栓
        out3 = sitk.GetImageFromArray(img_array3.astype(np.int16))
        sitk.WriteImage(out3, se3output)
        i = i + 1
        if i % 10 == 0:
            print('numbers:', i)

def statis():
    path = "/media/bit301/data/yml/data/p3/internal"  #p33为备份数据
    # path = "/media/bit301/backup/use/p3"
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "3.nii.gz" in path and "hnnk" not in path:
                # filee = root.split("/")[-1]
                path_list.append(path)
    i = 0
    category_counts = {}
    for se3output in path_list:
        read3 = sitk.ReadImage(se3output, sitk.sitkInt16)  #
        img_array3 = sitk.GetArrayFromImage(read3)  # 假腔
        # 获取图像的空间分辨率
        spacing = read3.GetSpacing()
        volume_per_pixel = np.prod(spacing)

        # 统计各个类别的数量
        unique_elements, counts_elements = np.unique(img_array3, return_counts=True)

        # 对每个类别进行判断
        for category in unique_elements:
            # 跳过背景像素（假设背景像素值为0）
            if category == 0:
                continue

            # 计算该类别的体积
            category_mask = img_array3 == category
            category_volume = np.sum(category_mask) * volume_per_pixel

            # 判断体积是否大于5000
            if category_volume > 500:
                # if category==4:
                #     print(se3output)
                # 更新类别计数器
                if category in category_counts:
                    category_counts[category] += 1
                else:
                    category_counts[category] = 1

        # # 打印统计结果
        # print(f"Processed file: {se3output}")
        # print(f"Unique categories and their counts for {se3output}:")
        # for category, count in zip(unique_elements, counts_elements):
        #     print(f"Category {category}: {count} pixels")
        i = i + 1
        if i % 10 == 0:
            print('numbers:', i)
    # 打印最终统计结果
    print("\nFinal category counts with volume greater than 5000:")
    for category, count in category_counts.items():
        print(f"Category {category}: {count} times")

def dilate_mask(mask, radius):
    """对mask进行膨胀处理"""
    selem = ball(radius)  # 半径为1的球形结构元素
    # selem = cube(3)  # 如果需要立方体结构元素，可以使用cube
    dilated_mask = binary_dilation(mask, selem)

    # structure = np.zeros((radius * 2 + 1,) * mask.ndim)
    # center = tuple(slice(radius, radius + 1) for _ in range(mask.ndim))
    # structure[center] = 1
    # dilated_mask = binary_dilation(mask, structure=structure)

    return dilated_mask

def remove_small_points(img, threshold_point=20):  # 80
    img_label, num = measure.label(img, return_num=True,connectivity=2)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等
    resMatrix = np.zeros(img_label.shape)
    dia = 6  # 60
    for i in range(1, len(props)):
        are = props[i].area
        if are > threshold_point:
            if are < 3400:
                dia = int(np.sqrt(are) / 2)
                tmp = (img_label == i + 1).astype(np.uint8)
                kernel = np.ones((dia, dia), np.uint8)
                tmp = cv2.dilate(tmp, kernel)  # [-1 1]
                # x, y = np.nonzero(tmp)
                # tmp[x[0] - dia:x[-1] + dia, y[0] - dia:y[-1] + dia] = 1  # resize
                resMatrix += tmp  # 组合所有符合条件的连通域
    resMatrix *= 1
    resMatrix[resMatrix > 1] = 1  #
    return resMatrix

def registor(patha, pathb):
    # 本例子展示的是将CTA配准至CT
    fixed_image = itk.imread(patha, itk.F)  # 这里itk.F是itk的一种图像类型，需要加itk.F代码才能在该类型精度下进行配准
    moving_image = itk.imread(pathb, itk.F)

    # 配准参数：rigid+affine+bspline
    # 注：只使用rigid也可以，加上affine+bspline允许一些弹性变形变化，会配得更好
    # 但在某些情况下使用affine+bspline会使得moving图像变形严重
    # parameter_object = itk.ParameterObject.New()
    parameter_object = itk.ParameterObject()
    parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(parameter_map_rigid)
    abc = parameter_object
    try:
        parameter_map_affine = abc.GetDefaultParameterMap('affine')
        abc.AddParameterMap(parameter_map_affine)
        parameter_object = abc

        parameter_map_bspline = abc.GetDefaultParameterMap('bspline')
        abc.AddParameterMap(parameter_map_bspline)
        parameter_object = abc
    except:
        print("registor have warning!")

    # result_registered_image就是配准之后的ct图像，可以直接输出保存
    # 同时配准结束后还会输出result_transform_parameters，是配准的变换参数，可以用它来变换标签
    result_registered_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image, parameter_object=parameter_object)
    itk.imwrite(result_registered_image, pathb)

# dicom to nii
def dcm2nii_sitk(path_read, path_save):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(path_read)
    N = len(seriesIDs)
    lens = np.zeros([N])
    for i in range(N):
        dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[i])
        lens[i] = len(dicom_names)
    N_MAX = np.argmax(lens)
    dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[N_MAX])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, path_save)
    # shutil.rmtree(path_read)

def resize_image_with_crop_or_pad(image, img_size=(512, 512), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """
    h, w, d = image.shape
    img_size = (img_size[0], img_size[1], d)
    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), 'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    # return np.pad(image[slicer], to_padding, **kwargs)
    return np.pad(image[tuple(slicer)], to_padding, **kwargs)

# if i<310:
#     i=i+1
#     continue
# make the list of nnunet to 3DUxnet
def json_to_list():
    import json
    five_fold = "/home/wangtao/yml/project/python39/p3/nnUNet/DATASET/nnUNet_preprocessed/Dataset301_Aorta/splits_final.json"
    with open(five_fold, 'r', encoding='UTF-8') as f:
        files = json.load(f)

    root_and_json = "/home/wangtao/yml/project/python39/p3/nnUNet/DATASET/nnUNet_raw/Dataset301_Aorta/train.json"
    for i in range(len(files)):
        train_files = files[i]['train']
        val_files = files[i]['val']
        train = "train" + str(i) + ".txt"
        val = "val" + str(i) + ".txt"
        f1 = open(train, "w")  # 564
        f2 = open(val, "w")  # 188

        with open(root_and_json, encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:  # 到 EOF，返回空字符串，则终止循环
                    break
                js = json.loads(line)
                for k, v in js.items():
                    if k in train_files:
                        f1.writelines(v + "\n")
                    else:
                        f2.writelines(v + "\n")
        f1.close()
        f2.close()

# get real input list
def get_files_list():
    path = "/media/bit301/data/yml/data/p3/external/cq/"  # CT_CTA internal external
    files_list = "cq.txt"
    f1 = open(files_list, "w")  # 564
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        # if "xm" in root or "p1/dmzyyh" in root or "hnnk" in root:
        #     if '0.nii.gz' in files:
        #         path_list.append(root.split("p2_nii")[1])
        for file in files:
            path = os.path.join(root, file)
            # if "xm" in path and "0.nii.gz" in path:
            if "0.nii.gz" in path:
                path_list.append(path)
    path_list = natsorted(path_list)
    for j in range(len(path_list)):
        f1.writelines(path_list[j] + "\n")
    # ff.close()
    f1.close()  # 关

# 划分训练-验证-测试数据
def split():
    # path="../../../data/p1_nii/dis/xz/"#CT_CTA disease
    # path = "../../../data/p2_nii/"  # CT_CTA disease p1_m
    path = "../../../data/p3/internal/"  #
    path_list = []
    aug_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        if "0.nii.gz" in files:
            path_list.append(root)

        # if "PA" in root and "hnnk" not in root:#将hnnk作为外部验证集
        #     if "0.nii.gz" not in files:
        #         continue
        #     if "aug" not in root:
        #         if "nor" in root or "dmzyyh" in root:
        #             path_list.append(root)
        #     else:
        #         aug_list.append(root)

    random.seed(0)
    random.shuffle(path_list)
    cross = 5
    # ff = open("p1_data.txt", "w")#564
    a = int(len(path_list) / cross)
    for i in range(cross):
        if i == 1:
            break
        train = "trainn" + str(i) + ".txt"
        # val = "abd_val" + str(i) + ".txt"
        test = "testn" + str(i) + ".txt"
        f1 = open(train, "w")  # 564
        f2 = open(test, "w")  # 188
        for j in range(len(path_list)):
            input_files = os.listdir(path_list[j])  # SE0 file list
            file_path = os.path.join(path_list[j], "0.nii.gz")
            # ff.writelines(file_path)
            if j >= (cross - 1 - i) * a and j < (cross - i) * a:
                f2.writelines(file_path + "\n")
            else:
                f1.writelines(file_path + "\n")

        for j in range(len(aug_list)):
            file_path = os.path.join(aug_list[j], "0.nii.gz")
            f1.writelines(file_path + "\n")
    # ff.close()
    f1.close()  # 关
    f2.close()

#用于级联模型
def process_stage1_mask():
    # path = "/media/bit301/backup/use/p3"  #
    path = "/media/bit301/data/yml/project/python310/p3/DATASET12/nnUNet_raw/Dataset301_Aorta/imagesTr"  # p33为备份数据
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "_0001.nii.gz" in path:
                path_list.append(path)
    # out="/media/bit301/data/yml/data/p3"
    print(len(path_list))
    ii = 0
    for se0output in path_list:
        read = sitk.ReadImage(se0output, sitk.sitkInt16)  #
        img_array = sitk.GetArrayFromImage(read)  # 假腔
        img_array= img_array*100  # have some caclified。只区分假腔血液，不管假腔的血栓
        out = sitk.GetImageFromArray(img_array.astype(np.int16))
        sitk.WriteImage(out, se0output)
        ii += 1
        if ii % 10 == 0:
            print(f'Processed {ii} files.')
    print("All files processed.")

def crop(ncct_npy,seg_npy2,seg_npy3,preseg):
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

    ncct_npy = ncct_npy[:, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
    seg_npy2= seg_npy2[:, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
    seg_npy3= seg_npy3[:, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
    preseg= preseg[:, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
    return ncct_npy,seg_npy2,seg_npy3,preseg

def crop_test_by_mask():#disscard
    path_list = []
    # inputs = input_dir  # 输入目录
    inputs = "/media/bit301/data/yml/data/p3/external/"  # pre
    for root, dirs, files in os.walk(inputs, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "0.nii.gz" in path:
                path_list.append(path)

    ii = 0
    for path in path_list:
        read = sitk.ReadImage(path, sitk.sitkInt16)
        ncct = sitk.GetArrayFromImage(read)
        path2=path.replace("0.nii.gz", "2.nii.gz")
        read = sitk.ReadImage(path2, sitk.sitkInt16)
        mask2 = sitk.GetArrayFromImage(read)

        path3=path.replace("0.nii.gz", "3.nii.gz")
        read = sitk.ReadImage(path3, sitk.sitkInt16)
        mask3 = sitk.GetArrayFromImage(read)

        path=path.replace("p3", "p3_crop_train")
        out_put=path.split("0.nii.gz")[0]
        if not os.path.isdir(out_put):
            os.makedirs(out_put)
        ncct,mask2,mask3,_=crop(ncct,mask2,mask3,mask3)#preseg
        ncct = sitk.GetImageFromArray(ncct.astype(np.int16))
        sitk.WriteImage(ncct, path)

        mask2 = sitk.GetImageFromArray(mask2.astype(np.int16))
        sitk.WriteImage(mask2, path.replace("0.nii.gz", "2.nii.gz"))

        mask3 = sitk.GetImageFromArray(mask3.astype(np.int16))
        sitk.WriteImage(mask3, path.replace("0.nii.gz", "3.nii.gz"))
        # 打印进度
        ii += 1
        if ii % 10 == 0:
            print(f'Processed {ii} files.')
    print("All files processed.")

def crop_by_preseg():
    path_list = []
    # inputs = input_dir  # 输入目录
    inputs = "/media/bit301/data/yml/data/p3/external"  # pre internal/
    for root, dirs, files in os.walk(inputs, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "0.nii.gz" in path:
                path_list.append(path)

    ii = 0
    for path in path_list:
        # if ii<957:
        #     ii=ii+1
        #     continue
        path_preseg = path.replace("p3","test2a/p3t/nnUNetTrainerMaCNN2").replace("0.nii.gz", "22.nii.gz")
        read = sitk.ReadImage(path_preseg, sitk.sitkInt16)
        preseg = sitk.GetArrayFromImage(read)

        read = sitk.ReadImage(path, sitk.sitkInt16)
        ncct = sitk.GetArrayFromImage(read)
        path2=path.replace("0.nii.gz", "2.nii.gz")
        read = sitk.ReadImage(path2, sitk.sitkInt16)
        mask2 = sitk.GetArrayFromImage(read)

        path3=path.replace("0.nii.gz", "3.nii.gz")
        read = sitk.ReadImage(path3, sitk.sitkInt16)
        mask3 = sitk.GetArrayFromImage(read)

        path=path.replace("p3", "p3_crop_preseg")
        out_put=path.split("0.nii.gz")[0]
        if not os.path.isdir(out_put):
            os.makedirs(out_put)
        ncct,mask2,mask3,preseg=crop(ncct,mask2,mask3,preseg)#preseg
        ncct = sitk.GetImageFromArray(ncct.astype(np.int16))
        sitk.WriteImage(ncct, path)

        mask2 = sitk.GetImageFromArray(mask2.astype(np.int16))
        sitk.WriteImage(mask2, path.replace("0.nii.gz", "2.nii.gz"))

        mask3 = sitk.GetImageFromArray(mask3.astype(np.int16))
        sitk.WriteImage(mask3, path.replace("0.nii.gz", "3.nii.gz"))

        preseg = sitk.GetImageFromArray(preseg.astype(np.int16))
        sitk.WriteImage(preseg, path.replace("0.nii.gz", "22.nii.gz"))
        # 打印进度
        ii += 1
        if ii % 10 == 0:
            print(f'Processed {ii} files.')
    print("All files processed.")

if __name__ == '__main__':
    # rmake_mask()#p2 task：整合管腔，钙化板块与血栓
    # dml_jc_xz_list()#找到之前将dml+jc、xz归纳为一类的数据。dml+jc归纳不是很合理
    # revert_list()#处理拉直数据
    # remove_cta()
    # process_mask1()#p3 task: 整合病灶mask
    # process_mask2()#p3 task: 整合病灶mask
    # copy_and_paste()#
    # rmask_plaque()#
    # reprocess2final()##将dml+jc、xz归为一类
    # crop_train()

    # crop_by_preseg()
    copy_and_paste2()
    # statis()
    # get_files_list()
    # process_stage1_mask()
    a = 1
