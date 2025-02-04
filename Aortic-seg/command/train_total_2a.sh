python /media/bit301/data/yml/project/python310/p3/LightMUNet/nnunetv2/dataset_conversion/Datasets301_Aorta_p3_2a.py
nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity #noet：修改size  尽可能的大
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 301 3d_lowres 0 -tr nnUNetTrainerMaCNN   #mamba+Mednext+concaten