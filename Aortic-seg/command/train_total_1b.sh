python /media/bit301/data/yml/project/python310/p3/LightMUNet/nnunetv2/dataset_conversion/Datasets301_Aorta_p3_1b.py
nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity #noet：修改size  尽可能的大

python ./nnunetv2/inference/predict_from_raw_data_1.py 0 nnUNetTrainerCNNSCA
python  media/bit301/data/yml/project/python310/p3/process/aortic_index_combine.py