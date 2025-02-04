Guidelines
1-install: details can be see[https://github.com/MIC-DKFZ/nnUNet]
2-data process:use test/data_process.py (1)resample and convert DICOM data to nii.gz file 
(2)make annotation and process,including mask Silhouette and multi-category mask merge 

3-training: 
(1)set correct path：Aortic-seg/nnunetv2/path.py # 
(2)data prepare：Aortic-seg/nnunetv2/dataset_conversion/Datasets301_Aorta.py #read the code carefully and figure out format of the output file
(3)data checking and training preparation:execute a command "nnUNetv2_plan_and_preprocess -d 301 --verify_dataset_integrity" under the path nnunet-v2 
(3)change the root to the Aortic-seg file and run bash ./command/train_total_1a.sh

4-inference: 
(1)python ./nnunetv2/inference/predict_from_raw_data_1.py 0 nnUNetTrainerCNNSCA  or run bash ./command/test_total_1b.sh
(2)reprocess: use process/data_process.py run postpossess function remove the small volume.
