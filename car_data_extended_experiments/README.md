Download the dataset from
(http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)
and put the folder `compcars` in the `data` folder.
Make sure to follow the steps in https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt.
For Mac or Linux, you may need to zip and then unzip a file.

1. Run 
	test1_without_feature_extraction.ipynb
	test1_without_finetune.ipynb
	to obtain experimental data.
2. Run
	test2_data_splitting_without_feature_extraction.ipynb
	test2_data_splitting_without_finetuning.ipynb
	to split datasets into training sets and test sets.
3. Run 
	test4_training_AE_AL_without_finetuning.ipynb
	test4_training_oracle_without_feature_extraction.ipynb
	to train models
4. Move AE_AL_result_dic.p from the codes under the folder "car_data." Run
	test5_Plot_AUC.ipynb
	to generate the plot