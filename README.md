# 02456_project_audioset_attention
Deep learning, DTU 02456. Autumn 2018.<br>

Final project working with Google AudioSet inspired in DCASE2018 Challange (Task 4).<br>
See project documentation: https://drive.google.com/drive/folders/1wYKw9w9nIngUnUSXUytIEmZBjXbPZIWh<br><br>


## !!! BEFORE CLONING THIS REPOSITORY !!!
We recommend you to create a folder in your computer (e.g. 'your_project') and then clone it inside this folder.<br><br>


1. Create a folder 'your_project'

		your_project


2. Clone this repository inside the folder

		your_project/
		    |_ 02456_project_audioset_attention (this repository)


3. Run 'setup.sh' (sh setup.sh) to download and clone the files (it might take a while). It will automatically do the following:

	3.1. You should download the previous link in the parent folder of this repository. It is called 'packed_features.zip'

	3.2. Clone 'audioset_classification' from https://github.com/qiuqiangkong/audioset_classification

	3.3. Clone 'dcase18_baseline' from https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/ We are interested in the metadata of Task4 because it is strongly labeled data

	3.4. Generate dataset

		your_project/
		    |_ 02456_project_audioset_attention (this repository)
		    |_ audioset_classification
		    |_ dcase2018_baseline
		    |_ packed_features
		    |_ dataset -----------------------> (might be included, not sure yet)


4. You are now ready to run the... ('runme.sh')

5. If you want to see some of our results open the Jupyter notebook 'Results.ipynb' and run it.



### Practical information

*If at some point you need to <b>download the dataset<\b> again but the setup is already done you can just run the shell script 'data_generator.sh'*


*If you want to <b>remove the folders<\b> created by during the setup you can just run the shell script 'remove_all.sh'. Then it will look like this again:*

	your_project/
	    |_ 02456_project_audioset_attention (this repository)






### Other sources of information:
Repositories:
https://github.com/DTUComputeCognitiveSystems/AI_playground/blob/master/notebooks/experiments/Sound%20Demo%203%20-%20Multi-label%20classifier%20pretrained%20on%20audioset.ipynb
.
.
.
