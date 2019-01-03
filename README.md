# 02456_project_audioset_attention
Deep learning, DTU 02456. Autumn 2018.<br>

The aim of this prokect is to test the hypothesis that: <i>training with weakly labeled outputs to predict strongly labeled outputs.</i><br>

<ul>
	<li><i>Weakly labeled output:</i> only say if a label is present in the whole audio clip or not</li>
	<li><i>Strongly labeled output:</i> per each label it says when, in time, each label appear in the audio clip (with a precision of 1s).</li>
</ul>

Final project working with Google AudioSet inspired in DCASE2018 Challange (Task 4).<br>
See project documentation: https://drive.google.com/drive/folders/1wYKw9w9nIngUnUSXUytIEmZBjXbPZIWh<br><br>


### !!! BEFORE CLONING THIS REPOSITORY !!!
We recommend you to create a folder in your computer (e.g. 'your_project') and then clone it inside this folder.<br>

1. Create a folder `your_project`

		your_project


2. Clone this repository inside the folder

		your_project/
		    |_ 02456_project_audioset_attention (this repository)


3. Run 'setup.sh' to download and clone the files (it might take a while). It will automatically do the following:

	3.1. Uncomment some lines to download the `packed_features.zip` from Google AudioSet project.

	3.2. Clone `audioset_classification` from https://github.com/qiuqiangkong/audioset_classification

	3.3. Clone `dcase18_baseline` from https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/ We are interested in the metadata of Task4 because it is strongly labeled data

	3.4. Create a symbolik link of `main_3.py` and `core_3.py` in the 'audioset_classification' repository. This are the files we modified to train the networks to be able to test our hypothesis.

	3.4. The directories should look like this:

		your_project/
		    |_ 02456_project_audioset_attention (this repository)
		    |_ audioset_classification
		    |_ dcase2018_baseline
		    |_ packed_features
<br>
### Run it and results
1. If you want to <b>train</b> the network you are now ready to run the `runme.sh` file. It was already done and we provide you the data from the trained models in the `data` folder of this repository.

2. The <b>results</b> of this project are presented in the Jupyter notebook `Results.ipynb`.

<br>

### Practical information

*If at some point you need to <b>download the dataset</b> again but the setup is already done you can just run the shell script 'data_generator.sh'*


*If you want to <b>remove the folders</b> created by during the setup you can just run the shell script 'remove_all.sh'. Then it will look like this again:*

	your_project/
	    |_ 02456_project_audioset_attention (this repository)

<br>

### Other sources of information:
Repositories:
https://github.com/DTUComputeCognitiveSystems/AI_playground/blob/master/notebooks/experiments/Sound%20Demo%203%20-%20Multi-label%20classifier%20pretrained%20on%20audioset.ipynb

<br>

### Contact us
Contact us if you want more information sending an e-mail to `paulopezribas@gmail.com`
