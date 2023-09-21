# [**Feedback Prize - English Language Learning**](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)

**Evaluating language knowledge of ELL students from grades 8-12**  

<img src="https://github.com/Nov05/pictures/blob/master/kaggle--feedback-prize-ell/2023-09-21%2010_40_27-Feedback%20Prize%20-%20English%20Language%20Learning%20_%20Kaggle-min.jpg?raw=true">  

<br>

## **Project Achievements**

1. [**Exploratory data analysis** (EDA)](https://github.com/Nov05/Google-Colaboratory/blob/master/20221012_Kaggle_FB3_ELL_EDA.ipynb) was conducted to the training data, which has 3911 unique entries, not a large dataset. According to the size, some simple traditional NLP approaches, such as `Bag-of-Words`, `tf-idf`, etc., could work supprisingly well. Another popular approach would be using fine-tuning pre-trained large language models, which have learnt human language deep patterns from huge training datasets and store the patterns in their tens of millions even billions of parameters, such as `DeBERTa-V3-Base` (86M).  

2. 7 different types of machine learning models were trained and submitted to Kaggle, with architectures from simple to complex, sizes from small to large, scores from low to very close to the top ones 0.433+ (my best score so far is **0.440395**, would rank around 1,108 of 2,654 teams). 

3. Among the 7 models, 
    * 5 models utilized a **scikit-learn (sklearn)** pipeline and 2 a regular neural network training class. The sklearn pipeline combines  
		* manually engineered features such as  
			* `unigrams count` (reflecting the english learners' vocabulary)  
			* `line breaks count` (for that essays with lower scores tend to have too few or too many line breaks)  
			* `I vs. i`  and `bad punctuation` (for that worse essays usually don't pay attention to the capitalization and punctuation rules), etc.  
			* `tf-idf` (a widely used statistical method), etc.  
        * a feature engineered with **fastText**, such as `english score`, to measure how much likely an essay is classified as English (for that essays with lower scores were written by non-native English speakers who tend to use more non-English words), etc.  
        * the output of a state-of-the-art natural language model, in this case, the pre-trained transformer-based DeBERTa-V3-Base model, as a "feature", and feed them into the relatively "traditional" simple machine learning regressors, such as `linear`, `xgboost`, LightGBM (`lgb`), and a 2-layer vanilla neural network (`nn`)   
    * 2 models each utilized a fine-tuned custom pre-trained **`DeBERTa`** model, which consists of the deberta-v3-base model, a pooling layer, and one or two fully connected layers.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With such a design, different types of models can be trained, evaluated, tested, and submitted with similar APIs.

**E.g.**   
* The scikit-learn pipeline  
```Python
def make_features_pipeline(fastext_model_path,
	                       deberta_config:MSFTDeBertaV3Config):
	pipes = [
		("unigrams_count", number_of_unigrams_pipe),
		("line_breaks_count", number_of_line_breaks_pipe),
		("english_score", make_english_score_pipe(fastext_model_path)), 
		("i_vs_I", i_pipe), 
		("bad_punctuation", bad_punctuation_pipe),
		("tf-idf", tf_idf_pipe)
	]
	if deberta_config.using_deberta:
		print("using deberta embedding")
		pipes += [("deberta_embedding", make_deberta_pipe(deberta_config))]
	else:
		print("not using deberta embedding")
	features_pipeline = FeatureUnion(pipes)
	return features_pipelines
```
* The neural network trainer clasess 
```Python
    inheritance relationship:
    ModelTrainer 
        |--> SklearnTrainer
                |--> SklearnRegressorTrainer (dummy, linear, xgb, lgb)
                |--> NNTrainer (nn)
        |--> DebertaTrainer (deberta1, deberta2)
```


4. A better workflow was established with [**GitHub Actions**](https://github.com/Nov05/kaggle--feedback-prize-ell/blob/main/.github/workflows/main.yml), which enables code firstly to be written in a local IDE, then committed to GitHub and automatically uploaded to Kaggle as a "dataset" (if the commit message title contains the string "**upload to kaggle**"), and finally imported to a Kaggle Notebook and executed. In the Kaggle Notebook, I just needed to type `$python main.py <model_type> <recast_scores> <using_deberta>` to choose different models to run. With the workflow, I could quickly iterate the code, test out different models with different hyperparameters. (The Kaggle dataset uploading public APIs [are not very user friendly](https://github.com/Nov05/Google-Colaboratory/blob/master/20230906_github_workflow_upload_dataset_to_kaggle_debug.ipynb).)  
[![watch the video](https://img.youtube.com/vi/3YpoEKYnzUE/0.jpg)](https://www.youtube.com/watch?v=3YpoEKYnzUE)    
<img src="https://github.com/Nov05/pictures/blob/master/kaggle--feedback-prize-ell/2023-09-20%2022_48_56-README.md%20-%20kaggle--feedback-prize-ell%20-%20Visual%20Studio%20Code-min.jpg?raw=true">  


5. For successful Kaggle submissions, I also had to figure out how to install Python libraries, import deberta-v3-base model without the Internet (as the competition required), and load the model checkpoints which were fine-tuned and saved in Google Colab or locally on my laptop (Kaggle's GPU weekly quota is 30 hours, and mine was solely used for submissions). It turned out all these files can be uploaded to Kaggle as "datasets", then you can `add data` in a Kaggle Notebook, and these "datasets" will be added to the `input` directory in different folders.   


**E.g.**  
* upload wheel files as Kaggle dataset `python`, then `add data` in the notebook, then install the library `sklego` from the Kaggle directory by executing the command `$pip install sklego --no-index --find-links=file:///kaggle/input/python`.  
<img src="https://github.com/Nov05/pictures/blob/master/kaggle--feedback-prize-ell/2023-09-20%2023_49_00-20230910_github%20repo%20(uploaded%20by%20github%20action)%20_%20Kaggle-min.jpg?raw=true" width=300>  


6. What can be improved? A lot of code refactoring can be done in the future, to make the training/evaluating/testing APIs and the training hyperparameters more unified, and the whole framework more flexible and automated. MLOps platforms such as **Weights & Biases** could be integrated, for better tracking and analysing of the training processes.  


**P.S.** 
* Kaggle leaderboard  
<img src="https://github.com/Nov05/pictures/blob/master/kaggle--feedback-prize-ell/2023-09-20%2013_24_32-Feedback%20Prize%20-%20English%20Language%20Learning%20_%20Kaggle-min.jpg?raw=true">

<br>

## **Project Artifacts**

* Project Proposal[【google docs】](https://docs.google.com/document/d/1euOWdw7vIrkO1fVCuqv4sPNscPVbE-fgF5VcMy9DsAs)
* Exploratory Data Analysis[【notebook】](https://github.com/Nov05/Google-Colaboratory/blob/master/20221012_Kaggle_FB3_ELL_EDA.ipynb)  
* Code in [【GitHub repo】](https://github.com/nov05/kaggle--feedback-prize-ell), or as [【Kaggle dataset】](https://www.kaggle.com/datasets/wenjingliu/kaggle--feedback-prize-ell)  
* Prototype and fine-tune Deberta (with Accelerate and W&B)【notebook】[V1](https://github.com/Nov05/Google-Colaboratory/blob/master/20230911_deberta_v3_base_accelerate_finetuning.ipynb), [V2](https://github.com/Nov05/Google-Colaboratory/blob/master/20230915_deberta_v3_base_accelerate_finetuning.ipynb)   
* **GitHub Actions** enable auto uploading repo to Kaggle[【how-to doc】](https://docs.google.com/document/d/1t5q14spGUW-xLo14hnDBK_gycsDRqdmO2POa0QzcbQE) [【github code】](https://github.com/Nov05/kaggle--feedback-prize-ell/blob/main/.github/workflows/main.yml)[【notebook for debugging】](https://github.com/Nov05/Google-Colaboratory/blob/master/20230906_github_workflow_upload_dataset_to_kaggle_debug.ipynb)


* Kaggle submissions  

| Notebook Version | **Run** | **Private Score** | **Public Score** | | 
|-|-|-|-|-|  
| [n1v1 - baseline](https://www.kaggle.com/code/wenjingliu/20221012-col-means-as-baseline?scriptVersionId=107904814) | 23.0s | 0.644705 | 0.618673 | column means as baseline |  
| [n1v20 - dummy](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=143608471) | 260.8s - GPU T4 x2 | 0.644891 | 0.618766 | train and infer on GPU |
| [n1v21 - linear](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=143608613) | 498.6s - GPU T4 x2 | 1.266085 | 1.254728 | train and infer on GPU |  
| [n1v27 - linear, no deberta embedding](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=143725868) | 85.5s - GPU T4 x2 | 0.778257 | 0.769154 | train and infer |
| [n1v25 - xgb](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=143643199) | 467.2s - GPU T4 x2 | 0.467965 | 0.471593 | train and infer on GPU |  
| [n1v28 - xgb, no deberta embedding](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=143725952) | 107.7s - GPU T4 x2 | 0.540446 | 0.531599 | train and infer |  
| [n1v23 - lgb](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=143616210) | 323.2s - GPU T4 x2 | <span style="color: green;">**0.458964**</span> | 0.459379 | train and infer on GPU |  
| [n1v26 - lgb, no deberta embedding](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=143707001) | 71.9s - GPU T4 x2 | 0.540557 | 0.528224 | train and infer |
| [n1v6 - nn](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=142544542) | 270.8s - GPU T4 x2 | 0.470323 | 0.466773 | train and infer on GPU |
| [n1v7 - nn](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action/?scriptVersionId=142545233) | 7098.2s | 0.470926 | 0.465280 | train and infer on CPU |  
| [n1v29 - nn, no deberta embedding](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=143730028) | 99.8s - GPU T4 x2 | 0.527629 | 0.515268 | train and infer on GPU |  
| [n1v12 - deberta 1 (invalid)](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=142990426) | 80.9s - GPU T4 x2 | <span style="color: red;">0.846934</span> | <span style="color: red;">0.836776</span> | fine-tuned custom deberta-v3-base, 7 epochs, with Accelerate, infer only |
| [n1v19 - deberta 1](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=143585342) | 86.9s - GPU T4 x2 | 0.474335 | 0.478700 | fine-tuned custom deberta, 30 epochs, infer only |  
| [n2v1 - deberta 2](https://www.kaggle.com/code/wenjingliu/fork-of-english-language-learning-157754/notebook?scriptVersionId=143479601) | 9895.7s - GPU P100 | <span style="color: green;">**0.440395**</span> | 0.441242 | fine-tuned custom, 5 epochs, Multilabel Stratified 4-Fold, train and infer |
| [n1v18 - deberta 2](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=143575345) | <span style="color: green;">**80.8s**</span> - GPU T4 x2 | 0.444191 | 0.443768 | fine-tuned custom, 5 epochs, infer only |

***Note:***  
*1. From the n1v7 Kaggle execution log we know that the time to train the simple hidden layer neural network on CPU is 4816-4806=10s, simliar to that on GPU, which means almost all time was spent on training data and testing data transformation, due to the size of the Deberta model. Hence there is no need to test out training on GPU and infering on CPU, which would be as slow as the scenario of both processes on CPU.*     

*2. From n1v12 to v13, with a few more epochs, there is a a little improvement. However, the scores are way larger than the column-mean baseline score is 0.644, which indicates some problem with the model or the training. it turned out that the shuffling of the test data caused the problem.*  

*3. Submission n2v1 and n1v18 used a different cutome model, which has one `attention` pooling layer and only one fully connected layer. there were also mixed techniques used during the traing, such as `gradient accumulation`, `layerwise learning rate decay`, `Automatic Mixed Precision`, `Multilabel Stratified K-Fold`, `Fast Gradient Method`, etc.. These techniques largely imporved the final score. With a pre-trained model, train only 5 epochs, less than 10,000 seconds, could get very close to the best score.*  

* Code repo structure explained  
	* `main.py` - provides a command-line interface for a package  
	* `config.py` - all the configurations  
	* `sklearn_transformers.py` - generate fasttext and deberta features  
	* `english_utils.py` - fasttext functions, text processing functions
	* `torch_utils.py` - dataset classes, neural network pooling layer classes
	* `pipelines.py` - all the scikit-learn pipelines (for model types: dummy, linear, xgb, lgb, nn)  
	* `deberta_models.py` - custom deberta models  
	* `trainers` folder - all the training classes  
	* `input` and `working` folders - simulate the Kaggle folders  
		* the Kaggle data files and sample submission files are stored in `\input\feedback-prize-english-language-learning`  
		* the Kaggle Notebook working and output direcotry is `\working`    
		* your own or someone else's datasets (which might include repos, python library wheel files, models, etc. uploaded to Kaggle) will be linked as sub-directories (by clicking on the `add data` button) under the `input` directory  
	* `main.yml` - GitHub Actions `upload github code to kaggle`  

<img src="https://raw.githubusercontent.com/Nov05/pictures/master/kaggle--feedback-prize-ell/2023-09-20%2001_38_46-Greenshot-min.jpg">

 <br>

## **Learning and Explorations**  

* The architecture of the BERT family models and how to train them, connect Google Colab with a local runtime on a docker image[【notebook】](https://github.com/Nov05/Google-Colaboratory/blob/master/20230814_huggingface_transformer_BERT_encoder_only.ipynb)  
* Weights & Biases MLOPS-001[【notebooks】](https://drive.google.com/drive/folders/17y-_5hB9CUjDO7HhOSWXBhB_RFTTb4HV)   
* **Scikit-lego** mega model example code[【notebook】](https://github.com/Nov05/Google-Colaboratory/blob/master/20230817_scikit_lego_meta_model_example_code.ipynb)
* Loading HuggingFace models[【notebook】](https://colab.research.google.com/drive/1GABUCj34h3OOjsC8vZ7ScOsYeYuMr7qR)  
* Scikit-learning CountVectorize, csr_matrix, np.matrix.A[【notebook】](https://github.com/Nov05/Google-Colaboratory/blob/master/20230910_sklearn_CountVectorize%2C_csr_matrix%2C_np_matrix_A.ipynb)  
* Imporve your Kaggle workflow with GitHub Actions[【Google Docs】](https://docs.google.com/document/d/1t5q14spGUW-xLo14hnDBK_gycsDRqdmO2POa0QzcbQE)  
* Kaggle dataset uploading public APIs[【notebook】](https://github.com/Nov05/Google-Colaboratory/blob/master/20230906_github_workflow_upload_dataset_to_kaggle_debug.ipynb)

[*... or check this Google Driver folder*](https://drive.google.com/drive/folders/1L-YlMhgc2LVWQTNyUweImgpNzIvURptD)

<br>

## **Repo update log** 

2023-09-21
1. project submitted to Udacity   

2023-09-19
1. added another fine-tuned cutome deberta-v3-base model (model_type=**'deberta2'**), which consists of deberta-v3-base (layer learning rate decay) + attention pooling + single fully-connected layer, trained 7 epochs on Google Colab GPU T4
2. bug fix: there is a large discrepancy bwtween the training score (around 0.45) and the testing score (around 0.8). went through the `deberta_trainer.py` code part by part, and finally figured out the problem was caused by the data loader. for testing data, `suffle=False` should be configured.   

2023-09-11
1. added fine-tuned deberta-v3-base model (model_type=**'derberta1'**), which consists of deberta-v3-base (frozen) + mean pooling + 2 fully-connected layer with ReLU and dropout layers inbetween, trained on Google Colab GPU T4. 5 epochs can reach a score around 4.8. typically for transferred learning, a wide fully-connected layer is enough, and more might be suboptimal. also ReLU and dropout might reduce the performance.  
2. train with Accelerate and Weights & Biases  

2023-09-10  
1. minor refactoring, bug fix, all done  
2. initial training, having all the current features and a simply vanilla neural network with hidden dimension [64] as the regressor yields the best result (has reached **MCRMSE=0.47+** at 200 epochs, the best score is **0.43+**) so far. refer to the [**training log**](https://gist.github.com/Nov05/146d7d53a3498e6fdeecc8a98c7da02b)   
3. fasttext and deberta (pre-trained, not fine-tuned) for feature extraction, and hyperparameters of the regressor could be fine-tuned for better result  
4. add requirements.txt (pip freeze > requirements.txt)     
5. submitted notebooks in kaggle   

2023-09-07   
1. forked then bug-fixed [the **github action**](https://github.com/Nov05/action-push-kaggle-dataset) ([issue and pull request](https://github.com/jaimevalero/push-kaggle-dataset/issues/14))   
2. updated the kaggle python api version in the action, from 1.5.12 to **1.5.16**  
3. the upload workflow will only be triggered if string "**upload to kaggle**" is found in the commit message (main.yml)  

## **Reference**

* Kaggle Notebook, [Using 🤗 Transformers for the first time | Pytorch by @BRYAN SANCHO](https://www.kaggle.com/code/bryansancho/using-transformers-for-the-first-time-pytorch)  
* Kaggle Notebook, [DeBERTa-v3-base | 🤗️ Accelerate | Finetuning by @SHREYAS DANIEL GADDAM](https://www.kaggle.com/code/shreydan/deberta-v3-base-accelerate-finetuning/notebook)
* Kaggle Notebook, [FB3 English Language Learning by @张HONGXU](https://www.kaggle.com/code/shufflecss/fb3-english-language-learning)  
* Kaggle Notebook, [0.45 score with LightGBM and DeBERTa feature by @FEATURESELECTION](https://www.kaggle.com/code/josarago/0-45-score-with-lightgbm-and-deberta-feature?scriptVersionId=113244660) 
* GitHub Actions repo, [jaimevalero/push-kaggle-dataset](https://github.com/jaimevalero/push-kaggle-dataset)  
* GitHub repo, https://github.com/microsoft/DeBERTa 