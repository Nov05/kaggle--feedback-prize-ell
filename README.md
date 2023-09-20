# [**Feedback Prize - English Language Learning**](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)

**Evaluating language knowledge of ELL students from grades 8-12**  

<br>

## **Project Artifacts**

* Project Proposal[【google docs】](https://docs.google.com/document/d/1euOWdw7vIrkO1fVCuqv4sPNscPVbE-fgF5VcMy9DsAs)
* Exploratory Data Analysis[【notebook】](https://github.com/Nov05/Google-Colaboratory/blob/master/20221012_Kaggle_FB3_ELL_EDA.ipynb)  
* Code in [【GitHub repo】](https://github.com/nov05/kaggle--feedback-prize-ell), or as [【Kaggle dataset】](https://www.kaggle.com/datasets/wenjingliu/kaggle--feedback-prize-ell)  
* Prototype and fine-tune Deberta (with Accelerate and W&B)【notebook】[V1](https://github.com/Nov05/Google-Colaboratory/blob/master/20230911_deberta_v3_base_accelerate_finetuning.ipynb), [V2](https://github.com/Nov05/Google-Colaboratory/blob/master/20230915_deberta_v3_base_accelerate_finetuning.ipynb)   


* Kaggle submissions  

| Notebook Version | **Run** | **Private Score** | **Public Score** | | 
|-|-|-|-|-|  
| [n1v1 - dummy](https://www.kaggle.com/code/wenjingliu/20221012-col-means-as-baseline?scriptVersionId=107904814) | 23.0s | 0.644705 | 0.618673 | column means as baseline | 
| [n1v6 - nn](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=142544542) | **270.8s** - GPU T4 x2 | **0.470323** | 0.466773 | train and infer on GPU |
| [n1v7 - nn](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action/?scriptVersionId=142545233) | 7098.2s | 0.470926 | 0.465280 | train and infer on CPU |
| [n1v12 - deberta 1 (invalid)](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=142990426) | **80.9s** - GPU T4 x2 | 0.846934 | 0.836776 | fine-tuned custom deberta-v3-base, 7 epochs, with Accelerate, infer only |
| [n1v13 - deberta 1](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=143052069) | 99.8s - GPU T4 x2 | 0.836189 | 0.795700 | fine-tuned custom deberta, 30 epochs, infer only |  
| [n2v1 - deberta 2](https://www.kaggle.com/code/wenjingliu/fork-of-english-language-learning-157754/notebook?scriptVersionId=143479601) | 9895.7s - GPU P100 | 0.440395 | 0.441242 | fine-tuned custom, 5 epochs, train and infer |
| [n1v18 - deberta 2](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=143575345) | 80.8s - GPU T4 x2 | 0.444191 | 0.443768 | fine-tuned custom, 5 epochs, infer only |

***Note:***  
*1. From the Notebook1-v7 execution log we know that the time to train the simple hidden layer neural network on CPU is 4816-4806=10s, simliar to that on GPU, which means almost all time was spent on training data and testing data transformation, due to the size of the Deberta model. Hence there is no need to test out training on GPU and infering on CPU, which would be as slow as the scenario of both processes on CPU.*     
*2. From v12 to v13, with a few more epochs, there is a a little improvement. However, the scores are way larger than the column-mean baseline score is 0.644, which indicates some problem with the model or the training.*  
*3. Submission n2v1 and n1v18 used a different cutome model, which has one `attention` pooling layer and only one fully connected layer. there were also mixed techniques used during the traing, such as `gradient accumulation`, `layerwise learning rate decay`, `Automatic Mixed Precision`, `Multilabel Stratified K-Fold`, `Fast Gradient Method`, etc.. These techniques largely imporved the final score. With a pre-trained model, train only 5 epochs, less than 10,000 seconds, could *  

 <br>

## **Learning and Explorations**  

* The architecture of the BERT family models and how to train them[【notebook】](https://github.com/Nov05/Google-Colaboratory/blob/master/20230814_huggingface_transformer_BERT_encoder_only.ipynb)  
* Weights & Biases MLOPS-001[【notebooks】](https://drive.google.com/drive/folders/17y-_5hB9CUjDO7HhOSWXBhB_RFTTb4HV)  

<br>

## **Repo update log** 

2023-09-11
1. added fine-tuned deberta-v3-base model  
2. train with Accelerate and Weights & Biases  

2023-09-10  
1. minor refactoring, bug fix, all done  
2. initial training, having all the current features and a simply vanilla neural network with hidden dimension [64] as the regressor yields the best result (has reached **MCRMSE=0.47+** at 200 epochs, the best score is **0.43+**) so far. refer to the [**training log**](https://gist.github.com/Nov05/146d7d53a3498e6fdeecc8a98c7da02b)   
3. fasttext and deberta (pre-trained, not fine-tuned) for feature extraction, and hyperparameters of the regressor could be fine-tuned for better result  
4. add requirements.txt (pip freeze > requirements.txt)     
5. submitted notebooks in kaggle   

2023-09-07   
1. forked then bug-fixed [the **github action**](https://github.com/Nov05/action-push-kaggle-dataset)   
2. updated the kaggle python api version in the action, from 1.5.12 to **1.5.16**  
3. the upload workflow will only be triggered if string "**upload to kaggle**" is found in the commit message (main.yml)  