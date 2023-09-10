# [**Feedback Prize - English Language Learning**](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)
Evaluating language knowledge of ELL students from grades 8-12  

* Project Proposal[【goodle docs】](https://docs.google.com/document/d/1euOWdw7vIrkO1fVCuqv4sPNscPVbE-fgF5VcMy9DsAs)
* Exploratory Data Analysis[【notebook】](https://github.com/Nov05/Google-Colaboratory/blob/master/20221012_Kaggle_FB3_ELL_EDA.ipynb)  


* Kaggle submissions  

|   | **Run** | **Private Score** | **Public Score** | | 
|-|-|-|-|-|  
| [Notebook 1](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=142544542) | **270.8s** - GPU T4 x2 | 0.470323 | 0.466773 | train and infer on GPU |
| [Notebook 2](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action/?scriptVersionId=142545233) | 7098.2s | 0.470926 | 0.465280 | train and infer on CPU |
| [Notebook 3](https://www.kaggle.com/code/wenjingliu/20230910-github-repo-uploaded-by-github-action?scriptVersionId=142564633) | - | - | - | train on GPU, infer on CPU |   

---

### Repo update log 

2023-09-10  
1. minor refactoring, bug fix, all done  
2. initial training, having all the current features and a simply vanilla neural network with hidden dimension [64] as the regressor yields the best result (has reached **MCRMSE=0.47+** at 200 epochs, the best score is **0.43+**) so far. refer to the [**training log**](https://gist.github.com/Nov05/146d7d53a3498e6fdeecc8a98c7da02b)   
3. fasttext and deberta (pre-trained, not fine-tuned) for feature extraction, and hyperparameters of the regressor could be fine-tuned for better result  
4. add requirements.txt (pip freeze > requirements.txt)     
5. submitted notebooks in kaggle  

<br>  

2023-09-07   
1. forked then bug-fixed [the **github action**](https://github.com/Nov05/action-push-kaggle-dataset)   
2. updated the kaggle python api version in the action, from 1.5.12 to **1.5.16**  
3. the upload workflow will only be triggered if string "**upload to kaggle**" is found in the commit message (main.yml)  