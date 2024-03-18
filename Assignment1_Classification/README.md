# Classification Models of Country Happiness Level
## Overview
This model is based on the world happiness index dataset, and tries to predict a country's happiness into 5 levels. 
A Random Forest classification model is used here. Grid search is used to fine tune the model. 
Finally, the model is submitted to aimodelshare to compare its performance with other models. 
*Note: the dataset can be found here [world happiness dataset](world_happiness_competition_data.zip) [other country variables](newcountryvars.csv)*
## Steps
### 1. Merging country variables into the dataset
Country Variables: 
![image](https://github.com/zesiii/Advanced-ML/assets/144854988/94f0a1f1-de60-4898-8b5a-b1b51ec3974f)
### 2. EDA
#### Distribution of Happiness Level
I examined the distribution of the independent variable, which is country happiness level. Happiness level is relatively evenly spread out. 
![image](https://github.com/zesiii/Advanced-ML/assets/144854988/30c2f75f-7705-4f0f-8ab0-ff131cf90528)
#### Pairplot of some predictor variables
With all the features that are intuitively considered to be positively correlated with happiness, the pairplot does roughly show this tendency.
![image](https://github.com/zesiii/Advanced-ML/assets/144854988/219600b7-9cdf-4681-a3d5-e73479757102)
### 3. Preprocessing
### 4. Establish the Random Forest model
### 5. Perform GridSearch
* Best mean cross-validation score: 0.683
* Best parameters: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 50}
* Accuracy: 47.06%
* F1 score: 46.02%
* Precision: 53.33%
* Recall: 48.43%
### 6. Submit the model
