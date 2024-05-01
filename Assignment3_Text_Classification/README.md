# Explore Various Deep Learning Approaches to Sentiment Classification Task
## Contents
- [Overview](#overview)
- [Structure of repo](#structure)
- [How to use the code](#guidance)
- [Steps](#steps)
    - EDA
    - Building Models
    - Training and Performance Evaluation
- [Conclusions](#conclusions)

## Overview
This project introduces different approaches (deep learning and non deep learning) to a sentiment classification task, tests their performance and analyzes the outcome.

The dataset used is Stanford Sentiment Treebank - Movie Review Classification Competition, where each review contains only one sentence and the sentiment class is polarized: "positive" or "negative".

The training dataset has 6920 rows of data. The classes are pretty balanced so no further processing is needed. 

We tried non-deep learning approaches to this problem first. SVC performs best among all other models. 

Then for deep learning approaches, we experimented with LSTM, keras tuner, GloVe embedding, Bert and T5. In conclusion, transformers significantly outperforms normal neural networks, and T5 is best among all models.

[Report](final_report.ipynb) is the detailed report comprising main models and corresponding preprocessing, visualization and model building details. Refer to this file for main process.

## Structure
At the root of the project, you will see: 
```text
├── back&froneted
│   ├── basicnn_model_weights.h5
│   └── bert_model_weights.h5
|   └──glove_model_weights.h5
|   └──glove.6B.200d.txt
|   └──main.py
|   └──models.py
|   └──performance_metrics.py
|   └──data_processing.py
|   └──utils.py
|   └──requirements.txt
|   └──front_end.ipynb
├── uncleaned_code
│   ├── assignment3_Part1_2_yz4691_qg2218
│   ├── assignment3_Part2_2_yz4691_qg2218
├── final_report.ipynb
└── README.md
```
## Guidance
All model weights and test data are included in this repo. You just have to run the [front_end.ipynb(back&fronted/front_end.ipynb) file, choose one of the three model choices we provide: `Glove200d`, `CNN` or `BERT`. It will preprocess the data based on the specific model, retrieve the model class and associated weights file path in the backend. The output will be prediction results and metrics as plots for you.

## Steps
### EDA
|Sentiment Class Balance|Review Length|
|:-:|:-:|
|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/1e1d18b3-f56f-48ae-9765-6c724385db7d)|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/50935d66-c942-4af3-85da-62359ec936ba)|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/be037ba1-61c9-4eea-9ab9-275114e113b9)|

The classes are mostly balanced, so there is no need for resampling. The review length of X_train varies, with longest review reaching 50 words. Therefore, as we proceed to step two, before buildiing deep learning models, we could consider padding the review into same length.

|Word Frequency in Reviews| |
|:-:|:-:|
|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/c340ca9a-5b7f-424f-ab85-90228d92d336)|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/15599270-ffbf-4aac-aa94-b4269777d2a1)|

Word frequency in positive and negative classed corpuses are different to a certain extent. Both contain theme words like "movie", "film", "story" and "character". Both contain common verbs and prepositions and adverbs like "feel", "like", "make" and "even". Positive corpuses contain specific words like "funny", "best", etc. Negative corpuses contain specific words like "bad".

### Non-deep learning models
Since this is not the emphasis of our project, I'll just include the metrics here. 
|Model|Accuracy|Training Accuracy|
|:-:|:-:|:-:|
|SVC|77%|95%|
|basic RF|74%|98%|
|specific RF|73%|89%|
|bagging classifier|72%|97%|
|grid RF|62%|68%|
|GBC|64%|72%|

SVC has the highest test accuracy, F1 score, precision, and recall, indicatint it is the most consistent across all metrics for the test set, suggesting it might be the best generalizer among the models evaluated for unseen data.

Overall, all models show obvious signs of overfitting. 

### Deep Learning Models
#### Tokenizer
We used keras tokenizer to preprocess and tokenize the input text. We then downloaded glove embedding and used it in the embedding layer of a basic convolutional neural network model. 

For bert and t5 -based models, we loaded their own pretrained tokenizer from keras to preprocess input text. 
#### Tuner
We use keras-tuner to tune hyperparameters of one LSTM model and build model based on the best set of hyperparameters.

#### Training History and Test Performance
|Model|Training History|Test Accuracy|
|:-:|:---:|:-:|
|LSTM model tuned with keras tuner|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/6046db08-c290-4ee0-92b2-0fb9241be1fb)|45%|
|Transfer model with GloVe embeddings|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/6e99d877-a2f8-4757-91bb-1433dc578ed9)|41%|
|Bert-based model|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/eb9e0dc2-5cc2-4475-bf3b-4dc3c084f814)|76%|
|T5-based model|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/b0873f4e-9d10-4c55-8799-83a9ac30c59a)|100%|

T5: T5 performs best with 100% accuracy. This can be foreseen since its training history performance is already unparalleled among other models. T5 is designed to be parameter-efficient compared to BERT. Therefore, it took less time in training and generates better results.

BERT: The BERT-based model doesn't show any sign of overfitting. We chose epoch=10 to save computation time, but it still hasn't converge, so an ideal solution is to increase epoch and see the train & validation performance throughout more epochs until it starts to overfit. BERT performs second best with a test accuracy of approximately 75.76% and a low loss of 0.413761. This might be attributed to its robust architecture and tokenization method since it utilizes subword tokenization, which mitigates the issue of out-of-vocabulary words. Therefore, BERT is able to maintain high performance when I used the test_sample.csv to test its performances.

Other Models : These models showed poorer performance compared to BERT, with accuracies ranging from 40% to 55%. I think this might be because when the training data vocab size (as well as the input dim) is set at 13836, and the testset vocab size is drastically reduces to 4864. This might causing these models to have large number of zero-padding inputs or unseen words, and may cause ineffective embedding lookups and suboptimal feature extraction.

## Conclusions
In terms of performance, transformer models significantly outperform other ones. Simple machine learning classifiers outperform simple deep learning. 

Since the dataset is not large and the task is rather simple, overfitting has been a great problem throughout all models. Only transformer models avoids showing overfitting in training.  This difference on overfitting between regular deep learning models and transformer models can be attributed to the following: 
- Attention Mechanism: helps in learning long-range dependencies more effectively compared to regular neural networks
- Pre-trained on Large Datasets: BERT and T5is trained on massive datasets, which helps in learning robust representations. The pre-trained embeddings often capture rich semantic and syntactic information, reducing the need for extensive training on task-specific data, which can help in avoiding overfitting.
- Regularization Techniques: Transformer models employ more complex and advanced regularization techniques during training.










