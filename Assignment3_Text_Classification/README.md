# Explore Various Approaches to Sentiment Classification (Deep Learning)
## Contents
- [Overview](#overview)
- [Structure of repo](#structure)
- [Steps](#steps)
- [Performance](#performance)
- [Conclusions](#Conclusion)
- [Insights](#Insights)


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
├── dir1
│   ├── file11.ext
│   └── file12.ext
├── dir2
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── dir3
├── final_report.ipynb
└── README.md
```

## Steps
jio
