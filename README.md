# Emotion-Classification-using-LLMs

## Contributors

Created by [Marc-Antoine Nadeau](https://github.com/ma-nadeau), [Jessie Kurtz](https://github.com/jkzcodes), and [Baicheng Peng](https://github.com/sivess)

## Overview

We investigate and compare the performance of the following models in emotion classification for the [GoEmotion](https://huggingface.co/datasets/google-research-datasets/go_emotions) dataset:

### Large Language Models (LLMs):
- [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert)
- [GPT2](https://huggingface.co/openai-community/gpt2)
- [Distilled-GPT2](https://huggingface.co/distilbert/distilgpt2)

### Traditional Models: 
- Naive Bayes
- RF
- SR
- XGBoost

## GoEmotions

The GoEmotions dataset, developed by Google Research, contains over 58,000 English Reddit comments, each labeled with one or more of 27 emotion categories or marked as neutral. This project focuses on classifying these emotions and evaluating performance using several algorithms and configurations.

## File Structure  
### How to Run
#### BERT
1. Navigate to the `src/BERT` folder.
2. Run the `PrepareDataset.py` file.

#### Distilled-GPT2
1. Navigate to the `src/Distilled-GPT2` folder.
2. Run the `PrepareDataset.py` file.

#### GPT2
1. Navigate to the `src/GPT2` folder.
2. Run the `PrepareDataset.py` file.

#### Word2Vec
1. Navigate to the `src/Word2Vec` folder.
2. Run the `word2vec.py` file.


### Results
Any resulting figures, graphs, and CSV will be saved in their respective folders:
- `Results-BERT`: Contains the results for the BERT LLM.
- `Results-Distilled-GPT2`: Contains the results for the Distilled-GPT2 LLM.
- `Results-GPT2`: Contains the results for the Distilled-GPT2 LLM.
- `Results-Naive-Bayes`: Contains the results for the Naive-Bayes.
- `Results-Word2Vec`: Contains the results for the Word2Vec.

## Prerequisites

Make sure you have the following Python libraries installed:
- `numpy`
- `matplotlib`
- `torch`
- `seaborn`
- `pandas`
- `transformers`
- `scikit-learn`

You can install them using pip:

```bash
pip install numpy matplotlib torch seaborn pandas transformers scikit-learn
```
