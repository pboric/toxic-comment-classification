# Toxic Comment Classification

Multi-label classification of online comments into six types of toxicity (toxic, severe toxic, obscene, threat,
insult, identity hate) with fine-tuned transformer models. One comment can belong to several classes, and the rare
classes (severe toxic, threat, identity hate) are the main difficulty.

## Results

| Version | Model and techniques | Macro F1 | ROC AUC per class |
|---|---|---:|---:|
| v1 | DistilBERT, class weights, early stopping, tuned decision thresholds | 0.68 | not reported |
| v2 | RoBERTa, focal loss, K-fold cross-validation, mixed precision | **0.73** | 0.993–0.996 |

Per-class F1 of the v2 model:

| Toxic | Severe toxic | Obscene | Threat | Insult | Identity hate |
|---:|---:|---:|---:|---:|---:|
| 0.89 | 0.42 | 0.87 | 0.69 | 0.82 | 0.68 |

The ranking of comments is excellent (ROC AUC above 0.99 for every class), but deciding *which* label applies is much
harder for the rare classes: severe toxicity is detected least well (F1 0.42, recall 0.32).

## Data

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
on Kaggle (Wikipedia talk-page comments). Download `train.csv` and put it in a `data/` folder; the data is not
included.

## Notebooks

| Notebook | Content |
|---|---|
| [`v1_distilbert.ipynb`](v1_distilbert.ipynb) | EDA, preprocessing, DistilBERT fine-tuning, class weights for imbalance, early stopping, threshold optimisation, per-class reports |
| [`v2_roberta.ipynb`](v2_roberta.ipynb) | RoBERTa with focal loss, K-fold cross-validation, mixed-precision training, final evaluation and example predictions |

## Run it

```bash
pip install -r requirements.txt
```

Then run the notebooks. A CUDA GPU is needed for reasonable training times (they were developed on an 8 GB GPU).

## License

MIT
