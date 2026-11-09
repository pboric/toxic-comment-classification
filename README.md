# Toxic Comment Classification

## Project Overview
This project implements a multi-label classifier to identify toxic comments in online forums using transformer-based models. The classifier categorizes comments into six types of toxicity:
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

Two implementations are provided:
1. Base implementation using DistilBERT
2. Enhanced implementation using RoBERTa with advanced features

## Features
- Multi-label text classification using transformer architectures
- Comprehensive text preprocessing pipeline
- Advanced training techniques:
  - Mixed precision training
  - K-fold cross validation
  - Focal Loss for handling class imbalance
  - Gradient accumulation
  - Early stopping
- Detailed performance visualization and analysis
- Production-ready prediction pipeline

## Requirements

### Hardware
- GPU with at least 8GB VRAM (tested on NVIDIA RTX 3070)
- 16GB RAM recommended
- Multi-core CPU (tested on i7 with 16 cores)

### Software
```
python>=3.8
torch>=1.9.0
transformers>=4.15.0
pandas>=1.3.0
numpy>=1.19.5
scikit-learn>=0.24.2
matplotlib>=3.4.3
seaborn>=0.11.2
nltk>=3.6.3
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/toxic-comment-classification.git
cd toxic-comment-classification
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset and place it in the `data/` directory:
```bash
mkdir data
# Place train.csv in the data directory
```

## Project Structure
```
toxic-comment-classification/
├── data/                    # Dataset directory
├── models/                  # Saved model checkpoints
├── notebooks/              
│   ├── baseline.ipynb      # DistilBERT implementation
│   └── enhanced.ipynb      # RoBERTa implementation
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Usage

### Training
Both implementations (baseline and enhanced) are provided as Jupyter notebooks with detailed documentation and explanations.

1. Base Implementation (DistilBERT):
```bash
jupyter notebook notebooks/baseline.ipynb
```

2. Enhanced Implementation (RoBERTa):
```bash
jupyter notebook notebooks/enhanced.ipynb
```

### Making Predictions
```python
from transformers import RobertaTokenizer
from model import ToxicClassifier  # Your model implementation

# Load model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = ToxicClassifier.load_from_checkpoint('models/final_model.pt')

# Make predictions
text = "Your text here"
predictions = predict_toxicity(text, model, tokenizer)
```

## Performance

### Base Model (DistilBERT)
- Macro F1 Score: 0.684
- Average AUC-ROC: 0.965

Class-wise F1 Scores:
- Toxic: 0.83
- Severe Toxic: 0.52
- Obscene: 0.81
- Threat: 0.49
- Insult: 0.78
- Identity Hate: 0.59

### Enhanced Model (RoBERTa)
- Macro F1 Score: 0.879
- Average AUC-ROC: 0.994

Class-wise F1 Scores:
- Toxic: 0.879
- Severe Toxic: 0.415
- Obscene: 0.874
- Threat: 0.554
- Insult: 0.816
- Identity Hate: 0.721

## Key Findings
1. Strong performance on main toxic categories
2. Challenging detection of rare toxic types (threats, severe toxic)
3. High AUC scores (>0.99) across all categories in enhanced model
4. Effective handling of class imbalance through Focal Loss
5. Successful implementation of memory-efficient training

## Future Improvements
1. Data augmentation for minority classes
2. Ensemble approaches
3. Model distillation for faster inference
4. Fine-tuning of class-specific thresholds
5. Collection of additional data for rare toxic categories

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset from [Kaggle's Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
- Hugging Face for transformer implementations
- PyTorch team for the deep learning framework

