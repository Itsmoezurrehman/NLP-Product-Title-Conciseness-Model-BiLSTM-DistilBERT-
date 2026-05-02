# NLP-Product-Title-Conciseness-Model(BiLSTM-DistilBERT)

Deep learning classifier for e-commerce product title conciseness, built for a dataset of ~36k product listings from Malaysia, Philippines, and Singapore. Framed as a regression task — the model outputs a probability in [0, 1] and is evaluated on RMSE.

## Task

Predict whether a product title is concise (1) or non-concise (0). A concise title gives clear, non-redundant information about a product in as few words as possible, with strong relevance to the product category. Non-concise titles either pad with filler words, repeat information, or omit key attributes.

## Models

**Model 1 — Multi-Input BiLSTM (Keras + TensorFlow)**
Text branch: learned embedding (vocab=30k, dim=128) → Bidirectional LSTM (64 units per direction) → Dropout(0.4). Structured branch: four numerical features (log price, country, product type, title word count) → Dense(32, ReLU) → Dropout(0.2). Both branches concatenated → Dense(64, ReLU) → Dropout(0.3) → Dense(1, sigmoid). Trained with MSE loss.

**Model 2 — Fine-tuned DistilBERT (PyTorch + HuggingFace)**
Pretrained `distilbert-base-uncased` fine-tuned end-to-end. [CLS] token output (768-dim) used as sentence representation. Same structured branch as BiLSTM. Concatenated → Dense(64, ReLU) → Dropout(0.3) → Dense(1, sigmoid). Low learning rate (2e-5) to preserve pretrained weights.

Both models use the same multi-input design: a text encoder handles the combined title + category path string, while a separate dense branch handles structured numerical features.

## Results

| Model | Val RMSE | Val Accuracy |
|---|---|---|
| Multi-Input BiLSTM | 0.35463 | 83.09% |
| Fine-tuned DistilBERT | 0.33257 | 83.93% |

DistilBERT outperforms BiLSTM on RMSE because its pretrained weights give it better-calibrated probability estimates — it is more confident when correct and less confident when wrong, which directly reduces RMSE even when binary accuracy is similar.

## Dataset

`CS5143-NLP_PA2_data_train.csv` — no header row. Columns: `country`, `sku_id`, `title`, `category_lvl_1`, `category_lvl_2`, `category_lvl_3`, `short_description`, `price`, `product_type`, `Concise_label`.

Upload as a Kaggle dataset before running. Update `DATA_PATH` in the loading cell to match your dataset path.

## How to Run

1. Upload `CS5143-NLP_PA2_data_train.csv` as a Kaggle dataset
2. Open the notebook on Kaggle and enable **GPU T4 x2** (Accelerator setting)
3. Run all cells top to bottom
4. BiLSTM trains in ~5 minutes. DistilBERT takes ~20–30 minutes on T4 x2.

**For test set inference (professor evaluation):**
Update `TEST_PATH` in the final cell to the location of the test CSV file, then run that cell. Predictions are saved to `test_predictions.csv` with both BiLSTM and DistilBERT outputs.

## Files

```
├── nlp-task-product-conciseness-classifier.ipynb   # main notebook
├── CS5143-NLP_PA2_data_train.csv                   # training data
└── PA2_Report_Moez_K247840.pdf                     # assignment report
```

Artifacts saved during training:
- `bilstm_conciseness.h5` — BiLSTM Keras model
- `distilbert_best.pt` — DistilBERT PyTorch weights (best epoch)
- `tokenizer_lstm.pkl` — Keras tokenizer
- `scaler.pkl` — StandardScaler for structured features
- `encoders.pkl` — LabelEncoders for country and product type

## Environment

- Python 3.10+
- TensorFlow 2.x (for BiLSTM)
- PyTorch + HuggingFace Transformers 5.x (for DistilBERT)
- scikit-learn, pandas, numpy, matplotlib, seaborn, beautifulsoup4

Install: `pip install transformers torch tensorflow scikit-learn pandas numpy matplotlib seaborn beautifulsoup4`

## Key Findings

The dominant error pattern is false positives — short titles that look clean but are missing category-expected attributes (e.g., a memory card title without read/write speed). The model picks up on brevity as a proxy for conciseness, which holds on average but fails for technically sparse short titles. A per-category required-attribute schema would be the natural next step for a production system, but is not available in the current dataset.

---
**Course:** CS5143 — Natural Language Processing, Spring 2026  
**Student:** Moez Ur Rehman 
**Institution:** FAST-NUCES
