# RobotDetection

Detecting bot bidders in online auctions using a Transformer-based deep learning model.  
Built for the [Facebook Recruiting IV: Human or Robot?](https://www.kaggle.com/competitions/facebook-recruiting-iv-human-or-bot/overview) Kaggle competition.

---

## Results

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.9347** |
| F1 Score (Bot) | 0.37 |
| Bot Recall | 0.95 |

---

## Problem

Online auction platforms are targeted by software-controlled bidders (bots) that manipulate prices through artificial bids. This project builds a binary classifier to distinguish human bidders from bots using behavioral bidding data.

Key challenge: severe class imbalance — only 5.1% of training bidders are bots (103 bots vs 1,910 humans).

---

## Approach

### Feature Engineering
19 behavioral features extracted per bidder from raw bid logs:
- Bidding frequency and volume
- Time intervals between consecutive bids (`gap_mean`, `gap_std`, `rapid_bid_ratio`)
- Device, IP, country, and URL diversity
- Bids-per-auction statistics

### Model Architecture
Transformer Encoder classifier (70K parameters):

```
Input Features (19)
      ↓
Linear Projection → d_model=64
      ↓
Transformer Encoder (2 layers, 4 heads)
      ↓
Global Average Pooling
      ↓
Classifier Head (64 → 32 → 1)
      ↓
Sigmoid → Probability [0, 1]
```

### Training
- Loss: `BCEWithLogitsLoss` with `pos_weight=18.6` to handle class imbalance
- Optimizer: Adam (`lr=1e-3`, `weight_decay=1e-4`)
- Early stopping at epoch 16, best val loss: 0.6207

---

## Repository Structure

```
├── WID3011_group16_RobotDetection.ipynb   # Full pipeline (preprocessing → training → evaluation → submission)
├── demo.py                                # Interactive demo using real bidder data
├── README.md
├── LICENSE
└── .gitignore
```

### Running the Demo

```bash
pip install torch scikit-learn pandas numpy
python demo.py
```

The demo loads `model.pt`, `scaler.pkl`, and `train_features.csv` from the same folder and lets you interactively explore real bidder predictions:

```
Choose:
  1. Random real HUMAN bidder
  2. Random real BOT bidder
  3. Show 5 humans in a row
  4. Show 5 bots in a row
  5. Compare human vs bot side by side
  6. Find a misclassified example
```

---

## ROC Curve

![ROC Curve](https://github.com/user-attachments/assets/9cb49d0c-2d62-4aae-b844-fdf525effc9e)
