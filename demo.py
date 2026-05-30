# Step 5: Interactive Demo — uses real bidders from train_features.csv
# Run: python 5_demo.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))

class BotDetector(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        x = self.proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(1)

with open(os.path.join(BASE, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

model = BotDetector(input_dim=19)
model.load_state_dict(torch.load(os.path.join(BASE, "model.pt"), map_location="cpu"))
model.eval()

FEATURE_COLS = [
    "total_bids", "unique_auctions", "unique_devices", "unique_countries",
    "unique_ips", "unique_urls", "unique_merchandise",
    "bids_per_auction_mean", "bids_per_auction_max", "bids_per_auction_std",
    "gap_mean", "gap_std", "gap_min", "gap_max", "rapid_bid_ratio",
    "device_diversity", "country_diversity", "ip_diversity", "top_country_ratio"
]

df     = pd.read_csv(os.path.join(BASE, "train_features.csv"))
humans = df[df["outcome"] == 0.0].reset_index(drop=True)
bots   = df[df["outcome"] == 1.0].reset_index(drop=True)

def predict_row(row):
    x = np.array([[row[f] for f in FEATURE_COLS]], dtype=np.float32)
    x = scaler.transform(x)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        return torch.sigmoid(model(x_t)).item()

def show_result(prob, true_label):
    pred    = "BOT" if prob >= 0.5 else "HUMAN"
    bar     = int(prob * 30)
    correct = "✓" if (prob >= 0.5) == (true_label == 1.0) else "✗"
    print(f"\n  Actual label : {'BOT' if true_label == 1.0 else 'HUMAN'}")
    print(f"  [{'#' * bar:<30}] {prob:.4f}")
    print(f"  Prediction   : >>> {pred} <<< {correct}")

print("\n" + "="*50)
print("  Bot Detection Demo  -  WID3011 Group 16")
print("="*50)

while True:
    print("\nChoose:")
    print("  1. Random real HUMAN bidder")
    print("  2. Random real BOT bidder")
    print("  3. Show 5 humans in a row")
    print("  4. Show 5 bots in a row")
    print("  5. Compare human vs bot side by side")
    print("  6. Find a misclassified example")
    print("  q. Quit")

    choice = input("\n> ").strip().lower()

    if choice == "q":
        print("Done.")
        break

    elif choice == "1":
        row = humans.sample(1).iloc[0]
        prob = predict_row(row)
        print(f"\n  bidder_id: {row['bidder_id'][:20]}...")
        print(f"  total_bids: {int(row['total_bids'])}  |  unique_devices: {int(row['unique_devices'])}  |  rapid_bid_ratio: {row['rapid_bid_ratio']:.3f}")
        show_result(prob, row["outcome"])

    elif choice == "2":
        row = bots.sample(1).iloc[0]
        prob = predict_row(row)
        print(f"\n  bidder_id: {row['bidder_id'][:20]}...")
        print(f"  total_bids: {int(row['total_bids'])}  |  unique_devices: {int(row['unique_devices'])}  |  rapid_bid_ratio: {row['rapid_bid_ratio']:.3f}")
        show_result(prob, row["outcome"])

    elif choice == "3":
        print("\n  --- 5 Random Human Bidders ---")
        for _, row in humans.sample(5).iterrows():
            prob    = predict_row(row)
            pred    = "HUMAN" if prob < 0.5 else "BOT"
            correct = "✓" if prob < 0.5 else "✗"
            print(f"  bids={int(row['total_bids']):6d}  devices={int(row['unique_devices']):5d}  rapid={row['rapid_bid_ratio']:.3f}  -> {prob:.3f} {pred} {correct}")

    elif choice == "4":
        print("\n  --- 5 Random Bot Bidders ---")
        for _, row in bots.sample(5).iterrows():
            prob    = predict_row(row)
            pred    = "BOT" if prob >= 0.5 else "HUMAN"
            correct = "✓" if prob >= 0.5 else "✗"
            print(f"  bids={int(row['total_bids']):6d}  devices={int(row['unique_devices']):5d}  rapid={row['rapid_bid_ratio']:.3f}  -> {prob:.3f} {pred} {correct}")

    elif choice == "5":
        h  = humans.sample(1).iloc[0]
        b  = bots.sample(1).iloc[0]
        ph = predict_row(h)
        pb = predict_row(b)
        print(f"\n  {'Feature':<25} {'HUMAN':>10} {'BOT':>10}")
        print(f"  {'-'*45}")
        for feat in ["total_bids", "unique_devices", "unique_ips", "rapid_bid_ratio", "ip_diversity"]:
            print(f"  {feat:<25} {h[feat]:>10.3f} {b[feat]:>10.3f}")
        print(f"  {'-'*45}")
        print(f"  {'Bot probability':<25} {ph:>10.4f} {pb:>10.4f}")
        print(f"  {'Prediction':<25} {'HUMAN':>10} {'BOT':>10}")

    elif choice == "6":
        print("\n  Looking for misclassified examples...")
        found = False
        for _, row in df.sample(min(200, len(df))).iterrows():
            prob       = predict_row(row)
            pred_bot   = prob >= 0.5
            actual_bot = row["outcome"] == 1.0
            if pred_bot != actual_bot:
                print(f"\n  bidder_id: {row['bidder_id'][:20]}...")
                print(f"  total_bids: {int(row['total_bids'])}  |  unique_devices: {int(row['unique_devices'])}  |  rapid_bid_ratio: {row['rapid_bid_ratio']:.3f}")
                show_result(prob, row["outcome"])
                found = True
                break
        if not found:
            print("  No misclassified example found in this sample. Try again!")

    else:
        print("Invalid choice.")
