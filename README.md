---
title: Social Network Ads Web Interface
emoji: 🏆
colorFrom: indigo
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---
# social-network-ads-web-interface
Hugging Face ML Deployment of Streamlit App for a Customer Purchase Prediction Model (Purchased - 1 or no Purchase - 0)
https://huggingface.co/spaces/saifhmb/Social-Network-Ads-Web-Interface

# Model description

This is a logistic regression classifier trained on social network ads dataset (https://huggingface.co/datasets/saifhmb/social-network-ads).
## Training Procedure
The preprocesing steps include using a train/test split ratio of 80/20 and applying feature scaling on all the features.
### Model Plot
![image](https://github.com/saifhmb/social-network-ads-web-interface/assets/111028776/bd23ae48-128f-48ad-a692-1dfdfea1c604)

## Evaluation Results

| Metric    |    Value |
|-----------|----------|
| accuracy  | 0.925    |
| precision | 0.944444 |
| recall    | 0.772727 |
### Confusion Matrix
![image](https://github.com/saifhmb/social-network-ads-web-interface/assets/111028776/1a4dc2f5-b2c5-4c52-a9d2-7a305922d66b)
