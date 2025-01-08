

# Model Description

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

### Model Explainability
SHAP was used to determine the important features that helps the model make decisions

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6662300a0ad8c45a1ce59190/ZoG4Wai4QeEBoBdwKsclW.png)

### Confusion Matrix
![image](https://github.com/saifhmb/social-network-ads-web-interface/assets/111028776/1a4dc2f5-b2c5-4c52-a9d2-7a305922d66b)
