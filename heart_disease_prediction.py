# ═══════════════════════════════════════════════════════════════
# Heart Disease Prediction with XGBoost + SHAP Explainability
# Dataset: UCI Heart Disease (Cleveland, 303 patients, 14 features)
# Developer: Aqib Hussain
# ═══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score, f1_score
)
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb

SEED = 42
np.random.seed(SEED)

# ── Load & Explore ─────────────────────────────────────────────────
df = pd.read_csv('heart.csv')   # Kaggle: heart-disease-uci
print(f"Shape: {df.shape}")
print(df.head())
print(df['target'].value_counts())

# ── Feature names ──────────────────────────────────────────────────
FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal']
TARGET   = 'target'

X = df[FEATURES].values
y = df[TARGET].values

# ── EDA: Correlation heatmap ───────────────────────────────────────
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix — Heart Disease Dataset', fontsize=13)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=120)
plt.show()

# ── Preprocessing ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

# SMOTE — handle class imbalance if present
smote = SMOTE(random_state=SEED)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

scaler        = StandardScaler()
X_train_sc    = scaler.fit_transform(X_train_sm)
X_test_sc     = scaler.transform(X_test)

print(f"Train: {X_train_sc.shape}  |  Test: {X_test_sc.shape}")
print(f"After SMOTE — class 0: {(y_train_sm==0).sum()}  class 1: {(y_train_sm==1).sum()}")

# ── XGBoost Model ─────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators       = 500,
    max_depth          = 5,
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    min_child_weight   = 5,
    eval_metric        = 'logloss',
    early_stopping_rounds = 30,
    random_state       = SEED,
    use_label_encoder  = False
)
model.fit(
    X_train_sc, y_train_sm,
    eval_set=[(X_test_sc, y_test)],
    verbose=False
)

# ── 5-Fold Cross-Validation ────────────────────────────────────────
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(
    xgb.XGBClassifier(n_estimators=300, max_depth=5,
                      learning_rate=0.05, random_state=SEED,
                      use_label_encoder=False, eval_metric='logloss'),
    X_train_sc, y_train_sm, cv=cv, scoring='accuracy'
)
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Evaluation ────────────────────────────────────────────────────
y_pred  = model.predict(X_test_sc)
y_prob  = model.predict_proba(X_test_sc)[:, 1]
acc     = accuracy_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_prob)
f1      = f1_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Test Accuracy : {acc:.4f}")
print(f"AUC-ROC       : {auc:.4f}")
print(f"F1-Score      : {f1:.4f}")
print(f"{'='*50}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['No Disease', 'Heart Disease']))

# ── Plots ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Heart Disease Prediction — XGBoost Results', fontsize=13)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Disease','Heart Disease'],
            yticklabels=['No Disease','Heart Disease'],
            annot_kws={'size':14})
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='red', lw=2.5, label=f'AUC = {auc:.3f}')
axes[1].plot([0,1],[0,1],'k--', alpha=0.5, label='Random')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='red')
axes[1].set_title('ROC Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(); axes[1].grid(alpha=0.3)

# Feature Importance (XGBoost native)
xgb_imp = model.feature_importances_
sorted_idx = np.argsort(xgb_imp)
axes[2].barh([FEATURES[i] for i in sorted_idx], xgb_imp[sorted_idx],
             color='steelblue', edgecolor='white')
axes[2].set_title('XGBoost Feature Importance')
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=120, bbox_inches='tight')
plt.show()

# ── SHAP Explainability ───────────────────────────────────────────
print("\nGenerating SHAP explanations...")
explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_test_sc)

# Global importance
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals, X_test_sc, feature_names=FEATURES,
                  plot_type='bar', show=False)
plt.title('SHAP — Global Feature Importance', fontsize=12)
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=120)
plt.show()

# Beeswarm — direction of impact
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_test_sc, feature_names=FEATURES, show=False)
plt.title('SHAP Beeswarm — Feature Impact Direction', fontsize=12)
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=120)
plt.show()

# Single patient explanation
high_risk = np.where(y_prob >= 0.8)[0]
if len(high_risk):
    idx = high_risk[0]
    shap.waterfall_plot(shap.Explanation(
        values=shap_vals[idx],
        base_values=explainer.expected_value,
        data=X_test_sc[idx],
        feature_names=FEATURES
    ), show=False)
    plt.title('SHAP Waterfall — Single High-Risk Patient')
    plt.tight_layout()
    plt.savefig('shap_waterfall.png', dpi=120)
    plt.show()

import pickle
with open('heart_model.pkl',  'wb') as f: pickle.dump(model, f)
with open('heart_scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
print("\n✅ Model saved → heart_model.pkl")
