"""
Model Training Module — Ensemble ML for Climate-Resilient Crop Prediction
Models: Random Forest, Gradient Boosting, XGBoost + VotingClassifier Ensemble
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import CropPreprocessor
from generate_data import generate_crop_dataset, generate_market_data

PALETTE = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
BG_COLOR = '#0f1923'
CARD_COLOR = '#1a2738'
TEXT_COLOR = '#e8f4f8'


def train_models():
    print("Generating dataset...")
    os.makedirs('../data', exist_ok=True)
    df = generate_crop_dataset(2000)
    df.to_csv('../data/crop_data.csv', index=False)
    market_df = generate_market_data()
    market_df.to_csv('../data/market_data.csv', index=False)

    print("Preprocessing...")
    preprocessor = CropPreprocessor()
    X, y = preprocessor.fit_transform(df)
    classes = preprocessor.get_classes()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5,
                         eval_metric='mlogloss', random_state=42, verbosity=0)
    svm = SVC(kernel='rbf', C=10, probability=True, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('xgb', xgb), ('svm', svm)],
        voting='soft'
    )

    models = {
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'XGBoost': xgb,
        'Support Vector Machine': svm,
        'Ensemble (Voting)': ensemble
    }

    print("Training models...")
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        metrics[name] = {'accuracy': round(acc * 100, 2), 'y_pred': y_pred}
        print(f"  {name}: {acc*100:.2f}%")

    feature_importance = dict(zip(preprocessor.feature_columns, rf.feature_importances_))

    os.makedirs('../models', exist_ok=True)
    preprocessor.save('../models/preprocessor.pkl')
    with open('../models/ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    with open('../models/metrics.pkl', 'wb') as f:
        pickle.dump({k: v['accuracy'] for k, v in metrics.items()}, f)
    with open('../models/feature_importance.pkl', 'wb') as f:
        pickle.dump(feature_importance, f)
    with open('../models/classes.pkl', 'wb') as f:
        pickle.dump(classes, f)

    os.makedirs('../results', exist_ok=True)
    _plot_model_comparison(metrics)
    _plot_feature_importance(feature_importance, preprocessor.feature_columns)
    _plot_confusion_matrix(y_test, metrics['Ensemble (Voting)']['y_pred'], classes)
    _plot_crop_distribution(df)

    print("\nAll models trained and saved successfully!")
    return metrics, feature_importance


def _styled_fig(w=10, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2c3e50')
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    return fig, ax


def _plot_model_comparison(metrics):
    short = ['RF', 'GB', 'XGB', 'SVM', 'Ensemble']
    accs  = [metrics[n]['accuracy'] for n in metrics.keys()]
    fig, ax = _styled_fig(11, 6)
    bars = ax.bar(short, accs, color=PALETTE, width=0.55, zorder=3, edgecolor='white', linewidth=0.6)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{acc:.1f}%', ha='center', va='bottom', color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax.set_ylim(85, 103)
    ax.set_title('Model Accuracy Comparison', fontsize=15, fontweight='bold', pad=14)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.grid(axis='y', alpha=0.25, color='white', zorder=0)
    plt.tight_layout()
    plt.savefig('../results/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_feature_importance(fi, columns):
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    feat_names = [x[0].replace('_', ' ') for x in sorted_fi]
    values     = [x[1] for x in sorted_fi]
    fig, ax = _styled_fig(10, 6)
    bars = ax.barh(feat_names, values, color=PALETTE * 2, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color=TEXT_COLOR, fontsize=9)
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.2, color='white')
    plt.tight_layout()
    plt.savefig('../results/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = _styled_fig(8, 7)
    im = ax.imshow(cm, interpolation='nearest', cmap='YlOrRd')
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=35, ha='right', color=TEXT_COLOR)
    ax.set_yticklabels(classes, color=TEXT_COLOR)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] < thresh else 'black', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix — Ensemble Model', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    plt.tight_layout()
    plt.savefig('../results/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_crop_distribution(df):
    counts = df['Crop'].value_counts()
    fig, ax = _styled_fig(8, 7)
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct='%1.1f%%',
        colors=PALETTE, startangle=140,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
        textprops=dict(color=TEXT_COLOR, fontsize=12)
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight('bold')
    ax.set_title('Crop Distribution in Dataset', fontsize=14, fontweight='bold', pad=12)
    ax.set_facecolor(CARD_COLOR)
    plt.tight_layout()
    plt.savefig('../results/crop_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    train_models()
