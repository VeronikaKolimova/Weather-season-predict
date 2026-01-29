# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf


def plot_feature_importance_dt(model, numeric_features, title="Важность признаков", save_path=None):
    try:
        ohe = model.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = ohe.get_feature_names_out(['Weather'])
        all_feature_names = numeric_features + list(cat_feature_names)
        importances = model.named_steps['classifier'].feature_importances_

        weather_importance = sum(
            imp for name, imp in zip(all_feature_names, importances) if name.startswith('Weather_'))
        other_importances = [imp for name, imp in zip(all_feature_names, importances) if
                             not name.startswith('Weather_')]
        other_names = [name for name in all_feature_names if not name.startswith('Weather_')]

        importances_combined = other_importances + [weather_importance]
        names_combined = other_names + ['Осадки']

        sorted_idx = np.argsort(importances_combined)[::-1]
        sorted_names = [names_combined[i] for i in sorted_idx]
        sorted_importances = [importances_combined[i] for i in sorted_idx]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_names)), sorted_importances)
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.title(title)
        plt.xlabel('Важность')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        print("Ошибка при построении важности признаков:", e)
        return None


def plot_accuracy_comparison(acc_knn, acc_dt, save_path=None):
    models = ['KNN', 'Дерево решений']
    accuracies = [acc_knn, acc_dt]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
    plt.ylim(0, 1)
    plt.title('Сравнение точности моделей')
    plt.ylabel('Точность')
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf


