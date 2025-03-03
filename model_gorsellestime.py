import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# Test tahminleri
y_probs = model.predict_proba(X_test)[:, 1]

# ROC eğrisi
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('Test Başarısızlık Tahmini ROC Eğrisi')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

# Özellik önem derecelerini görselleştirme
feature_importance = pd.DataFrame({
    'özellik': X.columns,
    'önem': model.feature_importances_
}).sort_values('önem', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='önem', y='özellik', data=feature_importance)
plt.title('Özellik Önem Dereceleri')
plt.tight_layout()
plt.savefig('feature_importance.png') 