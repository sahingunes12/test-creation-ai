from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Veriyi yükle
df = pd.read_csv('test_verileri.csv')
X = df.drop('basarisiz', axis=1)
y = df['basarisiz']

# Hiper-parametre optimizasyonu
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X, y)

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi doğruluk:", grid_search.best_score_)

# En iyi modeli kaydet
best_model = grid_search.best_estimator_
import pickle
with open('gelismis_test_model.pkl', 'wb') as f:
    pickle.dump(best_model, f) 