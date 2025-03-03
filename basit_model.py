import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Veriyi yükle
df = pd.read_csv('test_verileri.csv')

# Özellikler ve hedefi ayır
X = df.drop('basarisiz', axis=1)
y = df['basarisiz']

# Eğitim ve test verilerini böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahminler yap
y_pred = model.predict(X_test)

# Sonuçları değerlendir
print("Model doğruluğu:", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Modeli kaydet
import pickle
with open('test_tahmin_modeli.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model kaydedildi: test_tahmin_modeli.pkl") 