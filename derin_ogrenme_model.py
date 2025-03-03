import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Veri yükleme ve hazırlama
df = pd.read_csv('test_verileri.csv')
X = df.drop('basarisiz', axis=1).values
y = df['basarisiz'].values

# Verileri standartlaştırma
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Veriyi bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model oluşturma
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

# Modeli değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test doğruluğu: {accuracy:.4f}")

# Modeli kaydetme
model.save('derin_test_modeli.h5')
print("Derin öğrenme modeli kaydedildi.") 