import pandas as pd
import numpy as np

# Test verileri oluşturma örneği
def test_verisi_olustur():
    # Örnek: Test senaryoları ve sonuçları
    test_verileri = {
        'test_suresi': np.random.randint(10, 100, 1000),
        'kod_degisiklik_yuzde': np.random.randint(1, 50, 1000),
        'test_kapsamı': np.random.randint(50, 100, 1000),
        'onceki_hata_sayisi': np.random.randint(0, 20, 1000),
        'basarisiz': np.random.randint(0, 2, 1000)  # 0: Başarılı, 1: Başarısız
    }
    
    df = pd.DataFrame(test_verileri)
    return df

# Veriyi kaydet
df = test_verisi_olustur()
df.to_csv('test_verileri.csv', index=False)
print("Veri oluşturuldu ve kaydedildi.") 