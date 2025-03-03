import pickle
import pandas as pd

# Modeli yükle
with open('test_tahmin_modeli.pkl', 'rb') as f:
    model = pickle.load(f)

def test_basarisizligi_tahmin_et(test_suresi, kod_degisiklik_yuzde, test_kapsami, onceki_hata_sayisi):
    """
    # Function to predict test failure based on test metrics
    """
    veri = pd.DataFrame({
        'test_suresi': [test_suresi],
        'kod_degisiklik_yuzde': [kod_degisiklik_yuzde],
        'test_kapsamı': [test_kapsami],
        'onceki_hata_sayisi': [onceki_hata_sayisi]
    })
    
    tahmin = model.predict(veri)[0]
    olasilik = model.predict_proba(veri)[0][1]
    
    return tahmin, olasilik

# Örnek kullanım
sonuc, olasilik = test_basarisizligi_tahmin_et(45, 20, 75, 5)
if sonuc == 1:
    print(f"Test başarısız olabilir! Olasılık: {olasilik:.2f}")
else:
    print(f"Test muhtemelen başarılı olacak. Başarısızlık olasılığı: {olasilik:.2f}") 