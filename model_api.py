from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn

app = FastAPI(title="Test Tahmin API", description="Test başarısızlıklarını tahmin eder")

# Modeli yükle
with open('test_tahmin_modeli.pkl', 'rb') as f:
    model = pickle.load(f)

class TestOzellikleri(BaseModel):
    test_suresi: int
    kod_degisiklik_yuzde: int
    test_kapsami: int
    onceki_hata_sayisi: int

class TahminSonucu(BaseModel):
    basarisiz_olacak: bool
    basarisizlik_olasiligi: float

@app.post("/tahmin/", response_model=TahminSonucu)
def tahmin_yap(ozellikler: TestOzellikleri):
    """
    # Make a prediction for test failure based on the input test metrics
    """
    try:
        veri = pd.DataFrame({
            'test_suresi': [ozellikler.test_suresi],
            'kod_degisiklik_yuzde': [ozellikler.kod_degisiklik_yuzde],
            'test_kapsamı': [ozellikler.test_kapsami],
            'onceki_hata_sayisi': [ozellikler.onceki_hata_sayisi]
        })
        
        tahmin = bool(model.predict(veri)[0])
        olasilik = float(model.predict_proba(veri)[0][1])
        
        return TahminSonucu(
            basarisiz_olacak=tahmin,
            basarisizlik_olasiligi=olasilik
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("model_api:app", host="0.0.0.0", port=8000, reload=True) 