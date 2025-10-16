# ING Datathon Projesi

Bu depo, ING Datathon için özellik mühendisliği, modelleme ve topluluk (ensembling) kodlarını içerir.

## Hızlı Başlangıç (Windows PowerShell)

1) Sanal ortam oluştur/etkinleştir (önerilir):

```powershell
python -m venv .\venv
.\venv\Scripts\Activate.ps1
```

2) Bağımlılıkları kur:

```powershell
pip install -r requirements.txt
```

3) Tam hattı çalıştır (eğitim + çıkarım + gönderim):

```powershell
python .\main.py
```

Çıktılar `data/submissions/` ve `outputs/` altına yazılır (detaylar aşağıda).

## Klasör Yapısı

- `src/` — kaynak kod
  - `features/` — zaman güvenli özellik üretimi modülleri
  - `models/` — çapraz doğrulama, LightGBM/XGBoost/CatBoost/Two-Stage başlıkları
  - `ensemble/` — harmanlama, stacking, kalibrasyon araçları
  - `utils/` — yardımcılar: değerlendirme, veri kaydetme, ayar optimizasyonu
- `data/` — veri kümeleri ve gönderimler
  - `raw/` — ham girdiler (takip edilmez)
  - `processed/` — işlenmiş ara çıktılar (takip edilmez)
  - `submissions/` — üretilen gönderimler (takip edilmez)
  - `portfolio/` — portföy ile ilgili dosyalar (takip edilmez)
- `outputs/` — tahminler, raporlar ve günlükler (takip edilmez)
  - `predictions/` — tahmin paketleri ve OOF çıktıları
  - `reports/` — özellik önemleri, tanılama dosyaları
  - `catboost_info/` — CatBoost eğitim günlükleri
- `configs/` — ayar dosyaları ve optimize edilmiş parametreler
- `models/` — eğitilmiş model artefaktları (takip edilmez)
- `notebooks/` — keşif amaçlı defterler
- `scripts/` — yardımcı betikler ve CLI araçları

Proje giriş noktaları:
- `src/main.py` veya depo kökündeki `main.py` — uçtan uca eğitim/çıkarım
- `portfolio_runner.py` — gönderimleri kümeler, program planlar ve README günceller

## Veri akış diyagramı

```mermaid
flowchart LR
  A[Ham veri CSV'leri\n(data/raw/*.csv)] --> B[Yükleme & Ön İşleme\n(main.py)]
  B --> C[Özellik Mühendisliği\nfeatures.basic + features.advanced]
  C --> D[Matris Önbellekleme\nutils/save_training_data.py\noutputs/predictions/*.pkl]
  C --> E[Modeller\nLGB / XGB / CatBoost / Two-Stage\nZaman tabanlı ÇD]
  E --> F[OOF + Test Tahminleri]
  F --> G[Topluluk\nstacking.py + blend_submissions.py]
  G --> H[Kalibrasyon\nizotonik / beta + gamma taraması]
  H --> I[Gönderim CSV\n(data/submissions/submission*.csv)]
  E --> J[Raporlar\nÖzellik önemleri -> outputs/reports]
  E --> K[CatBoost günlükleri -> outputs/catboost_info]
```

## Çapraz doğrulama şeması (zaman tabanlı)

- Kronolojik, ay bazlı katlamalar; karıştırma yok.
- Doğrulama ayı t için, eğitim verisi t ayından kesinlikle önceki tüm verilerden oluşur.
- Tipik kurulum, son 5–6 ayı doğrulama katları olarak kullanır.

Örnek (6 ay):

```
Kat 1: Eğitim [A1..A4] -> Doğrulama A5
Kat 2: Eğitim [A1..A5] -> Doğrulama A6
...
```

Bu yaklaşım sızıntıdan arınmış değerlendirme sağlar ve liderlik tablosu ayını taklit eder.

## Sızıntı kontrolleri

- Tüm özellikler `ref_date` ve kıvrım (fold) eğitim kesimleriyle uyumlu, kesinlikle <= koşulu ile hesaplanır.
- Trend pencereleri ve dönem filtreleri, bitiş tarihlerinin kesimden önce olduğunu doğrular (`src/features/feature_engineering.py`).
- `src/features/advanced_features.py` içinde `validate_no_future_leakage(df, ref_date)` yardımcı işlevi sağlanır:
  - Tarih/dönem sütunlarını (örn. `window_end`, `date`, `month`) algılar ve herhangi bir değer `ref_date` >= ise hata fırlatır.
  - Herhangi bir dönemsellik veya kayan pencere veri çerçevesi oluşturduktan sonra kullanın.
- Trend/oran (ratio) özellikleri için aykırı değer kontrolü: gelişmiş özellikler aşamasında [1, 99] yüzdeliklerde winsorization uygulanır; geleceğe bakmadan eğitimi stabilize eder.

Minimal örnek:

```python
from src.features.advanced_features import validate_no_future_leakage
validate_no_future_leakage(recent_data, ref_date)
```

## `main.py` çalıştığında neler olur?

- Girdi CSV’leri öncelikle `data/raw/` içinden yüklenir (gerekirse `data/` veya kök klasöre geri döner).
- LightGBM, XGBoost, CatBoost ve Two-Stage başlıkları zaman tabanlı çapraz doğrulama ile eğitilir.
- Modeller harmanlanır ve tahminler kalibre edilir.
- Gönderim `data/submissions/submission.csv` dosyasına kaydedilir.
- Her çalıştırmada `data/submissions/last_update.txt` (zaman damgası + satır sayısı) güncellenir.
- Portföy araçları için `outputs/predictions/predictions_bundle.pkl` oluşturulur.
- Özellik önemleri `outputs/reports/feature_importance.csv` dosyasına yazılır.
- CatBoost günlükleri `outputs/catboost_info/` altına yazılır.

## Nasıl yeniden üretirim?

1) Sanal ortam oluştur ve bağımlılıkları kur (Windows PowerShell):

```powershell
python -m venv .\venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Girdi CSV’lerini `data/raw/` altına koyun (gerekirse `data/`/kök klasör yedekleri çalışır):
  - `customers.csv`
  - `customer_history.csv`
  - `reference_data.csv` ve `reference_data_test.csv` (varsa)

3) Uçtan uca hattı çalıştırın:

```powershell
python .\main.py
```

4) İsteğe bağlı ekler:
  - Daha hızlı yineleme için ara matrisleri önbelleğe al:

```powershell
python -m src.utils.save_training_data
```

  - Taban OOF tahminleri üzerinde stacking meta-öğrenici eğit:

```powershell
python -m src.ensemble.stacking --help
```

  - Bölüm bazlı (segment-wise) L2 kısıtlı, negatif olmayan harmanlama çalıştır:

```powershell
python -m src.ensemble.blend_submissions --segments tenure_bin,avg_active_products_bin --l2 1e-3
```

  - Optuna ile hiperparametre ayarı:

```powershell
python -m src.tuning.optuna_tuner --model lgb --trials 200 --last-n 6
python -m src.tuning.optuna_tuner --model xgb --trials 200 --last-n 6
```

5) Çıktıları nerede bulurum?
  - Gönderimler: `data/submissions/submission*.csv`
  - Tahmin paketleri ve OOF’lar: `outputs/predictions/`
  - Raporlar (özellik önemi, harman ağırlıkları): `outputs/reports/`
  - CatBoost günlükleri: `outputs/catboost_info/`

## Yapılandırma betiği

Üstteki klasörleri oluşturan ve (isteğe bağlı) artefaktları güvenle taşıyan bir yardımcı betik mevcuttur.

Ön izleme (taşınacakları yazdırır):

```powershell
python .\setup_structure.py
```

Uygula (taşı):

```powershell
python .\setup_structure.py --apply
```

Betik yalnızca yaygın, kod olmayan artefaktları taşır ve mevcut dosyaların üzerine yazmaz. Hedefte aynı isim varsa, sayısal bir ek eklenir.

## Notlar

- Büyük dosyalar ve üretilen artefaktlar `.gitignore` ile görmezden gelinir.
- Ham verileri sürüm kontrolünün dışında tutun. Yerelde `data/raw/` altına koyun.
- `src/` içinde yeniden düzenleme yaparsanız, içe aktarmaların çalışması için `__init__.py` dosyalarının olduğundan emin olun.

## Portföy araçları

`portfolio_runner.py` üzerinden gönderim ve planlama yardımcılarını kullanabilirsiniz:

```powershell
python .\portfolio_runner.py --help
```

Tipik alt komutlar `portfolio_tools/` içindeki yardımcıları bağlar (ör. gönderim kümeleri, planlama, README notları güncelleme).
