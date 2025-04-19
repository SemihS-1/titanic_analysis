# 🚢 Titanic Hayatta Kalma Tahmini Projesi

Bu proje, Titanic faciası sırasında yolcuların hayatta kalıp kalamayacağını tahmin etmek amacıyla **Random Forest** algoritması kullanılarak geliştirilmiştir. Kullanıcıdan sadece üç bilgi alınarak (`Pclass`, `Sex`, `Age`) basit ama etkili bir makine öğrenmesi tahmini yapılmaktadır.

---

## 📂 Veri Kümesi

- Kullanılan veri: [`train.csv`](https://www.kaggle.com/c/titanic/data) (Kaggle Titanic yarışmasından alınmıştır)
- Özellikler (sadece bazıları kullanılmıştır):
  - `Pclass`: Yolcunun bilet sınıfı (1, 2, 3)
  - `Sex`: Cinsiyet (male/female)
  - `Age`: Yaş
  - `Survived`: Hedef değişken (0 = Hayatta kalamadı, 1 = Hayatta kaldı)

---

## ⚙️ Kullanılan Kütüphaneler

- pandas
- numpy
- matplotlib
- scikit-learn

---

## 🧠 Modelleme Süreci

1. Veriler temizlendi ve eksik veriler dolduruldu.
2. Kategorik değişkenler sayısal değerlere dönüştürüldü (ör. cinsiyet: male → 1, female → 0).
3. Standartlaştırma (`StandardScaler`) uygulandı.
4. Model olarak `RandomForestClassifier` kullanıldı.
5. Başarı oranı `accuracy_score`, `classification_report` ve `confusion_matrix` ile ölçüldü.

---

## 🧪 Örnek Kullanım

Program çalıştırıldığında kullanıcıdan aşağıdaki bilgiler istenir:

```text
Yolcu Sınıfı (1 = üst, 2 = orta, 3 = alt): 2
Cinsiyet (male/female): male
Yaş: 27
