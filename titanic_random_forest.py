import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Veriyi Yükle
df = pd.read_csv("train.csv", sep=',', encoding='utf-8')

# 2. Gereksiz Kolonları At
df = df.drop(columns=['Name', 'Ticket', 'Cabin'], errors='ignore')

# 3. Eksik Verileri Doldur
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 4. Kategorik Verileri Sayısala Çevir
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])         # male -> 1, female -> 0
df['Embarked'] = le.fit_transform(df['Embarked'])  # S, C, Q -> 0,1,2

# 5. Giriş ve Çıkışları Ayır
X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

# 6. Ölçekleme (Tutarlılık için)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Eğitim ve Test Seti
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Random Forest Modeli
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_model.fit(X_train, y_train)

# 9. Tahmin ve Değerlendirme
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# 10. Confusion Matrix Görselleştirme
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Not Survived', 'Survived'])
fig, ax = plt.subplots(figsize=(8, 6))
disp_rf.plot(cmap=plt.cm.Greens, ax=ax, colorbar=False)
plt.title("Random Forest - Confusion Matrix")
plt.grid(False)
plt.show()

# === Kullanıcıdan Girdi Alarak Tahmin Yapma ===
print("\n--- Hayatta Kalma Tahmini ---")


pclass = int(input("Yolcu Sınıfı (1 = üst, 2 = orta, 3 = alt): "))
sex_input = input("Cinsiyet (male/female): ").strip().lower()
age = float(input("Yaş: "))

# Cinsiyeti sayısala çevir
if sex_input == "male":
    sex = 1
elif sex_input == "female":
    sex = 0



user_input = np.array([[pclass, sex, age]])
user_input_scaled = scaler.transform(user_input)
prediction = rf_model.predict(user_input_scaled)

print(f"\nTahmin: {'✅ Hayatta Kaldı' if prediction == 1 else '❌ Hayatta Kalamadı'}")

#KNN ile analiz etme

# 1. KNN Modeli Oluştur
knn_model = KNeighborsClassifier(n_neighbors=5)

# 2. Eğit
knn_model.fit(X_train, y_train)

# 3. Tahmin
y_pred_knn = knn_model.predict(X_test)

# 4. Değerlendirme
print("\nKNN SONUÇLARI")
print("Accuracy (KNN):", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report (KNN):\n", classification_report(y_test, y_pred_knn))
print("\nConfusion Matrix (KNN):\n", confusion_matrix(y_test, y_pred_knn))

# 5. Karışıklık Matrisi Görselleştir
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=['Not Survived', 'Survived'])

fig, ax = plt.subplots(figsize=(8, 6))
disp_knn.plot(cmap=plt.cm.Oranges, ax=ax, colorbar=False)
plt.title("KNN Confusion Matrix")
plt.grid(False)
plt.show()






