import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("REPORT_REDMINE_31-08-2024 - REPORT_REDMINE_31-08-2024.csv", encoding="ISO-8859-1")

# Gunakan kolom Perihal sebagai fitur dan Prioritas sebagai label
df = df.dropna(subset=["Perihal", "Prioritas"])
X_text = df["Perihal"]
y = df["Prioritas"]

# Ubah teks menjadi fitur numerik
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# Encode label prioritas menjadi angka
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Bagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Latih model SVM
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Simpan model dan alat bantu
joblib.dump(model, "svm_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(encoder, "label_encoder.pkl")
