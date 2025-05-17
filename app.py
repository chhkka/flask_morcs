from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model dan alat bantu
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

@app.route("/prediksi-keluhan", methods=["POST"])
def prediksi_keluhan():
    try:
        data = request.get_json()
        deskripsi = data["deskripsi"]

        # Proses deskripsi
        X = vectorizer.transform([deskripsi])
        y_pred = model.predict(X)
        label = encoder.inverse_transform(y_pred)

        return jsonify({
            "deskripsi": deskripsi,
            "hasil_prediksi_prioritas": label[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
