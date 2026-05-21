
import os, sys, pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess import CropPreprocessor
from market import get_market_info, get_all_market_data, get_top_market_crops

app = Flask(__name__)

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "models")

def load_artifacts():
    pre = CropPreprocessor.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    with open(os.path.join(MODEL_DIR, "ensemble_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "rb") as f:
        metrics = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_importance.pkl"), "rb") as f:
        fi = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "classes.pkl"), "rb") as f:
        classes = pickle.load(f)
    return pre, model, metrics, fi, classes

preprocessor, ensemble_model, model_metrics, feat_importance, crop_classes = load_artifacts()

@app.route("/")
def index():
    top_crops = get_top_market_crops()
    return render_template("index.html", top_crops=top_crops)

@app.route("/predict", methods=["GET"])
def predict_form():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "Temperature": float(request.form["temperature"]),
            "Rainfall": float(request.form["rainfall"]),
            "Humidity": float(request.form["humidity"]),
            "Soil_pH": float(request.form["soil_ph"]),
            "Nitrogen": float(request.form["nitrogen"]),
            "Phosphorus": float(request.form["phosphorus"]),
            "Potassium": float(request.form["potassium"]),
            "Altitude": float(request.form["altitude"]),
            "Region": int(request.form["region"]),
        }
        if not (-20 <= data["Temperature"] <= 50):
            raise ValueError("Temperature must be -20 to 50 C")
        if not (0 <= data["Humidity"] <= 100):
            raise ValueError("Humidity must be 0-100%")
        if not (3.5 <= data["Soil_pH"] <= 9.5):
            raise ValueError("Soil pH must be 3.5-9.5")

        X = preprocessor.transform(data)
        probs = ensemble_model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        predicted_crop = crop_classes[pred_idx]
        confidence = round(probs[pred_idx] * 100, 2)

        crop_probs = [{"crop": c, "prob": round(p * 100, 2)} for c, p in zip(crop_classes, probs)]
        crop_probs.sort(key=lambda x: x["prob"], reverse=True)

        market = get_market_info(predicted_crop)
        alt_recommendations = []
        for cp in crop_probs[1:3]:
            alt_info = get_market_info(cp["crop"])
            alt_info["probability"] = cp["prob"]
            alt_recommendations.append(alt_info)

        return render_template("result.html",
            predicted_crop=predicted_crop,
            confidence=confidence,
            market=market,
            crop_probs=crop_probs,
            input_data=data,
            alt_recommendations=alt_recommendations,
            region_name="Hilly" if data["Region"] == 1 else "Polar"
        )
    except (ValueError, KeyError) as e:
        return render_template("predict.html", error=str(e))

@app.route("/dashboard")
def dashboard():
    metrics_list = [{"name": k, "accuracy": v} for k, v in model_metrics.items()]
    fi_sorted = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)
    fi_list = [{"feature": k.replace("_", " "), "importance": round(v, 4)} for k, v in fi_sorted]
    market_data = get_all_market_data()
    return render_template("dashboard.html",
        metrics=metrics_list,
        feature_importance=fi_list,
        market_data=market_data,
        best_model=max(model_metrics, key=model_metrics.get),
        best_accuracy=max(model_metrics.values())
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        body = request.get_json()
        X = preprocessor.transform(body)
        probs = ensemble_model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        predicted = crop_classes[pred_idx]
        return jsonify({
            "predicted_crop": predicted,
            "confidence": round(float(probs[pred_idx]) * 100, 2),
            "all_probabilities": {c: round(float(p) * 100, 2) for c, p in zip(crop_classes, probs)},
            "market_info": get_market_info(predicted)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
