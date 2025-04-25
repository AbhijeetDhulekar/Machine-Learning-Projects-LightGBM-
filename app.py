import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_executor import Executor
import uuid
from threading import Lock
from datetime import datetime
import os


try:
    model = joblib.load("lgbm_model.joblib")
    scaler = joblib.load("scaler.joblib")
    print("INFO: Model and Scaler loaded successfully")
except Exception as e:
    print(f"ERROR: Failed to load model/scaler: {e}")
    model, scaler = None, None

app = Flask(__name__)
executor = Executor(app)


async_results = {}
results_lock = Lock()

def validate_and_preprocess(data):
    """Validate input data and preprocess for prediction."""
    required_fields = [
        "pclass", "sex", "age", "sibsp", "parch", "fare", 
        "embarked", "familysize", "is_alone", "deck"
    ]
    
    for field in required_fields:
        if field not in data:
            return {"error": f"Missing field: {field}"}, None

    try:
        if not (1 <= float(data["pclass"]) <= 3):
            return {"error": "Pclass must be 1, 2, or 3"}, None
            
        if not (0 <= float(data["sex"]) <= 1):
            return {"error": "Sex must be 0 (female) or 1 (male)"}, None
            
        if not (0 < float(data["age"]) <= 120):
            return {"error": "Age must be between 0-120"}, None
            
        if not (0 <= float(data["embarked"]) <= 2):
            return {"error": "Embarked must be 0-2 (C=0, Q=1, S=2)"}, None
            
        if not (0 <= float(data["deck"]) <= 7):
            return {"error": "Deck must be between 0-7"}, None

        features = [float(data[field]) for field in required_fields]
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        return None, features
        
    except ValueError as e:
        return {"error": f"Invalid input values: {str(e)}"}, None

@app.route("/")
def home():
    return render_template("titanic_prediction.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/health")
def health_check():
    status = {
        "status": "ready" if model and scaler else "error",
        "model_loaded": bool(model),
        "scaler_loaded": bool(scaler),
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(status)

@app.route("/titanic_sync", methods=["POST"])
def predict_sync():
    if model is None or scaler is None:
        return jsonify({
            "status": "error",
            "message": "Service unavailable",
            "details": "Model or Scaler not loaded"
        }), 503

    try:
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({
                "status": "error",
                "message": "Invalid request format",
                "details": "Expected { 'data': { ... } }"
            }), 400
        
        error, features = validate_and_preprocess(data["data"])
        if error:
            return jsonify({
                "status": "error",
                **error
            }), 400

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0, 1]
        
        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "survived": bool(prediction),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Prediction failed",
            "details": str(e)
        }), 500

@app.route("/titanic_async", methods=["POST"])
def predict_async():
    if model is None or scaler is None:
        return jsonify({
            "status": "error",
            "message": "Service unavailable"
        }), 503

    try:
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({
                "status": "error",
                "message": "Invalid request format"
            }), 400
            
        error, _ = validate_and_preprocess(data["data"])
        if error:
            return jsonify({
                "status": "error",
                **error
            }), 400

        job_id = str(uuid.uuid4())
        
        with results_lock:
            async_results[job_id] = {
                "status": "processing",
                "submitted_at": datetime.now().isoformat()
            }
        
        executor.submit(process_async_prediction, data, job_id)
        
        return jsonify({
            "status": "accepted",
            "job_id": job_id,
            "message": "Prediction processing started"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to submit job",
            "details": str(e)
        }), 500

def process_async_prediction(data, job_id):
    try:
        error, features = validate_and_preprocess(data.get("data", {}))
        
        if error:
            result = {
                "status": "error",
                **error,
                "completed_at": datetime.now().isoformat()
            }
        else:
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0, 1]
            
            result = {
                "status": "completed",
                "prediction": int(prediction),
                "probability": round(float(probability), 4),
                "survived": bool(prediction),
                "completed_at": datetime.now().isoformat()
            }
        
        with results_lock:
            async_results[job_id] = result
            
    except Exception as e:
        with results_lock:
            async_results[job_id] = {
                "status": "error",
                "message": "Processing failed",
                "details": str(e),
                "completed_at": datetime.now().isoformat()
            }

@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id):
    with results_lock:
        result = async_results.get(job_id, {
            "status": "not_found",
            "message": "Job ID not found"
        })
    
    if result["status"] == "processing":
        return jsonify({
            "status": "processing",
            "job_id": job_id,
            "message": "Prediction in progress"
        })
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5005)