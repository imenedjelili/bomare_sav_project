# dziribert_detection_service.py (Flask Service - DETECTION ONLY)
import logging
from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import os

# --- Configuration ---
DZIRIBERT_DETECTION_MODEL_NAME = os.getenv("DZIRIBERT_DETECTION_MODEL_NAME", "khaoula/dziribert")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

log = logging.getLogger("dziribert_detection_service")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

app = Flask(__name__)

# --- Global Model Variable ---
darija_classifier_pipeline = None

# --- Load Model on Startup ---
def load_detection_model():
    global darija_classifier_pipeline
    try:
        log.info(f"Loading Darija Detection model '{DZIRIBERT_DETECTION_MODEL_NAME}' on {DEVICE}...")
        darija_classifier_pipeline = pipeline(
            "text-classification",
            model=DZIRIBERT_DETECTION_MODEL_NAME,
            tokenizer=DZIRIBERT_DETECTION_MODEL_NAME,
            device=0 if DEVICE == "cuda" else -1
        )
        log.info("Darija Detection model loaded successfully.")
    except Exception as e:
        log.error(f"CRITICAL: Failed to load Darija Detection model: {e}", exc_info=True)

load_detection_model()

@app.route('/detect_darija', methods=['POST']) # Or keep as /process_darija if you prefer
def detect_darija_endpoint():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    text_input = data.get('text')
    if not text_input or not text_input.strip():
        return jsonify({"error": "Input text cannot be empty"}), 400

    if darija_classifier_pipeline is None:
        log.error("Darija detection model not available for processing.")
        return jsonify({"error": "Darija detection model not loaded", "is_darija": False, "confidence": 0.0}), 503

    try:
        log.info(f"Detecting Darija for text: '{text_input[:50]}...'")
        results = darija_classifier_pipeline(text_input)

        is_darija_pred = False
        confidence_pred = 0.0
        # --- !!! CRITICAL ADAPTATION POINT for your specific detection model's output !!! ---
        if results:
            top_result = results[0] if isinstance(results, list) else results
            if isinstance(top_result, dict) and 'label' in top_result and 'score' in top_result:
                label = str(top_result['label']).upper()
                score = top_result['score']
                # You need to know what label 'khaoula/dziribert' outputs for Darija.
                # Check its config.json or test it. Example:
                if "DARIJA" in label or label == "LABEL_1" or label == "ALGERIAN ARABIC": # MODIFY THIS
                    is_darija_pred = True
                confidence_pred = float(score)
                log.info(f"Detection raw result: label='{label}', score={score:.4f}")
        # --- End of CRITICAL ADAPTATION ---

        return jsonify({
            "is_darija": is_darija_pred,
            "confidence": confidence_pred,
            "processed_text": text_input
            # No translation fields here
        })
    except Exception as e:
        log.error(f"Error during Darija detection: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error during detection: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
