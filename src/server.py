from flask import Flask, request
import random, string, json, os
from models.bird_classification_service import _Bird_Classifier

# This constants must match the parameter used for preprocessing MFCCs for training
SAMPLE_RATE = 16000
WINDOW_LENGTH = 1

TRAINED_MODEL_PATH = './models/00-production_model/production_model.keras'

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions. Takes an audio file and returns multiple
    predictions, corresponding to each audio frame.
    Returns a JSON file with the following format:
    {
        {
            'label_code': int
            'species': str (scientific name)
            'probability': float (0-100)
            'time_section: (hh:mm:ss.mill, hh:mm:ss.mill)
        }
        {
            ...
        }
    }
    """
    # Get file from post request and save with random name
    audio_file = request.files['file']
    filename = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
    audio_file.save(filename)

    # Instantiate the classifier service
    classifier = _Bird_Classifier(
        model_path=TRAINED_MODEL_PATH,
        sr=SAMPLE_RATE,
        window_length=WINDOW_LENGTH,
    )

    # Make predictions and convert to JSON format
    predictions = classifier.predict(filename)
    predictions = json.dumps(predictions, indent=4)

    # Delete stored file
    os.remove(filename)

    return predictions


if __name__ == "__main__":
    app.run(debug=False)
