import requests
import json

# Endpoint url
URL = 'http://127.0.0.1:5000/predict'


# audio file we'd like to send for predicting keyword
FILE_PATH = './src/test files/antrostomus_rufus.mp3'
ext = FILE_PATH.split('.')[-1]


if __name__ == '__main__':
    # Open file
    file = open(FILE_PATH, 'rb')

    # Send POST request to server
    values = {'file': (FILE_PATH, file, f'audio/{ext}')}
    response = requests.post(URL, files=values)

    # Print predictions and save JSON
    data = response.json()
    with open('server_predictions_response.json', 'w') as jf:
        json.dump({'file': FILE_PATH, 'preictions': data}, jf, indent=2)

    for prediction in data:
        print(prediction)
