from flask import Flask, request, render_template
import torch.nn as nn
import torch
import pickle
import numpy as np
from torchvision import transforms
import cv2
import base64
import requests
from net import Net

app = Flask(__name__)
ML_MODEL = None
ML_MODEL_FILE = "model.pt"
TORCH_DEVICE = "cpu"

def get_model():
    """Loading the ML model once and returning the ML model"""
    global ML_MODEL
    if not ML_MODEL:
        ML_MODEL = Net()
        ML_MODEL.load_state_dict(
            torch.load(ML_MODEL_FILE, map_location=torch.device(TORCH_DEVICE))
        )

    return ML_MODEL

def freshness_label(freshness_percentage):
    if freshness_percentage > 90:
        return "Fresh"
    elif freshness_percentage > 65:
        return "Very Good"
    elif freshness_percentage > 50:
        return "Good"
    elif freshness_percentage > 0:
        return "Not Good"
    else:
        return "Bad or Not a Fruit"

def price_to_text(price):
    if price == 0:
        return "Free"
    return str(price)

def price_by_freshness_percentage(freshness_percentage):
    return int(freshness_percentage/100*10000)

def freshness_percentage_by_cv_image(cv_image):
    """
    Reference: https://github.com/anshuls235/freshness-detector/blob/4cd289fb05a14d3c710813fca4d8d03987d656e5/main.py#L40
    """
    mean = (0.7369, 0.6360, 0.5318)
    std = (0.3281, 0.3417, 0.3704)
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))
    image_tensor = transformation(image)
    batch = image_tensor.unsqueeze(0)
    out = get_model()(batch)
    s = nn.Softmax(dim=1)
    result = s(out)
    return int(result[0][0].item()*100)

def imdecode_image(image_file):
    return cv2.imdecode(
        np.frombuffer(image_file.read(), np.uint8),
        cv2.IMREAD_UNCHANGED
    )

def get_token(new=False):
    if new:
        url = "https://datalabs.siva3.io/api/v1/auth/token"

        payload = "grant_type=&username=temp&password=temp&scope=&client_id=&client_secret="
        headers = {
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        response = requests.post(url, headers=headers, data=payload)

        json_response = response.json() # Converts the response to JSON format

        with open('response_key.pickle', 'wb') as f:
            pickle.dump(json_response.get('access_token'), f)
        token = json_response.get('access_token')
        return token  

    try:
        with open('response_key.pickle', 'rb') as f:
            token = pickle.load(f)
            return token
    except:
        url = "https://datalabs.siva3.io/api/v1/auth/token"

        payload = "grant_type=&username=temp&password=temp&scope=&client_id=&client_secret="
        headers = {
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        response = requests.post(url, headers=headers, data=payload)

        json_response = response.json() # Converts the response to JSON format

        with open('response_key.pickle', 'wb') as f:
            pickle.dump(json_response.get('access_token'), f)
        token = json_response.get('access_token')
        return token    

def get_image_name(encoded_image):
    #encoded_image = base64.b64encode(image)
    data = {"image":encoded_image.decode('utf-8')}
    token = get_token()
    headers = {"Authorization": "Bearer "+token}
    response = requests.post('https://datalabs.siva3.io/api/v1/data_vision/object_detection',headers=headers,json=data)
    if response.status_code ==200:
        response = response.json()
        return response.get('detected_object')
    else:
        token = get_token(new=True)
        headers = {"Authorization": "Bearer "+token}
        response = requests.post('https://datalabs.siva3.io/api/v1/data_vision/object_detection',headers=headers,json=data)
        if response.status_code ==200:
            response = response.json()
            return response.get('detected_object')
    print("resp",response.text)
    return "Can't Identify Image/ File size too large"

def get_calories(query):
    if query == "Can't Identify Fruit" or query== "NoObjectDetected":
        return 'No Cals'
    TRIVIA_URL = 'https://api.api-ninjas.com/v1/nutrition?query={}'.format(query)
    resp = requests.get(TRIVIA_URL, headers={'X-Api-Key': '6HEKRbScTeIZV20E+zIoyg==FDmA97iqVPcia1OS'}).json()
    # Get first trivia result since the API returns a list of results.
    try:
        return resp[0].get('calories')
    except: return 'No cals'

def recognize_fruit_by_cv_image(cv_image):
    freshness_percentage = freshness_percentage_by_cv_image(cv_image)
    return {
        # TODO: change freshness_level to freshness_percentage
        "freshness_level": freshness_percentage,
        "price": price_by_freshness_percentage(freshness_percentage)
        
    }

## API

@app.route('/api/recognize', methods=["POST"])
def api_recognize():
    cv_image = imdecode_image(request.files["image"])
    return recognize_fruit_by_cv_image(cv_image)

## Web pages

@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/checkout", methods=["POST"])
def checkout_page():
    image = request.form['image']
    image = base64.b64decode(image.split("data:image/jpeg;base64,")[-1])
    with open("image.jpg", "wb") as f:
        f.write(image)
    f = open("image.jpg", "rb")
    cv_image = imdecode_image(f)
    fruit_information = recognize_fruit_by_cv_image(cv_image)
    # TODO: change freshness_level to freshness_percentage
    freshness_percentage = fruit_information["freshness_level"]

    # show submitted image
    image_content = cv2.imencode('.jpg', cv_image)[1].tobytes()
    encoded_image = base64.encodebytes(image_content)
    detected_objects = get_image_name(encoded_image)
    print('enc img',detected_objects,len(encoded_image))
    base64_image = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    cals = get_calories(detected_objects)
    
    return render_template(
        "checkout.html",
        freshness_percentage=freshness_percentage,
        freshness_label=freshness_label(freshness_percentage),
        base64_image=base64_image,
        price=price_to_text(fruit_information["price"]),
        object=detected_objects,
        calories=cals
    )
