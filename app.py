from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from ultralytics import YOLO
from PIL import Image
import torch
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from flask import Flask, request, jsonify, session
from flask_bcrypt import Bcrypt #pip install Flask-Bcrypt = https://pypi.org/project/Flask-Bcrypt/
from flask_cors import CORS, cross_origin #ModuleNotFoundError: No module named 'flask_cors' = pip install Flask-Cors
from models import db, User

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/images/'

app.config['SECRET_KEY'] = 'cairocoders-ednalan'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flaskdb.db'
 
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = True
  
bcrypt = Bcrypt(app) 
CORS(app, supports_credentials=True)
db.init_app(app)
  
with app.app_context():
    db.create_all()


yolo_model = YOLO("Model/best.pt")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

df = pd.read_csv('dataset/allMedicines.csv')
def extract_text(image_path):

    results = yolo_model.predict(source=image_path, conf=0.25, hide_labels=True)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    labels = results[0].boxes.cls.cpu().numpy()  # Class IDs
    image = Image.open(image_path)
    extracted_texts = []
    label_map = {0: "Company", 1: "Generic", 2: "Manufacturer"}
    mydict = {}
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        label_id = int(labels[i])
        label = label_map.get(label_id, "Unknown")
        cropped_image = image.crop((x_min, y_min, x_max, y_max))

        pixel_values = processor(cropped_image, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)

        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        extracted_texts.append(extracted_text)
        mydict[label]=extracted_text

    return extracted_texts, mydict

def check_fraudulent_product(mydict):
    names = df['name']
    found=False
    for label, value in mydict.items():
        value=value.lower()
        arr = value.split(' ')
        print(f"Label: {label}, Value: {value}")
        if label=="Company":
            print("inside label==company loop")
            print(f"list_value: {arr}")
            filtered_arr = [s for s in arr if "mg" not in s and "md" not in s and not any(
                char.isdigit() for char in s) and "Tablet" not in s]
            print(f"filtered_arr: {filtered_arr}")
            for name in names:
                for ite in filtered_arr:
                    if ite in name.lower():
                        found=True
                        break
                if found==True:
                    break
    return found


@app.route("/signup", methods=["POST"])
def signup():
    email = request.json["email"]
    password = request.json["password"]
 
    user_exists = User.query.filter_by(email=email).first() is not None
 
    if user_exists:
        return jsonify({"error": "Email already exists"}), 409
     
    hashed_password = bcrypt.generate_password_hash(password)
    new_user = User(email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
 
    session["user_id"] = new_user.id
 
    return jsonify({
        "id": new_user.id,
        "email": new_user.email
    })
 
@app.route("/login", methods=["POST"])
def login_user():
    email = request.json["email"]
    password = request.json["password"]
  
    user = User.query.filter_by(email=email).first()
  
    if user is None:
        return jsonify({"error": "Unauthorized Access"}), 401
  
    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Unauthorized"}), 401
      
    session["user_id"] = user.id
  
    return jsonify({
        "id": user.id,
        "email": user.email
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        extracted_texts, mydict = extract_text(image_path)
        fraud_check_results = check_fraudulent_product(mydict)
        return jsonify({'texts': extracted_texts, 'fraud_check': fraud_check_results})


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)


