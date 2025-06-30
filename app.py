import os
import torch
import pandas as pd
from flask import Flask, request, jsonify, session
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from models import db, User

# Base directory for safe file pathing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask App Initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'images')
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "default-dev-secret")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flaskdb.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True

# Initialize Extensions
CORS(app, supports_credentials=True)  # You can add origins=["https://your-frontend.azurestaticapps.net"]
bcrypt = Bcrypt(app)
db.init_app(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize DB
with app.app_context():
    db.create_all()

# Load Models and Data
yolo_model = YOLO(os.path.join(BASE_DIR, "Model", "best.pt"))
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset', 'allMedicines.csv'))

# Helper Functions
def extract_text(image_path):
    results = yolo_model.predict(source=image_path, conf=0.25, hide_labels=True)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()
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
        mydict[label] = extracted_text

    return extracted_texts, mydict

def check_fraudulent_product(mydict):
    names = df['name']
    found = False

    for label, value in mydict.items():
        value = value.lower()
        arr = value.split(' ')
        if label == "Company":
            filtered_arr = [s for s in arr if "mg" not in s and "md" not in s and not any(char.isdigit() for char in s) and "tablet" not in s]
            for name in names:
                for ite in filtered_arr:
                    if ite in name.lower():
                        found = True
                        break
                if found:
                    break
    return found


@app.route("/signup", methods=["POST"])
def signup():
    email = request.json.get("email")
    password = request.json.get("password")

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already exists"}), 409

    hashed_password = bcrypt.generate_password_hash(password)
    new_user = User(email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    session["user_id"] = new_user.id

    return jsonify({"id": new_user.id, "email": new_user.email})

@app.route("/login", methods=["POST"])
def login_user():
    email = request.json.get("email")
    password = request.json.get("password")

    user = User.query.filter_by(email=email).first()

    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Unauthorized"}), 401

    session["user_id"] = user.id
    return jsonify({"id": user.id, "email": user.email})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)
    extracted_texts, mydict = extract_text(image_path)
    fraud_check_results = check_fraudulent_product(mydict)

    return jsonify({'texts': extracted_texts, 'fraud_check': fraud_check_results})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200
