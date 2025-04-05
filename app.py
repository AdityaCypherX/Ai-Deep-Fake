import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from utils import preprocess_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define model path and check if it exists
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'fake_image_model.h5')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

try:
    # Load the model with error handling
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', result="No file part")
        
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', result="No selected file")
    
    # Get file extension
    _, file_extension = os.path.splitext(file.filename)
    
    # Create a unique filename to prevent overwrites
    unique_filename = f"image_{os.urandom(8).hex()}{file_extension}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        file.save(path)
        img = preprocess_image(path)
        prediction = model.predict(img)[0][0]

        if prediction >= 0.5:
            result = "Fake Image"
            confidence = f"{prediction * 100:.2f}%"
        else:
            result = "Real Image"
            confidence = f"{(1 - prediction) * 100:.2f}%"

        return render_template('index.html', result=f"{result} (Confidence: {confidence})")

    except Exception as e:
        return render_template('index.html', result=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
