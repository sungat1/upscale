from flask import Flask, request, jsonify, send_file, render_template
import os
import cv2
import numpy as np
import torch
from PIL import Image
import io
import zipfile
import urllib.request
from werkzeug.utils import secure_filename
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
WEIGHTS_FOLDER = 'weights'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)

# Model configurations
MODEL_URLS = {
    2: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    4: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    8: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
}

# Initialize models dictionary
models = {}

def download_model(scale):
    """Download the model if it doesn't exist"""
    model_path = os.path.join(WEIGHTS_FOLDER, f'RealESRGAN_x{scale}plus.pth')
    if not os.path.exists(model_path):
        print(f"Downloading model for {scale}x upscaling...")
        urllib.request.urlretrieve(MODEL_URLS[scale], model_path)
    return model_path

def get_model(scale):
    """Get or initialize the model for the specified scale"""
    if scale not in models:
        model_path = download_model(scale)
        
        # Initialize model architecture
        if scale == 8:  # anime model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        else:  # x2 and x4 models
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        
        # Initialize upsampler
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=400,  # Use tiles to handle large images
            tile_pad=10,
            pre_pad=0,
            half=device.type == 'cuda',  # Use half precision only with CUDA
            device=device
        )
        models[scale] = upsampler
    
    return models[scale]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upscale', methods=['POST'])
def upscale_images():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    scale = int(request.form.get('scale', 4))
    enhance = request.form.get('enhance', 'true').lower() == 'true'
    
    if not files:
        return jsonify({'error': 'No selected files'}), 400
    
    if scale not in [2, 4, 8]:
        return jsonify({'error': 'Invalid scale factor. Must be 2, 4, or 8'}), 400

    try:
        # Get the appropriate model
        upsampler = get_model(scale)
        
        # Process each file
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    input_path = os.path.join(UPLOAD_FOLDER, filename)
                    output_path = os.path.join(OUTPUT_FOLDER, f'upscaled_{filename}')
                    
                    try:
                        # Save the uploaded file
                        file.save(input_path)
                        
                        # Read image
                        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                        if img is None:
                            continue
                        
                        # Process with Real-ESRGAN
                        if scale == 8:
                            # For 8x, apply 4x twice
                            output, _ = upsampler.enhance(img, outscale=4)
                            output, _ = upsampler.enhance(output, outscale=2)
                        else:
                            output, _ = upsampler.enhance(img, outscale=scale)
                        
                        # Save the processed image
                        cv2.imwrite(output_path, output)
                        
                        # Add to zip file
                        zf.write(output_path, f'upscaled_{filename}')
                        
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                        continue
                    finally:
                        # Clean up
                        if os.path.exists(input_path):
                            os.remove(input_path)
                        if os.path.exists(output_path):
                            os.remove(output_path)
        
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='upscaled_images.zip'
        )
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
