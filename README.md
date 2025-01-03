# Batch Image Upscaler

A web application that allows users to batch upscale images using the Real-ESRGAN model.

## Features

- Drag and drop interface for uploading multiple images
- Batch processing of images
- Progress indication
- Download results as a zip file
- Modern and responsive UI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/xinntao/Real-ESRGAN.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model:
- Download the model from [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases)
- Place the downloaded model in the `weights` folder

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload images by either:
   - Clicking the "Choose Files" button
   - Dragging and dropping files into the upload area

4. Click "Upscale Images" to process the images

5. Download the zip file containing the upscaled images

## Requirements

- Python 3.7+
- Flask
- PyTorch
- OpenCV
- Pillow
- NumPy
- Real-ESRGAN dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.
