from flask import Flask, request, send_file
from RealESRGAN import RealESRGAN
from PIL import Image
import numpy as np
import torch
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

model_scale = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device=device, scale=model_scale)
model.load_weights(f'weights/RealESRGAN_x{model_scale}.pth')

IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMAGE_FORMATS)


@app.route('/superres', methods=['POST'])
def super_resolve():
    if 'image' not in request.files:
        return 'No file part', 400  # Bad Request

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400  # Bad Request

    if not is_image_file(file.filename):
        return 'Invalid file format', 400  # Bad Request

    image = Image.open(file.stream).convert('RGB')
    sr_image = model.predict(np.array(image))

    # Save super-resolved image locally
    sr_image_path = 'super_res_image.png'
    sr_image.save(sr_image_path)

    # Send the saved image file
    response = send_file(sr_image_path, mimetype='image/png')

    # Clean up the temporary file
    os.remove(sr_image_path)

    return response


if __name__ == '__main__':
    app.run(debug=True)
