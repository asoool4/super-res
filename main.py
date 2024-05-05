from flask import Flask, request, send_file
from RealESRGAN import RealESRGAN
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

model_scale = 4
device = 'cpu'

model = RealESRGAN(device=device, scale=model_scale)
model.load_weights(f'weights/RealESRGAN_x{model_scale}.pth')

@app.route('/superres', methods=['POST'])
def super_resolve():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    image = Image.open(file.stream).convert('RGB')
    sr_image = model.predict(np.array(image))

    # Save super-resolved image locally
    sr_image_path = 'super_res_image.png'
    sr_image.save(sr_image_path)

    # Send the saved image file
    return send_file(sr_image_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=False)
