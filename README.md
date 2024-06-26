**Flask Super-Resolution Service Documentation**

**Introduction:**
This documentation delineates the functionality and usage of a Flask-based web service tailored for super-resolution of images, leveraging the RealESRGAN model. The service empowers users to upload images, execute super-resolution processing, and seamlessly retrieve the enhanced images as output.

**Dependencies:**
- Flask: A micro web framework for Python.
- RealESRGAN: A deep learning-based model esteemed for image super-resolution.
- Pillow (PIL): Python Imaging Library for comprehensive image processing capabilities.
- NumPy: A quintessential library for proficient numerical operations in Python.
- PyTorch: A revered deep learning framework renowned for its versatility and performance.
- Flask-CORS: A Flask extension instrumental in facilitating Cross-Origin Resource Sharing (CORS).

**Usage:**
1. **Installation:**
   - Ensure Python is installed on your system.
   - Install the required dependencies using pip:
     ``` 
     pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
     pip install -r requirements.txt
     ```

2. **Starting the Service:**
   - Execute the provided Python script in your terminal or command prompt:
     ```
     python main.py
     ```

3. **Endpoint:**
   - The service exposes an endpoint at `/superres` meticulously crafted to orchestrate super-resolution processing.

4. **Input:**
   - Dispatch a POST request to `/superres` endpoint, meticulously attaching an image file as a form-data field named `image`.

5. **Output:**
   - Upon successful processing, the service graciously furnishes the super-resolved image as a response.

### Usage

---

Basic usage:

```python
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

path_to_image = 'inputs/lr_image.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('results/sr_image.png')
```

**Example Usage:**
- Utilizing Python requests library:
  ```python
  import requests

  url = 'http://localhost:5000/superres'
  files = {'image': open('input_image.jpg', 'rb')}
  response = requests.post(url, files=files)

  if response.status_code == 200:
      with open('super_res_image.png', 'wb') as f:
          f.write(response.content)
  ```

### Example

Low quality image:

![](inputs/lr_face.png)

Real-ESRGAN result:

![](results/sr_face.png)

---

**Notes:**
- The service adeptly discerns the available hardware (CPU/GPU) for model inference, ensuring optimal performance.
- Supported input image formats encompass PNG, JPG, JPEG, TIFF, BMP, and GIF.

**Conclusion:**
The Flask Super-Resolution Service presents a streamlined avenue for executing super-resolution processing on images employing the RealESRGAN model. Its seamless integration potential renders it indispensable for diverse applications aimed at augmenting image quality.

---

This documentation elucidates the service's functionality, usage, dependencies, and exemplary scenarios, sans the inclusion of the actual code.