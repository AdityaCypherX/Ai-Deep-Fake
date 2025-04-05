# from tensorflow.keras.preprocessing import image
# import numpy as np

# def preprocess_image(img_path, target_size=(128, 128)):
#     img = image.load_img(img_path, target_size=target_size)
#     img_array = image.img_to_array(img)
#     img_array = img_array / 255.0  # Normalize if required
#     img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)
#     img_array = img_array.reshape(1, -1)  # Flatten to shape: (1, 128*128*3 = 49152)
#     return img_array

from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path):
    target_size = (224, 224)  # this must match your model's expected input shape
    img = image.load_img(img_path, target_size=target_size)  # auto-resizes any image
    img_array = image.img_to_array(img)                      # shape: (224, 224, 3)
    img_array = img_array / 255.0                            # normalize pixels
    img_array = np.expand_dims(img_array, axis=0)            # shape: (1, 224, 224, 3)
    return img_array