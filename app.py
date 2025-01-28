import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the model
model = load_model('brain_tumor_model.h5')

# Load and preprocess the image
img_path = r'final_dataset\glioma\Te-gl_0010.jpg'  
img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale the image

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# If you have the class labels from your training data
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']  
predicted_label = class_labels[predicted_class[0]]

# Print the prediction
print(f'Predicted class: {predicted_label}')

# Display image with prediction
plt.imshow(img)
plt.title(f'Predicted: {predicted_label}')
plt.show()
