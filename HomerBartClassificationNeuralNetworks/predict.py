# -*- coding: utf-8 -*-
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


model = load_model('homer_bart_model.h5')
predict_folder = 'predict/'

for filename in os.listdir(predict_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  
        img_path = os.path.join(predict_folder, filename)
        
        
        img = load_img(img_path, target_size=(150, 150))
        
        
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array /= 255.0  

        predictions = model.predict(img_array)

        label = "Homer" if predictions[0][0] > 0.5 else "Bart"
        
        actual_label = "Homer" if "homer" in filename.lower() else "Bart"
        
        is_correct = "True" if label == actual_label else "False"
        print(f"Filename: {filename}, Predict: {label}, Actual: {actual_label}, Result: {is_correct}, Predict Value: {predictions[0][0]}")

