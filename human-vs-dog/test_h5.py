import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Source files https://github.com/AlejandroNunezArroyo/VisionTest
# Model in h5 https://github.com/AlejandroNunezArroyo/VisionTest/blob/main/FaceDataset/FaceDataset.h5
# Images https://github.com/AlejandroNunezArroyo/VisionTest/tree/main/FaceDataset/images
model=load_model("model_humandog.h5")

img = cv2.imread('images/289.jpg')

im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
rerect_sized=cv2.resize(im,(250,250))
normalized=rerect_sized/255.0
reshaped=np.reshape(normalized,(1,250,250,3))
reshaped = np.vstack([reshaped])
result=model.predict(reshaped)

if result[0][0] > 0.5:
  print(" Human ")
  cv2.imshow('Human',img)
else: 
  print(" Dog ")
  cv2.imshow('Dog',img)
print(result)
cv2.waitKey(0)
cv2.destroyAllWindows()