#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Loading necessary library
import pickle 
import numpy as np
import cv2 as cv


# In[5]:


# Loading model
pickle_in = open("digit_classification_model.p", "rb")
model = pickle.load(pickle_in)


# In[6]:


# declare size of image and threshold 
width = 640
height = 480
threshold = .80


# In[ ]:


# Preprocessing image
def preProcessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img/255
    return img


# Capturing image using webcam
cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

while True:
    success, img_original = cap.read()
    img = np.asarray(img_original)
    img = cv.resize(img, (28,28))
    img = preProcessing(img)
    cv.imshow("Webcam Imge", img)
    img = img.reshape(1,28,28,1)
    
    # predict
    class_index = int(model.predict_classes(img))
    #print(class_index)
    predictions = model.predict(img)
    #print(predictions)
    prob_value = np.amax(predictions)
    print(class_index, prob_value)
    
    if prob_value > threshold:
        cv.putText(img_original, str(class_index)+ str(prob_value), 
                   (50,50), cv.FONT_HERSHEY_COMPLEX,
                   1, (0,0,255), 1)
    
    cv.imshow('Original Image', img_original)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    

