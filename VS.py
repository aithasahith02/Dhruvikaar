#!/usr/bin/env python
# coding: utf-8

# # Importing the Dependencies

# In[ ]:


print("Importing the dependencies.....\n")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from imutils.video import VideoStream
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import smtplib
import imutils
import time
import cv2
import os


# # Loading the Dataset from the system
# 

# In[ ]:


Rn = input("Enter the Roll Number : ")
name = input("Enter the name : ")
email = input("Enter the Email ID : ")
'''
print("Loading the Dataset from the System....\n")
train = ImageDataGenerator(rescale = 1/255)
test = ImageDataGenerator(rescale = 1/255)

train_dataset = train.flow_from_directory('C:/Users/hp/MLOPS/PROJECT/DATASET/training/',
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'binary')
test_dataset = test.flow_from_directory('C:/Users/hp/MLOPS/PROJECT/DATASET/testing/',
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'binary')

print("The Classes are : \n ",train_dataset.classes)

print("Creating the CNN......\n")


# # Functions for Sending the mail

# In[ ]:

'''
def mail_for_valid() :
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("aithasahith0214@gmail.com","pwd")
    message = "Hello ",name," You are allowed and can enter into the class..."
    s.sendmail("aithasahith0214@gmail.com", mail, message)
    s.quit()
#Use this link to avoid error with smtplib Security issues
#https://myaccount.google.com/lesssecureapps?pli=1&rapt=AEjHL4NIDAYmVl3G3lMRH2NYlpioEKYPUfwamskAKpXcudrSSKCmxi-LzIAUyY9ApTtadSMm0CNuk08xRXqoXjqxTMLIb39R2A

def mail_for_invalid() :
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("aithasahith0214@gmail.com","pwd")
    message = "Hello ",name," You are not allowed and cant enter into the class..."
    s.sendmail("aithasahith0214@gmail.com", mail, message)
    s.quit()

'''
# # Building the Neural Network

# In[ ]:


print("Building the Neural Network..........\n")
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3) ,activation='relu',input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3) ,activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64,(3,3) ,activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512,activation='relu'),
                                    ##
                                    tf.keras.layers.Dense(1,activation='sigmoid')
                                   ]
                                  )

model.compile(loss = 'binary_crossentropy',
              optimizer = RMSprop(learning_rate=0.001),
              metrics = ['accuracy']
             )


# # Training the Neural Network

# In[ ]:


print("Training the Neural Network.........\n")
model_fit = model.fit(train_dataset,
                      steps_per_epoch=3,
                      epochs=10,
                     validation_data=test_dataset
                     )


# # Testing with a single image and Saving the model

# In[ ]:


print("Testing a Sample Image : \n")
img = image.load_img('C:/Users/hp/MLOPS/PROJECT/DATASET/training/valid/1.png',target_size=(200,200))
plt.imshow(img)


X = image.img_to_array(img)
print(X)
X = np.expand_dims(X,axis=0)
val = model.predict([X])
if val == 1:
    print("\n Dressed properly")
else :
    print("\n Not dressed properly")
    
    

print("Saving the Model with name : \"new_model.h5\".\n")
model.save('new_model.h5')


# # Loading the saved model and testing

# In[ ]:

'''
print("Loading the saved model from the file and testing.....\n")
my_model=load_model('new_model.h5')
'''
img = image.load_img('C:/Users/hp/MLOPS/PROJECT/DATASET/training/invalid/1.png',target_size=(200,200))
plt.imshow(img)

X = image.img_to_array(img)
X = np.expand_dims(X,axis=0)
val = my_model.predict([X])
if val == 1:
    print("dressed properly")
else :
    print("Not dressed properly")


# # Testing the model with Live VideoStream

# In[ ]:
'''

valid_counter=0
invalid_counter=0
print("Ready to steam Live .....\n ")
font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vs = VideoStream(src=0).start()
print("Starting the Camera.....\n ")
time.sleep(2.0)
while True :
    frame = vs.read()
    img=frame
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in faces:
        x1 = x - int(w/2)
        y1 = y + h
        w1 = w*2
        h1 = h*2
    frame = cv2.resize(frame,(200,200))
    X = np.expand_dims(frame,axis=0)
    val = my_model.predict([X])
    if(val==1):
        valid_counter+=1
        cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
        cv2.putText(img,'DRESSED PROPERLY',(50, 50), font, 1,(0, 255, 0),2,cv2.LINE_4)
        cv2.imshow('final',img)
        print("Properly dressed")
    else :
        invalid_counter+=1
        cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
        cv2.putText(img,'NOT DRESSED PROPERLY',(50, 50), font, 1,(0, 0, 255),2,cv2.LINE_4)
        cv2.imshow('final',img)

        print("Not dreassed properly")
    if invalid_counter == 50 :
        mail_for_invalid()
        break
    if valid_counter == 50 :
        mail_for_valid()
        break
    if cv2.waitKey(10) == 13 :
        break
cv2.destroyAllWindows()
print("Closing the Camera.....\nEnding the program.......\n")
vs.stop()


# In[ ]:




