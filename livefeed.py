import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('sign_language_model.h5')


def predict_sign(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    resized = cv2.resize(gray, (28, 28))  
    reshaped = resized.reshape(1, 28, 28, 1) / 255.0  
    result = model.predict(reshaped)  
    return np.argmax(result)  


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    
    
    roi = frame[100:300, 100:300]
    
    
    pred_class = predict_sign(roi)
    
   
    if pred_class >= 9:  
        pred_class += 1
    
    
    cv2.putText(frame, chr(pred_class + 65), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    
    cv2.imshow("Sign Language Recognition", frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
