import cv2
import sys
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model_mask.h5")
model2 = tf.keras.models.load_model("model_mask_2.h5")


video_capture = cv2.VideoCapture(0)

img_shape = (224, 224)

#dictionnaire
mask_labels_dict = {
    'with_mask':0,
    'without_mask':1
}

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #Retraitement de l'image
    frame2 = cv2.resize(frame, img_shape)
    frame2 = np.expand_dims(np.array(frame2), axis= 0)/255
    prob = model2.predict(frame2)
    prediction = prob.argmax(axis = -1)
    for k, val in mask_labels_dict.items():
        if prediction == val:
            #print(k)
            frame = cv2.putText(frame, k, (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2, cv2.LINE_AA)
            # Display the resulting frame
            cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
