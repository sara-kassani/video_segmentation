import os
import numpy as np
import cv2
import tensorflow as tf

video_path = "videos/seq_1_gt.mp4"
trained_model_path= "trained_model/simple_unet_model.h5"
model = tf.keras.models.load_model(trained_model_path)


vs = cv2.VideoCapture(video_path)
_, frame = vs.read()
H, W, _ = frame.shape
vs.release()

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('seq_1_pred.mp4', fourcc, 10, (W, H), True)

cap = cv2.VideoCapture(video_path)
idx = 0
while True:
    ret, frame = cap.read()
    if ret == False:
        cap.release()
        out.release()
        break

    H, W, _ = frame.shape
    frame = np.array(frame)
    frame = cv2.resize(frame, (512, 512))
    frame= frame / np.max(frame)
    frame=np.expand_dims(frame, axis=0)

    pred_proba = model.predict(frame)[0]
    pred = np.argmax(pred_proba, axis=3)
    pred = np.squeeze(pred, 0)
    mask = pred.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    fname= os.path.basename(video_path)
    cv2.imwrite(f"{fname}_{idx}.png", frame)
    idx += 1

    out.write(frame)
