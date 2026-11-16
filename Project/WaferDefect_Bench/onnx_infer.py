import onnxruntime as ort
import numpy as np
import cv2
import time

if __name__ == "__main__":
    sess = ort.InferenceSession("model.onnx")
    class_names = ['none', 'Loc', 'Edge-Loc', 'Center', 'Edge-Ring', 'Scratch', 'Random', 'Near-full', 'Donut']
    img = cv2.imread("test.png",0)
    img = cv2.resize(img,(24,24))
    img = img.astype(np.float32)/255.0
    input_tensor = np.expand_dims(np.expand_dims(img,0),0)

    start = time.time()
    out = sess.run(["output"], {"input":input_tensor})
    cost = time.time() - start
    pred = np.argmax(out[0])
    print(f"pred: {class_names[pred]}, cost: {cost*1000:.3f} ms")
