from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes 
import av
import cv2
import torch
import numpy as np
import time
import streamlit as st



model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='Models/weights/last.pt', force_reload=False)


Label = ['Casque_OK','Gilet_OK','Casque_NO','Gilet_NO']
# 0 pour Casque_OK
# 1 pour Gilet_OK
# 2 pour Casque_NO
# 3 pour Gilet_NO


font = cv2.FONT_HERSHEY_PLAIN

colors = np.random.uniform(0, 255, size=(2, 3))
classid = 0

CONFIDENCE_THRESHOLD = 0.7

st.title("Vérificateur de l'Uniforme")
number=[]


class VideoProcessor:

    def recv(self, frame):
 
        frm = frame.to_ndarray(format="bgr24")
        
        boxes = []
        class_ids = []
        results = model_yolo(frm)


        for i in range(0,len(results.pred[0])) :
            if results.pred[0][i,4] > CONFIDENCE_THRESHOLD :
                x = int(results.pred[0][i,0])
                y = int(results.pred[0][i,1])
                w = int(results.pred[0][i,2])
                h = int(results.pred[0][i,3])
                box = np.array([x, y, w, h])
                boxes.append(box)
                class_ids.append(int(results.pred[0][i,5]))

        for box, classid in zip(boxes,class_ids):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frm, box, color, 2)
            cv2.rectangle(frm, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frm, Label[classid], (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

            number.append(Label[classid])



            
            classe = number[-1]
            print(classe)
            if classe == "Gilet_NO" or classe == "Casque_NO" :
                msg = "Uniforme incomplet"
                print(msg)
            else : 
                msg = "Uniforme complet"
                print(msg)
            with open('file.txt', 'w') as file:
                file.write(msg)

            with open('file.txt') as f:
                first_line = f.readline()
                st.write(first_line)
            

        return av.VideoFrame.from_ndarray(frm, format='bgr24')

muted = st.checkbox("Mute")



webrtc_streamer( key="mute_sample", 
                video_processor_factory=VideoProcessor,
                video_html_attrs=VideoHTMLAttributes( autoPlay=True, controls=True, style={"width": "100%"}, muted=muted ), ) 

 

# bouton = affiche le texte
button = st.button('Lancer la détection :')
button_placeholder = st.empty()
# button_placeholder.write(f'Lancer la détection : ')
time.sleep(2)
button = False
button_placeholder.write(f'===> Détection en cours : ')


texte = st.empty()

while True :
    with open('file.txt') as f:
        first_line = f.readline()
        texte.write(first_line)
        time.sleep(5)