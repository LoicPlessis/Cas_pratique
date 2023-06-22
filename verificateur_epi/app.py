import cv2
import torch
import numpy as np
import streamlit as st


model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='exp7/weights/last.pt', force_reload=False)

Label = ['Casque_OK', 'Gilet_OK', 'Casque_NO', 'Gilet_NO']

font = cv2.FONT_HERSHEY_PLAIN

colors = np.random.uniform(0, 255, size=(2, 3))

CONFIDENCE_THRESHOLD = 0.7

st.title("Vérificateur EPI")

video_placeholder = st.empty()  # Espace réservé pour afficher la vidéo
message_placeholder = st.empty()  # Espace réservé pour afficher le message de l'uniforme

cap = None  # Objet pour la capture vidéo

play_button = st.button("Lancer la vidéo")
stop_button = st.button("Arrêter la vidéo")

if play_button:
    cap = cv2.VideoCapture(0)

while play_button and not stop_button:
    if cap is None:
        break

    ret, frame = cap.read()
    if not ret:
        break

    boxes = []
    class_ids = []
    results = model_yolo(frame)

    for i in range(0, len(results.pred[0])):
        if results.pred[0][i, 4] > CONFIDENCE_THRESHOLD:
            x = int(results.pred[0][i, 0])
            y = int(results.pred[0][i, 1])
            w = int(results.pred[0][i, 2])
            h = int(results.pred[0][i, 3])
            box = np.array([x, y, w, h])
            boxes.append(box)
            class_ids.append(int(results.pred[0][i, 5]))

    # Vérification de l'uniforme
    has_casque = False
    has_gilet = False

    for box, classid in zip(boxes, class_ids):
        color = colors[int(classid) % len(colors)]
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[:2] + box[2:]), color, 2)
        cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(frame, Label[classid], (box[0], box[1] - 5), font, 1, (0, 0, 0))

        if Label[classid] == "Casque_OK":
            has_casque = True
        if Label[classid] == "Gilet_OK":
            has_gilet = True

    # Affichage de l'état de l'uniforme
    if has_casque and has_gilet:
        msg = "Uniforme complet"
    else:
        msg = "Uniforme incomplet"

    video_placeholder.image(frame, channels="BGR")
    message_placeholder.markdown(f"<h2 style='text-align: center;'>{msg}</h2>", unsafe_allow_html=True)

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
