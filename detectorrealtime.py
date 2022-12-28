# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 10:29:35 2021

@author: jose_
"""

import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
with mp_face_detection.FaceDetection(
     min_detection_confidence=0.5) as face_detection:
     while True:
          ret, frame = cap.read()
          if ret == False:
               break
          frame = cv2.flip(frame, 1)
          height, width, _ = frame.shape
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = face_detection.process(frame_rgb)
          
          if results.detections is None:
              cv2.putText(frame, "Estado: No atento", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0,0,255), 1, cv2.LINE_AA)
          
          if results.detections is not None:
               for detection in results.detections:
                    # Ojo 1
                    x1 = int(detection.location_data.relative_keypoints[0].x * width)
                    y1 = int(detection.location_data.relative_keypoints[0].y * height)
                    # Ojo 2
                    x2 = int(detection.location_data.relative_keypoints[1].x * width)
                    y2 = int(detection.location_data.relative_keypoints[1].y * height)
                    
                    p1 = np.array([x1, y1])
                    p2 = np.array([x2, y2])
                    p3 = np.array([x2, y1])
                    
                    # Obtenemos las distancias de: d_eyes, l1
                    d_eyes = np.linalg.norm(p1 - p2)
                    l1 = np.linalg.norm(p1 - p3)
                    
                    # Calcular el ángulo formado por d_eyes y l1
                    angle = degrees(acos(l1 / d_eyes))
                    
                    # Determinar si el ángulo es positivo o negativo
                    if y1 < y2:
                         angle = - angle
                    
                    # Visualizar datos
                    cv2.putText(frame, "", (x1 - 60, y1), 1, 1.5, (0, 255, 0), 2)
                    cv2.putText(frame, "", (x2 + 10, y2), 1, 1.5, (0, 128, 255), 2)
                    #cv2.putText(frame, str(int(angle)), (x1 - 35, y1 + 15), 1, 1.2, (0, 255, 0), 2)
                    
                    # Linea
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                    # Círculos en cada uno de los vértices del triángulo
                    cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (x2, y2), 5, (0, 128, 255), -1)
                    
                    if -40 <= angle <= 40:
                        cv2.putText(frame, "Estado: Atento", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0,255,0), 1, cv2.LINE_AA)
                        
                    else:
                        cv2.putText(frame, "Estado: No atento", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0,0,255), 1, cv2.LINE_AA)
          
          cv2.imshow("Frame", frame)
          k = cv2.waitKey(1) & 0xFF
          if k == 27:
               break
cap.release()
cv2.destroyAllWindows()