import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

while True:
    sucess, img = cap.read()
    img_, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        print(img_)
        pointLeft = face[145]
        pointRight = face[374]
        cv2.circle(img_, pointLeft, 5, (255,0,255), cv2.FILLED)
        cv2.circle(img_, pointRight, 5, (255,0,255), cv2.FILLED)
        cv2.line(img_, pointLeft, pointRight, (0,200,0), 3)
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3
        #Focal Length
        f = 1000
        #Finding Distance

        d = (W*f)/w
        #print(d)

        #printtext
        cvzone.putTextRect(img, f'Depth: {int(d)}cm', (face[10][0]-100, face[10][1]-50), scale = 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
