import cv2
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
from skimage.transform import resize
from scipy.spatial import distance
from keras import models
from cvzone.FaceMeshModule import FaceMeshDetector
import cvzone

#cap = cv2.VideoCapture(0)
mesh_detector = FaceMeshDetector()

save = False
setting = False
track = False

model_path = "/Users/ray/Desktop/FaceDistanceMeasurement/facenet_keras.h5"
model = models.load_model(model_path)
margin = 6
image_size = 160

detector = FaceDetector(minDetectionCon=0.8)

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError("Dimension should be 3 or 4")
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def preProcess(img):
    whitenImg = prewhiten(img)
    whitenImg = whitenImg[np.newaxis, :]
    return whitenImg

def show_face(img,distance,x, y, w, h):
    #img, faces = detector.findFaces(img, draw=True)
    #(x, y, w, h) = faces[i]['bbox']
    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 2)
    cv2.putText(img, "{}".format(distance), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("vedio", img)

def find_face(img,embs_people=0):
    img_d, faces = detector.findFaces(img, draw=True)
    list=[]
    if faces != []:
        a = len(faces)
        for i in range(a):
            Error = False
            (x, y, w, h) = faces[i]['bbox']
            face = img[y:y + h, x:x + w]
            faceMargin = np.zeros((h + margin * 2, w + margin * 2, 3), dtype="uint8")

            try:
                faceMargin[margin:margin + h, margin:margin + w] = face
                aligned = resize(faceMargin, (image_size, image_size), mode='reflect')
                faceImg = preProcess(aligned)
                embs = l2_normalize(np.concatenate(model.predict(faceImg)))
                print(embs)
                if (embs_people is not 0):
                    distanceNum = distance.euclidean(embs_people, embs)
                    list.append(distanceNum)
                    #show_face(img, distanceNum, x, y, w, h)
            except ValueError:
                Error = True
                print("ValueError!")
        min = 2
        for a in list:
            if a < min:
                min = a
        if min <= 0.7:
            i = list.index(min)
            (x, y, w, h) = faces[i]['bbox']
            face = img[y:y + h, x:x + w]
            img_mesh, faces_mesh = mesh_detector.findFaceMesh(face, draw=True)
            # img_mesh, faces_mesh = mesh_detector.findFaceMesh(img, draw=True)
            #print("face_mesh",faces_mesh)
            if faces_mesh != []:
                face = faces_mesh[0]
                pointLeft = face[145]
                pointLeft = [pointLeft[0]+x,pointLeft[1]+y]
                pointRight = face[374]
                pointRight = [pointRight[0] + x, pointRight[1] + y]
                cv2.circle(img, pointLeft, 5, (255,0,255), cv2.FILLED)
                cv2.circle(img, pointRight, 5, (255,0,255), cv2.FILLED)
                cv2.line(img, pointLeft, pointRight, (0,200,0), 3)
                w_, _ = mesh_detector.findDistance(pointLeft, pointRight)
                W = 6.3
                #Focal Length
                f = 1000
                #Finding Distance

                d = (W*f)/w_
                print(d)

                #printtext
                cvzone.putTextRect(img, f'Depth: {int(d)}cm', (face[10][0]-100+x, face[10][1]-50+y), scale = 2)
                show_face(img, min, x, y, w, h)
        if (not Error):
            return embs


"""try:
                faceMargin[margin:margin + h, margin:margin + w] = face
                aligned = resize(faceMargin, (image_size, image_size), mode='reflect')
                faceImg = preProcess(aligned)
                embs = l2_normalize(np.concatenate(model.predict(faceImg)))
                if (embs_people is not 0):
                    distanceNum = distance.euclidean(embs_people, embs)
                    show_face(img, distanceNum)
                return embs
            except ValueError:
                print("ValueError!")"""


cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    cv2.imshow("vedio", img)
    if (not save):
        embs_people = find_face(img,0)
        save = True
    if (save):
        embs_test = find_face(img,embs_people)
        #distanceNum = distance.euclidean(embs_people, embs_test)
    #show_face(img,distanceNum)
    key = cv2.waitKey(1)
