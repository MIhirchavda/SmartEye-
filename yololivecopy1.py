import cv2
import numpy as np
import time
from gtts import gTTS
import os
from datetime import datetime
import HomeState
import audioio
import pickle
from threading import Timer

# cap = cv2.VideoCapture("/home/robo/VID_20201205_180147.mp4")
# Load YOLO
cap = cv2.VideoCapture(0)
### code of distance##

Known_distance = 76.2  # in cm
Known_width = 14.3
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("new_trainner.yml")
face_cascade = cv2.CascadeClassifier(
    'E:\OpenCv\Emma0 (updated backup) (copy)\cascades\data\haarcascade_frontalface_default.xml')
lables = {"person_name": 0}

with open("lables.pickel", 'rb') as f:
    og_lables = pickle.load(f)
    lables = {v: k for k, v in og_lables.items()}

#######
weights_path = "Models/Emma_models/yolov3.weights"
cfg_path = "Models/Emma_models/yolov3.cfg"
coco_path = "Models/Emma_models/coco.names"
net = cv2.dnn.readNet(weights_path, cfg_path)
classes = []
with open(coco_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_name = net.getLayerNames()
outputLayers = [layer_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
starting_time = time.time()

Object_Detection_frame = []


def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1


def img_estim(frame, thrshld):
    # print(np.mean(frame))
    # print("no")
    is_light = np.mean(frame) > thrshld
    # print("prob")
    # cv2.imshow('frame', frame)

    return 'light' if is_light else 'dark'


#######distance code#######


def objectDetection():
    img_id = 0

    def FocalLength(measured_distance, real_width, width_in_rf_image):
        focal_length = (width_in_rf_image * measured_distance) / real_width
        return focal_length

    # distance estimation function
    def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
        distance = (real_face_width * Focal_Length) / face_width_in_frame
        return distance

    def face_data(image):
        face_width = 0
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
        print(faces)
        face_width = []
        for (x, y, h, w) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), WHITE, 1)
            face_width.append(w)

        # faces = len(faces)
        print(face_width, "face data fun")
        return faces, face_width

    # reading reference image from directory
    ref_image = cv2.imread("IMG_20210313_115926.jpg")

    fc, ref_image_face_width = face_data(ref_image)
    Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width[0])
    print(Focal_length_found, "focal length found")
    cv2.imshow("ref_image", ref_image)

    while True:
        # img = cv2.imread("img.png")
        ret, img = cap.read()

        ### brightness detection
        color = (255, 255, 255)
        stroke = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        # print("0")
        output = str(img_estim(img, 50))
        # print("oo")
        output1 = str(np.mean(img))
        # print(output)
        cv2.putText(img, output + output1, (10, 30), font, 1, color, stroke, cv2.LINE_AA)
        # cv2.imshow('video', img)
        ###END

        img_id += 1
        # img = cv2.resize(img, None, fx=0.8, fy=0.8)
        height, width, channels = img.shape

        # detecting Image
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, False)

        # for b in blob:
        #    for n, img_blob in enumerate(b):
        #        cv2.imshow(str(n), img_blob)

        net.setInput(blob)
        outs = net.forward(outputLayers)
        # print(outs)

        class_ids = []
        confidences = []
        boxes = []
        # showing info on screen:
        # Object_Detection_info = []
        for out in outs:
            for i, detection in enumerate(out):
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # if confidence > 0.2:
                    # Object Detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # cv2.circle(img, (center_x,center_y), 10, (0, 255, 0), 2)

                    # Rectangle Co-Ordinates:
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    # print(class_id)
                    class_ids.append(class_id)

        # print(len(boxes))
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
        # print(indexes)
        number_objects_detected = len(boxes)
        font = cv2.FONT_HERSHEY_DUPLEX
        Object_Detection_info = []
        object_id = 0
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                # print(label)
                # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                # cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                object_name = label

                cv2.putText(img, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, color, 1)
                object_id += 1
                origin_x_y = (x, y)
                Object_Detection_info.append([object_id, object_name, origin_x_y, w, h])
                audioio.speek(object_name)
        print(Object_Detection_info)

        # od.setData(Object_Detection_info)
        # temp = listToString(Object_Detection_info)
        # audioio.speek(temp)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)

        fc, face_width_in_frame = face_data(img)
        face_len = len(fc)
        print(face_width_in_frame, "in While loop")

        for (x, y, w, h) in faces:
            print(x, y, w, h)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            id_, conf = recognizer.predict(roi_gray)
            if conf >= 45 and conf <= 80:
                print(id_)
                print(lables[id_])
                conf = str(conf)
                print(conf)
                # print(lables[conf])
                cv2.putText(img, str(lables[id_]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            audioio.speek(lables[id_])

            img_item = "DK.PNG"

            cv2.imwrite(img_item, roi_gray)

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)

        print(type(face_width_in_frame))
        if face_width_in_frame != 0:
            for i in range(face_len):
                if i == 0:
                    Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame[i])
                    print(i)
                    cv2.putText(img, f"Distance = {round(Distance, 2)} CM", (50, 50), fonts, 1, (WHITE), 2)

                if i == 1:
                    Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame[i])
                    print(i)
                    cv2.putText(img, f"Distance = {round(Distance, 2)} CM", (50, 70), fonts, 1, (WHITE), 2)
                if i == 2:
                    Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame[i])
                    print(i)
                    cv2.putText(img, f"Distance = {round(Distance, 2)} CM", (50, 90), fonts, 1, (WHITE), 2)
                if i == 3:
                    Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame[i])
                    print(i)
                    cv2.putText(img, f"Distance = {round(Distance, 2)} CM", (50, 110), fonts, 1, (WHITE), 2)
                if i == 4:
                    Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame[i])
                    print(i)
                    cv2.putText(img, f"Distance = {round(Distance, 2)} CM", (50, 130), fonts, 1, (WHITE), 2)
                if i == 5:
                    Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame[i])
                    print(i)
                    cv2.putText(img, f"Distance = {round(Distance, 2)} CM", (50, 150), fonts, 1, (WHITE), 2)
                Distance = round(Distance, 2)
                dist = str(Distance)
                audioio.speek(lables[id_] + ' at ' + dist + 'cm')

        setHome(output)

        # Detecting Objects
        elapsed_time = time.time() - starting_time
        fps = img_id / elapsed_time
        cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 1, (0, 0, 0), 1)
        cv2.imshow("Image", img)
        # cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


class ObjectData():
    def __init__(self):
        self.data = []

    def setData(self, data):
        self.data = data

    def getData(self):
        return self.data


def setHome(output):
    person_flag = False
    data = od.getData()

    now = datetime.now()
    current_time = now.strftime("%H:%M").split(":")
    if (int(current_time[0]) > 7 and int(current_time[0]) < 19):
        time = "Day"
    else:
        time = "Night"

    print(time)
    print(data)

    print("light : ", HomeState.light)
    print("fan : ", HomeState.fan)

    for object in data:
        if object[1] == "person":
            person_flag = True

    if (person_flag == False):
        HomeState.light = False
        HomeState.fan = False
        HomeState.force_light = 0
        HomeState.force_fan = 0

    else:
        if (HomeState.force_light == 0):
            if time == "Day":
                if (output == "light"):
                    HomeState.light = False
                else:
                    HomeState.light = True
            else:
                HomeState.light = False
                # if(person_flag!=sleeping):
                #     print("light = on")
                # else:
                #     print("light = off")

    #
    # if(flag==True and output=="light"):
    #     print("Nothing")
    #
    # elif(flag==True and output=="dark"):
    #     print("light on")
    #     speek("turnning light on")
    #
    # elif(flag==False and output=="light"):
    #     print("light off")
    #     speek("turnning light off")
    #
    # elif(flag==False and output=="dark"):
    #     print("Nothing")


def speek(text):
    tts = gTTS(text=text, lang="en")
    filename = "home.mp3"
    # print("file")
    tts.save(filename)
    # print("gen")
    os.system("play home.mp3")
    # print("fin")


def main():
    objectDetection()








if __name__ == "__main__":
    od = ObjectData()
    main()
Timer(30.0, objectDetection).start()


