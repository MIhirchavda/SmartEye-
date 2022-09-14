import cv2
import numpy as np
import time
from gtts import gTTS
import os
from datetime import datetime
import HomeState
import audioio

# cap = cv2.VideoCapture("/home/robo/VID_20201205_180147.mp4")
#Load YOLO
cap = cv2.VideoCapture(0)

weights_path = "Models/Emma_models/yolov3.weights"
cfg_path = "Models/Emma_models/yolov3.cfg"
coco_path = "Models/Emma_models/coco.names"
net =cv2.dnn.readNet(weights_path, cfg_path)
classes = []
with open(coco_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_name = net.getLayerNames()
outputLayers = [layer_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes), 3))

#Loading image
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
    #print(np.mean(frame))
    #print("no")
    is_light = np.mean(frame) > thrshld
    #print("prob")
    #cv2.imshow('frame', frame)

    return 'light' if is_light else 'dark'


def objectDetection():
    img_id = 0

    while True:
        #img = cv2.imread("img.png")
        ret, img = cap.read()

        ### brightness detection
        color = (255, 255, 255)
        stroke = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        #print("0")
        output = str(img_estim(img, 50))
        #print("oo")
        output1 = str(np.mean(img))
        #print(output)
        cv2.putText(img, output + output1, (10, 30), font, 1, color, stroke, cv2.LINE_AA)
        #cv2.imshow('video', img)
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
        #print(outs)

        class_ids = []
        confidences = []
        boxes = []
        # showing info on screen:
        #Object_Detection_info = []
        for out in outs:
            for i, detection in enumerate(out):
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                #if confidence > 0.2:
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
                    #print(class_id)
                    class_ids.append(class_id)

        # print(len(boxes))
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
        #print(indexes)
        number_objects_detected = len(boxes)
        font = cv2.FONT_HERSHEY_DUPLEX
        Object_Detection_info = []
        object_id = 0
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                #print(label)
                #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                #cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                object_name = label

                cv2.putText(img, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, color, 1)
                object_id += 1
                origin_x_y = (x, y)
                Object_Detection_info.append([object_id, object_name, origin_x_y, w, h])
                audioio.speek(object_name)
        print(Object_Detection_info)
        #od.setData(Object_Detection_info)
        #temp = listToString(Object_Detection_info)
        #audioio.speek(temp)

        setHome(output)

        # Detecting Objects
        elapsed_time = time.time() - starting_time
        fps = img_id / elapsed_time
        cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 1, (0, 0, 0), 1)
        cv2.imshow("Image", img)
        #cv2.imshow('image', img)
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
    if(int(current_time[0]) > 7 and int(current_time[0]) <19 ):
        time = "Day"
    else:
        time = "Night"

    print(time)
    print(data)

    print("light : ",HomeState.light)
    print("fan : ",HomeState.fan)

    for object in data:
        if object[1] == "person":
            person_flag = True

    if(person_flag == False):
        HomeState.light = False
        HomeState.fan = False
        HomeState.force_light = 0
        HomeState.force_fan = 0

    else:
        if(HomeState.force_light==0):
            if time == "Day":
                if(output == "light"):
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
    tts = gTTS(text= text, lang = "en")
    filename = "home.mp3"
    #print("file")
    tts.save(filename)
    #print("gen")
    os.system("play home.mp3")
    #print("fin")

def main():
    objectDetection()

if __name__ == "__main__":
    #od = ObjectData()
    main()
od = ObjectData()
