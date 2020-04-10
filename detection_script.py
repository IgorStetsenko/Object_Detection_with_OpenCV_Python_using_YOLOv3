import cv2
import numpy as np #import lib.

def load_yolov3():
    #load YOLO
    net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
    classes = []
    with open ('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return outputlayers, classes, net

def load_image(image_number):
    #loading image
    img = cv2.imread("img/"+str(image_number)+".jpg")
    #img = cv2.resize(img,None,fx=0.4,fy=0.3)
    height,width,channels = img.shape
    return height, width, channels, img

def detecting_objects(outputlayers, img, net):
    #detecting objects
    blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
    net.setInput(blob)
    outs = net.forward(outputlayers)
    return outs

def add_coordinate_objects(outs, width, height):
    # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids = []
    confidences = []
    boxes = []
    ind = w = h = x = y = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])  # put all rectangle areas
                confidences.append(float(confidence))  # how confidence was that object detected and show that percentage
                class_ids.append(class_id)  # name of the object tha was detected
                ind = cv2.dnn.NMSBoxes(boxes, confidences,0.4,0.6)
    return ind, boxes, class_ids, w, h, x, y

def draw_rectangle(img, indexes, boxes, classes, class_ids, w, h, x, y):
    #draw_rectangle
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    font = cv2.FONT_HERSHEY_PLAIN
    image_rectangle = None
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            image_rectangle = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            image_rectangle = cv2.putText(image_rectangle, label, (x, y + 30), font, 1, (255, 255, 255), 2)
    return  image_rectangle

def show_image(image_rectangle):
    #show_image
    cv2.imshow("Image_detected", image_rectangle)
    cv2.waitKey(10)
    cv2.destroyAllWindows()

def main_algoritm():
    image_number = 1
    while image_number <= 12:
        outputlayers, classes, net = load_yolov3()
        height, width, channels, img = load_image(image_number)
        outs = detecting_objects(outputlayers, img, net)
        ind, boxes, class_ids, w, h, x, y = add_coordinate_objects(outs, width, height)
        image_rectangle = draw_rectangle(img, ind, boxes, classes, class_ids, w, h, x, y)
        show_image(image_rectangle)
        image_number+=1

if __name__ == '__main__':

    main_algoritm()