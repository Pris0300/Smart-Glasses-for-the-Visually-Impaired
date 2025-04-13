## ----------------- ML TEST ----------------- ##
# Used to undestarnd the YOLO algorithm and how to use  the features it uses.
# The code is based on the tutorial from https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
# Also modified with other own ideas and codes.

## Way to call it: python Test.py -c yolov3.cfg -w yolov3.weights -cl yolov.txt

import cv2
import argparse
import numpy as np
import pyttsx3

# Argument parser to get the path to the image and the model.
ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image',
#                help = 'path to input image')
ap.add_argument('-c', '--config',
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights',
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes',
                help = 'path to text file containing class names')
args = ap.parse_args()

## ----------------- Functions ----------------- ##
# Function to get the output layer names in the YOLO architecture.
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# Function to draw bounding box on the detected object with class name.
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label +' '+str(round(confidence*100,2)), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Function to determine the position of an object in an image based on its coordinates
def determine_position(x, y, width, height):        
    # Calculate the size of each section of the image
    left_section = 0.3
    middle_section = 0.4
    
    # Calculate the bounds of each section
    left_bound = left_section * width
    middle_bound = left_bound + (middle_section * width)
    
    # Determine the position of the object
    if x < left_bound:
        return "Left"
    elif x > middle_bound:
        return "Right"
    else:
        return "Middle"

## ----------------- Init ----------------- ##
# Initialize the video capture object to get the real-time video feed from the webcam. / Change the 0 to the path to the video file to use a prerecorded video.
cap = cv2.VideoCapture (0)
# initialize the text-to-speech engine
engine = pyttsx3.init()

# Set the scale factor for the image and initialize the classes variable
scale = 0.00392
classes = None

# set properties for the engine
engine.setProperty("rate", 150) # set speaking rate
engine.setProperty("volume", 0.5) # set volume (0 to 1)

# Read the classes from the file and set a random color for each class.
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Read the pre-trained model and config file.
net = cv2.dnn.readNet(args.weights, args.config)

## ----------------- Main ----------------- ##
# Read the video frame by frame and process it.
while cap.isOpened ():
    _, frame = cap.read ()
    if frame is None: break
    # image = cv2.imread(args.image)
    image = frame
    Width = image.shape[1]
    Height = image.shape[0]

    # Convert the image into a blob that can be input to the YOLO model
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    
    # Set the input to the model
    net.setInput(blob)
    
    # Get the output layers from the YOLO model
    outs = net.forward(get_output_layers(net))

    # Initialize the lists for the class ids, confidences and the boxes.
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Loop through each output layer from the YOLO model
    for out in outs:
        for detection in out:
            # Get the class id, confidence and the bounding box for each detection.
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # This line uses Non-Maximum Suppression (NMS) algorithm to remove the overlapping bounding boxes and keep only the most confident ones for each object detected.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Loop through the indices of the remaining boxes after NMS
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]

        # Extract the x, y, w, and h coordinates from the box
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        # Display the image with bounding boxes and labels
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        position = determine_position(x, y, Width, Height)
        print('Pos: ',position)

        # # Speak the position of the object (Commented out because it is annoying)
        # engine.say(position)
        # engine.runAndWait()
        
    # Let the user see what we have done.
    cv2.imshow('title', frame)

    # If Q is pressed, exit the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tidy up OpenCV.
cap.release()
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()    


### ----------------- Tests ----------------- ###
# import cv2
# import pyaudio

# # Load the image
# img = cv2.imread('image.jpg')

# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Detect the objects in the image using a pre-trained object detector
# detector = cv2.CascadeClassifier('object_detector.xml')
# objects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# # Divide the image into three regions: left, middle, and right
# img_height, img_width, _ = img.shape
# left_boundary = int(img_width * 0.3)
# right_boundary = int(img_width * 0.7)

# # Group the objects into left, middle, or right
# left_objects = []
# middle_objects = []
# right_objects = []
# for (x, y, w, h) in objects:
#     if x + w < left_boundary:
#         left_objects.append((x, y, w, h))
#     elif x > right_boundary:
#         right_objects.append((x, y, w, h))
#     else:
#         middle_objects.append((x, y, w, h))

# ---------------------------------------------

# import cv2
# import numpy as np
# import pyttsx3

# # Load the image
# image = cv2.imread("example.jpg")

# # Get the dimensions of the image
# height, width, _ = image.shape

# # Define the regions of interest
# left_roi = [0, 0, int(width*0.3), height]
# middle_roi = [int(width*0.3), 0, int(width*0.4), height]
# right_roi = [int(width*0.7), 0, int(width*0.3), height]

# # Create a list to hold the coordinates of the objects
# object_coordinates = []

# # Convert the image to grayscale for better edge detection
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply thresholding to isolate the objects
# _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# # Find contours in the thresholded image
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Loop over the contours and find their bounding boxes
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     object_coordinates.append((x, y, x+w, y+h))

# # Group the objects based on their coordinates
# left_objects = []
# middle_objects = []
# right_objects = []
# for coord in object_coordinates:
#     if coord[0] < left_roi[2]:
#         left_objects.append(coord)
#     elif coord[0] > right_roi[0]:
#         right_objects.append(coord)
#     else:
#         middle_objects.append(coord)

# # Determine which group has the most objects
# if len(left_objects) > len(middle_objects) and len(left_objects) > len(right_objects):
#     message = "There are more objects on the left"
# elif len(middle_objects) > len(left_objects) and len(middle_objects) > len(right_objects):
#     message = "There are more objects in the middle"
# else:
#     message = "There are more objects on the right"

# ---------------------------------------------

# import cv2
# import numpy as np
# import pyttsx3

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Load image
# img = cv2.imread('path/to/image.jpg')

# # Convert image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Threshold image to binary
# _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# # Find contours in the image
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Calculate image dimensions
# height, width, _ = img.shape

# # Define percentage ranges for left, middle, and right groups
# left_range = int(width * 0.3)
# middle_range = int(width * 0.4)
# right_range = int(width * 0.3)

# # Initialize object count for each group
# left_count = 0
# middle_count = 0
# right_count = 0

# # Loop through each contour in the image
# for contour in contours:
#     # Get bounding rectangle coordinates for contour
#     x, y, w, h

# ---------------------------------------------

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Read the image
# img = cv2.imread('image.jpg')

# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply binary thresholding to segment the image
# _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# # Find the contours in the image
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Divide the image into three parts: left, middle, and right
# height, width = img.shape[:2]
# left = 0
# middle = int(0.4 * width)
# right = int(0.7 * width)

# # Group the objects based on their coordinates
# left_objs = []
# middle_objs = []
# right_objs = []

# for cnt in contours:
#     # Get the bounding box of the contour
#     x, y, w, h = cv2.boundingRect(cnt)
#     # Check the location of the bounding box
#     if x + w < middle:
#         left_objs.append(cnt)
#     elif x > right:
#         right_objs.append(cnt)
#     else:
#         middle_objs.append(cnt)

# # Count the number of objects in each group
# num_left_objs = len(left_objs)
# num_middle_objs = len(middle_objs)
# num_right_objs = len(right_objs)

# # Determine which group has the most objects
# if num_left_objs > num_middle_objs and num_left_objs > num_right_objs:
#     result = "left"
# elif num_middle_objs > num_left_objs and num_middle_objs > num_right_objs:
#     result = "middle"
# else:
#     result = "right"

# # Print the result and display the image with the bounding boxes
# print("The group with the most objects is:", result)

# # Draw the bounding boxes on the image
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt


