import cv2

def classify_signs(image, cascade_classifier):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    classified = cascade_classifier.detectMultiScale(gray)
    return classified

def show_box(image, classified, color=(150, 150, 255), thickness=2):
    for x,y,w,h in classified:
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
