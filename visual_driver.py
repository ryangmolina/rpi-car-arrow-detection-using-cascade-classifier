from motor import Motor
from car import Car
import numpy as np
import cv2
import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
import picamera
import time
import lane_detector
import sign_detector


def main():
    ena = 12
    enb = 16

    in1 = 7
    in2 = 11
    in3 = 13
    in4 = 15

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)

    right_wheel = Motor(ena, in1, in2)
    left_wheel = Motor(enb, in3, in4)

    car = Car(left_wheel, right_wheel)
    car.speed = 0
    # check engine
    car.start()
    car.stop()

    # camera resolution
    HEIGHT = 128
    WIDTH = 160

    with picamera.PiCamera(resolution=(WIDTH, HEIGHT),
                           sensor_mode=2) as camera:

        # initialization time of camera
        time.sleep(2)
        # video camera buffer
        rawCapture = PiRGBArray(camera, size=(WIDTH, HEIGHT))

        # trained cascade classifier for each object to detect
        left_cascade = cv2.CascadeClassifier('haar_trained_xml/left/cascade.xml')
        right_cascade = cv2.CascadeClassifier('haar_trained_xml/right/cascade.xml')

        detector = lane_detector.LaneDetector(lane_detector.HoughLineTransform(threshold=30))
        #detector = lane_detector.LaneDetector(lane_detector.HoughLineTransformP())
        #detector = lane_detector.LaneDetector(lane_detector.Contour())

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            # lane_roi = image[int(HEIGHT/2):HEIGHT,:].copy()    
            # lane_roi, action = detector.detect(lane_roi)
            
            # detect object in current frame
            left_signs = sign_detector.classify_signs(image, left_cascade)
            right_signs = sign_detector.classify_signs(image, right_cascade)
            
            # draw bounding box to the object detected
            sign_detector.show_box(image, left_signs)
            sign_detector.show_box(image, right_signs)
            
            # check if it is time to turn
            if len(left_signs) > 0:
                x,y,w,h = left_signs[0]
                distance = distance_to_camera(8, 50, w)-1
                print(distance)
                if distance < 5:
                    car.speed = 100
                    car.turn_left()
                    time.sleep(0.6)
                    car.stop()
            elif len(right_signs) > 0:
                x,y,w,h = right_signs[0]
                distance = distance_to_camera(8, 50, w)-1
                print(distance)
                if distance < 5:
                    car.speed = 100
                    car.turn_right()
                    time.sleep(0.6)
                    car.stop()
            
            # back to default action
            car.speed = 35
            car.forward()
          
            if action == "forward":
                car.forward()
            elif action == "turn_left":
                car.speed = 60
                car.turn_left()
                time.sleep(0.3)
            elif action == "turn_right":
                car.turn_right()
                car.speed = 60
                time.sleep(0.3)
            elif action == "stop":
                car.stop()
            cv2.imshow("Frame", image)
            cv2.imshow("Lane", lane_roi)

            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)
            if key == ord('q'):
                GPIO.cleanup()
                break


def find_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_gauss = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray_gauss, 50, 150)

    (contoured_image, contours, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    try:
        contour = max(contours, key=cv2.contourArea)
        return cv2.minAreaRect(contour)
    except:
        pass

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

if __name__ == "__main__":
    main()
