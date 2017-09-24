import time
class Car:

    def __init__(self, left_wheel, right_wheel, speed=0):
        self.left_wheel = left_wheel
        self.right_wheel = right_wheel
        self.speed = speed
        self.steer_speed = 40

    @property
    def speed(self):
        return self.__speed
        
    @speed.setter
    def speed(self, speed):
        if speed < 0:
            self.__speed = 0
        elif speed > 100:
            self.__speed = 100
        else:
            self.__speed = speed
        
        self.left_wheel.PWM.ChangeDutyCycle(self.__speed)
        self.right_wheel.PWM.ChangeDutyCycle(self.__speed)
        

    def start(self):
        self.left_wheel.PWM.start(self.speed)
        self.right_wheel.PWM.start(self.speed)
    
    def turn_forward_left(self):
        self.right_wheel.forward()
        self.steer(to_left=self.steer_speed)
        self.left_wheel.forward()
    
    def turn_forward_right(self):
        self.left_wheel.forward()
        self.steer(to_right=self.steer_speed)
        self.right_wheel.forward()
        
    def turn_reverse_left(self):
        self.right_wheel.reverse()
        self.steer(to_right=self.steer_speed)
        self.left_wheel.reverse()
    
    def turn_reverse_right(self):
        self.left_wheel.reverse()
        self.steer(to_left=self.steer_speed)
        self.right_wheel.reverse()

    def steer(self, to_left=100, to_right=100):
        self.right_wheel.PWM.ChangeDutyCycle(to_right)
        self.left_wheel.PWM.ChangeDutyCycle(to_left)
        
    def forward(self):
        self.left_wheel.forward()
        self.right_wheel.forward()

    def reverse(self):
        self.left_wheel.reverse()
        self.right_wheel.reverse()
    
    def turn_left(self):
        self.left_wheel.reverse()
        self.right_wheel.forward()

    def turn_right(self):
        self.left_wheel.forward()
        self.right_wheel.reverse()

    def stop(self):
        self.left_wheel.stop()
        self.right_wheel.stop()
    
