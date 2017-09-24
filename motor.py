import RPi.GPIO as GPIO
class Motor:
    """
    Add speed, acceleration rate.
    """
    def __init__(self, enable_pin, forward_pin, reverse_pin):
        self.forward_pin = forward_pin
        self.reverse_pin = reverse_pin
        self.enable_pin = enable_pin
        
        GPIO.setup(self.enable_pin, GPIO.OUT)
        GPIO.setup(self.forward_pin, GPIO.OUT)
        GPIO.setup(self.reverse_pin, GPIO.OUT)
        GPIO.output(self.enable_pin, True)
        self.PWM = GPIO.PWM(enable_pin, 1000)

    def forward(self):
        GPIO.output(self.forward_pin, True)
        GPIO.output(self.reverse_pin, False)
    
    def reverse(self):
        GPIO.output(self.forward_pin, False)
        GPIO.output(self.reverse_pin, True)

    def stop(self):
        GPIO.output(self.forward_pin, False)
        GPIO.output(self.reverse_pin, False)
        

