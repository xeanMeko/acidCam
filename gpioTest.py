import time
import serial

PORT = "/dev/ttyUSB0"   # Linux/Raspberry Pi
# PORT = "COM3"         # Windows example

ser = serial.Serial(PORT, 9600, timeout=1)

relay_on  = bytes([0xA0, 0x01, 0x01, 0xA2])
relay_off = bytes([0xA0, 0x01, 0x00, 0xA1])
print("Relay ON")
ser.write(relay_on)
time.sleep(1)

print("Relay OFF")
ser.write(relay_off)
time.sleep(1)

ser.close()
print("Done")