/*
 * Magic Wand Firmware
 * Module "Intelligent User Interfaces"
 * Project 2
 * 
 * Arduino Project
 * Board: Bluno Beetle v1.1
 * 
 * Tutorial: https://www.dfrobot.com/blog-283.html
 * 
 */

#include<Wire.h>
const int MPU = 0x68;
int16_t AcX, AcY, AcZ, Tmp, GyX, GyY, GyZ;
int16_t count;

int sampleDelay = 100;

void setupWire() {
  Wire.begin();
  Wire.beginTransmission(MPU);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
}

void setup() {
  Serial.begin(115200);
  setupWire();
  count = 0;
  Serial.println("Magic Wand setup done!");
}

void loop() {
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, 12, true);

  // Sample acceleration and gyroscope data in all three dimensions
  AcX = Wire.read() << 8 | Wire.read();
  AcY = Wire.read() << 8 | Wire.read();
  AcZ = Wire.read() << 8 | Wire.read();
  GyX = Wire.read() << 8 | Wire.read();
  GyY = Wire.read() << 8 | Wire.read();
  GyZ = Wire.read() << 8 | Wire.read();

  // Prepare the payload that is sent to other devices
  String payload = "#A," + String(AcX) + "," + String(AcY) + "," + String(AcZ)
                   + ",G," + String(GyX) + "," + String(GyY) + "," + String(GyZ) 
                   + ",T," + String(millis()) + "," + String(count) + "#";


  // Send the payload via bluetooth
  for (int i = 0; i < payload.length(); i++)
  {
    Serial.print(payload[i]);   // Push each char 1 by 1 on each loop pass
  }
  Serial.println("");

  count++;
  delay(sampleDelay);
}
