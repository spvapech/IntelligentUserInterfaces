# Project 1: Magic Wand


## Requirements

1. Arduino IDE
2. Wizard Wand and MicroUSB cable
3. Python Sensor Recorder


## Overview

### Recording of Spells

1. Arduino Firmware (*arduino-firmware-magicwand*)
2. Python Recorder (*python-alternative-recorder*)

### Dueling

1. Python Client (*python-client-wandduel*) - two wizards (clients) required
2. Python Server (*python-server-wandduel*) - one arena (server) required


## Folder Structure

* *arduino-firmware-magicwand* Arduino Project for the wizard wand firmware (use Arduino IDE to deploy the firmware onto the wizward wand)
* *python-recorder* For data recording (please specify the USB port to which the wizard wand is connected and run the script with a Python interpreter)
* *python-client-wandduel* Python script that each wizard has to call to cast a spell during a duell; here, you need to adapt the "yourcode.py" to add your classifier
* *python-server-wandduel* Python script that represent the arena; opens a webpage to which wizards can send their spells and shows the dueling outcome
* *recording-examples* Three example CSV files (lengths: 4.4 sec, 10.8 sec, 32.9 sec) to show that the data recording works without package loss for me (also shows the added id and time column

## Python Scripts

* To automatically install all Python packages required to execute a specific script, use "pip install -r requirements.txt" inside the the script's folder
* All Python scripts offer different settings to customize their behavior; they can be found in the first lines of each script