"""
Alternated version of recording script (modified for wand duels) 

Special thanks to Group F (SS22/IUI):
Nour-Eddine El Malki, Fabian Heinze, Backtash Fawad Mohammadi
"""

## configuration

show_serial_ports= True # IMPORTANT: if true, the available ports are printed to the console

# port value from console needs to be set below (examples of how the port value may look in comment)
arduino_port = "/dev/tty.usbmodem1101" # for windows, e.g., "COM1"; for mac/linux, e.g., "/dev/cu.usbmodem12345"
arduino_baudrate = 115200

delimiter = ';'

duel_server_ip = '127.0.0.1'
duel_server_url = 'http://' + duel_server_ip + ':5000/castspell'

## end of configuration



## imports
import os
from os import path

import sys
import csv
import glob
import requests
import threading

import numpy as np
import pandas as pd
import tkinter as tk

from tkinter import ttk
from datetime import datetime
from yourcode import process_spell

# pip install pyserial (in case the following two imports fail)
import serial
import serial.tools.list_ports as ports 



## variables for status
isConnected = False
isRecording = False

## store collected Bluno data
csv_lines = []

## generate header for storage file (CSV)
features = [
    "id", 
    "wizardName", "spellName", 
    "accX", "accY", "accZ", 
    "gyroX", "gyroY", "gyroZ", 
    "time"]


## get folder for battlelogs (create if it does not exist yet)
battlelogs_folder = path.join(path.curdir, 'battlelogs')
if not path.exists(battlelogs_folder):
    os.mkdir(battlelogs_folder)

## prints all available ports into the console (if settings is set to True)
if show_serial_ports:
    def serial_ports():
        """
        Function from Stackoverflow
        https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result

    print("Available ports are: ")
    print(serial_ports())

## method that sets the status label according to the current status
def set_state(state):
    label_status_value['text'] = f"{state}"

## method called to start connecting to the magic wand
def connect():
    set_state(f"Connecting to Port: {arduino_port}")
    connect_thread = threading.Thread(target=connect_wand_thread, daemon=True)
    connect_thread.start()

## method that starts the connection to the magic wand (called as a thread)
def connect_wand_thread():
    global isRecording, isConnected, csv_lines
    wand_device  = serial.Serial(port=arduino_port, baudrate=arduino_baudrate, timeout=.1)
    wand_device.flushInput()
    isConnected = False
    tries = 0
    while not isConnected and tries <= 30:
        line = str(wand_device.readline())
        if line.find('Magic Wand setup done') != -1:
            isConnected = True
        tries = tries + 1
    if isConnected:
        button_connect['state'] = "disabled"
        set_state(f"Connected to Port: {arduino_port}")
        print(delimiter.join(features))
        startTime = 0
        while isConnected:
            telemetry_line = str(wand_device.readline())
            if isRecording:
                if telemetry_line.find(',') != -1:
                    telemetry_data = telemetry_line.split(',')
                    if len(telemetry_data) > 9:
                        time = int(remove_escape_sequence(telemetry_data[9]))
                        if len(csv_lines) <= 0:
                            startTime = time
                        row = [len(csv_lines), '', '', float(telemetry_data[1]), float(telemetry_data[2]), float(telemetry_data[3]), float(telemetry_data[5]), float(telemetry_data[6]), float(telemetry_data[7]), time - startTime]
                        csv_lines.append(row)
                        print(row)
    else:
        button_connect['state'] = "normal"
        set_state("Disconnected")

## method that processes a switch in recording status
def toggle_recording():
    global isRecording, isConnected, csv_lines, count
    if not isConnected:
        return

    if not isRecording:
        isRecording = True
        button_cast['text'] = "Stop Casting"
        set_state(f"Recording from Port: {arduino_port}")
    else:
        isRecording = False
        button_cast['text'] = "Cast Spell"
        if not isConnected:
            button_connect['state'] = "normal"
            set_state(f"Disconnected")
        else:
            if len(csv_lines) > 0:
                now = datetime.now()
                filename = path.join(battlelogs_folder, f"battlelog-" + now.strftime("%Y%m%d-%H%M%S") + ".csv")
                with open(filename, "w", newline='') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=delimiter)
                    csv_writer.writerow(features)
                    csv_writer.writerows(csv_lines)
                csv_lines = []
                cast_spell_from_file(filename)
            button_connect['state'] = "disabled"
            set_state(f"Connected to Port: {arduino_port}")

## method that is called when testing with prerecorded example
def test():
    cast_spell_from_file(os.path.join('battlelogs', 'battlelog-example.csv'))

## method that cleans string of escape sequences at the end of a line
def remove_escape_sequence(val):
    str_val = str(val)
    str_val = str_val.replace("'", "")
    str_val = str_val.replace("\\n", "")
    str_val = str_val.replace("\\r", "")
    return str_val

def cast_spell_from_file(path):
    pandas_df = pd.read_csv(path, delimiter=delimiter)
    prediction = process_spell(pandas_df)
    params = {'teamname': entry_team.get(), 'slot': int(entry_slot.get()), 'spellname': prediction[1], 'spellclass': prediction[0]}
    try:
        r = requests.get(url=duel_server_url, params=params)
        if r.status_code == 200:
            print("Spell successfully send to server")
        else:
            print("ERROR: Spell could not be sent")
    except:
        print("ERROR: Spell could not be sent")


## graphical user interface

# create window frame
root = tk.Tk()
root.wm_title("Magic Wand | Duel Client")
root.minsize(width=350, height=280)

# create status label
frame_status = tk.Frame()
label_status = tk.Label(master=frame_status, text="Status:", width=15)
label_status.pack(side=tk.LEFT, padx=10)
label_status_value = tk.Label(master=frame_status, text="Disconnected")
label_status_value.pack(side=tk.LEFT, padx=10)
frame_status.pack(padx=10, pady=10)

# create button to connect to magic wand
frame_connect = tk.Frame()
button_connect = tk.Button(master=frame_connect, text="Connect", command=connect)
button_connect.pack()
separator = ttk.Separator(master=frame_connect, orient='horizontal')
separator.pack(fill='x', ipady=15)
frame_connect.pack()

# create entry field for team name
frame_team = tk.Frame()
label_team = tk.Label(master=frame_team, text="Wizard Team Name:", width=15)
label_team.pack(side=tk.LEFT, padx=10)
entry_team = tk.Entry(master=frame_team)
entry_team.insert(0, "ENTER your teamname")
entry_team.pack(side=tk.LEFT, padx=10)
frame_team.pack(padx=10, pady=5)

# create entry field for slot
frame_slot = tk.Frame()
label_slot = tk.Label(master=frame_slot, text="Slot (1=left, 2=right):", width=15)
label_slot.pack(side=tk.LEFT, padx=10)
entry_slot = tk.Entry(master=frame_slot)
entry_slot.insert(0, "1")
entry_slot.pack(side=tk.LEFT, padx=10)
frame_slot.pack(padx=10, pady=5)

# create button to start and stop casting
frame_cast = tk.Frame()
button_cast = tk.Button(master=frame_cast, text="Cast Spell", command=toggle_recording)
button_cast.pack()
separator = ttk.Separator(master=frame_cast, orient='horizontal')
separator.pack(fill='x', ipady=15)
frame_cast.pack()

# create button to start and stop casting
frame_test = tk.Frame()
button_test = tk.Button(master=frame_test, text="Test with Prerecording", command=test)
button_test.pack()
frame_test.pack(pady=10)

# GUI life loop
root.mainloop()