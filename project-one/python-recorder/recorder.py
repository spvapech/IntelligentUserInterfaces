## configuration

show_serial_ports= False # IMPORTANT: if true, the available ports are printed to the console

# port value from console needs to be set below (examples of how the port value may look in comment)
arduino_port = "/dev/tty.usbmodem1301" # for windows, e.g., "COM1"; for mac/linux, e.g., "/dev/cu.usbmodem12345"
arduino_baudrate = 115200

delimiter = ';'

## end of configuration

## imports
import os
from os import path
import sys
import csv
import glob
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import serial
import pandas as pd
import joblib
from scipy import stats


# MODELLE LADEN (einmalig beim Start)
scaler = joblib.load("../python-client-wandduel/scaler.pkl")
rf     = joblib.load("../python-client-wandduel/gesture_rf.pkl")

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

## get folder for recordings (create if it does not exist yet)
recording_folder = path.join(path.curdir, 'recordings')
if not path.exists(recording_folder):
    os.mkdir(recording_folder)

## optional: prints all available ports into the console
def serial_ports():
    if sys.platform.startswith('win'):
        ports_list = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith(('linux','cygwin')):
        ports_list = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports_list = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')
    result = []
    for p in ports_list:
        try:
            s = serial.Serial(p)
            s.close()
            result.append(p)
        except (OSError, serial.SerialException):
            pass
    return result

if show_serial_ports:
    print("Available ports are:")
    print(serial_ports())

## helper to update GUI status label
def set_state(state):
    label_status_value['text'] = f"{state}"

## connect button handler
def connect():
    set_state(f"Connecting to Port: {arduino_port}")
    threading.Thread(target=connect_wand_thread, daemon=True).start()

## thread method to open serial and read initial handshake
def connect_wand_thread():
    global isConnected, csv_lines
    wand = serial.Serial(port=arduino_port, baudrate=arduino_baudrate, timeout=0.1)
    wand.flushInput()
    # wait for setup completion message
    for _ in range(30):
        line = wand.readline().decode(errors='ignore')
        if 'Magic Wand setup done' in line:
            isConnected = True
            break
    if isConnected:
        button_connect['state'] = 'disabled'
        set_state(f"Connected to Port: {arduino_port}")
        startTime = None
        while isConnected:
            raw = wand.readline().decode(errors='ignore')
            if isRecording and ',' in raw:
                parts = raw.replace("#", "").split(',')
                if len(parts) >= 10:
                    t = int(parts[9])
                    if startTime is None:
                        startTime = t
                    row = [
                        len(csv_lines),
                        entry_wizard.get(),
                        entry_spell.get(),
                        float(parts[1]), float(parts[2]), float(parts[3]),
                        float(parts[5]), float(parts[6]), float(parts[7]),
                        t - startTime
                    ]
                    csv_lines.append(row)
    else:
        set_state("Disconnected")
        button_connect['state'] = 'normal'

## toggle recording start/stop
def toggle_recording(df=None):
    global isRecording, csv_lines
    if not isConnected:
        return
    if not isRecording:
        isRecording = True
        button_record['text'] = 'Stop Recording'
        set_state(f"Recording from Port: {arduino_port}")
    else:
        isRecording = False
        button_record['text'] = 'Record Spell'
        # save CSV
        if csv_lines:
            now = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"recording-{entry_wizard.get()}-{entry_spell.get()}-{now}.csv"
            full_path = path.join(recording_folder, filename)
            with open(full_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=delimiter)
                writer.writerow(features)
                writer.writerows(csv_lines)
            csv_lines = []
            set_state(f"Saved: {entry_spell.get()} + {entry_wizard.get()} + {now}")

            # ─── Prediction mit Random Forest ───────────────────────────
            df_rec = pd.read_csv(full_path, delimiter=delimiter)

            # 1) Outlier entfernen
            df_clean  = remove_outliers(df_rec)

            # 2) Lineare Interpolation
            df_interp = interpolate_df(df_clean)

            # 3) Features extrahieren
            feats = extract_features(df_interp).reshape(1, -1)

            # 4) Skalieren
            feats_s = scaler.transform(feats)

            # 5) Vorhersage
            pred = rf.predict(feats_s)[0]
            print(f"🔮 Prediction: {pred}")

            # plot automatically
            plot_recording(full_path)
        button_connect['state'] = 'disabled'




def remove_outliers(df, cols=['accX','accY','accZ','gyroX','gyroY','gyroZ']):
    zs = np.abs(stats.zscore(df[cols]))
    mask = (zs < 3).all(axis=1)
    return df[mask]

def interpolate_df(df):
    return df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

def extract_features(df):
    cols = ['accX','accY','accZ','gyroX','gyroY','gyroZ']
    X = df[cols].to_numpy()
    feats = []
    feats += list(X.mean(axis=0))
    feats += list(X.std(axis=0))
    feats += list(X.max(axis=0))
    feats += list(X.min(axis=0))
    return np.array(feats)

## function to plot a saved CSV
def plot_recording(file_path):
    df = pd.read_csv(file_path, delimiter=delimiter)
    df['t_s'] = df['time'] / 1000.0

    plt.figure()
    plt.plot(df['t_s'], df['accX'], label='Acc X')
    plt.plot(df['t_s'], df['accY'], label='Acc Y')
    plt.plot(df['t_s'], df['accZ'], label='Acc Z')
    plt.plot(df['t_s'], df['gyroX'], label='Gyro X')
    plt.plot(df['t_s'], df['gyroY'], label='Gyro Y')
    plt.plot(df['t_s'], df['gyroZ'], label='Gyro Z')
    plt.title(f"Acceleration – {path.basename(file_path)}")
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.tight_layout()
    plt.show()


def batch_plot_spells():
    spells = {'Stein': [], 'Schere': [], 'Papier': []}
    for f in glob.glob(path.join(recording_folder, '*.csv')):
        name = os.path.basename(f).lower()
        for key in spells:
            if key.lower() in name:
                spells[key].append(f)
    for key, files in spells.items():
        if not files:
            continue
        # Overlay Acceleration
        plt.figure()
        for file in files:
            df = pd.read_csv(file, delimiter=delimiter)
            df['t_s'] = df['time'] / 1000.0
            plt.plot(df['t_s'], df['accX'], alpha=0.6)
        plt.title(f"Acceleration Overlay – {key}")
        plt.xlabel('Time (s)')
        plt.ylabel('Acc X')
        plt.tight_layout()
        plt.show()
        # Overlay Gyroscope
        plt.figure()
        for file in files:
            df = pd.read_csv(file, delimiter=delimiter)
            df['t_s'] = df['time'] / 1000.0
            plt.plot(df['t_s'], df['gyroX'], alpha=0.6)
        plt.title(f"Gyroscope Overlay – {key}")
        plt.xlabel('Time (s)')
        plt.ylabel('Gyro X')
        plt.tight_layout()
        plt.show()

## GUI setup
root = tk.Tk()
root.wm_title("Magic Wand | Recorder")
root.minsize(350, 220)

# Status
frame_status = tk.Frame(root)
label_status = tk.Label(frame_status, text="Status:", width=15)
label_status.pack(side=tk.LEFT, padx=10)
label_status_value = tk.Label(frame_status, text="Disconnected")
label_status_value.pack(side=tk.LEFT, padx=10)
frame_status.pack(pady=10)

# Connect button
frame_connect = tk.Frame(root)
button_connect = tk.Button(frame_connect, text="Connect", command=connect)
button_connect.pack()
ttk.Separator(frame_connect, orient='horizontal').pack(fill='x', pady=10)
frame_connect.pack()

# Wizard name
frame_wiz = tk.Frame(root)
label_wiz = tk.Label(frame_wiz, text="Name of Wizard:", width=15)
label_wiz.pack(side=tk.LEFT, padx=10)
entry_wizard = tk.Entry(frame_wiz)
entry_wizard.insert(0, "Wizard #1")
entry_wizard.pack(side=tk.LEFT, padx=10)
frame_wiz.pack(pady=5)

# Spell name
frame_spell = tk.Frame(root)
label_spell = tk.Label(frame_spell, text="Name of Spell:", width=15)
label_spell.pack(side=tk.LEFT, padx=10)
entry_spell = tk.Entry(frame_spell)
entry_spell.insert(0, "Expelliarmus")
entry_spell.pack(side=tk.LEFT, padx=10)
frame_spell.pack(pady=5)

# Record button
frame_rec = tk.Frame(root)
button_record = tk.Button(frame_rec, text="Record Spell", command=toggle_recording)
button_record.pack()
frame_rec.pack(pady=10)

root.mainloop()
