import serial
import csv
import time

# Define the serial port and baudrate (replace 'COMx' with the correct port for your ESP32)
serial_port = 'COM7'  # For Windows, it could be 'COM3', 'COM4', etc.
# On Linux or Mac, it could be something like '/dev/ttyUSB0' or '/dev/tty.SLAB_USBtoUART'
baudrate = 115200  # The baud rate that matches your ESP32 serial setup

# Output CSV file
filename = 'serial_data.csv'

# Initialize the serial connection
ser = serial.Serial(serial_port, baudrate, timeout=1)

# Open the CSV file to write the data
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write header to CSV
    writer.writerow(['Time (s)', 'Voltage (V)'])
    
    print("Recording serial data to CSV... Press Ctrl+C to stop.")
    
    try:
        while True:
            if ser.in_waiting > 0:
                # Read data from serial (one line at a time)
                line = ser.readline().decode('utf-8').strip()  # Decode and strip any extra whitespace
                
                # If the line is in the format "Time (s)    Voltage (V)", we proceed
                if line:
                    print(line)  # Print to console for feedback
                    
                    # Split the data based on the tab space or adjust according to output format
                    data = line.split('\t')
                    
                    # Write the data to the CSV file
                    writer.writerow(data)
                    
                    # Flush to ensure immediate writing
                    file.flush()
                    
            time.sleep(0.1)  # Small delay to avoid overloading the serial reading
    except KeyboardInterrupt:
        print("\nRecording stopped.")
        ser.close()
