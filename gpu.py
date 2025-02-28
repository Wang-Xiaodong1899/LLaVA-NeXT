import signal
import time
from datetime import datetime

def print_time(signum, frame):
    print(f"Current time: {datetime.now()}")
    signal.alarm(10)

signal.signal(signal.SIGALRM, print_time)

signal.alarm(10)

print("Process is running. It will print the time every 5 seconds. Press Ctrl+C to exit.")
while True:
    signal.pause()