import time, logging
from core.config import PHONE_NUMBERS

# === Simulated GSM module ===
class GSM:
    def __init__(self, port=None, baudrate=None):
        self.port = port
        self.baudrate = baudrate
        print(f"[SIM] GSM module initialized on port {port} at {baudrate} baud")

    def send_sms(self, number, message):
        log_msg = f"[SIM] Sending SMS to {number}: {message}"
        print(log_msg)
        logging.info(log_msg)
        time.sleep(0.5)  # simulate delay

# === Initialize simulated GSM ===
gsm = GSM(port="COM3", baudrate=9600)