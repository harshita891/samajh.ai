import datetime

class Logger:
    def __init__(self, log_file='logs.txt', enable_logging=True):
        self.log_file = log_file
        self.enable_logging = enable_logging
    
    def log_event(self, message):
        if self.enable_logging:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_message = f"[{timestamp}] {message}\n"
            with open(self.log_file, 'a') as f:
                f.write(log_message)
            print(log_message)  # Optionally print to console as well

    def log_fps(self, fps):
        if self.enable_logging:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_message = f"[{timestamp}] FPS: {fps:.2f}\n"
            with open(self.log_file, 'a') as f:
                f.write(log_message)
            print(log_message)
