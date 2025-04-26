import logging
import time

class Logger:
    def __init__(self, log_file='detection_log.log', enable_logging=True):
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = logging.getLogger('DetectionLogger')
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def log_fps(self, fps):
        if self.enable_logging:
            self.logger.info(f"FPS: {fps:.2f}")

    def log_event(self, message):
        if self.enable_logging:
            self.logger.info(message)

    def log_time(self, start_time):
        if self.enable_logging:
            elapsed = time.time() - start_time
            self.logger.info(f"Elapsed Time: {elapsed:.4f} seconds")
