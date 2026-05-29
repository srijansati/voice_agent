import logging

def get_logger(name):
    # Create a logger specific to the module name
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File Handler
        file_handler = logging.FileHandler("app.log")
        file_handler.setFormatter(formatter)
        
        # Stream Handler (Console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
    return logger