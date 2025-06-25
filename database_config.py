from dotenv import load_dotenv
import os

class DatabaseConfig:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        load_dotenv()
        self.config = {
            'host': os.environ.get('HOST'),
            'database': os.environ.get('DATABASE'),
            'user': os.environ.get('USER'),
            'password': os.environ.get('PASSWORD')
        }
        self._initialized = True
    
    def get_config(self):
        return self.config.copy() 