import os

class Config:    
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = False
    
    MODEL_PATH = 'models/saved_models/trained_model.pkl'
    SCALER_PATH = 'models/saved_models/scaler.pkl'
        
    HIDDEN_LAYER_SIZE = 8
    LEARNING_RATE = 0.01
    EPOCHS = 1000
        
    RAW_DATA_PATH = 'data/raw/AdvertisingBudgetandSales.csv'
    PROCESSED_DATA_PATH = 'data/processed/scaled_data.csv'

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}