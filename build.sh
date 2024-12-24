#!/usr/bin/env bash
set -o errexit

# Create directories
mkdir -p models/saved_models
mkdir -p data/raw
mkdir -p data/processed

# Create sample dataset if not exists
if [ ! -f data/raw/AdvertisingBudgetandSales.csv ]; then
    echo "TV,Radio,Newspaper,Sales" > data/raw/AdvertisingBudgetandSales.csv
    echo "230.1,37.8,69.2,22.1" >> data/raw/AdvertisingBudgetandSales.csv
    echo "44.5,39.3,45.1,10.4" >> data/raw/AdvertisingBudgetandSales.csv
    echo "17.2,45.9,69.3,12.0" >> data/raw/AdvertisingBudgetandSales.csv
fi

# Install dependencies
pip install -r requirements.txt

# Train model
python3 train_model.py