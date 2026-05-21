# CropSense AI — Climate-Resilient Crop Prediction System

**Predicting Climate-Resilient Crops and Market Opportunities for Hilly and Polar Regions using an Ensemble Machine Learning Approach**

---

## Overview
CropSense AI is a full-stack web application that uses an ensemble of ML models (Random Forest, Gradient Boosting, XGBoost, SVM) to predict the best climate-resilient crop for a given region, with integrated market analysis.

## Crops Predicted
Rice | Ragi | Jowar | Corn | Wheat

## Features
- Ensemble VotingClassifier (RF + GB + XGBoost + SVM)
- 97%+ prediction accuracy
- 9-feature input (temperature, rainfall, humidity, pH, NPK, altitude, region)
- Market analysis: price, demand, profit, export potential
- Interactive dashboard with Chart.js visualizations
- REST API endpoint

## Project Structure
```
cropml/
├── app.py               # Flask application
├── requirements.txt
├── README.md
├── templates/           # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   └── dashboard.html
├── static/css/style.css
├── src/
│   ├── generate_data.py # Dataset simulation
│   ├── preprocess.py    # StandardScaler pipeline
│   ├── train.py         # Model training
│   └── market.py        # Market analysis module
├── models/              # Saved pickle models
├── data/                # CSV datasets
└── results/             # Generated charts
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (first time only)
cd src && python train.py && cd ..

# 3. Run Flask app
python app.py
```

Open browser at: http://localhost:5000

## API Usage
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Temperature":20,"Rainfall":800,"Humidity":60,"Soil_pH":6.5,
       "Nitrogen":60,"Phosphorus":40,"Potassium":40,"Altitude":1000,"Region":1}'
```

## Model Accuracies
| Model | Accuracy |
|-------|----------|
| Random Forest | 97.50% |
| Gradient Boosting | 97.75% |
| XGBoost | 96.75% |
| SVM | 95.50% |
| Ensemble (Voting) | 97.50% |
