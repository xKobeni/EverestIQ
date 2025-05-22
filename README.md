# Himalayan Expedition Prediction System

A comprehensive web application for analyzing and predicting various aspects of Himalayan mountain expeditions, including peak difficulty, summit time, and fatality risk assessment.

## 🌟 Features

- **Peak Difficulty Prediction**: Classifies mountain peaks into Easy, Moderate, and Hard categories based on historical expedition data
- **Summit Time Prediction**: Estimates the time required to reach the summit
- **Fatality Risk Assessment**: Evaluates the risk factors for expeditions
- **Interactive Dashboard**: Visualizes expedition statistics and trends
- **Data Analysis**: Comprehensive analysis of historical expedition data

## 📋 Prerequisites

- Python 3.8+
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib

## 🚀 Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Himalayan_Prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Himalayan_Prediction/
├── src/
│   ├── app.py                 # Main Flask application
│   ├── preprocess.py          # Data preprocessing utilities
│   ├── routes/                # Route handlers
│   │   ├── prediction_routes.py
│   │   ├── dashboard_routes.py
│   │   ├── fatality_risk_routes.py
│   │   ├── summit_time_routes.py
│   │   └── peak_difficulty_routes.py
│   ├── templates/             # HTML templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── dashboard.html
│   │   ├── peak_difficulty.html
│   │   ├── summit_time.html
│   │   └── fatality_risk.html
│   └── Machine_learning/      # ML models
│       ├── model_2/
│       │   ├── fatality_risk_model.py
│       │   └── train_model.py
│       └── model_4/
│           └── peak_difficulty_model.py
└── datasets/                  # Data files
    ├── exped.csv
    ├── peaks.csv
    ├── members.csv
    ├── cleaned_exped.csv
    └── himalayan_data_dictionary.csv
```

## 🏃‍♂️ Running the Application

1. Preprocess the data:
```bash
python src/preprocess.py
```

2. Start the Flask application:
```bash
python src/app.py
```

3. Access the application at `http://localhost:5000`

## 🎯 Training the Models

### Expedition Success Model (Model 1)
1. Navigate to the model directory:
```bash
cd src/Machine_learning/model_1
```

2. Run the training script:
```bash
python train_model.py
```

### Fatality Risk Model (Model 2)
1. Navigate to the model directory:
```bash
cd src/Machine_learning/model_2
```

2. Run the training script:
```bash
python train_model.py
```

### Summit Time Model (Model 3)
1. Navigate to the model directory:
```bash
cd src/Machine_learning/model_3
```

2. Run the training script:
```bash
python train_model.py
```

### Peak Difficulty Model (Model 4)
1. Navigate to the model directory:
```bash
cd src/Machine_learning/model_4
```

2. Run the training script:
```bash
python peak_difficulty_model.py
```

Note: Make sure you have preprocessed the data before training the models. The training process will:
- Load and prepare the data
- Train the models
- Save the trained models to the appropriate directories
- Generate model performance metrics


## 📊 Data Sources

The application uses several datasets:
- `exped.csv`: Raw expedition data
- `peaks.csv`: Mountain peak information
- `members.csv`: Expedition member data
- `cleaned_exped.csv`: Preprocessed expedition data
- `himalayan_data_dictionary.csv`: Data field descriptions

## 🔧 API Endpoints

- `/peak-difficulty`: Peak difficulty prediction form
- `/peak-difficulty/predict`: Predict difficulty for a peak
- `/peak-difficulty/peaks`: Get list of peaks with difficulty ratings
- `/summit-time`: Summit time prediction
- `/fatality-risk`: Fatality risk assessment
- `/dashboard`: Data visualization dashboard

## 📝 License

[Your License Here]

## 👥 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

[Your Contact Information] 