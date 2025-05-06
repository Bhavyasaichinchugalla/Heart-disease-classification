# Heart Disease Risk Prediction Web Application

## Project Description
This project is a web-based machine learning application designed to predict the risk of heart disease in individuals based on common clinical features. Users input 13 medical parameters, and the system provides a prediction (high or low risk) along with a confidence score and feature importance visualization.

The application is built using:
- **Flask** for backend web development
- **HTML/CSS/JavaScript** for the front-end user interface
- **scikit-learn** for machine learning modeling
- **matplotlib** for feature importance visualization
- **Gunicorn** for production deployment
- **Railway** for cloud hosting

The model used is a **RandomForestClassifier**, a supervised learning algorithm that supports both classification and regression. This model was trained on the Cleveland Heart Disease dataset. Key machine learning concepts applied include:
- Data preprocessing (handling missing values, encoding categorical variables)
- Model training and evaluation
- Model serialization using `joblib`
- Probability estimation using `predict_proba`
- Feature interpretation using feature importances

## Features
- Clean and responsive user interface
- Input validation and tooltips for guidance
- Confidence score with each prediction
- Feature importance chart for interpretability
- Hosted publicly with a production server

## Team Members and Banner IDs
- **Satya Sudha Pamulapati** — 001315945
- **Shreya Samir Kurani** — 001283918
- **Rajesh Adhi** — 001292324
- **Bhavyasai Chinchugalla** — 001321696
- **Hemavathi Karuppaiah** — 001358346

## Task Division
**Satya Sudha Pamulapati**
- Model training and testing
- Feature importance generation
- ML explanation logic

**Shreya Samir Kurani**
- Front-end HTML/CSS development
- UI enhancements (tooltips, modal popups)
- Spinner and layout styling

**Rajesh Adhi**
- Flask integration and form processing
- Handling form submissions and prediction routing
- Probability integration and session handling

**Bhavyasai Chinchugalla**
- Deployment setup (Railway, Gunicorn)
- `requirements.txt`, `Procfile`, and hosting configuration
- Debugging deployment issues

**Hemavathi Karuppaiah**
- Testing the app with various inputs
- SHAP fallback handling (feature importance)
- Documentation and README preparation

## How to Run Locally
1. Clone the repository
2. Install dependencies using `pip install -r requirements.txt`
3. Run the app with `python app.py`
4. Open `localhost:5000` in your browser

## Deployment
This application is deployed using [Railway](https://railway.app). Gunicorn is used as the production WSGI server, and a `Procfile` specifies the entry point.
## live website - [https://heart-disease-classification-production.up.railway.app/](https://heart-disease-classification-production.up.railway.app/)



## Dataset Source
[Cleveland Heart Disease Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)

## License
For educational use only.
