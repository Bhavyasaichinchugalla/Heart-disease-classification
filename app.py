from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong random key
model = joblib.load('model/heart_disease_model.pkl')
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['trestbps']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['restecg']),
                float(request.form['thalach']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]

            # Predict
            result = model.predict([features])[0]
            proba = model.predict_proba([features])[0][1]
            percentage = round(proba * 100, 1)
            risk = "High Risk" if result == 1 else "Low Risk"
            session['prediction'] = f"{risk} ({percentage}%)"

            # Feature Importance Plot (instead of SHAP)
            importances = model.feature_importances_
            plt.figure(figsize=(8, 6))
            plt.barh(feature_names, importances)
            plt.xlabel("Feature Importance")
            plt.title("Model Explanation (RandomForest)")
            plt.tight_layout()
            plt.savefig("static/shap_plot.png")
            plt.close()

        except Exception as e:
            session['prediction'] = f"Error: {e}"

        return redirect(url_for('index'))

    prediction = session.pop('prediction', None)
    return render_template("index.html", prediction=prediction)




# ✅ Add this so the app actually runs
if __name__ == '__main__':
    app.run(debug=True)
