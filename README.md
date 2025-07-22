# Autism Spectrum Disorder Predictor

This project is a web-based application for predicting Autism Spectrum Disorder (ASD) using a machine learning model. The application is built using Flask and provides a user-friendly interface for inputting data and viewing predictions.

## Features

- Predicts the likelihood of ASD based on user inputs.
- Preprocesses input data using trained encoders.
- Utilizes a trained machine learning model for predictions.
- Provides a clean and responsive web interface.

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Preprocessing**: Pandas, Label Encoding
- **Model Training**: RandomizedSearchCV, SMOTE

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/asd_app.git
   cd asd_app
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following files are present in the project directory:
   - `encoders.pkl`: Trained label encoders.
   - `best_model.pkl`: Trained machine learning model.
   - `feature_columns.pkl`: Feature column order used during training.

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000`.

## Usage

1. Fill in the form on the web interface with the required details, such as age, gender, ethnicity, and other relevant information.
2. Click the "Predict" button to get the prediction.
3. The application will display whether the individual is "Likely autistic" or "Likely NOT autistic."

## Project Structure

```
asd_app/
├── model/
│   ├── predicting.ipynb
│   ├── Autism_prediction.ipynb
├── asd_app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   ├── static/
│       └── (optional CSS/JS files)
├── README.md
└── requirements.txt
```

## Dataset

The model was trained on a dataset containing features such as age, gender, ethnicity, and responses to a series of questions. The dataset was preprocessed to handle missing values, outliers, and class imbalances.
