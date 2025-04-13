from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load encoders, model, and feature column order
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

def preprocess_input(data, encoders, feature_columns):
    df = pd.DataFrame([data])
    for col in df.columns:
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col])
            except ValueError:
                df[col] = encoders[col].transform([encoders[col].classes_[0]])
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.reindex(columns=feature_columns).fillna(0)
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        form_data = request.form.to_dict()

        # Convert input values
        for i in range(1, 11):
            form_data[f"A{i}_Score"] = form_data.get(f"A{i}_Score", "").lower()

        processed_input = preprocess_input(form_data, encoders, feature_columns)
        result = model.predict(processed_input)[0]
        prediction = "Likely autistic" if result == 1 else "Likely NOT autistic"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
