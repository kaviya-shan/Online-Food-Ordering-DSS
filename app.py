from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("food_dss_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form["Ease_of_Ordering"]),
                    float(request.form["Good_Taste"]),
                    float(request.form["Time_Saving"]),
                    float(request.form["Offers"]),
                    float(request.form["Age"])]
        prediction = model.predict([features])[0]
        return render_template("result.html", prediction_text=f"Prediction: {prediction}")
    except Exception as e:
        return render_template("result.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
