from flask import Flask, render_template, request
import elm

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods = ['GET', 'POST'])
def predict():
    if request.method == "POST":
        name = request.form.get("name")
        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        bloodPressure = float(request.form["bloodPressure"])
        skinThickness = float(request.form["skinThickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        diabetesPedigreeFunction = float(request.form["diabetesPedigreeFunction"])
        age = float(request.form["age"])

        data = elm.reshape(pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age)
        print(data)
        result, predicted = elm.elm_predict(data)
        print(predicted[0][0])
        result = "Positif" if result == 1 else "Negative"

    # data = [name, result]
    # return render_template("predict.html", d = data)

    if result == "Positif":
        data = [name, predicted]
        return render_template("result-pos.html", d = data)
    else:
        data = [name, predicted]
        return render_template("result-neg.html", d = data)

if __name__ == "__main__":
    app.run(debug=True)