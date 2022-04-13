from flask import Flask, request, render_template
import pickle
import numpy as np
import joblib

app = Flask(__name__, template_folder='templates')
model = joblib.load('Diabetes_Logistic_Regression_75_75.pkl')
#model = SVC(probability=True)


@app.route('/')
def home():
    return render_template("diabetes.html")


@app.route('/predict', methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final = [np.array(int_features)]

    #prediction = model.predict(final)
    prediction = model.predict_proba(final)
    print(prediction)

    confidence = int(abs((prediction[0][1]*100)-(prediction[0][0]*100)))
    #confidence = abs((prediction[0][1]*100)-(prediction[0][0]*100))
    print(confidence)

    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if output >= str(0.66):
        return render_template('result.html',
                               pred='You are at high risk of encountering Diabetes.\n'
                                    'Probability of Diabetes occurring is {}. \n The confidence of this prediction is : {}%.'.format(output, confidence))
    if output < str(0.33):
        return render_template('result.html',
                               pred='You are at low risk of encountering Diabetes.\n'
                                    'Probability of Diabetes occurring is {}. \n The confidence of this prediction is : {}%.'.format(output, confidence))
    if str(0.33) <= output < str(0.66):
        return render_template('result.html',
                               pred='You are at moderate risk of encountering Diabetes.\n '
                                    'Probability of Diabetes occurring is {}. \n The confidence of this prediction is : {}%.'.format(output, confidence))


if __name__ == '__main__':
    app.run(debug=True)
