from flask import Flask, render_template, request
import pickle
app = Flask(__name__)


# Sctandardising numerical features
pickle_in = open('standardization_scaler.pkl', 'rb')
scaler = pickle.load(pickle_in)

# loading in the model to predict on the data
with open('model_rf.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html', prediction = "Fill the form to predict")

@app.route('/predict', methods=["POST"])
def predict():

    Genders = {"Female":0, "Male":1, "Unknown":2}
    yes_no = {"no":0, "yes": 1}
    gender = request.form['gender']

    age = request.form['age']
    no_of_days_subscribed = request.form['no_of_days_subscribed']
    multi_screen = request.form['multi_screen']
    mail_subscribed = request.form["mail_subscribed"]
    weekly_mins_watched = request.form['weekly_mins_watched']
    minimum_daily_mins = request.form['minimum_daily_mins']
    maximum_daily_mins = request.form['maximum_daily_mins']
    weekly_max_night_mins = request.form['weekly_max_night_mins']
    videos_watched =  request.form['videos_watched']
    maximum_days_inactive = request.form['maximum_days_inactive']
    customer_support_calls = request.form['customer_support_calls']


    # Standardizing numerical features using scalar fitted n our train data
    [[age, no_of_days_subscribed, weekly_mins_watched, minimum_daily_mins, maximum_daily_mins, 
    weekly_max_night_mins, videos_watched, maximum_days_inactive, customer_support_calls]] = scaler.transform([[age, no_of_days_subscribed, weekly_mins_watched, minimum_daily_mins, maximum_daily_mins, weekly_max_night_mins, videos_watched, maximum_days_inactive, customer_support_calls]])


    prediction = model.predict([[Genders[gender], age, no_of_days_subscribed, yes_no[multi_screen], yes_no[mail_subscribed], weekly_mins_watched, 
                                minimum_daily_mins, maximum_daily_mins, weekly_max_night_mins, videos_watched, 
                                maximum_days_inactive, customer_support_calls]])
    print(prediction)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
 