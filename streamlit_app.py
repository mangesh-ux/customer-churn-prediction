
import pickle
import streamlit as st
import numpy as np



def welcome():
	return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs
def prediction(gender, age, no_of_days_subscribed, multi_screen, mail_subscribed, weekly_mins_watched, minimum_daily_mins, maximum_daily_mins, weekly_max_night_mins, videos_watched, maximum_days_inactive, customer_support_calls):

	# Sctandardising numerical features
	pickle_in = open('standardization_scaler.pkl', 'rb')
	scaler = pickle.load(pickle_in)

	# loading in the model to predict on the data
	with open('model_rf.pkl', 'rb') as file:
		model = pickle.load(file)

	[[age, no_of_days_subscribed, weekly_mins_watched, minimum_daily_mins, maximum_daily_mins, 
	weekly_max_night_mins, videos_watched, maximum_days_inactive, customer_support_calls]] = scaler.transform([[age, no_of_days_subscribed, weekly_mins_watched, minimum_daily_mins, maximum_daily_mins, weekly_max_night_mins, videos_watched, maximum_days_inactive, customer_support_calls]])
	prediction = model.predict([[gender, age, no_of_days_subscribed,multi_screen, mail_subscribed, weekly_mins_watched, minimum_daily_mins, maximum_daily_mins, 
	weekly_max_night_mins, videos_watched, maximum_days_inactive, customer_support_calls]])
	print(prediction)
	return prediction
	

# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	st.title("Customer Churn Prediction")

	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">Customer churn predictor ML App </h1>
	</div>
	"""

	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)

	Genders = {"Female":0, "Male":1, "Unknown":2}
	yes_no = {"No":0, "yes": 1}

	# the data required to make the prediction
	gender = st.radio("Gender", ("Male", "Female", "Unknown"))
	age = st.text_input("Age", "Type Here")
	no_of_days_subscribed = st.text_input("No. of Days Subscribed", "Type Here")
	multi_screen = st.radio("Multi Screen", ("Yes", "No"))
	mail_subscribed = st.radio("Mail Subscribed?", ("Yes", "No"))
	weekly_mins_watched = st.text_input("Weekly Minutes watched", "Type Here")
	minimum_daily_mins = st.text_input("Minimum Daily minutes", "Type Here") 	
	maximum_daily_mins = st.text_input("Maximum Daily minutes", "Type Here") 	
	weekly_max_night_mins = st.text_input("Weekly Max Minutes night watched", "Type Here")	
	videos_watched = st.text_input("Videos watched", "Type number") 	
	maximum_days_inactive = st.text_input("No. of days of inactivity", "Type here")	
	customer_support_calls = st.text_input("Customer Support calls", "How many customer support calls")
	result =""
	# the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	if st.button("Predict"):
		result = prediction(Genders[gender], age, no_of_days_subscribed, 
							yes_no[multi_screen], yes_no[mail_subscribed], 
							weekly_mins_watched, minimum_daily_mins, 
							maximum_daily_mins, weekly_max_night_mins, 
							videos_watched, maximum_days_inactive, 
							customer_support_calls)
		if result[0] == 1:
			st.success('Customer will churn')
		else:
			st.success('Customer will not churn')
if __name__=='__main__':
	main()
