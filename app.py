# Nick Marker, ncmarker@usc.edu
# ITP216 Spring 2024
# section 31885
# final Project 
# Description: a web app that uses password strength data and ML to predict the strength of a user's password,
# add it to the set of data, and displaya frequency chart of the current passwords

from flask import Flask, redirect, render_template, request, session, url_for, send_file
import pandas as pd
import csv

import graphs
import predict

app = Flask(__name__)

@app.route("/")
def home():
    """
    Render the home page that allows users to start interacting with the application.
    Returns the rendered 'home.html'.
    """
    return render_template("home.html")


@app.route("/strength-distribution", methods=["GET"])
def strength_distribution():
    """
    Generates and displays a histogram of password strength distributions.
    Calls a function from the graphs.py module to create the histogram image,
    which is then passed as base64 encoded data to the template.
    
    Returns:
        Rendered 'frequency.html' with histogram image source embedded.
    """
    hist_img_src = graphs.make_histogram()
    return render_template('frequency.html', hist_img_src = hist_img_src)


@app.route("/predict-strength", methods=["POST"])
def predict_strength():
    """
    Receives a password from a form submission and redirects to the prediction route
    to display the strength assessment of the given password.

    Returns:
        Redirect to the 'get_prediction' route with the password as a parameter.
    """
    password = request.form.get("password")
    return redirect(url_for('get_prediction', password=password))


@app.route("/prediction/<password>", methods=["GET"])
def get_prediction(password):
    """
    Displays the machine learning prediction results for a given password's strength.
    Uses the 'predict' module to obtain the prediction, and then generates a visualization
    of the result which is returned to the user.

    Args:
        password (str): The password for which strength is predicted.

    Returns:
        Rendered 'assess.html' with prediction results and an appropriate user message.
    """
    if not password:
            return render_template('home.html', message = 'No password provided.')
    messages = [
         'Your password is weak! Consider forming a stronger password.', 
         'Your password is medium. It will do, but is recommended to be stronger.', 
         'Your password is strong. Very nice job!'
    ]
    predicted_strength = predict.predict_strength(password)
    prediction_img_src = graphs.plot_prediction_strength(predicted_strength)
    return render_template('assess.html', prediction_img_src = prediction_img_src, user_message = messages[predicted_strength])


@app.route("/submit-password", methods=["POST"])
def submit_password():
    """
    Submits a user-provided password along with its evaluated strength to a CSV file.
    Validates the input and responds back to the user via the home page.

    Returns:
        Rendered 'home.html' with a message indicating the success or failure of the submission.
    """
    password = request.form.get("password")
    strength = int(request.form.get("strength"))
    if not password or not strength:
            return render_template('home.html', message = 'Insufficient data.')
    print(password, strength)
    append_to_csv_with_index(password, strength, './data/passwords.csv')
    return render_template('home.html', message = 'Password submitted.')


def append_to_csv_with_index(password, predicted_strength, file_path='./data/passwords.csv'):
    """
    Appends a new row with an auto-incremented index to a CSV file containing passwords
    and their strengths.

    Args:
        password (str): The password to record.
        predicted_strength (int): The evaluated strength of the password.
        file_path (str): Path to the CSV file where data is stored.

    Raises:
        FileNotFoundError: If the CSV file is not found and needs to be created.
    """
    try:
        existing_data = pd.read_csv(file_path)
       
        # Find the last index if the file is not empty
        if not existing_data.empty:
            last_index = existing_data.index[-1]
        else:
            last_index = -1

    except FileNotFoundError:
        # If the file does not exist start from index 0
        existing_data = pd.DataFrame(columns=['index', 'password', 'strength'])
        last_index = -1  

    new_data = pd.DataFrame({
        'index': [last_index + 1],
        'password': [password],
        'strength': [predicted_strength]
    })

    # Append the new DataFrame to the CSV file
    new_data.to_csv(file_path, mode='a', index=False, header=existing_data.empty, quoting=csv.QUOTE_ALL)
    print(f"Added new data to {file_path} with index {last_index + 1}: {password}, {predicted_strength}")
