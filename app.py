import os
import numpy as np
from flask import Flask, render_template, request, url_for, redirect, flash
import pandas as pd
import pickle
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Define the path to your model file
model_file_path = 'knn_model.pickle'

# Check if the model file exists
if os.path.exists(model_file_path):
    # Load the trained model
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
else:
    logging.error(f"File '{model_file_path}' does not exist.")

# Load the DataFrame
df = pd.read_csv("new_customers.csv")

# Define gender options
gender_options = {
    1: 'Male',
    0: 'Female'
}
app = Flask(__name__, template_folder='templates')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Define User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
        InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
        InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)


@app.route('/index', methods=['GET', 'POST'])
@login_required
def dashboard():
    # Prepare the data for the dropdowns in the form
    age = sorted(df['age'].unique())
    salary = sorted(df['salary'].unique())
    price = sorted(df['price'].unique())

    # Render the dashboard.html template with the data and gender_options
    return render_template('dashboard.html', age=age, salary=salary, price=price, gender_options=gender_options)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    age = int(request.form.get('age'))
    salary = int(request.form.get('salary'))
    price = int(request.form.get('price'))
    male = int(request.form.get('male'))

    # Log the received form data
    logging.debug(f"Received form data: age={age}, salary={salary}, price={price}, male={male}")

    # Map the selected gender to the model's expected format (1 for Male, 0 for Female)
    gender = male

    # Log the data being passed to the model
    logging.debug(f"Data passed to the model: {[[age, salary, price, gender]]}")

    # Make prediction if model exists
    if 'model' in globals():
        prediction = model.predict([[age, salary, price, gender]])
        logging.debug(f"Model prediction: {prediction}")
        result = "Customer will purchase" if prediction[0] == 1 else "Customer won't purchase"
        return result
    else:
        logging.error("Model not loaded.")
        return "Model not loaded."


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            print("Flashed message: Username already exists")
        else:
            hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
            new_user = User(username=form.username.data, password=hashed_password)

            try:
                db.session.add(new_user)
                db.session.commit()
                flash('User has been successfully registered!', 'success')
                print("Flashed message: User successfully registered")
                return redirect(url_for('login'))
            except Exception as e:
                db.session.rollback()
                print(f"Error occurred: {e}")
                flash('There was an issue adding the user to the database. Please try again later.', 'danger')
                print("Flashed message: Issue adding user to database")

    return render_template('register.html', form=form)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=False)
