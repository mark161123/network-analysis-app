import os
import io
import base64
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import certifi
import json

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from bson.objectid import ObjectId
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# --- App Initialization ---
app = Flask(__name__)

# --- Configuration ---
app.secret_key = os.urandom(24) 
app.config["MONGO_URI"] = "mongodb+srv://markfernandes710:markfernandes710@mark.jq8xavb.mongodb.net/mark?retryWrites=true&w=majority&appName=mark"

# --- Database & Serializer Setup ---
ca = certifi.where()
mongo = PyMongo(app, tlsCAFile=ca)
s = URLSafeTimedSerializer(app.secret_key)

# --- Gemini API Route (mocked) ---
@app.route('/generate_report', methods=['POST'])
def generate_report():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        data = request.get_json()
        accuracy = data.get('accuracy')
        importances = data.get('importances')

        if not accuracy or not importances:
            return jsonify({'error': 'Missing data for report generation.'}), 400

        mock_ai_response = f"""
        <p><b>Overall Performance:</b> The predictive model demonstrated a strong accuracy of <strong>{accuracy}</strong>, indicating it is highly effective at identifying patterns in the network traffic data.</p>
        <p><b>Key Findings:</b> The most influential factors in predictions are <strong>'{importances[0][0]}'</strong> and <strong>'{importances[1][0]}'</strong>. This suggests that monitoring failed login attempts and access to critical files are crucial for identifying potential threats.</p>
        <p><b>Recommendations:</b> It is recommended to closely investigate the origins of repeated failed logins and to review access permissions for the most frequently accessed sensitive files to enhance security.</p>
        """
        return jsonify({'report': mock_ai_response})
    except Exception as e:
        print(f"Error generating report: {e}")
        return jsonify({'error': 'Failed to generate AI report.'}), 500

# --- User Authentication Routes ---
@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user and check_password_hash(existing_user['password'], request.form['password']):
            session['logged_in'] = True
            session['username'] = request.form['username']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user:
            flash('That username already exists. Please choose another.', 'warning')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        hashed_answer = generate_password_hash(request.form['security_answer'].lower().strip(), method='pbkdf2:sha256')

        users.insert_one({
            'username': request.form['username'],
            'password': hashed_password,
            'security_question': request.form['security_question'],
            'security_answer': hashed_answer
        })

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        users = mongo.db.users
        username = request.form['username']
        user = users.find_one({'username': username})

        if not user:
            flash('Username not found.', 'danger')
            return redirect(url_for('forgot_password'))

        if 'security_answer' in request.form:
            submitted_answer = request.form['security_answer'].lower().strip()
            if check_password_hash(user['security_answer'], submitted_answer):
                token = s.dumps(user['username'], salt='password-reset-salt')
                return redirect(url_for('reset_password', token=token))
            else:
                flash('Incorrect answer. Please try again.', 'danger')
                question_map = {
                    "first_pet": "What was your first pet's name?",
                    "mother_maiden_name": "What is your mother's maiden name?",
                    "birth_city": "In what city were you born?"
                }
                question_text = question_map.get(user.get('security_question'))
                return render_template('forgot_password.html', username=username, question=question_text)
        else:
            question_map = {
                "first_pet": "What was your first pet's name?",
                "mother_maiden_name": "What is your mother's maiden name?",
                "birth_city": "In what city were you born?"
            }
            question_text = question_map.get(user.get('security_question'))
            if not question_text:
                flash('No security question found for this account.', 'danger')
                return redirect(url_for('forgot_password'))
            return render_template('forgot_password.html', username=username, question=question_text)
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        username = s.loads(token, salt='password-reset-salt', max_age=3600)
    except (SignatureExpired, BadTimeSignature):
        flash('The password reset link is invalid or has expired.', 'danger')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password = request.form['password']
        hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
        mongo.db.users.update_one({'username': username}, {'$set': {'password': hashed_password}})
        flash('Your password has been updated successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/about')
def about():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('about.html')

# --- File Upload & ML Analysis ---
@app.route('/upload', methods=['POST'])
def upload_file():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        data = pd.read_csv(file)
        features = ['num_failed_logins', 'hot', 'num_access_files', 'attack_type']
        target = 'label'

        if not all(col in data.columns for col in features + [target]):
            return jsonify({'error': 'Missing required columns'}), 400

        X = data[features]
        y = data[target]

        column_sums = X.sum().to_dict()
        bar_plot = io.BytesIO()
        plt.figure(figsize=(8, 5))
        pd.Series(column_sums).plot(kind='bar', color=['blue', 'green', 'orange', 'red'])
        plt.title('Sum of Features'); plt.ylabel('Sum'); plt.xlabel('Feature')
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(bar_plot, format='png'); plt.close(); bar_plot.seek(0)
        bar_plot_base64 = base64.b64encode(bar_plot.getvalue()).decode()

        heatmap = io.BytesIO()
        plt.figure(figsize=(10, 6))
        sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Heatmap'); plt.tight_layout()
        plt.savefig(heatmap, format='png'); plt.close(); heatmap.seek(0)
        heatmap_base64 = base64.b64encode(heatmap.getvalue()).decode()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        importance_plot = io.BytesIO()
        importances.plot(kind='barh', color='teal')
        plt.title('Feature Importance'); plt.xlabel('Importance'); plt.tight_layout()
        plt.savefig(importance_plot, format='png'); plt.close(); importance_plot.seek(0)
        importance_plot_base64 = base64.b64encode(importance_plot.getvalue()).decode()

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        os.makedirs('model', exist_ok=True)
        joblib.dump(model, 'model/random_forest_model.pkl')

        importance_data = [[name, round(value, 4)] for name, value in importances.items()]

        return jsonify({
            'accuracy': f'{accuracy * 100:.2f}%',
            'bar_plot': bar_plot_base64,
            'heatmap': heatmap_base64,
            'importance_plot': importance_plot_base64,
            'data_preview': data.head().to_html(classes='table table-bordered', index=False),
            'column_sums': column_sums,
            'importances': importance_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Run App (for local development only) ---
if __name__ == '__main__':
    app.run()
