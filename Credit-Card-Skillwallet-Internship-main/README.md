# SkillWallet Internship: Credit Card Approval Prediction

## Overview
This repository contains the complete machine learning pipeline for the **SkillWallet Internship - Credit Card Approval Prediction** project. The objective is to automate the credit card approval decision process using machine learning. By analyzing financial and demographic applicant data, the system evaluates the risk and likelihood of approval, significantly reducing manual review time.

## Project Structure
The project is organized into modular Python scripts that follow the internship Epic structure:

- **`data_collection.py`**: Generates and structures the `application_record.csv` and `credit_record.csv` datasets.
- **`epic2_eda.py`**: Performs Univariate, Multivariate, and Descriptive Exploratory Data Analysis, outputting graphs to the `plots/` directory.
- **`data_preprocessing.py`**: Cleans the data, drops duplicates, encodes categorical values, handles missing values, and merges datasets to output `processed_dataset.csv`.
- **`epic4_model_building.py`**: Trains, tests, and evaluates Logistic Regression, Random Forest, and Decision Tree classification algorithms.
- **`save_model.py`**: Persists the highest performing model (Random Forest) into `model.pkl` for deployment.
- **`app.py`**: A Flask backend that routes traffic and processes incoming applicant data through the trained model.
- **`models.py`**: The SQLAlchemy database schema mapping relationships between Users, Applicant Details, Credit History, ML Models, and Approval Predictions.

## Tech Stack
- **Data Science**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Visualization**: Matplotlib, Seaborn
- **Backend/Deployment**: Flask, Python Pickle
- **Database**: SQLite (via Flask-SQLAlchemy)

## Running the Application Locally
1. Activate your virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python app.py
   ```
3. Open a browser and navigate to `http://127.0.0.1:5000` to interact with the SkillWallet Credit Decision Engine UI.

## UI / Aesthetics
The frontend (`templates/index.html` & `templates/result.html`) features a modern, clean, minimalist Fintech design using a highly legible interface and glassmorphism styling elements to ensure a premium user experience.
