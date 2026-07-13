from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'Users'
    
    UserID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Name = db.Column(db.String(100), nullable=False)
    Email = db.Column(db.String(120), unique=True, nullable=False)
    Password = db.Column(db.String(200), nullable=False)
    Role = db.Column(db.String(50), nullable=False, default='User')
    
    # Relationship: 1 User to Many Applicant_Details
    applicant_details = db.relationship('ApplicantDetail', backref='user', lazy=True)

class ApplicantDetail(db.Model):
    __tablename__ = 'Applicant_Details'
    
    ApplicantID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    UserID = db.Column(db.Integer, db.ForeignKey('Users.UserID'), nullable=False)
    
    IncomeType = db.Column(db.String(100), nullable=False)
    EducationType = db.Column(db.String(100), nullable=False)
    FamilyStatus = db.Column(db.String(100), nullable=False)
    HousingType = db.Column(db.String(100), nullable=False)
    EmploymentDays = db.Column(db.Integer, nullable=False)
    
    # Relationships
    credit_histories = db.relationship('CreditHistory', backref='applicant', lazy=True)
    prediction = db.relationship('ApprovalPrediction', backref='applicant', uselist=False, lazy=True)

class CreditHistory(db.Model):
    __tablename__ = 'Credit_History'
    
    HistoryID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ApplicantID = db.Column(db.Integer, db.ForeignKey('Applicant_Details.ApplicantID'), nullable=False)
    
    MonthsBalance = db.Column(db.Integer, nullable=False)
    PaymentStatus = db.Column(db.String(50), nullable=False)
    OverdueStatus = db.Column(db.String(50), nullable=False)

class MLModel(db.Model):
    __tablename__ = 'ML_Model'
    
    ModelID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ModelName = db.Column(db.String(100), nullable=False)
    AlgorithmType = db.Column(db.String(100), nullable=False)
    Accuracy = db.Column(db.Float, nullable=False)
    ModelFile = db.Column(db.String(255), nullable=False) # e.g. path to model.pkl
    
    # Relationship
    predictions = db.relationship('ApprovalPrediction', backref='model', lazy=True)

class ApprovalPrediction(db.Model):
    __tablename__ = 'Approval_Prediction'
    
    PredictionID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ApplicantID = db.Column(db.Integer, db.ForeignKey('Applicant_Details.ApplicantID'), nullable=False)
    ModelID = db.Column(db.Integer, db.ForeignKey('ML_Model.ModelID'), nullable=False)
    
    ApprovalResult = db.Column(db.String(50), nullable=False)
    RiskCategory = db.Column(db.String(50), nullable=False)
    PredictionDate = db.Column(db.DateTime, default=datetime.utcnow)
