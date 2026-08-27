from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from flask_bcrypt import Bcrypt
import secrets

# Initialize extensions (to be used in app.py)
db = SQLAlchemy()
bcrypt = Bcrypt()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    
    # --- ROLE-BASED ACCESS CONTROL (NEW) ---
    # Can be 'normal' or 'expert'
    role = db.Column(db.String(20), default='normal', nullable=False)
    
    # --- PREMIUM & MONETIZATION FIELDS ---
    # tracks how many free queries the user has made
    attempts = db.Column(db.Integer, default=0)
    
    # boolean flag for premium status
    is_paid = db.Column(db.Boolean, default=False)
    
    # unique API key for external integrations
    api_key = db.Column(db.String(100), unique=True, nullable=True)

    def generate_api_key(self):
        new_key = f"MED-{secrets.token_urlsafe(24).upper()}"
        self.api_key = new_key
        return new_key

    def __repr__(self):
        return f'<User {self.username} - Role: {self.role} - Paid: {self.is_paid}>'