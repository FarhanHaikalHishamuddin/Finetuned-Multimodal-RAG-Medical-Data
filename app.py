import os
import base64
import json
import secrets
import re
import random
import io  # Added for image handling
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from qwen_vl_utils import process_vision_info
import torch

# Import Modular Components from your project files
from models import db, User, bcrypt
from engine import model, processor, retrieve_context
from utils import parse_output, calculate_metrics, strict_clean_answer, normalize
from expert_logger import log_expert_correction

app = Flask(__name__)

# --- CONFIGURATION ---
app.config['SECRET_KEY'] = 'medical_rag_secure_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
FREE_LIMIT = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- 💡 Inference using: {device} ---")


# Initialize Extensions
db.init_app(app)
bcrypt.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' 

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

# --- THE CORE AI ENGINE ---
def perform_inference(user_text, image_b64):
    # 1. Retrieve the expert case from your new FAISS index
    has_img = True if image_b64 else False
    
    # PASS the has_image flag to the engine!
    retrieved_docs = retrieve_context(user_text, has_image=has_img, k=1)
    
    full_reference = retrieved_docs[0] if retrieved_docs else ""
    
    # 2. Split the Reference (FAISS) into Thinking and Answer
    if "</think>" in full_reference:
        ref_parts = full_reference.split("</think>")
        extracted_reasoning = ref_parts[0].replace("<think>", "").strip()
        extracted_answer = ref_parts[-1].strip()
    else:
        extracted_reasoning = "Detailed expert reasoning not found."
        extracted_answer = full_reference

    # 3. Handle Image Input (NEW: Convert Base64 to PIL)
    image_obj = None
    if image_b64:
        try:
            image_data = base64.b64decode(image_b64)
            image_obj = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            print(f"Image Decode Error: {e}")

    # 4. Prepare the prompt for the AI
    prompt = (
        f"You are a medical AI assistant. Use the following Reference Case to guide your analysis.\n"
        f"Reference Case Reasoning: {extracted_reasoning}\n"
        f"Reference Case Answer: {extracted_answer}\n\n"
        f"Current Question: {user_text}\n"
        f"Think step-by-step inside <think> tags before giving the final answer."
    )

    # 5. Generate the AI Response 
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj} if image_obj else None,
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # Filter out None if no image
    messages[0]["content"] = [c for c in messages[0]["content"] if c is not None]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += "<think>\n" 

    inputs = processor(
        text=[text],
        images=[image_obj] if image_obj else None,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    # 6. Split the AI Response into Thinking and Answer
    if "</think>" in generated_text:
        gen_parts = generated_text.split("</think>")
        generated_reasoning = gen_parts[0].replace("<think>", "").strip()
        generated_answer = gen_parts[-1].strip()
    else:
        generated_reasoning = "AI logic synthesized from visual features."
        generated_answer = generated_text

    # 7. Calculate Live Metrics
    metrics = calculate_metrics(generated_answer, extracted_answer)

    return {
        "generated_reasoning": generated_reasoning,
        "generated_answer": generated_answer,
        "extracted_reasoning": extracted_reasoning,
        "extracted_answer": extracted_answer,
        "metrics": metrics
    }

# (Web UI Routes, Auth Routes, etc.)

@app.route('/')
@login_required
def home():
    return render_template('index.html', user=current_user, limit=FREE_LIMIT)

@app.route('/query', methods=['POST'])
@login_required
def query_model():
    if not current_user.is_paid and current_user.attempts >= FREE_LIMIT:
        return jsonify({"error": "QUOTA_EXCEEDED"}), 402
    
    data = request.json
    try:
        result = perform_inference(data.get('text', ''), data.get('image', None))
        
        current_user.attempts += 1
        db.session.commit()
        
        result["attempts"] = current_user.attempts
        result["user_role"] = current_user.role 

        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# (Other routes: expert_feedback, signup, login, logout, upgrade, external_api...)

@app.route('/expert_feedback', methods=['POST'])
@login_required
def expert_feedback():
    if current_user.role != 'expert':
        return jsonify({"error": "Forbidden"}), 403
    
    data = request.json
    success = log_expert_correction(
        image_name=data.get('image_name', 'default.jpg'),
        question=data.get('question'),
        expert_answer=data.get('expert_answer')
    )
    
    if success:
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 500

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        role = request.form.get('role', 'normal')

        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('signup'))
        
        hashed_pw = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
        new_user = User(username=username, password=hashed_pw, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and bcrypt.check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid credentials.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/docs')
@login_required
def api_docs():
    # This will look for 'docs.html' inside your /templates folder
    return render_template('docs.html', user=current_user)

@app.route('/upgrade', methods=['POST'])
@login_required
def upgrade():
    current_user.is_paid = True
    current_user.api_key = f"MED-{secrets.token_urlsafe(24).upper()}"
    db.session.commit()
    return jsonify({"status": "success", "api_key": current_user.api_key})


@app.route('/api/v1/analyze', methods=['POST'])
def external_api():
    api_key = request.headers.get('X-API-KEY')
    user = User.query.filter_by(api_key=api_key).first()
    
    if not user or not user.is_paid:
        return jsonify({"error": "Unauthorized or Invalid API Key"}), 401
    
    data = request.json
    try:
        result = perform_inference(data.get('text'), data.get('image'))
        user.attempts += 1
        db.session.commit()
        return jsonify({
            "status": "success", 
            "version": "v1.0",
            "results": result,
            "api_usage": user.attempts
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)