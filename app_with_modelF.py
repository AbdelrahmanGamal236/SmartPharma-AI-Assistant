import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
from functools import wraps
import requests
# from scraper import search_drugs
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from fuzzywuzzy import process
from medical_rag_model import search_medical_database, get_query_type
from gemini_api import check_with_gemini_api
from GoogleMapspy import GoogleMaps
import geopy.distance
from geopy.geocoders import Nominatim
from scraper import search_drugs_comprehensive, get_drug_complete_info




app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pharmacy.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)
####################### food interaction #####################
###################################New medical condition start :
@app.route('/medical_condition')
# @login_required
def medical_condition():
    return render_template('medical_condition.html')

@app.route('/search_medical_info', methods=['POST'])
def search_medical_info():
    query = request.form.get('query', '')
    search_type = request.form.get('search_type', 'symptoms')  # Default to symptoms if not specified

    # Optional: print debug info
    print(f"Received query: '{query}', Search type: '{search_type}'")

    if not query or query.strip() == "":
        return jsonify({
            "final_answer": "Please enter your symptoms or disease to get medical information."
        })

    # Search the database for relevant information
    db_results = search_medical_database(query, search_type)
    print(f"Database results: Found {len(db_results)} matches")

    final_answer = ""

    if db_results:
        db_result = db_results[0]
        if search_type == "symptoms":
            db_info = (f"Symptoms: {db_result['Symptoms/Question']}\n"
                       f"Predicted Disease: {db_result['Disease Prediction']}\n"
                       f"Recommended Medicines: {db_result['Recommended Medicines']}\n"
                       f"Medical Advice: {db_result['Advice']}")
        else:  # disease
            db_info = (f"Disease: {db_result['Disease Prediction']}\n"
                       f"Typical Symptoms: {db_result['Symptoms/Question']}\n"
                       f"Recommended Medicines: {db_result['Recommended Medicines']}\n"
                       f"Medical Advice: {db_result['Advice']}")

        # Use Gemini to enhance and verify the information
        if search_type == "symptoms":
            prompt = (f"A user has described these symptoms: '{query}'. "
                      f"Our medical database suggests the following information:\n\n{db_info}\n\n"
                      f"Based on this information, provide a comprehensive but concise response about the possible "
                      f"condition, recommended treatments, and advice. Format it in a friendly, informative way for "
                      f"a patient. Add any important medical information that might be missing, but remain consistent "
                      f"with the core diagnosis.")
        else:  # disease
            prompt = (f"A user is asking about this medical condition: '{query}'. "
                      f"Our medical database suggests the following information:\n\n{db_info}\n\n"
                      f"Based on this information, provide a comprehensive but concise response about this condition, "
                      f"its symptoms, recommended treatments, and advice. Format it in a friendly, informative way for "
                      f"a patient. Add any important medical information that might be missing.")

        gemini_response = check_with_gemini_api(prompt)
        if gemini_response and len(gemini_response) > 50:
            final_answer = gemini_response
        else:
            if search_type == "symptoms":
                final_answer = (f"Based on your symptoms, you may have: {db_result['Disease Prediction']}\n\n"
                               f"Recommended Medicines: {db_result['Recommended Medicines']}\n\n"
                               f"Medical Advice: {db_result['Advice']}")
            else:  # disease
                final_answer = (f"Information about {db_result['Disease Prediction']}:\n\n"
                               f"Typical Symptoms: {db_result['Symptoms/Question']}\n\n"
                               f"Recommended Medicines: {db_result['Recommended Medicines']}\n\n"
                               f"Medical Advice: {db_result['Advice']}")
    else:
        # No info found in database - use Gemini to generate an answer
        if search_type == "symptoms":
            prompt = (f"A user has described these symptoms: '{query}'. "
                      f"They're looking for information about possible medical conditions, treatments, and advice. "
                      f"Provide a comprehensive but concise response about the possible condition(s), recommended "
                      f"treatments, and advice. Format it in a friendly, informative way for a patient. "
                      f"Include a reminder that this is AI-generated advice and they should consult a healthcare professional.")
        else:  # disease
            prompt = (f"A user is asking about this medical condition: '{query}'. "
                      f"They're looking for information about its symptoms, treatments, and advice. "
                      f"Provide a comprehensive but concise response about this condition, its symptoms, "
                      f"recommended treatments, and advice. Format it in a friendly, informative way for a patient. "
                      f"Include a reminder that this is AI-generated advice and they should consult a healthcare professional.")

        gemini_response = check_with_gemini_api(prompt)
        if gemini_response and len(gemini_response) > 50:
            final_answer = gemini_response
        else:
            final_answer = (f"We couldn't find specific information about your query: '{query}'. "
                            f"Please try rephrasing or consult with a healthcare professional for personalized advice.")

    print(f"Final answer: '{final_answer[:100]}...'")
    return jsonify({"final_answer": final_answer})

###################################end of the new medical condation 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            flash('Please login as admin.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Database Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    first_name = db.Column(db.String(80), nullable=False)
    last_name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Medication(db.Model):
    __tablename__ = 'medications'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    allergy_test_required = db.Column(db.Boolean, default=False)
    storage_instructions = db.Column(db.Text)
    category = db.Column(db.String(50), nullable=False, default='antibiotics')

class Admin(db.Model):
    __tablename__ = 'admins'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Prescription(db.Model):
    __tablename__ = 'prescriptions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    message = db.Column(db.Text)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='pending')  # pending, approved, rejected
    admin_notes = db.Column(db.Text)
    review_date = db.Column(db.DateTime)

    user = db.relationship('User', backref=db.backref('prescriptions', lazy=True))

# Route for handling search functionality (API)
@app.route('/medicine_search', methods=['GET'])
def medicine_search_page():
    return render_template('search.html')
# Route for handling search functionality (API)
@app.route('/api_search_med', methods=['GET'])
def search_medicine():
    medicine_name = request.args.get('medicine_name', '').strip()

    if medicine_name:
        conn = sqlite3.connect('./instance/medicine_data.db')
        cursor = conn.cursor()

        # Use LIKE for partial matching, ignore case and handle extra spaces
        cursor.execute("""
        SELECT * FROM medicines
        WHERE name LIKE ? 
        """, ('%' + medicine_name + '%',))  # Match partial name

        result = cursor.fetchall()
        conn.close()

        if result:
            return jsonify([dict(zip([column[0] for column in cursor.description], row)) for row in result])
        else:
            return jsonify({"message": "No results found for the entered medicine name."}), 404
    else:
        return jsonify({"error": "No medicine name provided"}), 400

# Route for autocomplete suggestions (API)
@app.route('/autocomplete', methods=['GET'])
def autocomplete_medicine():
    term = request.args.get('term', '').strip()
    if not term:
        return jsonify({"error": "No term provided"}), 400

    conn = sqlite3.connect('./instance/medicine_data.db')
    cursor = conn.cursor()
    
    # First try to find suggestions that start with the term
    cursor.execute("""
    SELECT DISTINCT name FROM medicines
    WHERE name LIKE ? ORDER BY name ASC LIMIT 10
    """, (f'{term}%',))

    suggestions = [row[0] for row in cursor.fetchall()]
    
    # If no suggestions found, fall back to partial match
    if not suggestions:
        cursor.execute("""
        SELECT DISTINCT name FROM medicines
        WHERE name LIKE ? ORDER BY name ASC LIMIT 10
        """, (f'%{term}%',))
        suggestions = [row[0] for row in cursor.fetchall()]
    
    conn.close()

    return jsonify(suggestions)
# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # First check if it's an admin
        admin = Admin.query.filter_by(username=username).first()
        if admin and check_password_hash(admin.password, password):
            session['admin_id'] = admin.id
            session['is_admin'] = True
            flash('Welcome back, Admin!', 'success')
            return redirect(url_for('admin_dashboard'))
        
        # If not admin, check if it's a regular user
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['first_name'] = user.first_name
            flash(f'Welcome back, {user.first_name}!', 'success')
            return redirect(url_for('dashboard'))
            
        flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
            
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, first_name=first_name, last_name=last_name,
                       email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get(session['user_id'])
    medications = Medication.query.all()
    return render_template('dashboard.html', user=user, medications=medications)

@app.route('/admin')
@admin_required
def admin_dashboard():
    medications = Medication.query.all()
    return render_template('admin_dashboard.html', medications=medications)

@app.route('/admin/medications/add', methods=['POST'])
@admin_required
def add_medication():
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            price = float(request.form.get('price'))
            allergy_test_required = bool(request.form.get('allergy_test_required'))
            storage_instructions = request.form.get('storage_instructions')
            category = request.form.get('category', 'antibiotics')
            
            medication = Medication(
                name=name,
                price=price,
                allergy_test_required=allergy_test_required,
                storage_instructions=storage_instructions,
                category=category
            )
            db.session.add(medication)
            db.session.commit()
            
            flash('Medication added successfully!', 'success')
        except Exception as e:
            flash('Error adding medication: ' + str(e), 'error')
        
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/medications/edit/<int:id>', methods=['POST'])
@admin_required
def edit_medication(id):
    medication = Medication.query.get_or_404(id)
    if request.method == 'POST':
        try:
            medication.name = request.form.get('name')
            medication.price = float(request.form.get('price'))
            medication.allergy_test_required = bool(request.form.get('allergy_test_required'))
            medication.storage_instructions = request.form.get('storage_instructions')
            medication.category = request.form.get('category', 'antibiotics')
            
            db.session.commit()
            flash('Medication updated successfully!', 'success')
        except Exception as e:
            flash('Error updating medication: ' + str(e), 'error')
        
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/medications/delete/<int:id>')
@admin_required
def delete_medication(id):
    try:
        medication = Medication.query.get_or_404(id)
        db.session.delete(medication)
        db.session.commit()
        flash('Medication deleted successfully!', 'success')
    except Exception as e:
        flash('Error deleting medication: ' + str(e), 'error')
    
    return redirect(url_for('admin_dashboard'))


# Load saved models (ensure these are pre-trained and available)
processor = TrOCRProcessor.from_pretrained("trocr_processor")
model = VisionEncoderDecoderModel.from_pretrained("fineeeee")
similarity_model = SentenceTransformer("similarity_model")

def preprocess_image(image):
    """Convert the image to RGB format for processing."""
    return image.convert("RGB")

def extract_text_from_image(image):
    """Extract text from an image using the TrOCR model."""
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def match_medicine_name(extracted_text):
    """Find the closest matching medicine name from the list."""
    embeddings1 = similarity_model.encode([extracted_text], convert_to_tensor=True)
    embeddings2 = similarity_model.encode(medicine_names, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    best_match_idx = torch.argmax(cosine_scores).item()
    return medicine_names[best_match_idx], cosine_scores[0, best_match_idx].item()

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'prescription' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
            
        file = request.files['prescription']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
            
        try:
            # Open and process the image
            image = Image.open(file)
            image = preprocess_image(image)
            
            # Extract text and match medicine name
            extracted_text = extract_text_from_image(image)
            matched_medicine, similarity = match_medicine_name(extracted_text)
            
            # Save the file with a unique name
            original_filename = secure_filename(file.filename)
            filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{original_filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(file_path)
            
            # Create prescription record with medicine recognition results
            prescription = Prescription(
                user_id=session['user_id'],
                filename=filename,
                original_filename=original_filename,
                message=f"Extracted Text: {extracted_text}\nMatched Medicine: {matched_medicine} (Similarity: {similarity:.4f})"
            )
            
            db.session.add(prescription)
            db.session.commit()
            
            flash(f'Prescription uploaded successfully! Matched Medicine: {matched_medicine}', 'success')
            return redirect(url_for('upload'))
        
        except Exception as e:
            flash(f'Error processing prescription: {str(e)}', 'error')
            return redirect(request.url)
            
    # Get user's prescriptions for display
    user_prescriptions = Prescription.query.filter_by(user_id=session['user_id']).order_by(Prescription.upload_date.desc()).all()
    return render_template('upload.html', prescriptions=user_prescriptions)

@app.route('/download/<int:prescription_id>')
def download_prescription(prescription_id):
    # Check if user is logged in (either as admin or regular user)
    if not session.get('user_id') and not session.get('admin_id'):
        flash('Please login first.', 'error')
        return redirect(url_for('login'))

    prescription = Prescription.query.get_or_404(prescription_id)
    
    # Security check: ensure user can only download their own prescriptions or is admin
    if not session.get('is_admin') and prescription.user_id != session.get('user_id'):
        flash('Unauthorized access', 'error')
        return redirect(url_for('dashboard'))
        
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], prescription.filename)
        if not os.path.exists(file_path):
            flash('File not found', 'error')
            return redirect(url_for('admin_prescriptions') if session.get('is_admin') else url_for('dashboard'))
            
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        return send_from_directory(
            directory=directory,
            path=filename,
            as_attachment=True,
            download_name=prescription.original_filename
        )
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('admin_prescriptions') if session.get('is_admin') else url_for('dashboard'))

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    # Check if user is logged in (either as admin or regular user)
    if not session.get('user_id') and not session.get('admin_id'):
        flash('Please login first.', 'error')
        return redirect(url_for('login'))
        
    try:
        return send_from_directory(
            app.config['UPLOAD_FOLDER'],
            filename,
            as_attachment=False
        )
    except Exception as e:
        return f'Error serving file: {str(e)}', 404

@app.route('/delete_prescription/<int:prescription_id>')
@login_required
def delete_prescription(prescription_id):
    prescription = Prescription.query.get_or_404(prescription_id)
    
    # Security check: ensure user can only delete their own prescriptions
    if prescription.user_id != session['user_id'] and 'admin_id' not in session:
        flash('Unauthorized access', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        # Delete file from filesystem
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], prescription.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete database record
        db.session.delete(prescription)
        db.session.commit()
        
        flash('Prescription deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting prescription: {str(e)}', 'error')
    
    return redirect(url_for('upload'))

@app.route('/admin/prescriptions')
@admin_required
def admin_prescriptions():
    prescriptions = Prescription.query.order_by(Prescription.upload_date.desc()).all()
    return render_template('admin_prescriptions.html', prescriptions=prescriptions)

@app.route('/admin/prescriptions/review/<int:prescription_id>')
@admin_required
def review_prescription(prescription_id):
    prescription = Prescription.query.get_or_404(prescription_id)
    return render_template('review_prescription.html', prescription=prescription)

@app.route('/admin/prescriptions/update_status/<int:prescription_id>', methods=['POST'])
@admin_required
def update_prescription_status(prescription_id):
    prescription = Prescription.query.get_or_404(prescription_id)
    new_status = request.form.get('status')
    admin_notes = request.form.get('admin_notes', '')
    
    prescription.status = new_status
    prescription.admin_notes = admin_notes
    prescription.review_date = datetime.utcnow()
    
    db.session.commit()
    flash('Prescription review saved successfully!', 'success')
    return redirect(url_for('admin_prescriptions'))

@app.route('/pharmacy-finder')
@login_required
def pharmacy_finder():
    return render_template('pharmacy_finder.html')

def find_nearest_pharmacies(latitude, longitude, k=10):
    """
    Finds the nearest k pharmacies to the given coordinates.
    
    Args:
        latitude: The latitude of the target location.
        longitude: The longitude of the target location.
        k: The number of nearest pharmacies to find.
    
    Returns:
        A list of the k nearest pharmacies, sorted by distance.
    """
    try:
        maps = GoogleMaps(lang="ar", latitude=latitude, longitude=longitude, zoom=21, zoom_index=9)
        
        # Get location address for better search
        latlng = f"{latitude},{longitude}"
        geolocator = Nominatim(user_agent="geoapi")
        location = geolocator.reverse(latlng, language='ar')
        
        keyword = f"صيدلية بالقرب من {location.address}"
        print(f"Search keyword: {keyword}")
        
        all_pharmacies = maps.search(keyword)
        
        # Calculate distances and store as tuples (distance, place)
        pharmacies_with_distance = []
        target_coords = (latitude, longitude)
        
        for i, place in enumerate(all_pharmacies):
            if i >= 20:  # Limit initial search to prevent excessive API calls
                break
               
            place_coords = (place.latitude, place.longitude)
            distance = geopy.distance.distance(target_coords, place_coords).km
            pharmacies_with_distance.append((distance, place))
        
        # Sort by distance
        pharmacies_with_distance.sort(key=lambda item: item[0])
        print(f"Found {len(pharmacies_with_distance)} pharmacies")
        
        # Return the nearest k pharmacies
        return [place for distance, place in pharmacies_with_distance[:k]]
        
    except Exception as e:
        print(f"Error in find_nearest_pharmacies: {str(e)}")
        return []

@app.route('/find-nearest-pharmacy')
@login_required
def find_nearest_pharmacy():
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')

    if not latitude or not longitude:
        return jsonify({
            'success': False,
            'error': 'Location coordinates not provided',
            'pharmacy_name': 'Error',
            'pharmacy_address': 'Location coordinates not provided',
            'pharmacies': []
        }), 400

    try:
        lat = float(latitude)
        lng = float(longitude)
        nearest_pharmacies = find_nearest_pharmacies(lat, lng, k=10)

        if nearest_pharmacies:
            pharmacies_list = []
            target_coords = (lat, lng)
            
            for i, pharmacy in enumerate(nearest_pharmacies):
                pharmacy_coords = (pharmacy.latitude, pharmacy.longitude)
                distance = geopy.distance.distance(target_coords, pharmacy_coords).km

                pharmacy_data = {
                    'rank': i + 1,
                    'pharmacy_name': pharmacy.title,
                    'pharmacy_address': getattr(pharmacy, 'full_address_name', 'Address not available'),
                    'pharmacy_lat': pharmacy.latitude,
                    'pharmacy_lng': pharmacy.longitude,
                    'distance_km': round(distance, 2),
                    'rating': getattr(pharmacy, 'rating', None),
                    'url': getattr(pharmacy, 'url', None)
                }
                pharmacies_list.append(pharmacy_data)

            # Return both the old format for compatibility AND the new array format
            nearest_pharmacy = pharmacies_list[0]  # Get the closest one
            return jsonify({
                'success': True,
                'pharmacy_name': nearest_pharmacy['pharmacy_name'],
                'pharmacy_address': nearest_pharmacy['pharmacy_address'],
                'pharmacy_lat': nearest_pharmacy['pharmacy_lat'],
                'pharmacy_lng': nearest_pharmacy['pharmacy_lng'],
                'distance_km': nearest_pharmacy['distance_km'],
                'rating': nearest_pharmacy['rating'],
                'url': nearest_pharmacy['url'],
                'pharmacies': pharmacies_list  # Full list of all pharmacies
            })

        else:
            return jsonify({
                'success': False,
                'pharmacy_name': 'No pharmacies found nearby',
                'pharmacy_address': 'Try a different location or check your internet connection',
                'pharmacy_lat': lat,
                'pharmacy_lng': lng,
                'pharmacies': []
            })

    except ValueError:
        return jsonify({
            'success': False,
            'error': 'Invalid latitude or longitude format',
            'pharmacy_name': 'Error',
            'pharmacy_address': 'Invalid coordinates provided',
            'pharmacies': []
        }), 400
    except Exception as e:
        error_message = f'An error occurred: {str(e)}'
        print(f"Error in find_nearest_pharmacy: {error_message}")
        return jsonify({
            'success': False,
            'error': error_message,
            'pharmacy_name': 'Error',
            'pharmacy_address': error_message,
            'pharmacies': []
        }), 500
@app.route('/search')
def search_page():
    return render_template('search_drugs.html')

@app.route('/api/search_drugs')
def search_drugs_api():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Use the comprehensive search function from scraper
        results = search_drugs_comprehensive(query)
        
        # If no results found, return empty array with success status
        if not results:
            return jsonify([]), 200
            
        # Format results for frontend consumption
        formatted_results = []
        for drug in results:
            # Parse storage and allergy info
            storage_allergy = drug.get('storage_and_allergy', '')
            storage_info = ''
            allergy_info = ''
            
            if isinstance(storage_allergy, str):
                lines = storage_allergy.split('\n')
                storage_info = lines[0] if len(lines) > 0 else 'No storage information available'
                allergy_info = lines[1] if len(lines) > 1 else 'No allergy information available'
            
            formatted_drug = {
                'name': drug.get('name', ''),
                'name_en': drug.get('name_en', ''),
                'name_ar': drug.get('name_ar', ''),
                'price': float(drug.get('price', 0)),
                'category': drug.get('category', ''),
                'stock': int(drug.get('stock', 0)),
                'image': drug.get('image', ''),
                'share_link': drug.get('share_link', ''),
                'source': drug.get('source', ''),
                'drug_data': storage_allergy,  # Keep original format for backward compatibility
                'storage_info': storage_info,
                'allergy_info': allergy_info,
                'pregnancy_info': drug.get('pregnancy_info', {}),
                'food_interactions': drug.get('food_interactions', {})
            }
            formatted_results.append(formatted_drug)
            
        return jsonify(formatted_results)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred while searching: {str(e)}'}), 500


# Optional: Additional route for detailed drug information
@app.route('/api/drug_details/<drug_name>')
def get_drug_details(drug_name):
    try:
        result = get_drug_complete_info(drug_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

############
@app.route('/medicine_interaction')
def medicine_interaction():
    return render_template('medicine_interaction.html')

def init_db():
    with app.app_context():
        db.create_all()
        # Create default admin account if it doesn't exist
        if not Admin.query.filter_by(username='admin').first():
            admin = Admin(
                username='admin',
                password=generate_password_hash('admin')
            )
            db.session.add(admin)
            db.session.commit()
            print("Default admin account created - username: admin, password: admin")

if __name__ == '__main__':
    init_db()
    app.run(debug=True)