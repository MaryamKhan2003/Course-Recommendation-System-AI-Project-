"""
Course Recommendation System - Flask Web Application
Uses K-Nearest Neighbors (KNN) Algorithm Only
AIL 201 Final Project
"""

from flask import Flask, render_template, request, jsonify
import os

from recommender import KNNRecommender

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'knn_model.pkl')

# Load model
if os.path.exists(MODEL_PATH):
    knn = KNNRecommender.load(MODEL_PATH)
    print(f"✅ KNN Model loaded successfully!")
    print(f"📊 Test Accuracy: {knn.accuracy*100:.1f}%")
    print(f"📊 CV F1: {knn.cv_f1_mean*100:.1f}% ± {knn.cv_f1_std*100:.1f}%")
    
    # Fix: Check if course_data exists and get its length safely
    if knn.course_data is not None:
        print(f"📚 Total courses in database: {len(knn.course_data)}")
    else:
        print(f"📚 Total courses in database: 0")
else:
    print(f"❌ Model not found. Run: python recommender.py")
    knn = None

DIFFICULTY_INFO = {
    'Beginner': {'icon': '🌱', 'desc': 'Perfect for beginners. No prior experience needed.', 'color': '#28a745'},
    'Intermediate': {'icon': '📚', 'desc': 'For learners with some experience.', 'color': '#ffc107'},
    'Advanced': {'icon': '🎓', 'desc': 'For experienced learners.', 'color': '#dc3545'}
}


@app.route('/')
def index():
    return render_template('index.html', difficulty_info=DIFFICULTY_INFO)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Interest_Area': request.form.get('interest_area', ''),
            'desired_rating': float(request.form.get('desired_rating', 4.0)),
            'prefers_projects': 1 if request.form.get('prefers_projects') == 'on' else 0,
            'experience_level': request.form.get('experience_level', 'intermediate')
        }
        
        if not data['Interest_Area'] or len(data['Interest_Area'].strip()) < 3:
            return jsonify({'success': False, 'error': 'Please enter your interests (minimum 3 characters)'})
        
        if knn is None:
            return jsonify({'success': False, 'error': 'Model not loaded. Run: python recommender.py'})
        
        result = knn.predict(data)
        
        # Get total courses count safely
        if knn.course_data is not None:
            total_courses = len(knn.course_data)
        else:
            total_courses = 0
        
        return jsonify({
            'success': True,
            'recommended_difficulty': result['recommended'],
            'confidence': result['confidence'],
            'top3': result['top3'],
            'recommended_courses': result['recommended_courses'],
            'difficulty_info': DIFFICULTY_INFO.get(result['recommended'], DIFFICULTY_INFO['Intermediate']),
            'min_rating_used': result['min_rating_used'],
            'total_courses_available': total_courses
        })
        
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/metrics')
def metrics():
    """Return KNN model metrics as JSON"""
    if knn is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Get total courses safely
    if knn.course_data is not None:
        n_samples = len(knn.course_data)
    else:
        n_samples = 0
    
    return jsonify({
        'model_name': 'K-Nearest Neighbors (KNN)',
        'k_value': knn.k,
        'distance_metric': 'Euclidean Distance',
        'test_accuracy': round(knn.accuracy * 100, 2) if hasattr(knn, 'accuracy') else 0,
        'test_precision': round(knn.precision * 100, 2) if hasattr(knn, 'precision') else 0,
        'test_recall': round(knn.recall * 100, 2) if hasattr(knn, 'recall') else 0,
        'test_f1': round(knn.f1_score * 100, 2) if hasattr(knn, 'f1_score') else 0,
        'cv_accuracy_mean': round(knn.cv_accuracy_mean * 100, 2) if hasattr(knn, 'cv_accuracy_mean') else 0,
        'cv_accuracy_std': round(knn.cv_accuracy_std * 100, 2) if hasattr(knn, 'cv_accuracy_std') else 0,
        'cv_precision_mean': round(knn.cv_precision_mean * 100, 2) if hasattr(knn, 'cv_precision_mean') else 0,
        'cv_precision_std': round(knn.cv_precision_std * 100, 2) if hasattr(knn, 'cv_precision_std') else 0,
        'cv_recall_mean': round(knn.cv_recall_mean * 100, 2) if hasattr(knn, 'cv_recall_mean') else 0,
        'cv_recall_std': round(knn.cv_recall_std * 100, 2) if hasattr(knn, 'cv_recall_std') else 0,
        'cv_f1_mean': round(knn.cv_f1_mean * 100, 2) if hasattr(knn, 'cv_f1_mean') else 0,
        'cv_f1_std': round(knn.cv_f1_std * 100, 2) if hasattr(knn, 'cv_f1_std') else 0,
        'n_samples': n_samples,
        'n_features': 16,
        'n_classes': 3
    })


@app.route('/stats')
def stats():
    return render_template('stats.html')


@app.route('/health')
def health():
    # Get total courses safely
    if knn and knn.course_data is not None:
        total_courses = len(knn.course_data)
    else:
        total_courses = 0
    
    return jsonify({
        'status': 'running',
        'model_loaded': knn is not None,
        'model_type': 'K-Nearest Neighbors (KNN)',
        'total_courses': total_courses
    })


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 COURSE RECOMMENDATION SYSTEM")
    print("🤖 Model: K-Nearest Neighbors (KNN)")
    print("=" * 70)
    print("📍 Endpoints:")
    print("   Home:     http://localhost:5000/")
    print("   Stats:    http://localhost:5000/stats")
    print("   Metrics:  http://localhost:5000/metrics")
    print("=" * 70)
    
    if knn:
        print(f"\n✅ KNN Ready (k={knn.k})")
        if knn.course_data is not None:
            print(f"📚 Loaded {len(knn.course_data)} courses")
        else:
            print(f"📚 No course data loaded")
    else:
        print("\n⚠️  Run: python recommender.py first")
    
    print("\n🌐 Server starting...\n")
    app.run(debug=True, port=5000)