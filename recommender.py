"""
Course Recommender - K-Nearest Neighbors Implementation
AIL 201 Final Project
Only KNN Algorithm - No Model Comparison
"""

import pandas as pd
import numpy as np
import pickle
import os
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class KNNRecommender:
    """
    K-Nearest Neighbors Course Recommender
    Built from scratch using Euclidean distance
    """
    
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.course_data = None
        
        # Performance metrics
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        
        # Cross-validation metrics (mean ± std)
        self.cv_accuracy_mean = 0
        self.cv_accuracy_std = 0
        self.cv_precision_mean = 0
        self.cv_precision_std = 0
        self.cv_recall_mean = 0
        self.cv_recall_std = 0
        self.cv_f1_mean = 0
        self.cv_f1_std = 0
        
        # Individual fold scores
        self.cv_accuracy_scores = []
        self.cv_precision_scores = []
        self.cv_recall_scores = []
        self.cv_f1_scores = []
    
    def _euclidean_distance(self, a, b):
        """Calculate Euclidean distance between two vectors"""
        return np.sqrt(np.sum((a - b) ** 2))
    
    def predict_single(self, x):
        """Predict single sample using KNN"""
        if len(self.X_train) == 0:
            return 0
        
        distances = []
        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        
        distances.sort(key=lambda t: t[0])
        k_neighbors = min(self.k, len(distances))
        k_labels = [distances[i][1] for i in range(k_neighbors)]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    
    def predict_batch(self, X):
        """Predict multiple samples"""
        return [self.predict_single(x) for x in X]
    
    def _extract_user_features(self, user_data):
        """Extract 16 features from user input"""
        interest = user_data.get('Interest_Area', '').lower()
        experience_level = user_data.get('experience_level', 'intermediate').lower()
        desired_rating = user_data.get('desired_rating', 4.0)
        
        features = np.zeros(16)
        
        # Feature 0: Programming languages
        programming_keywords = ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'kotlin', 'swift', 'programming', 'coding']
        if any(word in interest for word in programming_keywords):
            features[0] = 1
        
        # Feature 1: Web Development
        web_keywords = ['web', 'html', 'css', 'react', 'angular', 'vue', 'node', 'frontend', 'backend', 'full-stack']
        if any(word in interest for word in web_keywords):
            features[1] = 1
        
        # Feature 2: Data Science
        data_keywords = ['data', 'analytics', 'sql', 'database', 'excel', 'tableau', 'power bi', 'data science']
        if any(word in interest for word in data_keywords):
            features[2] = 1
        
        # Feature 3: AI/ML
        ai_keywords = ['ai', 'machine learning', 'deep learning', 'neural', 'tensorflow', 'llm', 'generative', 'artificial intelligence']
        if any(word in interest for word in ai_keywords):
            features[3] = 1
        
        # Feature 4: Cybersecurity
        security_keywords = ['security', 'cyber', 'hacking', 'encryption', 'forensics', 'penetration', 'cybersecurity']
        if any(word in interest for word in security_keywords):
            features[4] = 1
        
        # Feature 5: Cloud Computing
        cloud_keywords = ['cloud', 'aws', 'azure', 'devops', 'docker', 'kubernetes', 'cloud computing']
        if any(word in interest for word in cloud_keywords):
            features[5] = 1
        
        # Feature 6: Business/Marketing
        business_keywords = ['business', 'marketing', 'management', 'seo', 'digital marketing', 'agile', 'scrum', 'finance']
        if any(word in interest for word in business_keywords):
            features[6] = 1
        
        # Feature 7: Mobile Development
        mobile_keywords = ['mobile', 'android', 'ios', 'app', 'swift', 'kotlin', 'react native']
        if any(word in interest for word in mobile_keywords):
            features[7] = 1
        
        # Feature 8: Game Development
        game_keywords = ['game', 'unity', 'gaming', '3d', 'virtual', 'augmented', 'game development']
        if any(word in interest for word in game_keywords):
            features[8] = 1
        
        # Feature 9: Networking
        network_keywords = ['network', 'tcp/ip', 'routing', 'protocol', 'networking']
        if any(word in interest for word in network_keywords):
            features[9] = 1
        
        # Feature 10: Design
        design_keywords = ['design', 'figma', 'photoshop', 'ui', 'ux', 'graphic', 'designing']
        if any(word in interest for word in design_keywords):
            features[10] = 1
        
        # Feature 11: IoT/Embedded
        iot_keywords = ['iot', 'embedded', 'raspberry', 'arduino', 'robotics']
        if any(word in interest for word in iot_keywords):
            features[11] = 1
        
        # Feature 12: Blockchain
        blockchain_keywords = ['blockchain', 'cryptocurrency', 'ethereum', 'smart contract']
        if any(word in interest for word in blockchain_keywords):
            features[12] = 1
        
        # Feature 13: Desired rating (normalized)
        features[13] = desired_rating / 5.0
        
        # Feature 14: Prefers projects
        features[14] = 1 if user_data.get('prefers_projects', False) else 0
        
        # Feature 15: Experience level
        exp_map = {'beginner': 0.0, 'intermediate': 0.5, 'advanced': 1.0}
        features[15] = exp_map.get(experience_level, 0.5)
        
        return features
    
    def _calculate_interest_score(self, course_row, user_interest):
        """Calculate how well a course matches user's interest"""
        interest = user_interest.lower()
        score = 0
        
        course_text = ""
        course_text += str(course_row.get('Course Name', '')).lower()
        course_text += " " + str(course_row.get('Course Description', '')).lower()
        course_text += " " + str(course_row.get('Skills', '')).lower()
        
        # Keyword matching
        keywords = interest.replace(',', ' ').split()
        for keyword in keywords:
            if len(keyword) > 2 and keyword in course_text:
                score += 3
        
        # Domain matching with weights
        domain_weights = {
            'python': 4, 'javascript': 4, 'java': 4,
            'web': 4, 'react': 5, 'angular': 5,
            'data': 4, 'sql': 4, 'database': 4,
            'ai': 5, 'machine': 5, 'deep': 5,
            'security': 4, 'cyber': 4, 'cloud': 4,
            'game': 5, 'mobile': 4, 'design': 3
        }
        
        for domain, weight in domain_weights.items():
            if domain in interest and domain in course_text:
                score += weight
        
        return min(score, 100)
    
    def get_course_recommendations(self, difficulty_level, user_interest="", min_rating=4.0, top_n=10):
        """Get top N course recommendations for a difficulty level"""
        
        if self.course_data is None or len(self.course_data) == 0:
            return self._get_fallback_courses(difficulty_level, top_n)
        
        diff_col = 'Difficulty Level'
        rating_col = 'Course Rating'
        
        # Filter by difficulty
        matching = self.course_data[self.course_data[diff_col].str.lower() == difficulty_level.lower()]
        if len(matching) == 0:
            matching = self.course_data[self.course_data[diff_col].str.lower().str.contains(difficulty_level.lower(), na=False)]
        
        if len(matching) == 0:
            return self._get_fallback_courses(difficulty_level, top_n)
        
        # Apply rating filter
        matching = matching[matching[rating_col] >= min_rating]
        if len(matching) == 0:
            matching = self.course_data[self.course_data[diff_col].str.lower() == difficulty_level.lower()]
        
        # Score courses
        scored_courses = []
        for idx, row in matching.iterrows():
            try:
                score = self._calculate_interest_score(row, user_interest)
                rating = row.get(rating_col, 4.0)
                if pd.notna(rating):
                    score += (rating - 3.0) * 5 if rating > 3.0 else 0
                scored_courses.append((score, row))
            except Exception as e:
                continue
        
        scored_courses.sort(key=lambda x: x[0], reverse=True)
        top_courses = scored_courses[:top_n]
        
        results = []
        for score, row in top_courses:
            try:
                results.append({
                    'course_name': str(row.get('Course Name', f'{difficulty_level} Course'))[:100],
                    'university': str(row.get('University', 'Coursera'))[:50],
                    'description': str(row.get('Course Description', ''))[:250] + ('...' if len(str(row.get('Course Description', ''))) > 250 else ''),
                    'skills': str(row.get('Skills', 'Various skills'))[:150],
                    'url': str(row.get('Course URL', '#')),
                    'rating': float(row.get('Course Rating', 4.5)),
                    'difficulty': difficulty_level,
                    'match_score': round(score, 1)
                })
            except Exception as e:
                continue
        
        return results
    
    def _get_fallback_courses(self, difficulty_level, top_n=10):
        """Fallback courses"""
        fallback = {
            'Beginner': [
                {'name': 'Python Programming for Beginners', 'university': 'University of Michigan',
                 'skills': 'Python, Programming', 'rating': 4.8,
                 'url': 'https://www.coursera.org/learn/python-programming',
                 'desc': 'Learn Python fundamentals including variables, loops, functions.'},
                {'name': 'JavaScript for Beginners', 'university': 'Meta',
                 'skills': 'JavaScript, Programming', 'rating': 4.7,
                 'url': 'https://www.coursera.org/learn/javascript-basics',
                 'desc': 'Learn JavaScript programming fundamentals.'},
                {'name': 'SQL for Data Science', 'university': 'UC Davis',
                 'skills': 'SQL, Database', 'rating': 4.7,
                 'url': 'https://www.coursera.org/learn/sql-for-data-science',
                 'desc': 'Learn SQL for data analysis.'},
                {'name': 'HTML and CSS for Beginners', 'university': 'Meta',
                 'skills': 'HTML, CSS', 'rating': 4.7,
                 'url': 'https://www.coursera.org/learn/html-css',
                 'desc': 'Build websites with HTML and CSS.'}
            ],
            'Intermediate': [
                {'name': 'Data Science Professional Certificate', 'university': 'IBM',
                 'skills': 'Data Science, Python, SQL', 'rating': 4.8,
                 'url': 'https://www.coursera.org/professional-certificates/ibm-data-science',
                 'desc': 'Complete data science curriculum.'},
                {'name': 'React Frontend Development', 'university': 'Meta',
                 'skills': 'React, JavaScript', 'rating': 4.8,
                 'url': 'https://www.coursera.org/specializations/meta-front-end-developer',
                 'desc': 'Build modern web apps with React.'},
                {'name': 'AWS Cloud Computing', 'university': 'AWS',
                 'skills': 'AWS, Cloud', 'rating': 4.8,
                 'url': 'https://www.coursera.org/specializations/aws-cloud-solutions-architect',
                 'desc': 'Master AWS cloud services.'}
            ],
            'Advanced': [
                {'name': 'Machine Learning Specialization', 'university': 'Stanford',
                 'skills': 'Machine Learning, Python, AI', 'rating': 4.9,
                 'url': 'https://www.coursera.org/specializations/machine-learning-introduction',
                 'desc': 'Master machine learning algorithms.'},
                {'name': 'Deep Learning Specialization', 'university': 'DeepLearning.AI',
                 'skills': 'Deep Learning, TensorFlow', 'rating': 4.9,
                 'url': 'https://www.coursera.org/specializations/deep-learning',
                 'desc': 'Master deep learning.'}
            ]
        }
        
        courses = fallback.get(difficulty_level, fallback['Intermediate'])
        results = []
        for course in courses[:min(top_n, len(courses))]:
            results.append({
                'course_name': course['name'],
                'university': course['university'],
                'description': course['desc'],
                'skills': course['skills'],
                'url': course['url'],
                'rating': course['rating'],
                'difficulty': difficulty_level,
                'match_score': 85.0
            })
        return results
    
    def cross_validate(self, X, y, cv_folds=5):
        """Perform K-fold cross-validation and calculate mean ± std"""
        
        print(f"\n{'='*60}")
        print(f"KNN CROSS-VALIDATION (k={self.k}, {cv_folds}-folds)")
        print(f"{'='*60}")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        self.cv_accuracy_scores = []
        self.cv_precision_scores = []
        self.cv_recall_scores = []
        self.cv_f1_scores = []
        
        fold_num = 1
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Predict using KNN from scratch
            y_pred_fold = []
            for x_val in X_val_fold:
                distances = []
                for i in range(len(X_train_fold)):
                    dist = self._euclidean_distance(x_val, X_train_fold[i])
                    distances.append((dist, y_train_fold[i]))
                distances.sort(key=lambda t: t[0])
                k_neighbors = min(self.k, len(distances))
                k_labels = [distances[i][1] for i in range(k_neighbors)]
                most_common = Counter(k_labels).most_common(1)
                y_pred_fold.append(most_common[0][0])
            
            acc = accuracy_score(y_val_fold, y_pred_fold)
            prec = precision_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0)
            rec = recall_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0)
            f1 = f1_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0)
            
            self.cv_accuracy_scores.append(acc)
            self.cv_precision_scores.append(prec)
            self.cv_recall_scores.append(rec)
            self.cv_f1_scores.append(f1)
            
            print(f"  Fold {fold_num}: Acc={acc*100:.2f}% | Prec={prec*100:.2f}% | Rec={rec*100:.2f}% | F1={f1*100:.2f}%")
            fold_num += 1
        
        # Calculate mean and standard deviation
        self.cv_accuracy_mean = np.mean(self.cv_accuracy_scores)
        self.cv_accuracy_std = np.std(self.cv_accuracy_scores)
        self.cv_precision_mean = np.mean(self.cv_precision_scores)
        self.cv_precision_std = np.std(self.cv_precision_scores)
        self.cv_recall_mean = np.mean(self.cv_recall_scores)
        self.cv_recall_std = np.std(self.cv_recall_scores)
        self.cv_f1_mean = np.mean(self.cv_f1_scores)
        self.cv_f1_std = np.std(self.cv_f1_scores)
        
        print(f"\n{'─'*60}")
        print(f"CROSS-VALIDATION SUMMARY ({cv_folds}-folds):")
        print(f"{'─'*60}")
        print(f"  Accuracy:  {self.cv_accuracy_mean*100:.2f}% ± {self.cv_accuracy_std*100:.2f}%")
        print(f"  Precision: {self.cv_precision_mean*100:.2f}% ± {self.cv_precision_std*100:.2f}%")
        print(f"  Recall:    {self.cv_recall_mean*100:.2f}% ± {self.cv_recall_std*100:.2f}%")
        print(f"  F1-Score:  {self.cv_f1_mean*100:.2f}% ± {self.cv_f1_std*100:.2f}%")
        print(f"{'─'*60}")
        
        return {
            'accuracy': {'mean': self.cv_accuracy_mean, 'std': self.cv_accuracy_std},
            'precision': {'mean': self.cv_precision_mean, 'std': self.cv_precision_std},
            'recall': {'mean': self.cv_recall_mean, 'std': self.cv_recall_std},
            'f1': {'mean': self.cv_f1_mean, 'std': self.cv_f1_std}
        }
    
    def predict(self, user_data):
        """Get course recommendation for a user"""
        features = self._extract_user_features(user_data)
        features_2d = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features_2d)
        
        pred_encoded = self.predict_single(features_scaled[0])
        pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        user_interest = user_data.get('Interest_Area', '')
        user_experience = user_data.get('experience_level', 'intermediate').lower()
        min_rating = user_data.get('desired_rating', 4.0)
        
        # Override with user's experience level
        exp_map = {'beginner': 'Beginner', 'intermediate': 'Intermediate', 'advanced': 'Advanced'}
        expected = exp_map.get(user_experience, 'Intermediate')
        
        if pred_label != expected:
            pred_label = expected
            confidence = 70.0
        else:
            # Calculate confidence
            distances = []
            for i in range(len(self.X_train)):
                dist = self._euclidean_distance(features_scaled[0], self.X_train[i])
                distances.append((dist, self.y_train[i]))
            distances.sort(key=lambda t: t[0])
            
            label_counts = Counter([self.label_encoder.inverse_transform([d[1]])[0] 
                                   for d in distances[:min(15, len(distances))]])
            total = sum(label_counts.values())
            confidence = (label_counts[pred_label] / total * 100) if pred_label in label_counts else 50
        
        # Top 3 difficulty levels
        top3 = [
            {'difficulty': expected, 'match_score': confidence},
            {'difficulty': 'Intermediate' if expected != 'Intermediate' else 'Beginner', 'match_score': round((100 - confidence) / 2, 1)},
            {'difficulty': 'Advanced' if expected != 'Advanced' else 'Beginner', 'match_score': round((100 - confidence) / 2, 1)}
        ]
        
        # Get course recommendations (returns 10)
        courses = self.get_course_recommendations(pred_label, user_interest, min_rating, top_n=10)
        
        return {
            'recommended': pred_label,
            'confidence': round(confidence, 1),
            'top3': top3[:3],
            'recommended_courses': courses,
            'min_rating_used': min_rating
        }
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def load_and_preprocess_data(filepath='coursera_courses.csv'):
    """Load and preprocess the Coursera dataset"""
    
    # Try multiple possible locations
    possible_paths = [
        filepath,
        'data/coursera_courses.csv',
        '../coursera_courses.csv',
        os.path.join(os.path.dirname(__file__), filepath),
        os.path.join(os.path.dirname(__file__), 'data', 'coursera_courses.csv')
    ]
    
    df = None
    used_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, encoding='utf-8')
                used_path = path
                print(f"✅ Loaded dataset from: {path}")
                break
            except Exception as e:
                print(f"⚠️ Could not read {path}: {e}")
                continue
    
    if df is None:
        print(f"❌ Dataset not found. Tried locations:")
        for path in possible_paths:
            print(f"   - {path}")
        print("\nPlease ensure 'coursera_courses.csv' is in the current directory.")
        return None, None, None
    
    print(f"📊 Loaded {len(df)} courses")
    
    diff_col = 'Difficulty Level'
    rating_col = 'Course Rating'
    
    # Check if required columns exist
    required_cols = [diff_col, rating_col, 'Course Name', 'Course URL']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return None, None, None
    
    # Clean data
    df = df.dropna(subset=[diff_col])
    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce').fillna(4.5)
    df[rating_col] = df[rating_col].clip(0, 5)
    
    features = []
    target = []
    skipped = 0
    
    for idx, row in df.iterrows():
        try:
            text = (str(row.get('Course Description', '')) + " " +
                    str(row.get('Skills', '')) + " " +
                    str(row.get('Course Name', ''))).lower()
            
            feature_vec = [
                1 if any(w in text for w in ['python', 'java', 'javascript', 'c++', 'programming', 'coding']) else 0,
                1 if any(w in text for w in ['web', 'html', 'css', 'react', 'angular', 'vue', 'node']) else 0,
                1 if any(w in text for w in ['data', 'sql', 'database', 'analytics', 'excel', 'tableau']) else 0,
                1 if any(w in text for w in ['ai', 'machine learning', 'deep learning', 'neural', 'tensorflow']) else 0,
                1 if any(w in text for w in ['security', 'cyber', 'hacking', 'encryption']) else 0,
                1 if any(w in text for w in ['cloud', 'aws', 'azure', 'devops', 'docker']) else 0,
                1 if any(w in text for w in ['business', 'marketing', 'management', 'finance']) else 0,
                1 if any(w in text for w in ['mobile', 'android', 'ios', 'app', 'swift', 'kotlin']) else 0,
                1 if any(w in text for w in ['game', 'unity', 'gaming', '3d']) else 0,
                1 if any(w in text for w in ['network', 'tcp/ip', 'routing']) else 0,
                1 if any(w in text for w in ['design', 'figma', 'photoshop', 'ui', 'ux']) else 0,
                1 if any(w in text for w in ['iot', 'embedded', 'raspberry', 'arduino']) else 0,
                1 if any(w in text for w in ['blockchain', 'cryptocurrency', 'ethereum']) else 0,
                row[rating_col] / 5.0,
                1 if 'project' in text else 0,
                0.0 if str(row[diff_col]).strip() == 'Beginner' else (0.5 if str(row[diff_col]).strip() == 'Intermediate' else 1.0)
            ]
            
            features.append(feature_vec)
            target.append(str(row[diff_col]).strip())
        except Exception as e:
            skipped += 1
            continue
    
    if len(features) == 0:
        print("❌ No valid features extracted from dataset")
        return None, None, None
    
    X = np.array(features, dtype=np.float64)
    y = np.array(target)
    
    print(f"📈 Feature matrix: {X.shape}")
    print(f"🎯 Target distribution:")
    for label, count in pd.Series(y).value_counts().items():
        print(f"   {label}: {count} courses")
    
    if skipped > 0:
        print(f"⚠️ Skipped {skipped} rows due to errors")
    
    return X, y, df


def train_model():
    """Train the KNN model"""
    
    print("=" * 70)
    print("🤖 KNN COURSE RECOMMENDATION SYSTEM - TRAINING")
    print("=" * 70)
    
    # Load dataset from current directory
    X, y, df = load_and_preprocess_data('coursera_courses.csv')
    
    if X is None:
        print("\n❌ Cannot proceed without dataset!")
        print("\n📝 Please make sure 'coursera_courses.csv' is in the same folder as this script.")
        print("   The file should have these columns:")
        print("   - Course Name")
        print("   - University")
        print("   - Difficulty Level (Beginner/Intermediate/Advanced)")
        print("   - Course Rating")
        print("   - Course URL")
        print("   - Course Description")
        print("   - Skills")
        return None
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    print(f"   Features: {X.shape[1]}")
    
    # Create and train KNN model
    knn = KNNRecommender(k=5)
    knn.scaler = scaler
    knn.label_encoder = le
    knn.course_data = df
    knn.train(X_train, y_train)
    
    # Cross-validation
    knn.cross_validate(X_scaled, y_encoded, cv_folds=5)
    
    # Test on hold-out set
    print(f"\n📊 Testing on Hold-Out Set:")
    y_pred = knn.predict_batch(X_test)
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)
    
    knn.accuracy = accuracy_score(y_test_labels, y_pred_labels)
    knn.precision = precision_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
    knn.recall = recall_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
    knn.f1_score = f1_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
    
    print(f"   Accuracy:  {knn.accuracy*100:.2f}%")
    print(f"   Precision: {knn.precision*100:.2f}%")
    print(f"   Recall:    {knn.recall*100:.2f}%")
    print(f"   F1-Score:  {knn.f1_score*100:.2f}%")
    
    # Save model
    os.makedirs('model', exist_ok=True)
    knn.save('model/knn_model.pkl')
    
    print(f"\n{'='*70}")
    print("✅ KNN MODEL TRAINED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"🔢 K Value: {knn.k}")
    print(f"📏 Distance: Euclidean Distance")
    print(f"📈 CV F1: {knn.cv_f1_mean*100:.2f}% ± {knn.cv_f1_std*100:.2f}%")
    print(f"🎯 Test F1: {knn.f1_score*100:.2f}%")
    print(f"\n💾 Model saved to: model/knn_model.pkl")
    print(f"\n🚀 Run: python app.py")
    print(f"🌐 Open: http://localhost:5000")
    
    return knn


if __name__ == "__main__":
    train_model()