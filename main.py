from flask import Flask, request, render_template_string, jsonify, send_from_directory
from flask_cors import CORS
import requests
import nltk
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
import os
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
# from  import NEWS_API_KEY
import re
import traceback

app = Flask(__name__, static_folder='.')

# Enable CORS for all routes with specific settings
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# Initialize NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            subscription TEXT DEFAULT 'Free',
            usage_count INTEGER DEFAULT 5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_reset DATE DEFAULT CURRENT_DATE
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user(email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()
    
    if user:
        return {
            'id': user[0],
            'name': user[1],
            'email': user[2],
            'password_hash': user[3],
            'subscription': user[4],
            'usage_count': user[5],
            'created_at': user[6],
            'last_reset': user[7]
        }
    return None

def create_user(name, email, password):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        password_hash = hash_password(password)
        c.execute('INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)', 
                 (name, email, password_hash))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def update_user_subscription(email, subscription='Premium'):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('UPDATE users SET subscription = ? WHERE email = ?', 
             (subscription, email))
    conn.commit()
    conn.close()

def update_user_usage(email, usage_count):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('UPDATE users SET usage_count = ? WHERE email = ?', 
             (usage_count, email))
    conn.commit()
    conn.close()

def reset_daily_usage():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    today = datetime.now().date()
    c.execute('''UPDATE users 
                 SET usage_count = 10, last_reset = ? 
                 WHERE subscription = 'Free' AND last_reset < ?''', 
             (today, today))
    conn.commit()
    conn.close()

class NewsVerifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def query_keywords(self, statement):
        try:
            stop_words = set(nltk.corpus.stopwords.words('english'))
        except:
            stop_words = set()
            
        question_words = {'is', 'are', 'was', 'were', 'did', 'do', 'does', 'has', 
                         'have', 'had','can', 'could', 'should', 'would', 'what', 
                         'when', 'where', 'who', 'why', 'how'}
        
        try:
            words = nltk.word_tokenize(statement.lower())
        except:
            words = statement.lower().split()
            
        filtered = [w for w in words if w.isalpha() and w not in stop_words and w not in question_words]
        return ' '.join(filtered)

    def get_news_articles(self, query):
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&pageSize=100&language=en&sortBy=relevancy"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = []
                for article in data.get('articles', []):
                    title = article.get('title', '')
                    description = article.get('description', '')
                    content = article.get('content', '')
                    # Combine all available text
                    text = f"{title} {description} {content}".strip()
                    if text:
                        articles.append(text)
                
                print(f"Retrieved {len(articles)} articles for query '{query}'")
                return articles
            else:
                print(f"NewsAPI Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def verify_statement(self, statement):
        query = self.query_keywords(statement)
        print(f"Query keywords: {query}")
        
        if not query:
            return {
                'statement': statement,
                'verification': 'Inconclusive',
                'confidence': 0.0,
                'reason': 'Unable to extract keywords from the statement.',
                'sources': [],
                'articles_analyzed': 0
            }
        
        articles = self.get_news_articles(query)
        
        if not articles:
            return {
                'statement': statement,
                'verification': 'Inconclusive',
                'confidence': 0.0,
                'reason': 'No relevant news articles found to verify this statement. Try rephrasing or checking the topic.',
                'sources': [],
                'articles_analyzed': 0
            }
        
        try:
            all_texts = [statement] + articles
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            statement_vector = tfidf_matrix[0]
            article_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(statement_vector, article_vectors)[0]
            avg_similarity = similarities.mean()
            max_similarity = similarities.max()
            
            print(f"Average similarity: {avg_similarity:.4f}, Max similarity: {max_similarity:.4f}")
            
            # Improved verification logic
            if max_similarity > 0.4 or avg_similarity > 0.25:
                verification = 'Likely True'
                confidence = min(max(max_similarity * 100, avg_similarity * 150), 95)
                reason = f'Strong correlation found with {len(articles)} news articles. The claim is well-supported by multiple sources.'
            elif max_similarity > 0.2 or avg_similarity > 0.12:
                verification = 'Uncertain'
                confidence = max_similarity * 80
                reason = f'Moderate correlation found in {len(articles)} articles. The claim has some support but lacks strong evidence.'
            else:
                verification = 'Likely False'
                confidence = (1 - avg_similarity) * 40
                reason = f'Little to no correlation found in {len(articles)} articles. The claim lacks credible news support.'
            
            # Get top matching sources
            top_indices = similarities.argsort()[-5:][::-1]
            sources = []
            for i in top_indices:
                if i < len(articles) and similarities[i] > 0.05:
                    source_text = articles[i][:150] + "..."
                    sources.append(source_text)
            
            return {
                'statement': statement,
                'verification': verification,
                'confidence': round(confidence, 2),
                'reason': reason,
                'sources': sources[:3],
                'articles_analyzed': len(articles)
            }
            
        except Exception as e:
            print(f"Error in verification: {e}")
            return {
                'statement': statement,
                'verification': 'Error',
                'confidence': 0.0,
                'reason': f'Error processing statement: {str(e)}',
                'sources': [],
                'articles_analyzed': 0
            }

# Initialize
init_db()
verifier = NewsVerifier()

# Routes for serving HTML files
@app.route('/')
def home():
    return send_from_directory('.', 'home.html')

@app.route('/index.html')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        claim = data.get('claim', '').strip()
        user_email = data.get('user_email')
        
        if not claim:
            return jsonify({'error': 'No claim provided'}), 400
        
        # Check user usage
        if user_email:
            user = get_user(user_email)
            if user:
                # Reset usage if it's a new day
                today = str(datetime.now().date())
                if user['last_reset'] != today:
                    update_user_usage(user_email, 5)
                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    c.execute('UPDATE users SET last_reset = ? WHERE email = ?', (today, user_email))
                    conn.commit()
                    conn.close()
                    user['usage_count'] = 5
                
                # Check limits for free users
                if user['subscription'] == 'Free' and user['usage_count'] <= 0:
                    return jsonify({'error': 'Daily usage limit reached. Please upgrade to premium.'}), 403
        
        # Perform verification
        result = verifier.verify_statement(claim)
        
        # Update usage for free users
        if user_email:
            user = get_user(user_email)
            if user and user['subscription'] == 'Free':
                new_count = max(0, user['usage_count'] - 1)
                update_user_usage(user_email, new_count)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in verify endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password required'})
        
        user = get_user(email)
        if not user:
            return jsonify({'success': False, 'message': 'Invalid email or password'})
        
        if user['password_hash'] != hash_password(password):
            return jsonify({'success': False, 'message': 'Invalid email or password'})
        
        # Reset daily usage if it's a new day
        today = str(datetime.now().date())
        if user['last_reset'] != today and user['subscription'] == 'Free':
            update_user_usage(email, 5)
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('UPDATE users SET last_reset = ? WHERE email = ?', (today, email))
            conn.commit()
            conn.close()
            user['usage_count'] = 5
        
        return jsonify({
            'success': True,
            'user': {
                'email': user['email'],
                'name': user['name'],
                'subscription': user['subscription'],
                'usage_count': user['usage_count']
            }
        })
        
    except Exception as e:
        print(f"Error in login: {e}")
        return jsonify({'success': False, 'message': 'Server error'})

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        if not all([name, email, password]):
            return jsonify({'success': False, 'message': 'All fields are required'})
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'})
        
        if create_user(name, email, password):
            return jsonify({'success': True, 'message': 'User created successfully'})
        else:
            return jsonify({'success': False, 'message': 'Email already exists'})
            
    except Exception as e:
        print(f"Error in register: {e}")
        return jsonify({'success': False, 'message': 'Server error'})

@app.route('/subscribe', methods=['POST'])
def subscribe():
    """Subscribe endpoint - Not functional in demo"""
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({'success': False, 'message': 'Email required'})
    
    # This endpoint exists but doesn't actually upgrade users
    # It's kept for future payment integration
    return jsonify({
        'success': False, 
        'message': 'Payment processing is not available in this demo version'
    })

@app.route('/update_usage', methods=['POST'])
def update_usage():
    try:
        data = request.get_json()
        email = data.get('email')
        usage_count = data.get('usage_count')
        
        if not email or usage_count is None:
            return jsonify({'success': False, 'message': 'Email and usage_count required'})
        
        update_user_usage(email, usage_count)
        return jsonify({'success': True, 'message': 'Usage updated'})
        
    except Exception as e:
        print(f"Error in update_usage: {e}")
        return jsonify({'success': False, 'message': 'Server error'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'TruthFort API is running'})

@app.route('/favicon.png')
def favicon():
    return send_from_directory('.', 'favicon.png', mimetype='image/png')

if __name__ == '__main__':
    reset_daily_usage()
    print("TruthFort Server Starting...")
    print("Database initialized")
    print("Server running on http://localhost:8000")
    print("Home page: http://localhost:8000")
    print("Main app: http://localhost:8000/index.html")

    app.run(debug=True, port=8000, host='0.0.0.0')
