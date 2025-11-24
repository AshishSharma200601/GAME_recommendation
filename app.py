# app.py
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from functools import wraps
import pandas as pd
import numpy as np
import os
import joblib
import re
import sqlite3
import hashlib
from contextlib import closing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from datetime import datetime

# -----------------------------
# Paths & setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Local CSV (your project) preferred
LOCAL_CSV = os.path.join(DATA_DIR, "games.csv")
# Uploaded CSV path used earlier in this conversation (exists only in the hosted environment)
UPLOADED_CSV_PATH = "/mnt/data/bb93fff5-1e5e-46a8-a575-ed550f26360c.csv"

# Choose CSV: prefer local file, else fallback to uploaded path if present
if os.path.exists(LOCAL_CSV):
    CSV_PATH = LOCAL_CSV
elif os.path.exists(UPLOADED_CSV_PATH):
    CSV_PATH = UPLOADED_CSV_PATH
else:
    CSV_PATH = LOCAL_CSV  # default location (will raise if not found later)

# -----------------------------
# UTIL
# -----------------------------
def safe_lower(x):
    try:
        return str(x).lower()
    except Exception:
        return ""

def normalize_series(s):
    s = s.astype(float)
    if s.max() > s.min():
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    else:
        return np.zeros(len(s))

def simple_clean(text):
    # basic cleaning to remove weird chars and collapse whitespace
    if pd.isna(text):
        return ""
    t = str(text)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

# -----------------------------
# Load CSV & prepare catalog
# -----------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH} - please place your games CSV at data/games.csv")

catalog = pd.read_csv(CSV_PATH)

# Diagnostics (helpful when debugging)
print(f"Loaded CSV: {CSV_PATH} — rows: {len(catalog)}")
print("Columns:", list(catalog.columns)[:50])

# Normalize column names (strip spaces)
catalog.columns = [c.strip() for c in catalog.columns]

# Map common column names (case-insensitive) from CSV to the internal names
# so the template/recommend endpoint can rely on `title`, `description`, etc.
col_lookup = {c.strip().lower(): c for c in catalog.columns}
def map_col(target, candidates):
    for cand in candidates:
        key = cand.strip().lower()
        if key in col_lookup:
            catalog[target] = catalog[col_lookup[key]]
            return True
    return False

map_col('title', ['title'])
map_col('description', ['description', 'summary'])
map_col('genres', ['genres'])
map_col('tags', ['tags'])
map_col('platforms', ['platforms'])
map_col('image_url', ['image_url', 'image', 'cover', 'cover_url'])
map_col('game_id', ['game_id', 'id'])
# reviews_total may appear under different names
map_col('reviews_total', ['reviews_total', 'number of reviews', 'number_of_reviews', 'reviews'])
# rating/score column
map_col('rating', ['rating', 'score', 'user_rating', 'metacritic_score'])

# release_year: try to extract from a 'release_date' or 'release date' column if present
if 'release_year' not in catalog.columns:
    if map_col('release_date', ['release_date', 'release date', 'Release Date'.lower()]):
        try:
            catalog['release_year'] = pd.to_datetime(catalog['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
        except Exception:
            catalog['release_year'] = 0
    else:
        # try any column that looks like a year
        for c in catalog.columns:
            try:
                if catalog[c].astype(str).str.match(r"^\d{4}$").any():
                    catalog['release_year'] = pd.to_numeric(catalog[c], errors='coerce').fillna(0).astype(int)
                    break
            except Exception:
                continue
    if 'release_year' not in catalog.columns:
        catalog['release_year'] = 0

# Ensure required text columns exist and fill NAs
text_cols = ["title", "description", "genres", "tags", "platforms"]
for c in text_cols:
    if c not in catalog.columns:
        catalog[c] = ""
catalog[text_cols] = catalog[text_cols].fillna("").astype(str)

# create a cleaned combined text field used for TF-IDF
catalog["text_core"] = (
    catalog["title"].fillna("").astype(str) + " "
    + catalog["description"].fillna("").astype(str) + " "
    + catalog["genres"].fillna("").astype(str) + " "
    + catalog["tags"].fillna("").astype(str)
)
catalog["text_core"] = catalog["text_core"].apply(simple_clean).astype(str)

# If text_core is empty for many rows, fallback to title|genres|tags
if catalog["text_core"].str.strip().apply(len).sum() == 0:
    catalog["text_core"] = (
        catalog["title"].fillna("").astype(str) + " "
        + catalog["genres"].fillna("").astype(str) + " "
        + catalog["tags"].fillna("").astype(str)
    ).apply(simple_clean)

# Ultimate guard: deterministic tokens from game_id or index
if catalog["text_core"].str.strip().apply(len).max() == 0:
    if "game_id" in catalog.columns:
        catalog["text_core"] = catalog["game_id"].astype(str).apply(lambda gid: f"game_{gid}")
    else:
        # FIX: apply on Series, not Index
        catalog["text_core"] = pd.Series(catalog.index.astype(str)).apply(lambda i: f"game_{i}")

# Cast numeric fields if present (safe)
for col in ["price", "reviews_total", "release_year", "rating"]:
    if col in catalog.columns:
        catalog[col] = pd.to_numeric(catalog[col], errors="coerce").fillna(0.0)
    else:
        catalog[col] = 0.0

# Lowercased versions for quick comparisons
catalog["_title_lc"] = catalog["title"].fillna("").astype(str).str.lower()
catalog["_tags_lc"] = catalog["tags"].fillna("").astype(str).str.lower()
catalog["_genres_lc"] = catalog["genres"].fillna("").astype(str).str.lower()
catalog["_platforms_lc"] = catalog["platforms"].fillna("").astype(str).str.lower()

# -----------------------------
# TF-IDF + persistence
# -----------------------------
VECT_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
X_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.joblib")  # using joblib for both

def train_vectorizer_and_matrix(docs):
    """
    Fit a TF-IDF vectorizer on docs and return (vectorizer, X)
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, stop_words="english")
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        # try without english stop words if empty vocab
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0, stop_words=None)
        X = vec.fit_transform(docs)
    # persist
    joblib.dump(vec, VECT_PATH)
    joblib.dump(X, X_PATH)
    return vec, X

# Try to load persisted artifacts; if missing, train now.
if os.path.exists(VECT_PATH) and os.path.exists(X_PATH):
    try:
        vectorizer = joblib.load(VECT_PATH)
        X = joblib.load(X_PATH)
    except Exception as e:
        print("Failed loading artifacts (retraining):", e)
        vectorizer, X = train_vectorizer_and_matrix(catalog["text_core"].tolist())
else:
    vectorizer, X = train_vectorizer_and_matrix(catalog["text_core"].tolist())

# Precompute normalized popularity (reviews) and recency (year)
catalog["_rev_norm"] = normalize_series(catalog["reviews_total"])
catalog["_year_norm"] = normalize_series(catalog["release_year"])
# hybrid popularity score
catalog["_pop_score"] = 0.7 * catalog["_rev_norm"] + 0.3 * catalog["_year_norm"]

# -----------------------------
# Image resolver - tries to get real game images
# -----------------------------
import hashlib
import urllib.parse

def resolve_image_url(row):
    """
    Resolves game cover image URL.
    Priority:
    1. Direct image_url from CSV if available
    2. Real game image from API (when configured)
    3. Deterministic placeholder based on game title
    
    To use real game images, integrate with:
    - RAWG API: https://rawg.io/apidocs (free tier: 20,000 requests/month)
    - IGDB API: https://www.igdb.com/api (free registration)
    - Steam API: requires API key
    """
    # row may be a Series (when used with apply axis=1)
    try:
        if "image_url" in row and isinstance(row["image_url"], str) and row["image_url"].strip():
            url = row["image_url"].strip()
            if url.startswith(('http://', 'https://')):
                return url
    except Exception:
        pass
    
    # Try to get game title for image search
    title = ""
    try:
        title = row.get("title", "") or (row["title"] if "title" in row else "")
    except Exception:
        try:
            title = row["title"] if "title" in row else ""
        except Exception:
            title = ""
    
    if title:
        # TODO: Integrate with real game image API
        # Example with RAWG API (uncomment and add your API key):
        # try:
        #     import requests
        #     title_encoded = urllib.parse.quote(title)
        #     api_key = "YOUR_RAWG_API_KEY"  # Get from https://rawg.io/apidocs
        #     url = f"https://api.rawg.io/api/games?search={title_encoded}&key={api_key}"
        #     response = requests.get(url, timeout=2)
        #     if response.status_code == 200:
        #         data = response.json()
        #         if data.get('results') and len(data['results']) > 0:
        #             return data['results'][0].get('background_image', '')
        # except Exception:
        #     pass
        
        # For now: deterministic placeholder based on title hash
        # This ensures same game always gets same image
        title_hash = hashlib.md5(title.encode()).hexdigest()[:12]
        return f"https://picsum.photos/seed/game_{title_hash}/800/1000"
    
    # Ultimate fallback: deterministic placeholder
    seed = row.get("game_id", None) or title or ""
    return f"https://picsum.photos/seed/{str(seed).replace(' ','_')}/800/1000"

# -----------------------------
# Authentication setup - SQL Database
# -----------------------------
DB_FILE = os.path.join(DATA_DIR, "users.db")
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=os.path.join(BASE_DIR, "static"))
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production-' + str(os.urandom(16)))
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

def init_db():
    """Initialize database and create users table if it doesn't exist"""
    with closing(get_db()) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        print(f"Database initialized at {DB_FILE}")

def hash_password(password):
    """Hash password using SHA256 (for production, use bcrypt)"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, email, password):
    """Create a new user in the database"""
    try:
        with closing(get_db()) as conn:
            cursor = conn.cursor()
            hashed_password = hash_password(password)
            cursor.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, hashed_password)
            )
            conn.commit()
            return True
    except sqlite3.IntegrityError as e:
        if 'UNIQUE constraint failed: users.username' in str(e):
            raise ValueError("Username already exists")
        elif 'UNIQUE constraint failed: users.email' in str(e):
            raise ValueError("Email already registered")
        raise
    except Exception as e:
        print(f"Error creating user: {e}")
        return False

def get_user_by_username(username):
    """Get user by username"""
    try:
        with closing(get_db()) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

def get_user_by_email(email):
    """Get user by email"""
    try:
        with closing(get_db()) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    except Exception as e:
        print(f"Error getting user by email: {e}")
        return None

def verify_user(username_or_email, password):
    """Verify user credentials - accepts either username or email"""
    # Try to find user by username first
    user = get_user_by_username(username_or_email)
    
    # If not found by username, try email
    if not user:
        user = get_user_by_email(username_or_email)
    
    # Verify password if user found
    if user and user['password'] == hash_password(password):
        return user
    return None

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# -----------------------------
# Authentication routes
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username_or_email = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        remember = request.form.get("remember") == "on"
        
        if not username_or_email or not password:
            flash("Please fill in all fields", "error")
            return render_template("login.html")
        
        # Verify user credentials from database (accepts username or email)
        user = verify_user(username_or_email, password)
        if user:
            session["username"] = user["username"]
            session["email"] = user["email"]
            session["user_id"] = user["id"]
            if remember:
                session.permanent = True
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username/email or password", "error")
        
        return render_template("login.html")
    
    # If already logged in, redirect to home
    if "username" in session:
        return redirect(url_for("home"))
    
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        # Validation
        if not username or not email or not password:
            flash("Please fill in all fields", "error")
            return render_template("register.html")
        
        if len(username) < 3 or len(username) > 20:
            flash("Username must be between 3 and 20 characters", "error")
            return render_template("register.html")
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            flash("Username can only contain letters, numbers, and underscores", "error")
            return render_template("register.html")
        
        if len(password) < 6:
            flash("Password must be at least 6 characters long", "error")
            return render_template("register.html")
        
        if password != confirm_password:
            flash("Passwords do not match", "error")
            return render_template("register.html")
        
        # Email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            flash("Please enter a valid email address", "error")
            return render_template("register.html")
        
        # Check if username already exists
        if get_user_by_username(username):
            flash("Username already exists", "error")
            return render_template("register.html")
        
        # Check if email already exists
        if get_user_by_email(email):
            flash("Email already registered", "error")
            return render_template("register.html")
        
        # Create new user in database
        try:
            if create_user(username, email, password):
                flash("Account created successfully! Please login.", "success")
                return redirect(url_for("login"))
            else:
                flash("Error creating account. Please try again.", "error")
                return render_template("register.html")
        except ValueError as e:
            flash(str(e), "error")
            return render_template("register.html")
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return render_template("register.html")
    
    # If already logged in, redirect to home
    if "username" in session:
        return redirect(url_for("home"))
    
    return render_template("register.html")

@app.route("/logout")
def logout():
    username = session.get("username", "User")
    session.clear()
    flash(f"Goodbye, {username}! You have been logged out.", "success")
    return redirect(url_for("login"))

# -----------------------------
# Main routes
# -----------------------------
@app.route("/")
@login_required
def home():
    # simple index.html must exist in templates/ — otherwise return a short message
    if os.path.exists(os.path.join(TEMPLATES_DIR, "index.html")):
        return render_template("index.html", username=session.get("username"))
    return "<h3>Game recommender running. Use /recommend?q=…</h3>"

@app.get("/recommend")
@login_required
def recommend():
    """
    Query params:
      - q: free-text query (title/description/tags)
      - k: top-k (default 12)
      - genre: filter by genre substring (case-insensitive)
      - platform: filter by platform substring (case-insensitive)
      - max_price: float filter
    """
    q = request.args.get("q", default="", type=str).strip()
    k = request.args.get("k", default=12, type=int)
    k = max(1, min(int(k), 200))
    offset = request.args.get("offset", default=0, type=int)
    offset = max(0, int(offset))
    genre = request.args.get("genre", default="", type=str).strip().lower()
    platform = request.args.get("platform", default="", type=str).strip().lower()
    max_price = request.args.get("max_price", default=None, type=str)

    # filtering mask
    mask = pd.Series(True, index=catalog.index)
    if genre:
        mask &= catalog["_genres_lc"].str.contains(genre, na=False)
    if platform:
        mask &= catalog["_platforms_lc"].str.contains(platform, na=False)
    if max_price:
        try:
            max_price_val = float(max_price)
            mask &= catalog["price"].astype(float) <= max_price_val
        except Exception:
            pass

    filtered = catalog[mask].copy()
    if filtered.empty:
        return jsonify({"total": 0, "results": []})

    # If query provided: TF-IDF similarity + exact-match boosts + popularity blending
    if q:
        q_clean = simple_clean(q)
        q_vec = vectorizer.transform([q_clean])
        # compute similarity only on filtered rows (X supports row indexing by list)
        try:
            sims = linear_kernel(q_vec, X[filtered.index]).ravel()  # shape (len(filtered),)
        except Exception:
            # fallback: compute against full X then pick filtered positions
            sims_full = linear_kernel(q_vec, X).ravel()
            sims = sims_full[filtered.index]

        # boost exact title match or strong containment in title/tags
        q_lc = q.lower()
        boosts = np.zeros(len(filtered), dtype=float)
        q_words = set(q_lc.split())
        for i, idx in enumerate(filtered.index):
            title = catalog.at[idx, "_title_lc"]
            tags = catalog.at[idx, "_tags_lc"]
            # exact title
            if title == q_lc:
                boosts[i] += 0.5
            # title contains all query words
            if q_words and q_words.issubset(set(title.split())):
                boosts[i] += 0.25
            # tags contain query tokens
            if any(token in tags for token in q_words):
                boosts[i] += 0.2

        # normalize sims to 0..1
        if sims.max() > sims.min():
            sims_n = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
        else:
            sims_n = np.zeros_like(sims)

        pop_scores = filtered["_pop_score"].values
        # final combined score: mostly similarity, plus pop and boosts
        final = 0.75 * sims_n + 0.2 * pop_scores + 1.0 * boosts
        # small tie-breaker by reviews (so popular exact-match titles float up)
        final = final + 1e-6 * filtered["reviews_total"].values

    else:
        # No query: rank by popularity score primarily
        final = filtered["_pop_score"].values.copy()
        # additional small boost to newer titles
        final = final + 0.01 * filtered["release_year"].values  # tiny preference for newer titles

    # choose ordered indices and then apply offset/limit (pagination)
    order_all = np.argsort(final)[::-1]
    
    # Deduplicate by game_id or title before pagination
    seen = set()
    unique_indices = []  # positions in filtered DataFrame
    for idx in order_all:
        # Use game_id if available, otherwise use title
        if "game_id" in filtered.columns:
            identifier = str(filtered.iloc[idx]["game_id"]).strip().lower()
        else:
            identifier = str(filtered.iloc[idx]["title"]).strip().lower()
        
        if identifier and identifier not in seen:
            seen.add(identifier)
            unique_indices.append(idx)
    
    total = int(len(unique_indices))
    # compute slice
    start = offset
    end = offset + k
    sel = unique_indices[start:end]
    out_rows = filtered.iloc[sel].copy()
    out_rows["score"] = [float(final[i]) for i in sel]
    out_rows["image_url"] = out_rows.apply(resolve_image_url, axis=1)

    cols = [
        col for col in [
            "game_id", "title", "description", "genres", "tags", "platforms", "price",
            "reviews_total", "release_year", "rating", "score", "image_url"
        ] if col in out_rows.columns
    ]
    return jsonify({"total": total, "results": out_rows[cols].to_dict(orient="records")})

@app.get("/health")
@login_required
def health():
    return jsonify({"status": "ok", "rows": int(len(catalog)), "csv": CSV_PATH})

# optional endpoint to retrain vectorizer on the current CSV (useful if you update the CSV)
@app.post("/retrain")
@login_required
def retrain():
    global vectorizer, X, catalog
    # reload CSV in case it's been updated
    if not os.path.exists(CSV_PATH):
        return jsonify({"status": "error", "msg": f"CSV not found at {CSV_PATH}"}), 400

    catalog = pd.read_csv(CSV_PATH)
    catalog.columns = [c.strip() for c in catalog.columns]
    
    # Re-map columns (same logic as initial load)
    col_lookup = {c.strip().lower(): c for c in catalog.columns}
    def map_col(target, candidates):
        for cand in candidates:
            key = cand.strip().lower()
            if key in col_lookup:
                catalog[target] = catalog[col_lookup[key]]
                return True
        return False
    
    map_col('title', ['title'])
    map_col('description', ['description', 'summary'])
    map_col('genres', ['genres'])
    map_col('tags', ['tags'])
    map_col('platforms', ['platforms'])
    map_col('image_url', ['image_url', 'image', 'cover', 'cover_url'])
    map_col('game_id', ['game_id', 'id'])
    map_col('reviews_total', ['reviews_total', 'number of reviews', 'number_of_reviews', 'reviews'])
    map_col('rating', ['rating', 'score', 'user_rating', 'metacritic_score'])
    
    # Handle release_year
    if 'release_year' not in catalog.columns:
        if map_col('release_date', ['release_date', 'release date', 'Release Date'.lower()]):
            try:
                catalog['release_year'] = pd.to_datetime(catalog['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
            except Exception:
                catalog['release_year'] = 0
        else:
            catalog['release_year'] = 0

    # rebuild minimal combined text
    for c in ["title", "description", "genres", "tags", "platforms"]:
        if c not in catalog.columns:
            catalog[c] = ""
    catalog[["title","description","genres","tags"]] = catalog[["title","description","genres","tags"]].fillna("").astype(str)

    catalog["text_core"] = (
        catalog["title"].fillna("").astype(str) + " "
        + catalog["description"].fillna("").astype(str) + " "
        + catalog["genres"].fillna("").astype(str) + " "
        + catalog["tags"].fillna("").astype(str)
    ).apply(simple_clean)

    # ultimate guard
    if catalog["text_core"].str.strip().apply(len).max() == 0:
        if "game_id" in catalog.columns:
            catalog["text_core"] = catalog["game_id"].astype(str).apply(lambda gid: f"game_{gid}")
        else:
            catalog["text_core"] = pd.Series(catalog.index.astype(str)).apply(lambda i: f"game_{i}")

    vectorizer, X = train_vectorizer_and_matrix(catalog["text_core"].tolist())
    # recompute numeric/popularity fields
    for col in ["price", "reviews_total", "release_year", "rating"]:
        if col in catalog.columns:
            catalog[col] = pd.to_numeric(catalog[col], errors="coerce").fillna(0.0)
        else:
            catalog[col] = 0.0
    catalog["_rev_norm"] = normalize_series(catalog["reviews_total"])
    catalog["_year_norm"] = normalize_series(catalog["release_year"])
    catalog["_pop_score"] = 0.7 * catalog["_rev_norm"] + 0.3 * catalog["_year_norm"]
    catalog["_title_lc"] = catalog["title"].fillna("").astype(str).str.lower()
    catalog["_tags_lc"] = catalog["tags"].fillna("").astype(str).str.lower()
    catalog["_genres_lc"] = catalog["genres"].fillna("").astype(str).str.lower()
    catalog["_platforms_lc"] = catalog["platforms"].fillna("").astype(str).str.lower()

    return jsonify({"status": "retrained", "rows": int(len(catalog))})

if __name__ == "__main__":
    # Initialize database on startup
    init_db()
    print("Database initialized. Starting Flask app...")
    # Run the Flask app
    app.run(host="0.0.0.0", port=8000, debug=True)
