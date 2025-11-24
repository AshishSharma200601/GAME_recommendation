👨‍💻 Developed By

Ashish Sharma
Course: CSE AI-FT
Submitted To: Mr. Mudrik Kaushik

📌 1. Overview

The goal of this project is to simplify game discovery for users in the massively growing gaming industry. By combining natural-language processing, genre/tag matching, and live video game data, the system provides accurate and personalized recommendations.

The final build includes:

A polished frontend UI

A complete Flask backend

RAWG API integration

A custom ranking-based recommendation engine

Fully functional input/output pipeline

⚙️ 2. Technology Stack
Layer	Technologies	Purpose
Frontend	HTML5, CSS3, JavaScript	Visual UI, dynamic user interactions
Backend	Python, Flask, Requests	API server, query processing, recommendation logic
Data Source	RAWG Video Games API	Real-time game details, tags, ratings, trailers
Design	Google Fonts (Orbitron), Neon Theme	Futuristic gaming aesthetic
🧠 3. Features
✔ Fully Functional Frontend

Neon dark mode UI

Responsive layout

Stylish input bar with animations

Auto-generated game cards

Screenshots, ratings & trailer buttons

Smooth transitions and effects

✔ Completed Backend

REST API with Flask

/recommend endpoint

Natural-language keyword extraction

RAWG API integration

Trailer retrieval

Clean JSON output

✔ Working Recommendation Algorithm

Keyword → Tag mapping

RAWG search + filter

Multi-factor scoring:

Genre match

Tag similarity

Rating weight

Popularity

Final ranking of top 5–7 games

✔ RAWG Data Processing

Game details

Screenshots

Preview images

Platforms

Trailers

🏗️ 4. System Workflow
User Query → Frontend → Flask API → Keyword Processing → RAWG API → 
Tag/Genre Matching → Scoring Algorithm → Ranked Results → Frontend Cards

📂 5. Project Structure
AI-Game-Recommender/
│
├── backend/
│   ├── app.py
│   ├── recommender.py
│   ├── rawg_api.py
│   ├── utils.py
│   └── requirements.txt
│
└── frontend/
    ├── index.html
    ├── styles.css
    ├── script.js
    └── assets/

🚀 6. Running the Project
Step 1: Install Dependencies
pip install flask requests

Step 2: Add RAWG API Key

In app.py:

API_KEY = "YOUR_RAWG_API_KEY"

Step 3: Start the Server
python app.py

Step 4: Open Frontend

Open the file index.html in any browser.

🎯 7. Example Queries

Try typing:

“anime adventure RPG”

“first-person horror game”

“open world samurai story rich”

“multiplayer racing low-end pc”

The system will return matching high-quality recommendations.

🎉 8. Output Example

Each recommendation card includes:

Game Title

Genres & Tags

Supported Platforms

Rating

Cover Image

YouTube Trailer (button)

📈 9. Future Updates

Implement ML-based embedding similarity

Add personalization based on user history

Host project on AWS / Vercel

Add multi-query comparison

Add caching layer for faster load

📜 10. License

This project is free for academic and educational use.
