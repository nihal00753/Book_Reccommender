# 📚 Book Recommender System

An end-to-end machine learning project that recommends books using **popularity-based ranking** and **item-based collaborative filtering**, deployed as a live, publicly accessible web application.

🔗 **Live Demo:** [https://nihal0753.pythonanywhere.com/](https://nihal0753.pythonanywhere.com/)

---

## Overview

This project demonstrates the complete lifecycle of a machine learning product — from raw data to a deployed, user-facing application. Rather than stopping at a Jupyter notebook with offline metrics, the trained models were serialized and integrated into a Flask web app, then deployed to a live server.

The app provides two recommendation experiences:
- **Top Books** — a curated list of highly-rated, frequently-reviewed books, useful for new users with no rating history (addressing the classic "cold start" problem).
- **Similar Book Recommendations** — given a book the user likes, the app returns similar titles based on collaborative filtering computed from user rating behavior.

---

## What Was Achieved

- **Built a working recommendation engine from scratch**, covering data cleaning, exploratory analysis, model design, and evaluation.
- **Designed two complementary recommendation strategies** — popularity-based and collaborative filtering — to handle both new and returning users.
- **Engineered the system for performance**, precomputing and caching expensive operations (similarity matrices, pivot tables) so the live app responds instantly without retraining on each request.
- **Built and integrated a backend web application** (Flask) that serves the trained models through a usable interface.
- **Deployed the application to a live production environment** (PythonAnywhere), making it accessible to anyone via a public URL — not just runnable locally.

---

## How It Was Built

### 1. Data Cleaning & Preparation
Worked with a books/users/ratings dataset, filtering out users with very few ratings and books with very few reviews. This reduced noise and sparsity, improving the quality and reliability of the recommendations.

### 2. Popularity-Based Model
Calculated each book's number of ratings and average rating, applied a minimum vote threshold, and ranked the results to produce a "Top Books" list (`popular.pkl`). This gives every user a useful starting point, even without any personalization data.

### 3. Collaborative Filtering Model
Constructed a user–book pivot table (`pt.pkl`) representing rating patterns across users, then computed pairwise **cosine similarity** between books (`similarity_scores.pkl`). For any given book, the system retrieves the most similar titles based on how users rated them — a standard item-based collaborative filtering approach.

### 4. Backend Application
Built a **Flask** application (`app.py`) that loads the precomputed models at startup and exposes routes for:
- the home/top books page, and
- the recommendation search and results page.

HTML templates (Jinja2) handle the presentation layer.

### 5. Deployment
Configured the project for production using a `Procfile` and deployed it on **PythonAnywhere**, turning a local script into a live, shareable web application.

---

## Tech Stack

| Layer | Tools / Technologies |
|---|---|
| Language | Python |
| Data Processing & ML | Pandas, NumPy, scikit-learn |
| Backend | Flask |
| Frontend | HTML, CSS (Jinja2 templates) |
| Model Persistence | Pickle |
| Deployment | PythonAnywhere |

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![PythonAnywhere](https://img.shields.io/badge/PythonAnywhere-1D9FD7?style=for-the-badge&logo=python&logoColor=white)

---

## Project Structure

Book_Reccommender/

│

├── app.py                          # Flask application & routes

├── book-recommender-system.ipynb   # Data cleaning, EDA, model building

├── books.pkl                       # Book metadata

├── popular.pkl                     # Precomputed popularity rankings

├── pt.pkl                          # User–book pivot table

├── similarity_scores.pkl           # Cosine similarity matrix

├── templates/                      # HTML templates

├── requirements.txt

└── Procfile                        # Deployment configuration

---

## Running Locally

```bash
git clone https://github.com/nihal00753/Book_Reccommender.git
cd Book_Reccommender
pip install -r requirements.txt
python app.py
```

The app will be available at `http://127.0.0.1:5000/`

---

## Potential Future Enhancements

- Hybrid recommendations incorporating book genres/descriptions (content-based filtering)
- Improved frontend design and UX
- User accounts with personalized recommendation history
- Better handling of new/low-data books (cold-start improvements)

---

## Live Demo

👉 [https://nihal0753.pythonanywhere.com/](https://nihal0753.pythonanywhere.com/)
