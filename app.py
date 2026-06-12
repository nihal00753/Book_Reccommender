from flask import Flask, render_template, request
import pickle
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

app = Flask(__name__)

# --- NLP enhancement: TF-IDF based content similarity (title + author) ---
# Used as a fallback when a book isn't in the collaborative-filtering table
content_df = books.drop_duplicates('Book-Title').reset_index(drop=True)
content_df['content'] = content_df['Book-Title'] + ' ' + content_df['Book-Author'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(content_df['content'])
title_to_idx = {title: i for i, title in enumerate(content_df['Book-Title'])}


def get_close_match(user_input, choices):
    """Fuzzy match user input against known titles (handles typos/case)."""
    matches = difflib.get_close_matches(user_input, choices, n=1, cutoff=0.6)
    return matches[0] if matches else None


def content_based_recommend(book_title, n=4):
    idx = title_to_idx.get(book_title)
    if idx is None:
        return []
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sims.argsort()[::-1][1:n+1]
    data = []
    for i in similar_idx:
        row = content_df.iloc[i]
        data.append([row['Book-Title'], row['Book-Author'], row['Image-URL-M']])
    return data


@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/recommend_books', methods=['post'])
def recommend():
    user_input = request.form.get('user_input', '').strip()
    message = None
    data = []

    # Try exact match first
    matched_title = user_input if user_input in pt.index else None

    # Fuzzy match if exact fails (handles typos / wrong case)
    if matched_title is None:
        matched_title = get_close_match(user_input, list(pt.index))

    if matched_title:
        index = np.where(pt.index == matched_title)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])),
                                key=lambda x: x[1], reverse=True)[1:5]

        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
            data.append(item)

        if matched_title.lower() != user_input.lower():
            message = f"Showing results for '{matched_title}'"
    else:
        # Fallback: content-based (TF-IDF) recommendation using overall book list
        content_match = get_close_match(user_input, list(content_df['Book-Title']))
        if content_match:
            data = content_based_recommend(content_match)
            message = f"'{user_input}' not found in user ratings — showing similar books to '{content_match}' based on title/author"
        else:
            message = f"No matches found for '{user_input}'. Try another title."

    return render_template('recommend.html', data=data, message=message, searched=user_input)


if __name__ == '__main__':
    app.run(debug=True)