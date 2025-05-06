import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from functions import sparse_matrix, similar_users_suggestions, similar_books


# Configuration
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Datasets
ratings = pd.read_csv('book_crossing/book_crossing/book_ratings.dat', delimiter = '\t')
history = pd.read_csv('book_crossing/book_crossing/book_history.dat', delimiter = '\t') # Books and readers history.
items = pd.read_csv('book_crossing/book_crossing/items_info_clean.dat', delimiter = '\t', on_bad_lines='skip')  # list of books and Ids (Primary id key)


# Book mapping
book_titles = dict(zip(items['Book_ID'], items['Book-Title']))
inv_book_titles = dict(zip(items['Book-Title'], items['Book_ID']))


# collecting users input
available_book_list = [book_titles[i] for i in ratings['item'].unique()]
# user_item_list = []
user_rating_list = []


# Streamlit text sections, header, tite, sub headers
st.title('`Book Recommendation Application`')
st.write('This is a book recommendation application that suggests any number of books you want. You can request suggestions /' \
'based on the books you have read in the past or based on a perticular book you want to read something similar to again. For /' \
'suggestions based on the books you have read in the past, use the User based suggestion section, for similar books, use the /' \
'Book based suggestion section')

st.write('Please note that the dataset this application was built upon was collected in 2004 so more recent books would not appear/' \
' here. However, the dataset contains over 16,000 books so, dig deep, I am sure you will find something you have read here.')



st.subheader('User Based Suggestions')
st.write('This section creates a temporary user persona for you then suggests books similar users have enjoyed in the past.')
st.write('Select as many books that you have read and rate them from 1 - 10. Then select how many book suggestions you want')


selected_books = st.multiselect('Search for books you have read in the past:', options=available_book_list)
for book in selected_books:
    user_rated = st.number_input(f'I will give {book} a rating of :', min_value=1, max_value=10, step=1)
    user_rating_list.append(user_rated)


k = st.slider('How many book recommendations do you want?:')

book_suggestions = similar_users_suggestions(selected_books, user_rating_list, ratings, 'user', k)

st.write('Based on the books you have rated, I suggest you read the following books:')

for i in range(len(book_suggestions)):
    st.write(f'{i+1}. {book_suggestions[i]}')

st.subheader('Book Based Suggestions')
st.write('This section suggests book similar to the your input. The approach used is item based collaborative /' \
' filtering so the suggestions might not always be from the same genre.')

book = st.selectbox('What book do you want to see suggestions for?:', available_book_list)
k = st.slider('How many book recommendations do you want?:')

book_suggestions = similar_books(book, ratings, k)

st.write('Based on the books you have rated, I suggest you read the following books:')

for i in range(len(book_suggestions)):
    st.write(f'{i+1}. {book_suggestions[i]}')
    
st.markdown('---')
st.markdown('This application was created by Abatan Ayodeji (Agroall).')