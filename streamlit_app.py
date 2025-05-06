import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from functions import sparse_matrix, similar_users_suggestions, similar_books


# Configuration
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
with open('style.css') as f:
    st.markdown(f'<style>(f.read())</style>', unsafe_allow_html=True)


# Datasets
ratings = pd.read_csv('book_crossing/book_crossing/book_ratings.dat', delimiter = '\t')
history = pd.read_csv('book_crossing/book_crossing/book_history.dat', delimiter = '\t') # Books and readers history.
items = pd.read_csv('book_crossing/book_crossing/items_info_clean.dat', delimiter = '\t', on_bad_lines='skip')  # list of books and Ids (Primary id key)


# Book mapping
book_titles = dict(zip(items['Book_ID'], items['Book-Title']))
inv_book_titles = dict(zip(items['Book-Title'], items['Book_ID']))


# collecting users input
available_book_list = [book_titles[i] for i in ratings['item'].unique()]
user_item_list = []
user_rating_list = []


# Streamlit text sections, header, tite, sub headers





st.markdown('---')
st.markdown('This application was created by Abatan Ayodeji (Agroall).')