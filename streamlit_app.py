# import streamlit as st
# import pandas as pd
# from scipy.sparse import csr_matrix
# from sklearn.neighbors import NearestNeighbors
# from functions import sparse_matrix, similar_users_suggestions, similar_books


# # Configuration
# st.set_page_config(layout='wide', initial_sidebar_state='expanded')
# with open('style.css') as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# # Datasets
# ratings = pd.read_csv('book_crossing/book_crossing/book_ratings.dat', delimiter = '\t')
# history = pd.read_csv('book_crossing/book_crossing/book_history.dat', delimiter = '\t') # Books and readers history.
# items = pd.read_csv('book_crossing/book_crossing/items_info_clean.dat', delimiter = '\t', on_bad_lines='skip')  # list of books and Ids (Primary id key)


# # Book mapping
# book_titles = dict(zip(items['Book_ID'], items['Book-Title']))
# inv_book_titles = dict(zip(items['Book-Title'], items['Book_ID']))


# # collecting users input
# available_book_list = [book_titles[i] for i in ratings['item'].unique()]
# # user_item_list = []
# user_rating_list = []


# # Streamlit text sections, header, tite, sub headers
# st.title('`Book Recommendation Application`')
# k = st.slider('How many book recommendations do you want?:')
# st.write('This is a book recommendation application that suggests any number of books you want. You can request suggestions /' \
# 'based on the books you have read in the past or based on a perticular book you want to read something similar to again. For /' \
# 'suggestions based on the books you have read in the past, use the User based suggestion section, for similar books, use the /' \
# 'Book based suggestion section')

# st.write('Please note that the dataset this application was built upon was collected in 2004 so more recent books would not appear/' \
# ' here. However, the dataset contains over 16,000 books so, dig deep, I am sure you will find something you have read here.')



# st.subheader('User Based Suggestions')
# st.write('This section creates a temporary user persona for you then suggests books similar users have enjoyed in the past.')
# st.write('Select as many books that you have read and rate them from 1 - 10. Then select how many book suggestions you want')


# selected_books = st.multiselect('Search for books you have read in the past:', options=available_book_list)
# for book in selected_books:
#     user_rated = st.number_input(f'I will give {book} a rating of :', min_value=1, max_value=10, step=1)
#     user_rating_list.append(user_rated)



# book_suggestions = similar_users_suggestions(selected_books, user_rating_list, ratings, 'user', k)

# st.write('Based on the books you have rated, I suggest you read the following books:')

# for i in range(len(book_suggestions)):
#     st.write(f'{i+1}. {book_suggestions[i]}')

# st.subheader('Book Based Suggestions')
# st.write('This section suggests book similar to the your input. The approach used is item based collaborative /' \
# ' filtering so the suggestions might not always be from the same genre.')

# book = st.selectbox('What book do you want to see suggestions for?:', available_book_list)
# # k_book = st.slider('How many book recommendations do you want?:')

# book_suggestions = similar_books(book, ratings, k)

# st.write('Based on the book you chose, I suggest you read the following books:')

# for i in range(len(book_suggestions)):
#     st.write(f'{i+1}. {book_suggestions[i]}')

# st.markdown('---')
# st.markdown('This application was created by Abatan Ayodeji (Agroall).')


import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from functions import sparse_matrix, similar_users_suggestions, similar_books

# Page Configuration
st.set_page_config(
    page_title="Book Recommendation System",
    layout='wide',
    initial_sidebar_state='expanded'
)

# Load custom CSS
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load datasets
ratings = pd.read_csv('book_crossing/book_crossing/book_ratings.dat', delimiter='\t')
history = pd.read_csv('book_crossing/book_crossing/book_history.dat', delimiter='\t')
items = pd.read_csv('book_crossing/book_crossing/items_info_clean.dat', delimiter='\t', on_bad_lines='skip')

# Book title mapping
book_titles = dict(zip(items['Book_ID'], items['Book-Title']))
inv_book_titles = dict(zip(items['Book-Title'], items['Book_ID']))
available_book_list = sorted([book_titles[i] for i in ratings['item'].unique() if i in book_titles])

# Title and description
st.title("📚 Book Recommendation App")
st.markdown("""
Welcome to the **Book Recommendation System**!  
Get book suggestions based on your past reading preferences or find books similar to one you already love.

**Note**: The dataset used is from 2004 and includes over **16,000 books**. While you may not find newer releases here, there's a treasure trove of classics to explore!
""")

st.markdown("---")

# Sidebar for general settings
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.header("🎛️ General Settings")
k = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10)

# --- USER-BASED SUGGESTIONS ---
st.subheader("👤 User-Based Recommendations")
st.markdown("""
Select books you've read in the past and rate them on a scale of 1 to 10.  
We'll create a temporary user profile and find people with similar tastes to suggest new books you might enjoy.
""")

selected_books = st.multiselect("🔎 Search & select books you've read:", options=available_book_list)
user_rating_list = []
for book in selected_books:
    rating = st.slider(f"Rate **{book}**", 1, 10, 7)
    user_rating_list.append(rating)

if selected_books and user_rating_list:
    with st.spinner("Finding similar readers and their favorite books..."):
        book_suggestions = similar_users_suggestions(
            selected_books, user_rating_list, ratings, 'user', k
        )
    st.success("Here are your personalized book suggestions:")
    for i, suggestion in enumerate(book_suggestions):
        st.markdown(f"**{i+1}. {suggestion}**")
else:
    st.info("Select and rate at least one book to get user-based recommendations.")

st.markdown("---")

# --- BOOK-BASED SUGGESTIONS ---
st.subheader("📖 Book-Based Recommendations")
st.markdown("""
Looking for books similar to a specific one you loved?  
We'll use collaborative filtering to find items often liked together, even if they're from different genres.
""")

selected_book = st.selectbox("📘 Choose a book:", options=available_book_list)

if selected_book:
    with st.spinner(f"Finding books similar to **{selected_book}**..."):
        similar_books_list = similar_books(selected_book, ratings, k)
    st.success("Readers who liked that book also enjoyed:")
    for i, title in enumerate(similar_books_list, 1):
        st.markdown(f"**{i}. {title}**")

st.markdown("---")
st.caption("Developed with ❤️ by Abatan Ayodeji (Agroall) • Built using Streamlit")

