import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv('book_crossing/book_crossing/book_ratings.dat', delimiter = '\t')
history = pd.read_csv('book_crossing/book_crossing/book_history.dat', delimiter = '\t') # Books and readers history.
items = pd.read_csv('book_crossing/book_crossing/items_info_clean.dat', delimiter = '\t', on_bad_lines='skip')  # list of books and Ids (Primary id key)

# Book mapping
book_titles = dict(zip(items['Book_ID'], items['Book-Title']))
inv_book_titles = dict(zip(items['Book-Title'], items['Book_ID']))

# Sparse matrix function
def sparse_matrix(df, user_id_name, item_id_name, rating_column_name):
    """
    Create a sparse user-item matrix from a dataframe.
    Returns the item and user matrices, mapping dictionaries, and inverse mappings.
    """

    # Factorize ensures a consistent mapping
    df[user_id_name], user_idx = pd.factorize(df[user_id_name])
    df[item_id_name], item_idx = pd.factorize(df[item_id_name])

    user_map = dict(zip(user_idx, range(len(user_idx))))
    item_map = dict(zip(item_idx, range(len(item_idx))))
    inv_user_map = dict(enumerate(user_idx))
    inv_item_map = dict(enumerate(item_idx))

    user_index = df[user_id_name]
    item_index = df[item_id_name]
    ratings = df[rating_column_name].astype(float)

    item_matrix = csr_matrix((ratings, (item_index, user_index)))
    user_matrix = csr_matrix((ratings, (user_index, item_index)))

    return item_matrix, user_matrix, user_map, item_map, inv_user_map, inv_item_map


# User similarity function
def similar_users_suggestions(user_item_list, user_rating_list, df, user_id, n_similar_users, metric='cosine'):
    """
    Generates personalized book recommendations by identifying similar users using K-Nearest Neighbors (KNN).

    This function creates a synthetic user based on their rated books and computes their similarity
    to existing users in the dataset. It then recommends the top-rated book from each of the most similar users.

    Args:
        user_item_list (list of str): A list of book titles the new user has rated.
        user_rating_list (list of float): A list of ratings corresponding to the `user_item_list`.
        df (pd.DataFrame): The user-item ratings dataframe with columns ['user', 'item', 'rating'].
        user_id (str): The name of the user ID column in `df`.
        n_similar_users (int): Number of similar users to retrieve.
        metric (str, optional): Similarity metric to use for KNN. Defaults to 'cosine'.

    Returns:
        list of str: A list of top-rated book titles from each of the `n_similar_users` most similar users.

    Notes:
        - `book_titles` and `inv_book_titles` must be defined globally:
            - `book_titles` maps book ID to title
            - `inv_book_titles` maps title to book ID
        - If a neighbor has no rated books, they are skipped.
    """

    # Step 1: Create a new user ID and copy dataframe to avoid mutation
    user = df[user_id].max() + 1
    df = df.copy()

    # Step 2: Map book titles to book IDs
    user_item_id = [inv_book_titles[i] for i in user_item_list]

    # Step 3: Add the new user's ratings to the DataFrame
    for index, item_id in enumerate(user_item_id):
        new_row = {'user': user, 'item': item_id, 'rating': user_rating_list[index]}
        df.loc[len(df)] = new_row

    # Step 4: Create new sparse matrices from updated df
    item_matrix, user_matrix, user_map, item_map, inv_user_map, inv_item_map = sparse_matrix(
        df, 'user', 'item', 'rating'
    )

    # Step 5: Fit KNN to user matrix
    k = n_similar_users + 1  # +1 to include the new user
    # user_loc = user_map[user]
    user_vector = user_matrix[user]

    KNN = NearestNeighbors(n_neighbors=k, algorithm='brute', metric=metric)
    KNN.fit(user_matrix)
    neighbours = KNN.kneighbors(user_vector.reshape(1, -1), return_distance=False)

    # Step 6: Get similar user IDs, skipping the first (which is the new user)
    neighbouring_user_ids = [inv_user_map[n] for n in neighbours.flatten()[1:]]

    # Step 7: Get top-rated books from each neighbour
    book_suggestions = []
    for uid in neighbouring_user_ids:
        user_books = df[df['user'] == uid]
        if not user_books.empty:
            best_book = user_books.sort_values(by='rating', ascending=False)['item'].head(1)
            book_id = int(best_book.values[0])
            book_suggestions.append(book_titles[book_id])

    return book_suggestions


# item similarity fuction
def similar_books(book, df, k, metric='cosine'):
    """
    Finds k books similar to the given book based on item-item collaborative filtering.

    Args:
        book (str): The title of the book for which to find similar books.
        df (pd.DataFrame): Ratings DataFrame with columns ['user', 'item', 'rating'].
        k (int): Number of similar books to return.
        metric (str, optional): Distance metric for KNN. Default is 'cosine'.

    Returns:
        List[str]: List of book titles similar to the input book.
    
    Notes:
        - `book_titles` and `inv_book_titles` must be defined globally.
    """
    # Convert book title to item ID
    book_id = inv_book_titles[book]

    # Build item-user matrix
    item_matrix, user_matrix, user_map, item_map, inv_user_map, inv_item_map = sparse_matrix(
        df, 'user', 'item', 'rating'
    )

    # Get matrix row index for the book
    book_loc = item_map[book_id]
    book_vector = item_matrix[book_loc]

    # Fit KNN
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric=metric)
    knn.fit(item_matrix)

    # Find neighbors
    neighbours = knn.kneighbors(book_vector.reshape(1, -1), return_distance=False).flatten()

    # Skip the first neighbor (it will be the book itself)
    similar_ids = [inv_item_map[idx] for idx in neighbours[1:]]

    # Map item IDs back to book titles
    return [book_titles[item_id] for item_id in similar_ids]

