# Sparse matrix function
def sparse_matrix(df, user_id_name, item_id_name, rating_column_name):
    """
    This function helps to create a sparse matrix (a matrix largely populated by zeroes) from your ratings dataset.
    
    Parameters:
    df: Pandas DataFrame
    user_id_name (str): Column name of the user ID.
    item_id_name (str): Column name of the item ID.
    rating_column_name (str): Column name of the ratings.
    
    Returns:
    - matrix: the resulting sparse matrix
    - user_map: a dictionary mapping original user IDs to matrix row indices
    - item_map: a dictionary mapping original item IDs to matrix column indices
    - inv_user_map: inverse mapping from row indices back to original user IDs
    - inv_item_map: inverse mapping from column indices back to original item IDs
    """
    
    # Stores the number of unique users and items in the dataset.
    # This is used in determining the shape of the sparse matrix.
    n_users = df[user_id_name].nunique()
    n_items = df[item_id_name].nunique()

    # Creates a map of the user/item IDs to new sequential indices.
    user_map = dict(zip(df[user_id_name].unique(), list(range(n_users))))
    item_map = dict(zip(df[item_id_name].unique(), list(range(n_items))))

    # Creates the inverse map of user/item IDs for referencing purposes.
    inv_user_map = dict(zip(list(range(n_users)), df[user_id_name].unique()))
    inv_item_map = dict(zip(list(range(n_items)), df[item_id_name].unique()))

    # Applies the new IDs to create index lists.
    user_index = [user_map[i] for i in df[user_id_name]]
    item_index = [item_map[i] for i in df[item_id_name]]

    # Creates the sparse matrix using Compressed Sparse Row (csr) format.
    item_matrix = csr_matrix((df[rating_column_name], (item_index, user_index)), shape=(n_items, n_users))
    user_matrix = csr_matrix((df[rating_column_name], (user_index, item_index)), shape=(n_users, n_items))

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
    user_loc = user_map[user]
    user_vector = user_matrix[user_loc]

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

