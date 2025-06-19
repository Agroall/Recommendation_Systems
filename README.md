# 📚 Book Recommendation System

A collaborative filtering book recommender system built using Python and Streamlit. 
This project provides user-based and item-based recommendation engines using real-world data from the [Book-Crossing dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).

> ✅ Try the app live: [Click here to get your next book recommendation!](https://get-your-book-recommendations.streamlit.app/)

---

## 🚀 Features

- 📖 **Book-based recommendations** using item-item collaborative filtering  
- 👤 **User-based recommendations** using user similarity scores  
- 📊 Works with cleaned Book-Crossing dataset: 17,000+ books and 270,000+ user interactions  
- 💡 Explains the theory and methods behind recommendation engines  
- 🌐 Deployed using Streamlit Cloud  

---

## 📂 Dataset Used

The app uses a cleaned version of the **Book-Crossing dataset**, which includes:

- **Books** (`items_info_clean.dat`)
- **Ratings** (`book_ratings.dat`)
- **User history** (`book_history.dat`)

Data was cleaned following the MovieLens 100K preprocessing methods.

---

## ⚙️ Installation

To run the app locally:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/recommendation_systems.git
   cd recommendation_systems

2. Install the required packages:
   ```bash
   pip install -r requirements.txt

3. Run the app:
   ```bash
   streamlit run streamlit_app.py


## 📌 How It Works
The app allows you to select and rate books you've read, then recommends similar titles based on other users’ preferences.

You can also get book-based suggestions: select a book you liked and the app finds similar ones.

Uses cosine similarity to compare user or item vectors in a sparse ratings matrix.

## 🧪 Future Improvements
1. Update the Dataset with scraped data from Goodreads.
2. Handling suggestions for users without prior reading experience.

## 📬 Try It Out!
Curious what to read next?
👉 Try the app live: (https://get-your-book-recommendations.streamlit.app/)

## 🧑‍💻 Author
Built by Abatan Ayodeji (Agroall)
Feel free to connect or contribute!

## 📜 License
This project is open source under the MIT License.
