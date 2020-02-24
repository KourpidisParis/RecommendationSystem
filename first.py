import warnings

import nltk
import pandas as pd
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import jaccard_similarity_score

warnings.simplefilter(action='ignore', category=FutureWarning)


def drop_books(ratings, books):
    books_counts = ratings.groupby('ISBN').bookRating.count().sort_values(ascending=False)
    final_books = pd.merge(books_counts, books, on='ISBN')
    indexBooks = final_books[final_books['bookRating'] < 10].index
    final_books.drop(indexBooks, inplace=True)
    return final_books


def drop_users(ratings, users):
    users_votes = ratings.groupby('userID').ISBN.count().sort_values(ascending=False)
    final_users = pd.merge(users_votes, users, on='userID')
    indexVotes = final_users[final_users['ISBN'] < 5].index
    final_users.drop(indexVotes, inplace=True)
    return final_users


def transform_to_lower_case(final_books):
    final_books['bookTitle'] = final_books['bookTitle'].str.lower()


def identify_tokens(row):
    review = row['bookTitle']
    tokens = nltk.word_tokenize(review)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words


def stem_list(row):
    stemming = PorterStemmer()
    my_list = row['words']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return stemmed_list


def remove_stops(row):
    stops = set(stopwords.words("english"))
    my_list = row['stemmed_words']
    meaningful_words = [w for w in my_list if not w in stops]
    return meaningful_words


def make_a_list_of_lists_one_list(list):
    final_list = []
    for sub_list in list:
        for item in sub_list:
            final_list.append(item)

    return final_list


def calculate_jaccard_similarity(temp_kewywords, temp_authors, temp_years, temp):
    score = []
    for i in range(len(temp)):
        keys = temp.iloc[i]["meaningful_words"]
        author = temp.iloc[i]["bookAuthor"]
        year = temp.iloc[i]['yearOfPublication']
        t = jaccard_similarity(temp_kewywords, keys) *0.2# + equals_authors(temp_authors,
        #                                                                     author) * 0.4 + min_difference(temp_years,
        #                                                                                                    year) * 0.4
        score.append(t)
    return score


# Για κάποιο λόγο η έτοιμη συνάρτηση jaccard απο την βιβλιοθηκη sklearn ,δεν παίζει,συνεπώς την δμιούργησα
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    if len(s1.union(s2)) == 0:
        return 0
    else:
        return len(s1.intersection(s2)) / len(s1.union(s2))


def equals_authors(authors, author):
    for a in authors:
        if a == author:
            return 1
        else:
            return 0


def min_difference(years, year):
    array = []
    for y in years:
        t = 1 - (abs(year - y) / 2005)
        array.append(t)
    return min(array)


def calculate_dice_cofficient(temp_kewywords, temp_authors, temp_years, temp):
    score = []
    keys = temp.iloc[i]["meaningful_words"]
    author = temp.iloc[i]["bookAuthor"]
    year = temp.iloc[i]['yearOfPublication']

    t = dice_coefficiency(temp_kewywords, keys) * 0.5 + equals_authors(temp_authors,
                                                                        author) * 0.3 + min_difference(temp_years,
                                                                                                       year) * 0.2
    score.append(t)
    return score

def dice_coefficiency(l1,l2):
    s1 = set(l1)
    s2 = set(l2)
    if len(s1.union(s2)) == 0:
        return 0
    else:
        return 2*len(s1.intersection(s2)) / len(s1.union(s2))


if __name__ == "__main__":
    users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding='latin-1')
    users.columns = ['userID', 'Location', 'Age']

    ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding='latin-1')
    ratings.columns = ['userID', 'ISBN', 'bookRating']

    books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding='latin-1')
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM',
                     'imageUrlL']
    books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'], axis=1, inplace=True)

    final_books = drop_books(ratings, books)
    final_users = drop_users(ratings, users)

    ratings_new = ratings[ratings.ISBN.isin(final_books.ISBN)]
    ratings_new = ratings_new[ratings_new.userID.isin(users.userID)]

    # -----TRANSFORM TO LOWER CASE----
    # transform_to_lower_case(final_books)

    # -----TOKENIZE-----
    i = 0
    for i in range(len(final_books)):
        identify_tokens(final_books.iloc[i])
    final_books['words'] = final_books.apply(identify_tokens, axis=1)

    # -----STEMMING-----
    i = 0
    for i in range(len(final_books)):
        stem_list(final_books.iloc[i])
    final_books['stemmed_words'] = final_books.apply(stem_list, axis=1)

    # -----STOPWORDS REMOVAL-----
    i = 0
    for i in range(len(final_books)):
        remove_stops(final_books.iloc[i])
    final_books['meaningful_words'] = final_books.apply(remove_stops, axis=1)

    # -----RECOMENDED SYSTEM------

    # Επιλέγω ένα χρήστη τυχαία ,απο αυτόυς που απέμειναν μετά τον αποκλεισμό (<5)
    top3 = ratings_new.loc[ratings_new['userID'] == 153662].sort_values(by='bookRating', ascending=False)
    # Απομονώνω τα τρία αντικείμενα  που έχουν την μεγαλύτερη βαθμολογία
    top3 = top3.head(3)

    top3 = pd.merge(top3, final_books, on='ISBN')

    key_words = []
    authors = []
    years = []
    for i in range(len(top3)):
        key_words.append(top3.loc[i]['meaningful_words'])
        authors.append(top3.loc[i]['bookAuthor'])
        years.append(top3.loc[i]['yearOfPublication'])

    print(list(top3))
    print(top3.loc[0]['bookTitle'])
    print(top3.loc[1]['bookTitle'])
    print(top3.loc[2]['bookTitle'])


    # Η λίστα key_words είναι μια λίστα απο λίστες.Την κάνω μια λίστα
    key_words2 = make_a_list_of_lists_one_list(key_words)
    #
    # # Απο το παρακάτω σύνολο δεδομένων θα προτείνουμε στο χρήστη 10 αντικείμενα
    check = ratings_new.loc[ratings_new['userID'] != 153662]
    check1 = pd.merge(check, final_books, on='ISBN')
    print(list(check1))
    # jaccard similarity
    score = calculate_jaccard_similarity(key_words2, authors, years, check1)
    check1["score"] = score
    print(list(check1))
    check1 = check1.sort_values(by='score', ascending=False)
    recommends = check1.head(10)
    print(recommends.bookTitle)