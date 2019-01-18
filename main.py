import requests
import json
import pandas
from dateutil.parser import parse
from sklearn.neighbors import KNeighborsClassifier

API_KEY = "39cdec2682946ef94e836b69ca784765"
movies_info = ['film_id', 'remote_film_id', 'name', 'budget', 'popularity',
               'release_date', 'revenue', 'vote_average', 'vote_count']

names = ['id', 'user_id', 'film_id', 'rate']

genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
          'Fantasy', 'History', 'Horror', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie',
          'Thriller', 'War', 'Western', 'Music']


def show_json(value):
    print(json.dumps(value, indent=4, sort_keys=True))


def get_movie_link_by_id(id):
    return "https://api.themoviedb.org/3/movie/{}?api_key={}".format(id, API_KEY)


def create_movies_database():
    data = pandas.read_csv(
        'data/movie.csv',
        header=None,
        names=movies_info + genres,
        delimiter=';',
        index_col='film_id',
        dtype='object',
    )

    for index, row in data.iterrows():
        print(str(index) + ". Film id - {}".format(row['remote_film_id']))
        r = requests.get(get_movie_link_by_id(row['remote_film_id']))
        if r.status_code == 200:
            json = r.json()
            data.at[index, 'budget'] = json['budget']
            data.at[index, 'popularity'] = json['popularity']
            data.at[index, 'revenue'] = json['revenue']
            data.at[index, 'vote_average'] = json['vote_average']
            data.at[index, 'vote_count'] = json['vote_count']
            set_film_genre(data, index, json['genres'])
            set_film_year(data, index, json['release_date'])
        else:
            print("Film not found")

    data = data.fillna(0)
    data[list(genres)].astype(int)
    print(data.head())
    data.to_csv('movies.csv', sep=';')


def set_film_genre(data, index, genres):
    for genre in genres:
        data.at[index, genre['name']] = 1


def set_film_year(data, index, date):
    parsed = parse(date)
    data.at[index, 'release_date'] = parsed.year


def join_train_with_films():
    train = pandas.read_csv(
        'data/train.csv',
        header=None,
        names=names,
        delimiter=';',
        index_col='id',
    )
    films = pandas.read_csv(
        'movies.csv',
        delimiter=';',
        index_col='film_id',
    )
    combine = pandas.merge(train, films, on='film_id')
    combine.to_csv(
        'features.csv',
        sep=';',
        # header=None
    )


def create_features():
    create_movies_database()
    join_train_with_films()


def load_features():
    global features
    features = pandas.read_csv(
        'features.csv',
        delimiter=';',
        index_col='id',
    )


def load_task():
    global task
    task = pandas.read_csv(
        'data/task.csv',
        header=None,
        names=names,
        delimiter=';',
        index_col='id',
    )


def save_as_submission():
    task.rate = task.rate.astype(int)
    print(task.head())
    task.to_csv('submission.csv', sep=';', header=None, )


def start_prediction():
    load_features()
    load_task()
    rate_tasks()
    save_as_submission()


def rate_tasks():
    columns_to_drop = ['rate', 'name', 'user_id', 'film_id', 'remote_film_id']
    for index, row in task.iterrows():
        print(index)
        user_id = task.at[index, 'user_id']
        film_id = task.at[index, 'film_id']
        rated_films = features[features.user_id == user_id]

        films_features = rated_films.drop(columns_to_drop, axis=1)
        rates = rated_films['rate']

        nbrs = KNeighborsClassifier(n_neighbors=90)  # 30 was also ok
        nbrs.fit(films_features, rates)
        film = features[features.film_id == film_id]
        film_features = film.drop(columns_to_drop, axis=1)
        prediction = nbrs.predict(film_features.head(1))
        task.at[index, 'rate'] = prediction[0]


def start():
    # create_features()
    start_prediction()


if __name__ == "__main__":
    start()
