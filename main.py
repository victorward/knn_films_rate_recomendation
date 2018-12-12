import requests
import json
import pandas

API_KEY = "39cdec2682946ef94e836b69ca784765"
names = ['id', 'user_id', 'film_id', 'rate']


def show_json(value):
    print(json.dumps(value, indent=4, sort_keys=True))


def get_movie_link_by_id(id):
    return "https://api.themoviedb.org/3/movie/{}?api_key={}".format(id, API_KEY)


def create_movies_database():
    train = pandas.read_csv(
        'data/train.csv',
        header=None,
        names=names,
        delimiter=';',
        index_col='id'
    )
    task = pandas.read_csv(
        'data/task.csv',
        header=None,
        names=names,
        delimiter=';',
        index_col='id'
    )
    films = pandas.concat([train, task])
    films_id = films['film_id']

    movies_db = pandas.DataFrame(columns=['id', 'budget'])
    for index in films_id:
        print("Film id - {}".format(index))
        r = requests.get(get_movie_link_by_id(index))
        json = r.json()
        show_json(json)
        movies_db.append({'id': index, 'budget': json['budget']}, ignore_index=True)
        print("Film budget - {}".format(json['budget']))
        if index == 3:
            break

    print(movies_db.head())


def start():
    create_movies_database()


if __name__ == "__main__":
    start()
