import numpy as np
from datetime import datetime


class Movie:
    def __init__(self, title="", year=0, runtime=0):
        if runtime < 0:
            runtime = 0
        self.title = title
        self.year = year
        self.runtime = runtime

    def __repr__(self):
        return(f"{self.title}({self.year}) - {self.runtime} mins")

    def get_runtime(self):
        hours = round(self.runtime / 60)
        mins = self.runtime - (hours * 60)
        return hours, mins


def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    # random = Random()

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100

        # Lets have the views range from 100-10000
        views = np.random.randint(100, 10000)
        stars = np.random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array

def create__movie_list(num_movies=5):
    return [ Movie() for _ in range(num_movies) ]

if __name__ == "__main__":
    movies = create__movie_list()

    long_movies = [ movie for movie in movies if movie.runtime > 150 ]

    stars = []
    for movie in movies:
        temp_dict = dict()
        temp_dict['title'] = movie.title
        temp_dict['stars'] = np.random.rand(0, 5)

        stars.append(temp_dict)

    array = get_movie_data()
    print(array.shape)
    print(array, "\n")
    print(array[:2], "\n")
    print(array[:, [-2,-1]], "\n")

    new_list = array[:,2].tolist()
    print(new_list)
