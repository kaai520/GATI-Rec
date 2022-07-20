import numpy as np
import scipy.sparse as sp
import pickle as pkl
import h5py
import os
import pandas as pd
from zipfile import ZipFile
from urllib.request import urlopen
import random

try:
    from BytesIO import BytesIO
except ImportError:
    from io import BytesIO


def download_dataset(dataset, files, data_dir):
    """ Downloads dataset if files are not present. """

    if not np.all([os.path.isfile(data_dir + f) for f in files]):
        url = "http://files.grouplens.org/datasets/movielens/" + dataset.replace('_', '-') + '.zip'
        request = urlopen(url)

        print('Downloading %s dataset' % dataset)

        if dataset in ['ml_100k', 'ml_1m']:
            target_dir = os.path.abspath(os.path.join(data_dir, '..', dataset.replace('_', '-')))
        elif dataset == 'ml_10m':
            target_dir = os.path.abspath(os.path.join(data_dir, '..', 'ml-10M100K'))
        else:
            raise ValueError('Invalid dataset option %s' % dataset)
        with ZipFile(BytesIO(request.read())) as zip_ref:
            zip_ref.extractall(os.path.abspath(os.path.join(data_dir, '..')))

        os.rename(target_dir, data_dir)


def load_official_trainvaltest_split(root, dataset, rating_map, use_feature=False):
    '''
    Loads official train/test split and uses 10% of training samples for validation
    :param dataset: ml_100k
    :return:
    '''
    sep = '\t'
    # Check if files exist and download otherwise
    files = ['/u1.base', '/u1.test', '/u.item', '/u.user']
    fname = dataset
    data_dir = os.path.join(root, dataset)

    download_dataset(fname, files, data_dir)

    dtypes = {'u_nodes': np.int64, 'v_nodes': np.int64, 'ratings': np.float32, 'timestamp': np.float64}

    filename_train = os.path.join(data_dir, 'u1.base')
    filename_test = os.path.join(data_dir, 'u1.test')

    data_train = pd.read_csv(filename_train, sep=sep, header=None, names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'],
                             dtype=dtypes)

    data_test = pd.read_csv(filename_test, sep=sep, header=None, names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'],
                            dtype=dtypes)

    data_total = pd.concat([data_train, data_test])
    unique_u_nodes = data_total['u_nodes'].unique()
    unique_v_nodes = data_total['v_nodes'].unique()
    num_users = len(unique_u_nodes)
    num_items = len(unique_v_nodes)
    u_nodes_map = dict(zip(unique_u_nodes, range(num_users)))
    v_nodes_map = dict(zip(unique_v_nodes, range(num_items)))

    data_train['u_nodes'] = data_train['u_nodes'].map(u_nodes_map)
    data_train['v_nodes'] = data_train['v_nodes'].map(v_nodes_map)

    data_test['u_nodes'] = data_test['u_nodes'].map(u_nodes_map)
    data_test['v_nodes'] = data_test['v_nodes'].map(v_nodes_map)

    data_train['rating_categories'] = data_train['ratings'].map(rating_map)
    # data_test['rating_categories'] = data_test['ratings'].map(rating_map)

    rating_mx_train = sp.csr_matrix((data_train['rating_categories'], [data_train['u_nodes'], data_train['v_nodes']]),
                                    shape=(num_users, num_items))

    if use_feature:
        # movie feature
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown',
                         'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
                         'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                         'Western']
        movie_df = pd.read_csv(os.path.join(data_dir, 'u.item'), sep=r'|', header=None, names=movie_headers,
                               engine='python')
        genres_headers = ['movie id', 'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime',
                          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                          'Sci-Fi', 'Thriller', 'War', 'Western']

        movie_df['movie id'] = movie_df['movie id'].map(v_nodes_map)
        genres_df = movie_df[genres_headers]
        items_df = genres_df.sort_values(by=['movie id']).reset_index(drop=True)

        # user feature
        users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
        users_df = pd.read_csv(os.path.join(data_dir, 'u.user'), sep=r'|', header=None, names=users_headers,
                               engine='python')

        unique_ages = users_df['age'].unique()
        cut_ages = np.array(pd.cut(unique_ages, 6, labels=list(range(6))))
        ages_map = dict(zip(unique_ages, cut_ages))

        unique_occupation = users_df['occupation'].unique()
        occupation_map = dict(zip(unique_occupation, range(len(unique_occupation))))

        unique_gender = users_df['gender'].unique()
        gender_map = dict(zip(unique_gender, range(len(unique_gender))))

        users_df['user id'] = users_df['user id'].map(u_nodes_map)
        users_df['age'] = users_df['age'].map(ages_map)
        users_df['gender'] = users_df['gender'].map(gender_map)
        users_df['occupation'] = users_df['occupation'].map(occupation_map)
        users_df = users_df[['user id', 'age', 'gender', 'occupation']]
        return rating_mx_train, data_train, data_test, users_df, items_df

    return rating_mx_train, data_train, data_test


def load_data_monti(root, dataset, rating_map, use_feature=False):
    data_dir = os.path.join(root, dataset, 'training_test_dataset.mat')
    db = h5py.File(data_dir, 'r')
    # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
    M = np.asarray(db['M']).astype(np.float32).T
    train_M = np.asarray(db['Otraining']).astype(np.float32).T
    test_M = np.asarray(db['Otest']).astype(np.float32).T
    
    num_users = M.shape[0]
    num_items = M.shape[1]
    

    train_idx = np.where(train_M)
    test_idx = np.where(test_M)
    data_train = {'u_nodes': train_idx[0], 'v_nodes': train_idx[1], 'ratings': M[train_idx]}
    data_train = pd.DataFrame(data_train)
    data_test = {'u_nodes': test_idx[0], 'v_nodes': test_idx[1], 'ratings': M[test_idx]}
    data_test = pd.DataFrame(data_test)

    data_train['rating_categories'] = data_train['ratings'].map(rating_map)
    rating_mx_train = sp.csr_matrix((data_train['rating_categories'], [data_train['u_nodes'], data_train['v_nodes']]),
                                    shape=(num_users, num_items))
    if use_feature:
        if dataset == 'flixster':
            Wrow = np.asarray(db['W_users']).astype(np.float32).T
            Wcol = np.asarray(db['W_movies']).astype(np.float32).T
            u_features = Wrow
            v_features = Wcol
        elif dataset == 'douban':
            Wrow = np.asarray(db['W_users']).astype(np.float32).T
            u_features = Wrow
            v_features = np.eye(num_items).astype(np.float32)
        elif dataset == 'yahoo_music':
            Wcol = np.asarray(db['W_tracks']).astype(np.float32).T
            u_features = np.eye(num_users).astype(np.float32)
            v_features = Wcol
        return rating_mx_train, data_train, data_test, u_features, v_features
    db.close()
    return rating_mx_train, data_train, data_test


def load_split_data(root, dataset, rating_map, seed=1234, use_feature=False):
    sep = r'\:\:'
    # Check if files exist and download otherwise
    files = ['/ratings.dat', '/movies.dat', '/users.dat']
    fname = dataset
    data_dir = os.path.join(root, dataset)

    download_dataset(fname, files, data_dir)
    dtypes = {'u_nodes': np.int64, 'v_nodes': np.int64, 'ratings': np.float32, 'timestamp': np.float64}
    filename = os.path.join(data_dir, 'ratings.dat')
    data = pd.read_csv(filename, sep=sep, header=None, names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'],
                       dtype=dtypes, engine='python')
    # shuffle here like cf-nade paper with python's own random class
    # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
    data_array = data.values.tolist()
    random.seed(seed)
    random.shuffle(data_array)
    data_array = np.array(data_array)
    u_nodes = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])

    unique_u_nodes = np.unique(u_nodes)
    unique_v_nodes = np.unique(v_nodes)
    num_users = len(unique_u_nodes)
    num_items = len(unique_v_nodes)
    u_nodes_map = dict(zip(unique_u_nodes, range(num_users)))
    v_nodes_map = dict(zip(unique_v_nodes, range(num_items)))

    num_test = int(np.ceil(ratings.shape[0] * 0.1))
    num_train = ratings.shape[0] - num_test

    u_nodes_train = pd.Series(u_nodes[:num_train]).map(u_nodes_map)
    u_nodes_test = pd.Series(u_nodes[num_train:]).map(u_nodes_map)
    v_nodes_train = pd.Series(v_nodes[:num_train]).map(v_nodes_map)
    v_nodes_test = pd.Series(v_nodes[num_train:]).map(v_nodes_map)
    ratings_train = ratings[:num_train]
    ratings_test = ratings[num_train:]

    data_train = pd.DataFrame({'u_nodes': u_nodes_train, 'v_nodes': v_nodes_train, 'ratings': ratings_train})
    data_test = pd.DataFrame({'u_nodes': u_nodes_test, 'v_nodes': v_nodes_test, 'ratings': ratings_test})
    data_train['rating_categories'] = data_train['ratings'].map(rating_map)
    rating_mx_train = sp.csr_matrix((data_train['rating_categories'], [data_train['u_nodes'], data_train['v_nodes']]),
                                    shape=(num_users, num_items))
    if use_feature:
        # user feature
        users_headers = ['user id', 'gender', 'age', 'occupation', 'zip code']
        u_df = pd.read_csv(os.path.join(data_dir, 'users.dat'), sep='::', header=None, names=users_headers,
                           engine='python')

        gender_map = {'M': 0, 'F': 1}
        unique_age = u_df['age'].unique()
        age_map = dict(zip(sorted(unique_age), range(len(unique_age))))
        users_df = pd.DataFrame({'user id': u_df['user id'].map(u_nodes_map), 'age': u_df['age'].map(age_map),
                                 'gender': u_df['gender'].map(gender_map), 'occupation': u_df['occupation']})
        # movie feature
        movies_headers = ['movie id', 'title', 'genres']
        movie_dtypes = {'movie id': np.int64, 'title': str, 'genres': str}
        movie_df = pd.read_csv(os.path.join(data_dir, 'movies.dat'), sep='::', header=None, names=movies_headers,
                               engine='python', dtype=movie_dtypes)
        v_nodes_set = set(unique_v_nodes)
        movie_id_arr = movie_df['movie id'].values
        save_idx = [i for i in range(len(movie_id_arr)) if movie_id_arr[i] in v_nodes_set]
        movie_df = movie_df.iloc[save_idx, :]
        movie_df['movie id'] = movie_df['movie id'].map(v_nodes_map)
        movie_df = movie_df.reset_index(drop=True)

        items_df_arr = np.zeros((num_items, 19), dtype=np.int64)  # (3706, 19)
        items_df_arr[:, 0] = range(num_items)

        genres_list = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama',
                       'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                       'Western']

        genres_map = dict(zip(genres_list, range(len(genres_list))))
        for i in range(num_items):
            movie_genres = movie_df.iloc[i, 2]
            movie_genres_list = movie_genres.split('|')
            for genre in movie_genres_list:
                header_idx = genres_map[genre] + 1
                items_df_arr[i, header_idx] = 1

        genres_headers = ['movie id', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
                          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                          'Sci-Fi', 'Thriller', 'War', 'Western']
        items_df = pd.DataFrame(items_df_arr, columns=genres_headers)
        return rating_mx_train, data_train, data_test, users_df, items_df

    return rating_mx_train, data_train, data_test


if __name__ == '__main__':
    # rating_mx_train, data_train, data_test = load_data_monti('../raw_data', 'yahoo_music', {x: int(x) for x in np.arange(1., 100.01, 1)})
    # print(data_train)
    # print(np.unique(data_train['rating_categories']))
    # print(np.unique(data_test['ratings']))

    rating_mx_train, data_train, data_test = load_split_data('../raw_data', 'ml_1m',
                                                             {x: int(x) for x in np.arange(1., 5.01)})
    print(data_train)
    print(data_test)
    print(rating_mx_train.data)
