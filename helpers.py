from sklearn import manifold
from sklearn.metrics import euclidean_distances
import numpy as np

def read_file_and_build_distance(nv,filename):
    with open(filename , mode ="r") as file:
        input_data = [d.strip().split() for d in file.readlines()]
        distances = [[None for j in range(nv)] for i in range(nv)]
        for i in range(nv):
            for j in range(nv):
                distances[i][j] = int(input_data[i][j])
    return distances

def read_file_and_build_distance_(nv,filename):
    with open(filename , mode ="r") as file:
        input_data = [d.strip().split() for d in file.readlines()]
        distances = [[None for j in range(nv)] for i in range(nv)]
        for i in range(nv):
            row = 0
            for j in range(i+1,nv):
                distances[i][j] = int(input_data[i][j])
                distances[j][i] = distances[i][j]
                row += 1 
        for i in range(nv):
            distances[i][i] = - 10 
        # print(distances[-1])
    return distances

def inversion_list_cities(cities,city_1,city_2):
    for k in range(city_1,city_1 +1+ int((city_2 - city_1)/2)):
        tmp = cities[k] 
        cities[k] = cities[city_2 + city_1 -k]
        cities[city_2 + city_1 -k] = tmp
    return cities


def evaluate(distance,cities,size):
    tmp = 0 
    for i in range(size - 1):
        tmp += distance[cities[i]][cities[i+1]]
    tmp += distance[cities[0]][cities[size - 1]]
    return tmp

if __name__ == "__main__":
    nb_villes = 100
    filename = "distances_entre_villes_{}.txt".format(str(nb_villes))
    distances = read_file_and_build_distance(nb_villes,filename)
    for i in range(100):
        data = " ".join(list(map(str,distances[i])))
        print(data)
    # cities =[]
    # score = evaluate(distances,cities,nb_villes)

    # embedding = manifold.MDS(n_components=2)
    # distance_transformed = embedding.fit_transform(np.array(distances))
    # distance_transformed.shape
    # print(distance_transformed)

    # l = [i for i in range(20)]
    # l_r = inversion_list_cities(l,5,10)
    # print(l_r)

    # ll = [i for i in range(20)]
    # ll[5:10+1] = ll[5:10+1][::-1]
    # print(ll)