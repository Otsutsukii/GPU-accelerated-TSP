import networkx as nx 
from random import choice,randint
import cupy as cp
from numba import cuda,jit,njit,vectorize,guvectorize
import numba
import numpy as np
from numpy import int32,int64
import matplotlib.pyplot as plt

class TabuSearch_GPU():

    def __init__(self,nbiter,dt,nv,filename):
        self.nbiterations = nbiter
        self.current_iter = 0
        self.tabu_duration = dt 
        self.size_solution = nv

        self.distance_matrix = self.read_file_and_build_distance(nv,filename)

        self.current = solution_GPU(nv)
        tmp = self.current.evaluate(self.distance_matrix)
        self.current.fitness = tmp
        
        self.gpu_matrix_indexes = [[] for _ in range(1024)]
        size = round(nv*nv/1024)
        indexe_tmp = 0 
        for i in range(nv):
            for j in range(nv):
                if len(self.gpu_matrix_indexes[indexe_tmp]) == size:
                    indexe_tmp+=1
                else:
                    self.gpu_matrix_indexes[indexe_tmp].append([i,j])


        print("the initial random solution is")
        self.current.show()

        self.tabu_list2 = cuda.to_device(np.array([[-1 for j in range(self.size_solution)] for i in range(self.tabu_duration)]))
        self.tabu_check = cuda.to_device(np.array([0 for j in range(self.tabu_duration)]))

        # self.current.plot_graph(self.distance_matrix)

    def read_file_and_build_distance(self,nv,filename):
        with open(filename , mode ="r") as file:
            input_data = [d.strip().split() for d in file.readlines()]

            distances = [[None for j in range(nv)] for i in range(nv)]
            for i in range(nv):
                row = 0
                for j in range(i+1,nv):
                    distances[i][j] = int(input_data[i][row])
                    distances[j][i] = distances[i][j]
                    row += 1 
            for i in range(nv):
                distances[i][i] = - 10 
        # return numba.cuda.to_device(np.array(distances))
        return distances


    def update_tabu_list_2(self,solution,position):
        for i in range(self.size_solution):
            self.tabu_list2[position][i] = solution.cities[i]
        position += 1 
        if position == self.tabu_duration:
            position = 0 
        return position
    
  
    def neighbors_2opt(self,best_i,best_j, best_vois = 100000):
        
        for i in range(self.size_solution):
            for j in range(i+1,self.size_solution):
                if (((i != 0) or (j != self.size_solution -1 )) and ((i!=0) or (j!=self.size_solution -2))) :
                    self.current.inversion_list_cities(i,j)
                    tmp = self.current.evaluate(self.distance_matrix)
                    self.current.fitness = tmp 
                    # threadsperblock = 50
                    # blockpergrid = (self.tabu_duration + (threadsperblock - 1)) // threadsperblock
                    # if(self.not_tabu(i,j) and (self.current.fitness < best_vois)):
                    # not_tabu2[blockpergrid,threadsperblock](self.tabu_list2,self.current.cities,self.tabu_check,self.size_solution,self.tabu_list2.shape[0],self.tabu_list2.shape[1])
                    if(self.not_tabu2(self.current) and (self.current.fitness < best_vois)):
                        best_vois = self.current.fitness
                        best_i  = i 
                        best_j = j

                    self.current.inversion_list_cities(i,j)
                    tmp = self.current.evaluate(self.distance_matrix)
                    self.current.fitness = tmp
        return best_i,best_j

                    # if local_minima%10 == 0:
                    #     a,b = np.random.randint(1,99),np.random.randint(1,99)
                    #     a,b = min(a,b),max(a,b)
                    #     part = self.current.cities[a:b]
                    #     shuffle(part)
                    #     self.current.cities[a:b] = part
    def optimize(self):
        local_minima = 0 
        improved = 0 
        first = True 
        descent = False
        improve_sol = -1 
        n = self.size_solution
        best_solution = solution_GPU(n)

        best_i = 0 
        best_j = 0 
        best_eval = self.current.fitness
        f_before = 10000000

        position = 0 
        for current_iter in range(self.nbiterations):
            best_i,best_j = self.neighbors_2opt(best_i,best_j)
            self.current.inversion_list_cities(best_i,best_j)
            self.current.order()
            tmp = self.current.evaluate(self.distance_matrix)
            self.current.fitness = tmp
            f_after = self.current.fitness

            if self.current.fitness < best_eval :
                best_eval = self.current.fitness
                best_solution.cities = self.current.cities
                best_solution.evaluate(self.distance_matrix)
                improve_sol = current_iter
                improved += 1 
            else:
                if ((f_before<f_after) and (descent==True)) or ((f_before == f_after)and(first)) :
                    print("minimum local at iteration",current_iter,"min = ",f_before)
                    print("best search from previous search ",best_eval,"km")
                    first = False
                    local_minima +=1
                
                if(f_before <= f_after):
                    descent = False
                else:
                    descent = True
                if (f_before != f_after) and (not first ):
                    first = True

            # self.tabu_list[best_i][best_j] = current_iter + self.tabu_duration
            f_before = f_after
            position = self.update_tabu_list_2(self.current,position)
            print(current_iter,self.current.fitness,best_eval)
            self.current_iter = current_iter

        return best_solution

TPB = 100 

@cuda.jit
def kernel2opt(distance_matrix,cities,tour,counter,gpu_matrix_indexes,opt_indexes,opt_iterations,best,best_tour):
    local_id = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    max_ = counter 
    city_coords = cuda.shared.array(shape=(TPB, TPB), dtype=int32)
    t = cuda.shared.array(shape=(1, TPB), dtype=int32)
    best_block = cuda.shared.array(shape=(TPB, 3), dtype=int32)

    best_local = 100000


    for i in range(cuda.threadIdx.x, cities, cuda.blockDim.x):
        t[i] = tour[i]
        tmp_mat = gpu_matrix_indexes[cuda.threadIdx.x]
        for ii in range(len(tmp_mat)):
            city_coords[tmp_mat[ii][0]][tmp_mat[ii][1]] = distance_matrix[tmp_mat[ii][0]][tmp_mat[ii][1]]
    cuda.syncthreads()

    for i in range(opt_iterations):
        if local_id+i < max_:
            x = opt_indexes[local_id+i][0]
            y = opt_indexes[local_id+i][1]

            change = city_coords[x][(y-1)%len(t)] + city_coords[x][(y+1)%len(t)] + city_coords[y][(x+1)%len(t)] + city_coords[y][(x-1)%len(t)] 
            no_change = city_coords[x][(x-1)%len(t)] + city_coords[x][(x+1)%len(t)] + city_coords[y][(y+1)%len(t)] + city_coords[y][(y-1)%len(t)]
             
            if change < best_local:
                best_block[local_id] = [x,y,change]

    cuda.syncthreads()

    k = cuda.blockDim.x >>1 

    while k != 32 : 
        if cuda.threadIdx.x < k :
            best_block[local_id] = best_block[local_id+k] if best_block[local_id+k][3] < best_block[local_id][3] else best_block[local_id]
        
        cuda.syncthreads()
        k>>=1

    if cuda.threadIdx.x <= 32 :
        best_block[local_id] = best_block[local_id+32] if best_block[local_id+32][3] < best_block[local_id][3] else best_block[local_id]
        best_block[local_id] = best_block[local_id+16] if best_block[local_id+16][3] < best_block[local_id][3] else best_block[local_id]
        best_block[local_id] = best_block[local_id+8] if best_block[local_id+8][3] < best_block[local_id][3] else best_block[local_id]
        best_block[local_id] = best_block[local_id+4] if best_block[local_id+4][3] < best_block[local_id][3] else best_block[local_id]
        best_block[local_id] = best_block[local_id+2] if best_block[local_id+2][3] < best_block[local_id][3] else best_block[local_id]
        best_block[local_id] = best_block[local_id+1] if best_block[local_id+1][3] < best_block[local_id][3] else best_block[local_id]

    if cuda.threadIdx.x == 0 :
        best_tour[cuda.blockIdx.x] = best_block[cuda.threadIdx.x]





class solution_GPU():

    def __init__(self,nv):
        self.restart = True 
        self.size = nv
        self.cities = [None for i in range(nv)]
        self.cities[0] = 0 
        self.fitness = 1000000
        for i in range(1,self.size):
            self.restart = True 
            while(self.restart):
                self.restart = False
                a = randint(0,self.size-1)
                for j in range(0,i):
                    if a == int(self.cities[j]) :
                        self.restart = True
            self.cities[i] = a
        self.cities = np.array(self.cities)

    def evaluate(self,distance):
        tmp = 0 
        for i in range(self.size - 1):
            tmp += distance[self.cities[i]][self.cities[i+1]]
        tmp += distance[self.cities[0]][self.cities[self.size - 1]]
        self.fitness = tmp 
        return tmp


    def order(self):
        if self.cities[0] != 0:
            position_0 = 0 
            city_c = [None for _ in range(self.size)]
            for i in range(self.size):
                city_c[i] = self.cities[i]
                if self.cities[i] == 0:
                    position_0 = i
            k = 0 
            for i in range(position_0,self.size):
                self.cities[k] = city_c[i]
                k+=1
            
            for i in range(0,position_0):
                self.cities[k] = city_c[i]
                k+=1
        
        if self.cities[1] > self.cities[self.size-1]:
            for k in range(1,2+int((self.size-2)/2)):
                inter = self.cities[k]
                self.cities[k] = self.cities[self.size - k]
                self.cities[self.size - k] = inter 


    def inversion_list_cities(self,city_1,city_2):
        for k in range(city_1,city_1 +1+ int((city_2 - city_1)/2)):
            tmp = self.cities[k] 
            self.cities[k] = self.cities[city_2 + city_1 -k]
            self.cities[city_2 + city_1 -k] = tmp

                    

    def show(self):
        string = "-".join(map(str,self.cities))
        print(string)
        print("-->",self.fitness,"km")


    def plot_graph(self,distance_matrix):
        for i in range(len(distance_matrix)):
            distance_matrix[i][i] = 0
        A = np.array(distance_matrix)
        G = nx.from_numpy_matrix(A)

        edges = []
        for i in range(len(self.cities)):
            edges.append((self.cities[i-1],self.cities[i]))
        edges.append((self.cities[-1],self.cities[0]))
        labels = nx.get_edge_attributes(G,'weight')
        # G = nx.drawing.nx_agraph.to_agraph(G)  spectral_layout(G)  nx.spring_layout(G,iterations= 10000,scale=1000.0)
        #  nx.kamada_kawai_layout(G)
        nx.draw(G,  with_labels = True,pos=nx.kamada_kawai_layout(G)  ,node_size=300, node_color='green', prog='neato',edgelist=edges)
        # # nx.draw_networkx_nodes(G,with_labels = True,pos=nx.spring_layout(G),prog='neato',node_size=300, node_color='green')
        # nx.draw_networkx_edges(G,pos=nx.spring_layout(G),edgelist=edges,edge_color="red",prog='neato')
        plt.show()


if __name__ == "__main__":
    nb_iteration = 100
    duree_tabou  = 100
    nb_villes = 100
    filename = "distances_entre_villes_100.txt"
    
    tabu = TabuSearch_GPU(nb_iteration,duree_tabou,nb_villes,filename)
    best_sol = tabu.optimize()

    best_sol.show()
    best_sol.plot_graph(tabu.distance_matrix)

