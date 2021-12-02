import networkx as nx 
from random import choice,randint
import cupy as cp
from numba import cuda,jit,njit,vectorize,guvectorize,int32
import numba
from numpy import int32,int64
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import euclidean_distances
from sklearn import manifold
# from random import randint 
from random import shuffle 
from time import time

class TabuSearch():

    def __init__(self,nbiter,dt,nv,filename,alpha = 1):

        # parameters 
        self.nbiterations = nbiter
        self.current_iter = 0
        self.tabu_duration = dt 
        self.size_solution = nv
        
        # tabu modification report 2
        self.lower_bound = round(alpha*0.5*np.sqrt(nv))
        self.upper_bound = round(alpha*1.5*np.sqrt(nv))
        self.bounds = np.random.randint(self.lower_bound,self.upper_bound,size=self.nbiterations)
        #tabu frequency 
        self.frequencies = [[0 for i in range(nv)] for j in range(nv)]
        self.FTD1 = lambda x,y:10 + round(alpha*(self.frequencies[x][y]/self.size_solution))

        #tabu frequency 2 
        self.frequencies2 = [0 for j in range(nv)]
        self.FTD2 = lambda x:10 + round(alpha*(self.frequencies2[x]/self.size_solution))

        #gpu distance matrix 
        self.points2D = []
        self.index_2d_points = 0
        if filename[0:8] == "tsp-size":
            self.distance_matrix = self.read_file_build_from_points(nv,filename)
            self.city_duration = [[] for i in range(nv)]
        else:
            self.distance_matrix = self.read_file_and_build_distance(nv,filename)
            dist = self.read_file_and_build_distance_(nv,filename)
            self.distance_matrix = dist
            self.city_duration = [[] for i in range(nv)]
        # embedding = manifold.MDS(n_components=2)
        # distance_transformed = [[int(coord[0]),int(coord[1])] for coord in embedding.fit_transform(np.array(self.distance_matrix))]
        # self.gps = distance_transformed
        # self.distance_matrix2 = [[distance_transformed[j] for j in range(self.size_solution)] for i in range(self.size_solution) ]
        # self.distance_GPU = cuda.to_device(np.array(self.distance_matrix))
        
        # #shared memeory gpu
        # self.gpu_matrix_indexes = [[] for _ in range(1024)]
        # size = round(nv*nv/1024)
        # indexe_tmp = 0 
        # for i in range(nv):
        #     for j in range(nv):
        #         if len(self.gpu_matrix_indexes[indexe_tmp]) == size*2:
        #             indexe_tmp+=1
        #         else:
        #             self.gpu_matrix_indexes[indexe_tmp].append(i)
        #             self.gpu_matrix_indexes[indexe_tmp].append(j)
        # self.gpu_matrix_indexes = [e for e in self.gpu_matrix_indexes if e != []]
        # self.gpu_matrix_indexes[-1] = [self.gpu_matrix_indexes[-1][0] for _ in range(size*2)]
        # # print(self.gpu_matrix_indexes)
        # self.gpu_matrix_indexes = cuda.to_device(np.array(self.gpu_matrix_indexes))
        # # print(self.gpu_matrix_indexes)
        # # print("gpu matrix",self.gpu_matrix_indexes[0])

        # # indexes for 2opt on GPU 
        
        # self.indexes_2opt_gpu = []
        # for i in range(self.size_solution):
        #     for j in range(i+1,self.size_solution):
        #         self.indexes_2opt_gpu.append([i,j])
        # self.nb_iteration_per_thread_2opt = round(len(self.indexes_2opt_gpu)/1024)
        # self.counter_2opt = len(self.indexes_2opt_gpu)
        # # print(self.indexes_2opt_gpu)
        # # print(self.nb_iteration_per_thread_2opt)
        # self.indexes_2opt_gpu = cuda.to_device(np.array(self.indexes_2opt_gpu.copy()))
        # # print(self.indexes_2opt_gpu)
        

        # #indexes for 3opt GPU 
        self.indexes_3opt_gpu = [[i, j, k] for i in range(self.size_solution) for j in range(i + 2, self.size_solution) for k in range(j + 2, self.size_solution + (i > 0))]
        # self.nb_iteration_per_thread_3opt = round(len(self.indexes_3opt_gpu)/1024)
        # self.counter_3opt = len(self.indexes_3opt_gpu)
        # # print(self.indexes_3opt_gpu)
        # # print(self.nb_iteration_per_thread_3opt)
        # # self.indexes_3opt_gpu = cuda.to_device(np.array(self.indexes_3opt_gpu))

        # result of the gpu 
        # tmp = [[0,0,0] for i in range(self.nb_iteration_per_thread_2opt)]
        # self.bpb = cuda.to_device(np.array(tmp))

        # # self.block_best = cuda.to_device(np.array([[0,0,99999] for i in range(1024*5)]))
        # self.block_best = cuda.to_device(np.array([0,0,999999]))
        # initialization
        self.current = solution(nv)
        tmp = self.current.evaluate(self.distance_matrix)
        self.current.fitness = tmp
        
        # self.helpers_data = cuda.to_device(np.array([nv,len(self.indexes_2opt_gpu),tmp,1]))
        
        print("the initial random solution")
        self.current.show()

        # self.tabu_list = cuda.to_device(np.array([[-1 for j in range(nv)] for i in range(nv)]))
        self.tabu_list = [[-1 for j in range(nv)] for i in range(nv)]
        self.tabu_list_GPU = cuda.to_device([[-1 for j in range(nv)] for i in range(nv)])
        self.tabu_list2 = {}

        # self.current.plot_graph(self.distance_matrix)

    def read_file_build_from_points(self,nv,filename):
        with open(filename, mode = "r") as file:
            input_data = [d.strip().split() for d in file.readlines()]
            self.points2D = input_data
            distances = [[None for j in range(nv)] for i in range(nv)]
            initial_data = input_data[0]
            initial_data = [float(x) for x in initial_data]
            initial_data = [round(x,2) for x in initial_data]

            ii=0
            for i in range(1,len(initial_data),2):
                jj= 0 
                for j in range(1,len(initial_data),2):
                    distances[ii][jj]= round(abs(initial_data[i-1]-initial_data[j-1])+abs(initial_data[i]-initial_data[j]),2)*100
                    jj+=1
                ii+=1
            for i in range(nv):
                distances[i][i] = -10
            
        return distances

    def read_next_from_list(self,nv,i):
        data = self.points2D[i]
        distances = [[None for j in range(nv)] for i in range(nv)]
        initial_data = data
        initial_data = [float(x) for x in initial_data]
        initial_data = [round(x,2) for x in initial_data]
        ii=0
        for i in range(1,len(initial_data),2):
            jj= 0 
            for j in range(1,len(initial_data),2):
                distances[ii][jj]= round(abs(initial_data[i-1]-initial_data[j-1])+abs(initial_data[i]-initial_data[j]),2)*100
                jj+=1
            ii+=1
        for i in range(nv):
            distances[i][i] = -10
        self.distance_matrix = distances
        return distances

    def read_file_and_build_distance(self,nv,filename):
        with open(filename , mode ="r") as file:
            input_data = [d.strip().split() for d in file.readlines()]
            if len(input_data[0]) != len(input_data[-1]):
                distances = [[None for j in range(nv)] for i in range(nv)]
                for i in range(nv):
                    row = 0
                    for j in range(i+1,nv):
                        distances[i][j] = int(input_data[i][row])
                        distances[j][i] = distances[i][j]
                        row += 1 
                for i in range(nv):
                    distances[i][i] = - 10 
            elif len(input_data[0]) == len(input_data[-1]):
                distances = [[None for j in range(nv)] for i in range(nv)]
                for i in range(nv):
                    for j in range(nv):
                        distances[i][j] = int(input_data[i][j])
                for i in range(nv):
                    print(distances[i])
        return distances
        # return numba.cuda.to_device(np.array(distances))

    def read_file_and_build_distance_(self,nv,filename):
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
            # print(distances[-1])
        return distances

    def not_tabu2_hash(self,solution):
        hashed_str = "_".join(map(str,solution.cities[1:]))
        hashed_str_reversed = "_".join(map(str,reversed(solution.cities[1:])))

        if hashed_str in self.tabu_list2 or hashed_str_reversed in self.tabu_list2:
            return False
        
        return True 

    def not_tabu(self,i,j):
        return self.tabu_list[i][j] < self.current_iter


    def update_tabu_list_2(self,solution,position):
        if self.tabu_duration != 0:
            for i in range(self.size_solution):
                self.tabu_list2[position][i] = solution.cities[i]
            position+=1
            if position == self.tabu_duration - 1:
                return 0
            return position
    
    def dist_city(self,point1,point2):
        return abs(point1[0]-point2[0]) + abs(point1[1]-point2[1])

    def neighbors_2opt_(self,best_i,best_j):
        best_vois = 100000
        current = self.current.fitness
        for i in range(self.size_solution):
            for j in range(i+1,self.size_solution):
                if (((i != 0) or (j != self.size_solution -1 )) and ((i!=0) or (j!=self.size_solution -2))) :

                    city_i, city_i_1= self.current.cities[i],self.current.cities[(i-1)%self.size_solution]
                    city_j, city_j1 = self.current.cities[j],self.current.cities[(j+1)%self.size_solution]
                    current_distance = self.distance_matrix[city_i][city_i_1] + self.distance_matrix[city_j][city_j1]
                    changed_distances = self.distance_matrix[city_i][city_j1] + self.distance_matrix[city_j][city_i_1]
                    
                    if(self.not_tabu(i,j) and (current + (changed_distances - current_distance) < best_vois )):
                    # if(self.not_tabu2_hash(self.current) and current + (changed_distances - current_distance) < best_vois):
                        # print(changed_distances,current_distance)
                        best_vois = current + (changed_distances - current_distance)
                        best_i  = i 
                        best_j = j
        # print("restricted sol",count_not_tabou)
        return best_i,best_j

    def neighbors_2opt(self,best_i,best_j):
        best_vois = 100000
        visit = 0 
        for i in range(self.size_solution):
            for j in range(i+1,self.size_solution):
                if (((i != 0) or (j != self.size_solution -1 )) and ((i!=0) or (j!=self.size_solution -2))) :
                    # self.count_nn +=1
                    visit += 1
                    self.current.inversion_list_cities(i,j)
                    tmp = self.current.evaluate(self.distance_matrix)
                    self.current.fitness = tmp 
                    if(self.not_tabu2_hash(self.current) and (self.current.fitness < best_vois)):
                    
                        best_vois = self.current.fitness
                        best_i  = i 
                        best_j = j

                    self.current.inversion_list_cities(i,j)
                    tmp = self.current.evaluate(self.distance_matrix)
                    self.current.fitness = tmp
        # print("visit" , visit)
        return best_i,best_j

    def optimize(self,opt3 = False):
        tabu_list_3opt = [[[-1 for k in range(self.size_solution)] for j in range(self.size_solution)] for i in range(self.size_solution)]

        local_minima = 0 
        improved = 0 
        first = True 
        descent = False
        improve_sol = -1 
        n = self.size_solution
        best_solution = solution(n)
        tmp = best_solution.evaluate(self.distance_matrix)
        best_cities = []
        best_iteration = 0
        n3opt_improvement = 0 
        best_iteration_3opt = 0

        restart = 0 
        best_i = 0 
        best_j = 0 
        best_eval = self.current.fitness
        f_before = 10000000

        position = 0 
        for current_iter in range(self.nbiterations):
            
            # block_per_grid = self.nb_iteration_per_thread_2opt 
            # thread_per_block = 1024
            # out = self.bpb
            # t1=time()
            # kernel2opt[block_per_grid,thread_per_block](self.distance_GPU,self.tabu_list_GPU, self.current.cities_GPU, self.indexes_2opt_gpu, self.block_best, self.helpers_data)

            # print("GPU time",time()-t1)
            # res = self.block_best.copy_to_host()
            # print("l",[x for x in res if int(x[2]) == 9999999])
            # res = [x for x in res if self.not_tabu(x[0],x[1])]
            # res = min(res,key=lambda x:x[2])
            # print("2opt GPU",res)
            # best_i,best_j = int(res[0]),int(res[1])
            # self.block_best = cuda.to_device(np.array([[0,0,self.current.fitness] for i in range(1024*5)]))

            # self.current.inversion_list_cities(best_i,best_j)
            # print(self.current.cities)
            # t2=time()
            best_i,best_j = self.neighbors_2opt_(best_i,best_j)
            # print("CPU time",time()-t1)
            self.current.inversion_list_cities(best_i,best_j)

            # self.current.order()
            tmp = self.current.evaluate(self.distance_matrix)
            self.current.fitness = tmp
            # self.helpers_data[2] = self.current.fitness
            f_after = self.current.fitness

            # for i in range(len(self.current.cities)):
            #     self.current.cities_GPU[i] = self.current.cities[i]

            if tmp < best_eval :
                best_eval = tmp
                best_cities = self.current.cities.copy()
                best_solution.cities = self.current.cities
                tmp = best_solution.evaluate(self.distance_matrix)
                improve_sol = current_iter
                improved += 1 
                best_iteration = current_iter
                restart = restart - (restart%10) + 1 
                # best_solution.plot_graph(self.distance_matrix,name=str(self.current.fig)+"_"+str(best_solution.fitness)+"_2opt_"+str(current_iter)+"_")
                
            else:
                if ((f_before<f_after) and (descent==True)) or ((f_before == f_after)and(first)) :
                    # print("minimum local at iteration",current_iter,"min = ",f_before)
                    # print("best search from previous search ",best_eval,"km")
                    first = False
                    local_minima +=1
                    restart +=1
                    route = self.current.cities
                    if restart%10 == 0:
                        a,b = np.random.randint(1,99),np.random.randint(1,99)
                        a,b = min(a,b),max(a,b)
                        part = self.current.cities[a:b]
                        shuffle(part)
                        self.current.cities[a:b] = part
                    if current_iter < 100000 :
                        for i in range(1):
                            route,distance,_ = self._3opt(route,best_eval)
                            # print("3 opt min local",distance)
                            self.current.cities = route
                            tmp = self.current.evaluate(self.distance_matrix)
                            if tmp < best_eval:
                                best_cities = route.copy()
                                best_iteration_3opt = current_iter
                                n3opt_improvement+=1
                                # print("3 opt improvement ",tmp)
                                best_eval = tmp
                                best_solution.cities = route
                                best_solution.evaluate(self.distance_matrix)
                                # best_solution.plot_graph(self.distance_matrix,name=str(self.current.fig)+"_"+str(best_solution.fitness)+"_3opt_"+str(current_iter)+"_")
                                # self.current.fig+=1

                
                if(f_before <= f_after):
                    descent = False
                else:
                    descent = True
                if (f_before != f_after) and (not first ):
                    first = True
#04 92 96 90 37 
            # self.tabu_list[best_i][best_j] = current_iter + self.tabu_duration

            # self.tabu_list_GPU[best_i][best_j] = current_iter + self.tabu_duration
            # self.helpers_data[3] = current_iter+1
            # self.tabu_list[best_i][best_j] = current_iter + self.bounds[current_iter]
            # self.city_duration[best_i].append(self.bounds[current_iter])
            # self.city_duration[best_j].append(self.bounds[current_iter])

            self.frequencies[best_i][best_j] +=1
            self.tabu_list[best_i][best_j] = current_iter + self.FTD1(best_i,best_j)
            self.city_duration[best_i].append(self.FTD1(best_i,best_j))
            self.city_duration[best_j].append(self.FTD1(best_i,best_j))

            # self.frequencies2[best_i]+=1
            # self.tabu_list[best_i][best_j] = current_iter + self.FTD2(best_i)
            # self.city_duration[best_i].append(self.FTD2(best_i))
            # self.city_duration[best_j].append(self.FTD2(best_j))
            # self.current.plot_graph(self.distance_matrix,"Graph"+str(self.current.fig)+"_")
            f_before = f_after
            # position = self.update_tabu_list_2(self.current,position)

            # self.tabu_list2["_".join(map(str,self.current.cities[1:])) ] = current_iter + self.tabu_duration
            # self.tabu_list2["_".join(map(str,reversed(self.current.cities[1:]))) ] = current_iter + self.tabu_duration

            # self.tabu_list2["_".join(map(str,self.current.cities[1:])) ] = current_iter + self.bounds[current_iter]
            # self.tabu_list2["_".join(map(str,reversed(self.current.cities[1:]))) ] = current_iter + self.bounds[current_iter]

            # print(current_iter,self.current.fitness,best_eval)
            self.current_iter = current_iter
        print("------------------------------------------")
        print("best score",best_eval)
        print("best iteration",best_iteration)
        print("best iteration by 3opt", best_iteration_3opt)
        print("number of local minima",local_minima)
        # print("mean value for RTD bounds",sum(self.bounds)/len(self.bounds))
        print("improved times",improved)
        print("3 opt improved times",n3opt_improvement)
        print("------------------------------------------")
        best_solution.cities = best_cities
        best_solution.evaluate(self.distance_matrix)
        return best_solution, best_eval, max(best_iteration,best_iteration_3opt), local_minima, self.city_duration

    def calc_tour_cost(self,route):
        tmp = 0 
        for i in range(self.size_solution-1):
            tmp += self.distance_matrix[route[i]][route[i+1]]
        return tmp + self.distance_matrix[route[0]][route[self.size_solution-1]]



    def _3opt(self,route,best):
        best_score = 10000
        best_tour = route
        best_ijk = []
        # combi = ((i, j, k) for i in range(self.size_solution) for j in range(i + 2, self.size_solution) for k in range(j + 2, self.size_solution + (i > 0)))
        for (i, j, k) in self.indexes_3opt_gpu:
            A, B, C, D, E, F = route[i-1], route[i], route[j-1], route[j], route[k-1], route[k% self.size_solution]

            d0 = self.distance_matrix[A][B]+ self.distance_matrix[C][D] + self.distance_matrix[E][F]
            d1 = self.distance_matrix[A][C]+ self.distance_matrix[B][D] + self.distance_matrix[E][F]
            d2 = self.distance_matrix[A][B]+ self.distance_matrix[C][E] + self.distance_matrix[D][F]
            d3 = self.distance_matrix[A][D]+ self.distance_matrix[E][B] + self.distance_matrix[C][F]
            d4 = self.distance_matrix[F][B]+ self.distance_matrix[C][D] + self.distance_matrix[E][A]
            l = [d1,d2,d3,d4]
            min_idx = l.index(min(l))

            # if best + ( l[min_idx] - d0 ) < best_score:
            if l[min_idx] < d0:
                # print("min 3opt found during run",l[min_idx])
                best_score = best + ( l[min_idx] - d0 )
                if min_idx == 0 :
                    best_tour[i:j] = reversed(route[i:j]) 
                elif min_idx == 1 :
                    best_tour[j:k] = reversed(route[j:k])
                elif min_idx == 2:
                    tmp = route[j:k] + route[i:j]
                    best_tour[i:k] = tmp
                elif min_idx == 3:
                    best_tour[i:k] = reversed(route[i:k])
                best_ijk = [i,j,k]
                best_score = self.calc_tour_cost(best_tour)

        return best_tour,best_score,best_ijk



@cuda.jit
def kernel2opt(distances,tabu_list, t, opt_indexes, best_block,helper_data):
    local_id = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    city_l = helper_data[0]
    max_ = helper_data[1]
    current = helper_data[2]
    iteration = helper_data[3]

    if local_id < max_:
        x = opt_indexes[local_id][0]
        y = opt_indexes[local_id][1]
        if tabu_list[x][y] < iteration :
            Ai,Bi,Aj,Bj = t[x], t[(x-1)%city_l], t[y], t[(y+1)%city_l]
            change = distances[Ai][Bj] + distances[Aj][Bi]
            no_change = distances[Ai][Bi] + distances[Aj][Bj]
            tmp = current + (change - no_change)
            # best_block[local_id][0] = t[x]
            # best_block[local_id][1] = t[y]
            # best_block[local_id][2] = tmp
            if tmp < best_block[2]:
                best_block[0] = t[x]
                best_block[1] = t[y]
                best_block[2] = tmp
            else:
                best_block[0] = t[x]
                best_block[1] = t[y]
                best_block[2] = 999999




class solution():
    
    def __init__(self,nv):
        self.fig = 0
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
        
        self.order()

        self.cities_GPU = cuda.to_device(np.array(self.cities))

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


    def plot_graph(self,distance_matrix,name=""):
        
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
        nx.draw(G,  with_labels = True,pos=nx.kamada_kawai_layout(G,scale=3)  ,node_size=200, node_color='green', prog='neato',edgelist=edges)
        # # nx.draw_networkx_nodes(G,with_labels = True,pos=nx.spring_layout(G),prog='neato',node_size=300, node_color='green')
        # nx.draw_networkx_edges(G,pos=nx.spring_layout(G),edgelist=edges,edge_color="red",prog='neato')
        plt.savefig("Graph_RL_comparison"+name+str(self.fig)+".png", format="PNG")
        self.fig+=1
        plt.close()




if __name__ == "__main__":
    nb_iteration = 1000
    duree_tabou  = 10
    nb_villes =  10
    scores = []
    filenames = ["tsp-size-1000-len-10-test.txt","tsp-size-1000-len-20-test.txt","tsp-size-1000-len-50-test.txt","tsp-size-1000-len-100-test.txt"]
    for i in range(5):

        # filename = "distances_entre_villes_{}.txt".format(nb_villes)
        # filename = "half100.txt"
        # filename = "distances_entre_villes_10.txt"
        filename = "tsp-size-1000-len-10-test.txt"
        tabu = TabuSearch(nb_iteration,duree_tabou,nb_villes,filename)
        distance_matrix = tabu.read_next_from_list(nb_villes,i)
        best_sol,best_eval,best_it,local_minima,city_duration = tabu.optimize()
        scores.append([best_eval,best_it,local_minima,city_duration])
        best_sol.show()
        best_sol.plot_graph(tabu.distance_matrix,name="final_"+ "nb_cities {}".format(nb_villes) +"_"+str(best_sol.fitness)+"_it" + str(i)+"_"+"RTF_")
    print("best score mean",sum([x[0] for x in scores])/len(scores))
    print("best iterations mean",sum([x[1] for x in scores])/len(scores))
    print("local minima mean",sum([x[2] for x in scores])/len(scores))
    # for i in range(10):
    #     for j in range(len(scores[i][3])):
    #         scores[i][3][j] = sum(scores[i][3][j])/len(scores[i][3][j])
    # meanCities = [0 for i in range(nb_villes)]
    # x = ["{}".format(i) for i in range(nb_villes)]
    # for i in range(10):
    #     for j in range(len(scores[i][3])):
    #         meanCities[j]+=scores[i][3][j]
    #     plt.bar(x,scores[i][3])
    #     # plt.xticks(np.arange(len(x)),x)
    #     plt.savefig("CityMean_run_{}_city_{}_alpha{}".format(i,nb_villes,1))
    #     plt.gcf().autofmt_xdate()
    #     plt.clf()
    # plt.bar(x,[x/10 for x in meanCities])
    # plt.gcf().autofmt_xdate()
    # plt.savefig("CityMean_mean_city_{}_alpha{}".format(nb_villes,1))
    # plt.clf()

    # duree_tabou = 200
    # scores = []
    # for i in range(1):
    #     filename = "distances_entre_villes_{}.txt".format(nb_villes)
    #     # filename = "distances_entre_villes_10.txt"
        
    #     tabu = TabuSearch(nb_iteration,duree_tabou,nb_villes,filename,alpha=5)
    #     best_sol,best_eval,best_it,local_minima,city_duration = tabu.optimize()
    #     scores.append([best_eval,best_it,local_minima,city_duration])
    #     best_sol.show()
    #     best_sol.plot_graph(tabu.distance_matrix,name="final"+ "nb_cities {}".format(nb_villes) +"_"+str(best_sol.fitness)+"_it" + str(i)+"_"+"RTF_")
    # print("best score mean",sum([x[0] for x in scores])/len(scores))
    # print("best iterations mean",sum([x[1] for x in scores])/len(scores))
    # print("local minima mean",sum([x[2] for x in scores])/len(scores))
    # for i in range(10):
    #     for j in range(len(scores[i][3])):
    #         scores[i][3][j] = sum(scores[i][3][j])/len(scores[i][3][j])
    # meanCities = [0 for i in range(nb_villes)]
    # x = ["{}".format(i) for i in range(nb_villes)]
    # for i in range(10):
    #     for j in range(len(scores[i][3])):
    #         meanCities[j]+=scores[i][3][j]
    #     plt.bar(x,scores[i][3])
    #     plt.gcf().autofmt_xdate()
    #     plt.savefig("CityMean_run_{}_city_{}_alpha{}".format(i,nb_villes,5))
    #     plt.clf()
    # plt.bar(x,[x/10 for x in meanCities])
    # plt.gcf().autofmt_xdate()
    # plt.savefig("CityMean_mean_city_{}_alpha{}".format(nb_villes,5))
    # plt.clf()


