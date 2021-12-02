import networkx as nx 
from random import choice,randint
import cupy as cp
from numba import cuda,jit,njit,vectorize,guvectorize
import numba
from numpy import int32,int64
import numpy as np
import matplotlib.pyplot as plt

class TabuSearch():

    def __init__(self,nbiter,dt,nv,filename):

        # GPU settings 
        self.threadsperblock = 32
        # self.blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock



        self.nbiterations = nbiter
        self.current_iter = 0
        self.tabu_duration = dt 
        self.size_solution = nv

        self.distance_matrix = self.read_file_and_build_distance(nv,filename)
        self.current = solution(nv)
        tmp = self.current.evaluate(self.distance_matrix)
        self.current.fitness = tmp

        print("the initial random solution is")
        self.current.show()

        self.tabu_list = cuda.to_device(np.array([[-1 for j in range(nv)] for i in range(nv)]))
        self.tabu_3opt = cuda.to_device(np.array([[[-1 for j in range(nv)] for i in range(nv)] for k in range(nv)]))
        self.tabu_list2 = [[-1 for j in range(self.size_solution)] for i in range(self.tabu_duration)]

        self.current.plot_graph(self.distance_matrix)

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

        return numba.cuda.to_device(np.array(distances))

    def not_tabu2(self,solution):
        for i in range(self.tabu_duration):
            for j in range(1,self.size_solution):
                if self.tabu_list2[i][j] != solution.cities[j]:
                    break
                elif j == self.size_solution - 1 :
                    return False
        
        for i in range(self.tabu_duration):
            for j in range(1,self.size_solution):
                if self.tabu_list2[i][j] != solution.cities[self.size_solution - j ]:
                    break
                elif j == self.size_solution - 1 :
                    return False
        
        return True 
             

    def not_tabu(self,i,j):
        return self.tabu_list[i][j] < self.current_iter


    def update_tabu_list_2(self,solution , position):
        if self.tabu_duration != 0:
            for i in range(self.size_solution):
                self.tabu_list2[position][i] = solution.cities[i]
            position+=1
            if position == self.tabu_duration - 1:
                return 0
            return position

    def neighbors_2opt(self,best_i,best_j):
        best_vois = 100000

        for i in range(self.size_solution):
            for j in range(i+1,self.size_solution):
                if (((i != 0) or (j != self.size_solution -1 )) and ((i!=0) or (j!=self.size_solution -2))) :
                    self.current.inversion_list_cities(i,j)
                    tmp = self.current.evaluate(self.distance_matrix)
                    self.current.fitness = tmp 
                    # if(self.not_tabu(i,j) and (self.current.fitness < best_vois)):
                    if(self.not_tabu2(self.current) and (self.current.fitness < best_vois)):
                        best_vois = self.current.fitness
                        best_i  = i 
                        best_j = j

                    self.current.inversion_list_cities(i,j)
                    tmp = self.current.evaluate(self.distance_matrix)
                    self.current.fitness = tmp
        return best_i,best_j


    def optimize(self):
        local_minima = 0 
        improved = 0 
        first = True 
        descent = False
        improve_sol = -1 
        n = self.size_solution
        best_solution = solution(n)

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
            # self.current.plot_graph(self.distance_matrix)
        return best_solution


# @guvectorize(["int32[:],int32[:,:],int32,buf[:],int32[:],int32[:,:],int32[:,:,:],int32[:,:,:],int32,int32[:]"]
#                 , "(n),(n,n),(),(p),(m),(d,n),(n,n,n),(n,n,n),()->(n)", target = "cuda")
# def neighbors_3opt(tour, distance, n ,buf,cost,tmp , tabu_list3,buffer,best ,out_cities):
    
#     for i in range(n-3):
#         for j in range(i+2,n-2):
#             for k in range(j+2,n -1):
#                 # A, B, C, D, E, F = tour[i-1], tour[i], tour[j-1], tour[j], tour[k-1], tour[k]
#                 cost[0] = 0 
#                 for i in range(len(tour)-1):
#                     cost+= distance[tour[i]][tour[i+1]]
#                 cost[0]+=distance[tour[0]][n-1] 
                
#                 dist = abs(i-j)
#                 for ii in range(n):
#                     if ii < i and ii > j :
#                         tmp[0][ii] = tour[ii]
#                     else:
#                         tmp[0][ii] = tour[j-dist]
#                         dist-=1
                
#                 cost[1] = 0 
#                 for i in range(len(tour)-1):
#                     cost+= distance[tmp[0][i]][tmp[0][i+1]]
#                 cost[1]+=distance[tmp[0][0]][n-1] 


#                 dist = abs(j-k)
#                 for ii in range(n):
#                     if ii < j and ii > k :
#                         tmp[1][ii] = tour[ii]
#                     else:
#                         tmp[1][ii] = tour[j-dist]
#                         dist-=1

#                 cost[2] = 0 
#                 for i in range(len(tour)-1):
#                     cost+= distance[tmp[1][i]][tmp[1][i+1]]
#                 cost[2]+=distance[tmp[1][0]][n-1] 
                
#                 dist = abs(i-k)
#                 for ii in range(n):
#                     if ii < i and ii > k :
#                         tmp[3][ii] = tour[ii]
#                     else:
#                         tmp[3][ii] = tour[j-dist]
#                         dist-=1

#                 cost[4] = 0 
#                 for i in range(len(tour)-1):
#                     cost+= distance[tmp[3][i]][tmp[3][i+1]]
#                 cost[4]+=distance[tmp[3][0]][n-1] 
                
#                 buf = tour[j:k] + tour[i:j] 
#                 tmp[2] = tour 
#                 tmp[2][i:k] = buf 


#                 cost[3] = 0 
#                 for i in range(len(tour)-1):
#                     cost+= distance[tmp[2][i]][tmp[2][i+1]]
#                 cost[3]+=distance[tmp[2][0]][n-1]
                
#                 best_tmp = min(cost)
                
#                 for iii in range(5):
#                     if best_tmp == cost[iii]:
#                         if iii != 0 and best > best_tmp:
#                             best = best_tmp
#                             out_cities = tmp[iii]
                
                    

                        


                # d1 = distance[A][C] + distance[B][D] + distance[E][F] 
                # d2 = distance[A][B] + distance[C][E] + distance[D][F] 
                # d3 = distance[A][D] + distance[E][B] + distance[C][F] 
                # d4 = distance[F][B] + distance[C][D] + distance[E][A]

                




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
    
    def move_1_city(self,city_1,city_2):
        inter = self.cities[city_1]
        if city_1 < city_2:
            for k in range(city_1,city_2):
                self.cities[k] = self.cities[k+1]
        else:
            for k in range(city_2,city_1):
                self.cities[k] = self.cities[k+1]
        self.cities[city_2] = inter 
    
    def inversion_list_cities(self,city_1,city_2):
        for k in range(city_1,city_1 +1+ int((city_2 - city_1)/2)):
            tmp = self.cities[k] 
            self.cities[k] = self.cities[city_2 + city_1 -k]
            self.cities[city_2 + city_1 -k] = tmp
    
    def identity(self,solution):
        for i in range(1,self.size):
            if int(solution.cities[i]) != int(self.cities[i]):
                return False
        return True 
    

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
        nx.draw(G,  with_labels = True,pos=nx.kamada_kawai_layout(G)  ,node_size=300, node_color='green', prog='neato',edgelist=edges)
        # # nx.draw_networkx_nodes(G,with_labels = True,pos=nx.spring_layout(G),prog='neato',node_size=300, node_color='green')
        # nx.draw_networkx_edges(G,pos=nx.spring_layout(G),edgelist=edges,edge_color="red",prog='neato')
        plt.savefig("Graph"+name+str(self.fig)+".png", format="PNG")
        self.fig+=1
        plt.close()




if __name__ == "__main__":
    nb_iteration = 100
    duree_tabou  = 100
    nb_villes = 100
    filename = "distances_entre_villes_100.txt"
    # filename = "distances_entre_villes_10.txt"
    
    tabu = TabuSearch(nb_iteration,duree_tabou,nb_villes,filename)
    best_sol = tabu.optimize()

    best_sol.show()
    best_sol.plot_graph(tabu.distance_matrix,name="fi")