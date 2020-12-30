import networkx as nx
import matplotlib.pyplot as plt
import random
from Utils import *
import numpy as np
G = nx.Graph()
for i in range(100):
    for j in range(100):
        r = random.random()
        if i != j and r>0.5:
            G.add_edge(i,j)

filenames = ["road-euroroad.edges"]#,"road-euroroad.edges","road-usroads.mtx","road-roadNet-CA.mtx","road-germany-osm.mtx"]
dicts = []
for filename in filenames:
    print("----------------------------------------------------------------------")
    print("For {0}".format(filename))
    filepath = os.path.join("/Users/scott/Data/networkScience/origin",filename)
    name = filename.split(".",1)[0]
    save_path = "/Users/scott/Data/networkScience/pickle"
    G=read_file(filepath,"Network Repository") #["SNAP", "Network Repository"]
    # add_features(G,name,save_path)
    path = os.path.join(save_path, name + ".pkl")

    Gc = lifeline(G)
    print("NN:{0}, NE:{1}".format(G.number_of_nodes(),G.number_of_edges()))
    print("NN:{0}, NE:{1}".format(Gc.number_of_nodes(),Gc.number_of_edges()))
    save_double_graph(G,Gc,save_path,name="catch")
    # save_graph(G,save_path,name="try")
    # save_graph(Gc, save_path, name="try2")

    # dict = pickle.load(open(path, 'rb'))
    # add_average_shortest_path_length(G,name,save_path)
    # dicts.append(dict)




# def compute_distance_matrix(G,edge):
#     dis_mat = np.zeros((nx.number_of_nodes(G),nx.number_of_nodes(G)))
#     for from_node in nx.shortest_path(G):
#         for to_node in nx.shortest_path(G)[from_node]:
#             print("dismat {0} to {1} of edge {2}".format(from_node,to_node,edge),end='\r')
#             dis_mat[from_node][to_node] = len(nx.shortest_path(G)[from_node][to_node])
#     return dis_mat
# edge_list = []
# dis_mat_list = []
# for edge in nx.edges(G):
#     print("Processing {0}".format(edge),end="\r")
#     Gc = G.copy()
#     Gc.remove_edge(*edge)
#     dis_mat = compute_distance_matrix(Gc,edge)
#     edge_list.append(edge)
#     dis_mat_list.append(dis_mat)
# for edge1,dis_mat1 in zip(edge_list,dis_mat_list):
#     for edge2,dis_mat2 in zip(edge_list,dis_mat_list):
#         if (dis_mat1 - dis_mat2).all():
#             print("{0} is euqual to {1}".format(edge1,edge2))

