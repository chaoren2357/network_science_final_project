import networkx as nx
import matplotlib.pyplot as plt
import random
from Utils import *
import numpy as np
# G = nx.Graph()
#
# for i in range(100):
#     for j in range(100):
#         r = random.random()
#         if i != j and r>0.5:
#             G.add_edge(i,j)
#
# filenames = ["road-chesapeake.mtx"]#,"road-euroroad.edges","road-usroads.mtx","road-roadNet-CA.mtx","road-germany-osm.mtx"]
# dicts = []
# for filename in filenames:
#     print("----------------------------------------------------------------------")
#     print("For {0}".format(filename))
#     filepath = os.path.join("/Users/scott/Data/networkScience/origin",filename)
#     name = filename.split(".",1)[0]
#     save_path = "/Users/scott/Data/networkScience/pickle"
#     # G=read_file(filepath,"Network Repository") #["SNAP", "Network Repository"]
#     # add_features(G,name,save_path)
#     path = os.path.join(save_path, name + ".pkl")
#     dict = pickle.load(open(path, 'rb'))
#     dicts.append(dict)
# plot_betweenness_distribution(dicts,save_path,filenames)
import time
for n in [10,100,1000,100000]:
    G=nx.fast_gnp_random_graph(n, 1./n)
    current_time = time.time()
    a=nx.betweenness_centrality(G)
    print(time.time()-current_time)
