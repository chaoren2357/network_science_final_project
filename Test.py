# # import networkx as nx
# # import matplotlib.pyplot as plt
# # import random
# # G = nx.Graph()
# #
# # for i in range(100):
# #     for j in range(100):
# #         r = random.random()
# #         if i != j and r>0.5:
# #             G.add_edge(i,j)
# # # G.add_edge(1,2)
# # # G.add_edge(1,3)
# # # G.add_edge(2,3)
# #
# # a = nx.eccentricity(G)
# # print(a)
# # nx.draw(G, with_labels=False, font_weight='bold',node_size=10)
# # plt.show()
# from matplotlib import pylab
# import networkx as nx
# import matplotlib.pyplot as plt
# import random
# import time
# import numpy as np
# # num=2000
# # G = nx.Graph()
# # s = time.time()
# # for i in range(num):
# #     print("Now i = {0}".format(i),end = "\r")
# #     for j in range(num):
# #         r = random.random()
# #         if i != j and r>0.5:
# #             G.add_edge(i,j)
# # e=time.time()
#
# def save_graph(graph,file_name):
#     #initialze Figure
#     plt.figure(num=None, figsize=(500, 500), dpi=80)
#     plt.axis('off')
#     fig = plt.figure(1)
#     pos = nx.spring_layout(graph)
#     nx.draw_networkx_nodes(graph,pos)
#     nx.draw_networkx_edges(graph,pos)
#     nx.draw_networkx_labels(graph,pos)
#
#     cut = 1.00
#     xmax = cut * max(xx for xx, yy in pos.values())
#     ymax = cut * max(yy for xx, yy in pos.values())
#     xmin = cut * min(xx for xx, yy in pos.values())
#     ymin = cut * min(yy for xx, yy in pos.values())
#     print(xmin,xmax,ymin,ymax)
#     plt.xlim(xmin, xmax)
#     plt.ylim(ymin, ymax)
#
#     plt.savefig(file_name,bbox_inches="tight")
#     pylab.close()
#     del fig
# def plot_betweenness_distribution(G,flag=""):
#     print("Plotting betweenness distribution...")
#     betweenness_dict = nx.betweenness_centrality(G)
#
#     p = np.array(nx.degree_histogram(G))/G.number_of_nodes()
#     if flag == "loglog":
#         plt.loglog([i for i in range(len(p))],p)
#     elif flag == "log":
#         plt.yscale("log")
#         plt.plot([i for i in range(len(p))],p)
#     elif flag == "":
#         plt.plot([i for i in range(len(p))],p)
#     plt.xlabel("k")
#     plt.ylabel(r"$P_k$")
#     plt.title("Digree Distribution")
#     plt.show()
# def plot_degree_distribution(G,flag = "bar"):
#     print("Plotting degree distribution...")
#     p = np.array(nx.degree_histogram(G))/G.number_of_nodes()
#     if flag == "loglog":
#         plt.loglog([i for i in range(len(p))],p)
#     elif flag == "log":
#         plt.yscale("log")
#         plt.plot([i for i in range(len(p))],p)
#     elif flag == "bar":
#         plt.bar([i for i in range(len(p))],p)
#     plt.xlabel("k")
#     plt.ylabel(r"$P_k$")
#     plt.title("Digree Distribution")
#     plt.show()
# #Assuming that the graph g has nodes and edges entered
# # save_graph(G,"/Users/scott/Desktop/my_graph.pdf")
# # plot_degree_distribution(G,flag="log")
# # nx.draw(G,with_labels=True, font_weight='bold')
# # plt.show()
# # # print(np.log(10))
# # print(nx.clustering(G))
#
# def heatmap(data, row_labels, col_labels, ax=None,
#             cbar_kw={}, cbarlabel="", **kwargs):
#     """
#     Create a heatmap from a numpy array and two lists of labels.
#
#     Parameters
#     ----------
#     data
#         A 2D numpy array of shape (N, M).
#     row_labels
#         A list or array of length N with the labels for the rows.
#     col_labels
#         A list or array of length M with the labels for the columns.
#     ax
#         A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
#         not provided, use current axes or create a new one.  Optional.
#     cbar_kw
#         A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
#     cbarlabel
#         The label for the colorbar.  Optional.
#     **kwargs
#         All other arguments are forwarded to `imshow`.
#     """
#
#     if not ax:
#         ax = plt.gca()
#
#     # Plot the heatmap
#     im = ax.imshow(data, **kwargs)
#
#     # Create colorbar
#     cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#     cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
#
#     # We want to show all ticks...
#     ax.set_xticks(np.arange(data.shape[1]))
#     ax.set_yticks(np.arange(data.shape[0]))
#     # ... and label them with the respective list entries.
#     ax.set_xticklabels(col_labels)
#     ax.set_yticklabels(row_labels)
#
#     # Let the horizontal axes labeling appear on top.
#     ax.tick_params(top=True, bottom=False,
#                    labeltop=True, labelbottom=False)
#
#     # Rotate the tick labels and set their alignment.
#     # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
#     #          rotation_mode="anchor")
#
#     # Turn spines off and create white grid.
#     for edge, spine in ax.spines.items():
#         spine.set_visible(False)
#
#     ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
#     ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
#     ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
#     ax.tick_params(which="minor", bottom=False, left=False)
#
#     return im, cbar
# # print("Generate Time:{0}s".format(format(e-s,".2f")))
# # s=time.time()
# # prob = np.zeros((num,num))
# # for i,j in G.edges:
# #     prob[i][j] = prob[i][j]+1
# # prob = prob/nx.number_of_edges(G)
# # e=time.time()
# # print("Calculate Time:{0}s".format(format(e-s,".2f")))
#
#
# vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
#               "potato", "wheat", "barley"]
# farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
#            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
#
# harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
#
# degree_corr = np.random.random((30,30))
# row_labels = np.arange(30)+1
# col_labels = np.arange(30)+1
#
# fig, ax = plt.subplots()
#
# # im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
# #                    cmap="YlGn", cbarlabel="harvest [t/year]")
# im, cbar = heatmap(degree_corr, row_labels, col_labels, ax=ax,
#                    cmap="YlGn", cbarlabel="degree_corr")
#
#
# fig.tight_layout()
# plt.show()
# # We want to show all ticks...
# # ax.set_xticks(np.arange(len(farmers)))
# # ax.set_yticks(np.arange(len(vegetables)))
# # ... and label them with the respective list entries
# # ax.set_xticklabels(farmers)
# # ax.set_yticklabels(vegetables)
#
# # Rotate the tick labels and set their alignment.
# # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
# #          rotation_mode="anchor")
#
# # Loop over data dimensions and create text annotations.
# # for i in range(len(vegetables)):
# #     for j in range(len(farmers)):
# #         text = ax.text(j, i, harvest[i, j],
# #                        ha="center", va="center", color="w")
#
# # ax.set_title("Harvest of local farmers (in tons/year)")
# fig.tight_layout()
# plt.show()
# print(nx.edges(G))
#
#
import os
import pickle
import numpy as np
filepath = "/Users/scott/Desktop/Classes/NetworkScience/Data/road-chesapeake.mtx"
name = "road-chesapeake"
save_path = "/Users/scott/Desktop"
# G=read_file(filepath,"Network Repository") #["SNAP", "Network Repository"]
# get_basic_information(G,name,save_path)
# print("------------------Results-------------------")
path = os.path.join(save_path, name + ".pkl")
dict = pickle.load(open(path, 'rb'))
k_num = len(dict["DH"][1])-1
prob = np.zeros((k_num,k_num))
print("k_nUm:{0},e_num:{1},node num:{2}".format(k_num,len(dict["E"][1]),dict["NN"][1]))

for i,j in dict["E"][1]:
    prob[dict["D"][1][i]-1][dict["D"][1][j]-1] = prob[dict["D"][1][i]-1][dict["D"][1][j]-1] + 1

prob = prob/len(dict["E"][1])
print(sum(sum(prob)))
