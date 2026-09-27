# Networks-project
import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt 

fh = open("out.ucidata-zachary", "rb")
ZKC_graph = nx.read_edgelist(fh, comments='%')
fh.close()

ZKC_graph = nx.karate_club_graph() 
print("The number of nodes is", ZKC_graph.number_of_nodes()) 
print("The number of edges is", ZKC_graph.number_of_edges()) 

pos_ZKC = nx.spring_layout(ZKC_graph) 
nx.draw(ZKC_graph, pos_ZKC, cmap=plt.cm.prism) 

Adj_ZKC = nx.to_numpy_array(ZKC_graph) 
Adj_ZKC = np.where(Adj_ZKC >= 1, 1, 0) 
plt.spy(Adj_ZKC) 
Adj_ZKC.shape
np.unique(Adj_ZKC)

def deg_node(A,i):
    k = 0
    for j in range(A.shape[0]):
        k = k + A[i][j]
    return k

def A_minus_AT(A):
    B = A- np.transpose(A)
    return np.unique(B)
A_minus_AT(Adj_ZKC)

def Degree_list(A):
    lista = [0] * A.shape[0]
    for i in range(A.shape[0]):
        lista[i] = int(deg_node(A,i))
    return lista

Degree_list(Adj_ZKC)

plt.hist(Degree_list(Adj_ZKC), bins=20)
plt.xlabel("Degree of node")
plt.ylabel("Frequency")
plt.title("Histogram on degree of node against frequency")
plt.show()

k_seq = ZKC_graph.degree()
k_seq_np = np.array(k_seq)
k_seq_1darray = k_seq_np[:,1]
nx.draw(ZKC_graph, pos_ZKC,node_color=k_seq_1darray, cmap=plt.cm.Blues, with_labels=True)
print(k_seq)

print("The mean of the degree sequence is", int(np.mean(Degree_list(Adj_ZKC))))
print("The standard deviation of the degree sequence is", int(np.std(Degree_list(Adj_ZKC))))
print("The maximum of the degree sequence is", int(np.max(Degree_list(Adj_ZKC))))
print("The minimum of the degree sequence is", int(np.min(Degree_list(Adj_ZKC))))

def Laplacian(A):
    L = np.diag(Degree_list(A)) - A
    return L

print(np.sum(Laplacian(Adj_ZKC),1)) 
print(np.sum(Laplacian(Adj_ZKC),0))

np.linalg.eigh(Laplacian(Adj_ZKC))

eigenvalues, eigenvectors = np.linalg.eigh(Laplacian(Adj_ZKC))
print(eigenvectors[:,0])
print(eigenvectors[:,1])

def spectral_bipartitioning(A, k_seq):
    laplacian_matrix = np.diagflat(k_seq) - A                       
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)    
    index_1 = np.argsort(eigenvalues)[1]                            
    partition = [val >= 0 for val in eigenvectors[:, index_1]]      
    partition_array = 1*np.reshape(partition, A.shape[0])           
    return partition_array

bip_ZKC=spectral_bipartitioning(Adj_ZKC, k_seq_1darray)
nx.draw(ZKC_graph, pos_ZKC,node_color=bip_ZKC, cmap=plt.cm.prism,
                            with_labels=True)

fh = open("out.contiguous-usa", "rb")
US_graph_Konect = nx.read_edgelist(fh, comments='%')
fh.close()
print("The number of nodes is", US_graph_Konect.number_of_nodes()) 
print("The number of edges is", US_graph_Konect.number_of_edges())

fh = open("out.contiguous-usa", "rb")
US_graph_Konect = nx.read_edgelist(fh, comments='%')
US_graph = nx.to_numpy_array(US_graph_Konect) 
US_graph = np.where(US_graph >= 1, 1, 0) 

US_seq = US_graph_Konect.degree()
US_seq = np.array(US_seq)
US_seq=US_seq.astype(int)
US_seq_1darray = US_seq[:,1]

US_graph_Konect=nx.relabel_nodes(US_graph_Konect,statenames)
pos_us = nx.spring_layout(US_graph_Konect)
bip_us=spectral_bipartitioning(US_graph, US_seq_1darray)
nx.draw(US_graph_Konect, pos_us,node_color=bip_us, cmap=plt.cm.prism, with_labels=True)

from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics import jaccard_score
US_graph_Konect
G = ZKC_graph
c_frozenlists = list(greedy_modularity_communities(G))
c_lists = np.array([list(comm) for comm in c_frozenlists], dtype=object)
print(c_lists)

nodes = sorted(G.nodes())

greedy_labels = np.zeros(len(nodes), dtype=int)
for i, comm in enumerate(c_frozenlists):
    for node in comm:
        idx = nodes.index(node)
        greedy_labels[idx] = i
        
print(greedy_labels)

score_macro = jaccard_score(greedy_labels, bip_ZKC, average='macro')
score_micro = jaccard_score(greedy_labels, bip_ZKC, average='micro')

print("Jaccard score (macro) is", score_macro)
print("Jaccard score (micro) is", score_micro)

nx.draw(ZKC_graph, pos_ZKC,node_color=greedy_labels, cmap=plt.cm.prism, with_labels=True)

from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics import jaccard_score

G = US_graph_Konect
c_frozenlists = list(greedy_modularity_communities(G))
c_lists = np.array([list(comm) for comm in c_frozenlists], dtype=object)
print(c_lists)

nodes = sorted(G.nodes())

greedy_labels = np.zeros(len(nodes), dtype=int)
for i, comm in enumerate(c_frozenlists):
    for node in comm:
        idx = nodes.index(node)
        greedy_labels[idx] = i
        
print(greedy_labels)

score_macro = jaccard_score(greedy_labels, bip_us, average='macro')
score_micro = jaccard_score(greedy_labels, bip_us, average='micro')

print("Jaccard score (macro) is", score_macro)
print("Jaccard score (micro) is", score_micro)

nx.draw(US_graph_Konect, pos_us,node_color=greedy_labels, cmap=plt.cm.RdYlBu, with_labels=True)

