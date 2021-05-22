import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import Coarse_Graining as cg
import Partitioning as pt


# Returns a boolean array of size N with the binary representation of i
def in_binary(i,N):
    if i >= 2**N:
        raise ValueError('Value is not too large to fit into the binary representation!')
    b = list(np.binary_repr(i, N))
    b = np.array(b, dtype=int)
    b = np.array(b, dtype=bool)
    return b


# Creates the laplacian of the SIS dynamic on G with parameters gamma and tau. The indexes are assigned by binary rep.
def sis_laplacian(G, tau, gamma):
    N = len(G)
    row = []
    col = []
    data = []
    for i in range(2**N):
        b = in_binary(i,N)
        rates = tau * np.matmul(b, G)
        for j in range(N):
            if b[j]:
                row.append(i)
                col.append(i - 2**(N-1-j))
                data.append(gamma)
            elif rates[j] > 0:
                row.append(i)
                col.append(i + 2**(N-1-j))
                data.append(rates[j])
        row.append(i)
        col.append(i)
        leaving_rate = np.sum(rates) - np.sum(rates[b]) + gamma * np.sum(b)
        data.append(-leaving_rate)

    A = sp.csc_matrix((data,(row, col)), (2**N, 2**N), dtype=float)
    return A


# Example code that generates the results from the thesis. Takes about 1.5 minutes on my device.

graph = nx.read_gml('Toy_Network.gml')
G = nx.adjacency_matrix(graph).toarray()

n = 13
A = sis_laplacian(G, 1, 1)
x0 = np.zeros(2**n)
x0[971] = 1
# x0[np.random.randint(2**n)] = 1
T = 10
m = 500

# Computing the dominant eigenpairs and QSD of A
e, v = sp.linalg.eigs(A.transpose(), 3, which='SM')
order = np.argsort(np.abs(e))
e = e[order]
v = v.transpose()[order]
v[0] = v[0] * np.sign(v[0][0].real)
v[1] = -v[1] * np.sign(v[1][0].real)
if np.abs(v[0][0] - 1) > 1e-12 or np.abs(e[0]) > 1e-12:
    raise ValueError('Leading Eigenpair has an unexpected value!')
if np.max(np.abs(v[0].imag)) > 0 or np.max(np.abs(v[1].imag)) > 0:
    raise ValueError('Leading Eigenpair has an unexpected value!')
val = e
vec = v
qsd = vec[1].real


# Setting up the coarse-graining methods
cluster = [[0,1,2,3], [4,5,6,7],[8,9,10,11,12]]
obs = np.array([vec[1].real]).transpose()

partition = pt.partinion_edge_based(cluster, G)
edge_method = cg.Method(partition)

k = np.max(partition) + 1
partition = pt.partition_k_means(k, qsd)
kmeans_edge = cg.Method(partition)

partition = pt.partinion_node_based(cluster)
node_method = cg.Method(partition)

k = np.max(partition) + 1
partition = pt.partition_k_means(k, qsd)
kmeans_node = cg.Method(partition)

methods = [edge_method, kmeans_edge, node_method, kmeans_node]


exact_sol = cg.compare_methods(methods, x0, A, T, m)

methods[0].calc_leading_ev(val, vec)
methods[1].calc_leading_ev(val, vec)
methods[2].calc_leading_ev(val, vec)
methods[3].calc_leading_ev(val, vec)

plt.figure(figsize=(5,5))
t = np.linspace(0,T,m)
plt.plot(t, methods[0].error_norm, lw=2)
plt.plot(t, methods[1].error_norm, lw=2, ls='--')
plt.plot(t, methods[2].error_norm, lw=2)
plt.plot(t, methods[3].error_norm, lw=2, ls='--')

plt.legend(['Edge-Based', 'k-Means 3772', 'Node-Based', 'k-Means 150'])



