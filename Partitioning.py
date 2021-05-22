import numpy as np
from sklearn.cluster import KMeans


# Returns a boolean array of size N with the binary representation of i
def in_binary(i,N):
    if i >= 2**N:
        raise ValueError('Value is not too large to fit into the binary representation!')
    b = list(np.binary_repr(i, N))
    b = np.array(b, dtype=int)
    b = np.array(b, dtype=bool)
    return b


# Returns the matrix A reduced to the indicated columns and rows
def sub_matrix(A, rows, cols):
    return A[rows].transpose()[cols].transpose()


# Creates a partition in which sates with the same id are grouped
# ids: Array of size n (number of states). Each id is an array of fixed size s
def partition_by_id(ids):
    n = len(ids)
    ids = ids.transpose()
    partition = np.full(n, -1, dtype=int)

    anchor = (ids.ndim == 1)
    if not anchor:
        if len(ids) == 1:
            anchor = True
            ids = ids[0]

    if ids.ndim == 1:
        args = np.argsort(ids)
        prev = ids[args[0]]

        c = 0
        for i in range(n):
            if ids[args[i]] != prev:
                c = c+1
                prev = ids[args[i]]
            partition[args[i]] = c

        return partition

    row = ids[0]
    args = np.argsort(row)
    prev = row[args[0]]

    current = []
    max = 0

    def rescursion(curr):
        curr = np.array(curr)
        sub_ids = ids[1:].transpose()[curr]
        return partition_by_id(sub_ids) + np.full(len(curr), max)

    for i in range(n):
        if row[args[i]] != prev:
            sub_partition = rescursion(current)
            partition[current] = sub_partition
            max = np.max(sub_partition) + 1
            prev = row[args[i]]
            current = [args[i]]
        else:
            current.append(args[i])

    partition[current] = rescursion(current)

    return partition


# Returns a partition of the state-space based on the number of infected nodes
# cluster: Array of size m (number of clusters) with arrays resembling classes containing their nodes
def partinion_node_based(cluster):
    m = len(cluster)
    sizes = np.zeros(m, dtype=int)
    for i in range(m):
        sizes[i] = len(cluster[i])
    N = np.sum(sizes)

    partition = np.zeros(2**N, dtype=int)

    for i in range(2**N):
        b = in_binary(i,N)
        s = 0
        factor = 1
        for p in range(m):
            s = s + factor * np.sum(b[cluster[p]])
            factor = factor * (sizes[p] + 1)
        partition[i] = s

    return partition


# Returns a partition of the state-space based on the number of edges of each type
# cluster: Array of size m (number of clusters) with arrays resembling classes containing their nodes
# G: The adjecency matrix of the graph. Should only contain the values 0, and 1, with diagonal only 0's
def partinion_edge_based(cluster, G):
    N = len(G)
    m = len(cluster)

    sizes = np.zeros(m)
    for i in range(m):
        sizes[i] = len(cluster[i])

    G = G + G.transpose()
    G = np.array(G, dtype=bool)
    for i in range(N):
        G[i][i]= False

    ids = np.zeros((2**N, m*m + m), dtype=int)

    for i in range(2**N):
        b = in_binary(i,N)
        b_inv = np.invert(b)
        id = np.zeros(m*m + m)
        for j in range(m):
            for k in range(m):
                sub = sub_matrix(G, cluster[j], cluster[k])
                sub = np.array(sub, dtype=int)
                II = b[cluster[j]]@sub@b[cluster[k]]
                SS = b_inv[cluster[j]]@sub@b_inv[cluster[k]]
                factor = np.sum(G)
                id[j*m + k] = SS * factor + II

            id[m*m + j] = np.sum(b[cluster[j]])

        ids[i] = id

    partition = partition_by_id(ids)

    return partition


# Returns a partition that groups states by their value in the QSD.
# k: number of classes in the partition, i.e. dimension of the coarse-grained system.
# qsd: second leading eigenvector. Should contain the zero state.
def partition_k_means(k, qsd):
    n = len(qsd)
    partition = np.zeros(n, dtype=int)

    obs = qsd[1:].reshape(-1,1)
    partition[1:] = KMeans(k-1).fit(obs).labels_ + np.ones(n-1)

    return partition
