import numpy as np

from kdtree import KdTree


UNDEFINED = None
NOISE = -1

def dbscan(points, eps, minpts):

    labels = [UNDEFINED for _ in range(len(points))]

    kdtree = KdTree(points)

    cluster_label = 0

    for i, p in enumerate(points):

        if labels[i] != UNDEFINED:
            continue

        neighboring_idxs = set(kdtree.range_query(p, eps, return_idx=True))

        if len(neighboring_idxs) < minpts:
            label[i] = NOISE
            continue

        labels[i] = cluster_label

        neighboring_idxs.remove(i)

        for q in neighboring_idxs:
            if labels[q] == NOISE:
                labels[q] = cluster_label
            if labels[q] != UNDEFINED:
                continue

            other_neighbors = set(kdtree.range_query(points[q], eps, return_idx=True))
            labels[q] = cluster_label

            if len(other_neighbors) < minpts:
                continue

            neighboring_idxs = neighboring_idxs | other_neighbors

        cluster_label += 1

    return np.array(labels, dtype=np.int)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    def create_clusters(n_clusters, eps, minpts):

        npts_per_cluster = minpts * 2
        pts = np.random.randn(n_clusters * npts_per_cluster, 2)

        offset = 2*eps     
        for ith_cluster in range(1, n_clusters):

            beg_idx = npts_per_cluster * ith_cluster
            end_idx = beg_idx + npts_per_cluster

            pts[beg_idx:end_idx] += offset

            offset += 2*eps

        return pts

    nclusters = 3
    eps = 5
    minpts = 5

    pts = create_clusters(nclusters, eps, minpts)

    labels = dbscan(pts, eps, minpts)

    plt.scatter(pts[:,0], pts[:,1])
    for i in range(nclusters):
        
        idxs = labels == i
        plt.scatter(pts[idxs,0], pts[idxs,1])

    plt.show()
            
        
