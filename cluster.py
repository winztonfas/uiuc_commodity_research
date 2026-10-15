from narwhals import corr
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def hrp_1(ret):
    # correlation matrix
    corr = ret.corr()

    if not np.isfinite(corr.values).all():
        raise ValueError("Correlation matrix contains non-finite values")

    corr_values = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_values, 1)
    corr = pd.DataFrame(corr_values, index=corr.index, columns=corr.columns)
    
    # distance matrix
    d_corr = np.sqrt(0.5*(1-corr))

    dist = squareform(d_corr.values, checks=False)
    link = linkage(dist, method="single")

    return d_corr, link

def size_controlled_clusters(
    d_corr,
    link,
    base_distance=0.43,
    stricter_distance=0.35,
    min_size=2,
    max_size=4):

    labels = fcluster(link, t=base_distance, criterion="distance")
    cluster_map = pd.Series(labels, index=d_corr.index)

    final_clusters = []

    for _, members in cluster_map.groupby(cluster_map).groups.items():
        members = list(members)

        if len(members) > max_size:
            final_clusters.extend(
                split_large_clusters(
                    d_corr,
                    members,
                    max_size=max_size,
                    stricter_distance=stricter_distance,))
        else:
            final_clusters.append(members)

    final_clusters = [
        c for c in final_clusters
        if min_size <= len(c) <= max_size]

    return final_clusters

def split_large_clusters(
    d_corr,
    products,
    max_size=4,
    stricter_distance=0.35,
    min_distance=0.05,
    distance_decay=0.9):
    if len(products) <= max_size:
        return [products]

    # quasi, re-order 
    sub_d = d_corr.loc[products, products]
    dist = squareform(sub_d.values, checks=False)
    sub_link = linkage(dist, method="single")

    labels = fcluster(sub_link, t=stricter_distance, criterion="distance")

    sub_clusters = []
    for label in sorted(set(labels)):
        members = sub_d.index[labels == label].tolist()

        if len(members) > max_size and stricter_distance > min_distance:
            sub_clusters.extend(
                split_large_clusters(
                    d_corr,
                    members,
                    max_size=max_size,
                    stricter_distance=stricter_distance * distance_decay,
                    min_distance=min_distance,
                    distance_decay=distance_decay,
                )
            )
        else:
            sub_clusters.append(members)

    return sub_clusters

def get_cluster(
    d_corr,link,
    base_distance=0.43,stricter_distance=0.35,
    min_size=2, max_size=4):
    
    clusters = size_controlled_clusters(
    d_corr, link,
    base_distance=base_distance,
    stricter_distance=stricter_distance,
    min_size=min_size,
    max_size=max_size,)
    
    return clusters

def main():
    