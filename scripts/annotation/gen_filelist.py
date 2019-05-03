import os
import random
import numpy as np

def gen_filelist(subcluster_rslt_path, anno_file_name, total_annos=3000):
    cluster_names_all = os.listdir(subcluster_rslt_path)
    cluster_names = []
    for cname in cluster_names_all:
        if os.path.isdir(os.path.join(subcluster_rslt_path, cname)):
            cluster_names.append(cname)
    num_clusters = len(cluster_names)
    # first sample 1 images from each clusters
    sampled_cluster_ids = []

    # the distribution of cluster, beginning clusters have more weights
    cluster_bag = np.arange(0, num_clusters) 
    alpha = 10/num_clusters;
    cluster_dis = 1/(1+np.exp(alpha*(cluster_bag-num_clusters/2)))
    cluster_dis = cluster_dis / cluster_dis.sum()
    for i in range(total_annos - num_clusters):
        sampled_cluster_ids.append(np.random.choice(cluster_bag, p=cluster_dis))

    sampled_cluster_ids.extend(list(cluster_bag))
    sampled_cluster_ids.sort()
    sampled_cluster_ids = np.array(sampled_cluster_ids)

    # sample within each clusters
    sampled_flakes = []

    for c in range(num_clusters):
        n_c = (sampled_cluster_ids == c).sum()
        assert n_c >= 1

        c_fp = open(os.path.join(subcluster_rslt_path, str(c), 'names.txt'))
        c_names = c_fp.readlines()
        c_nimg = len(c_names)
        c_flake_ids = np.random.permutation(c_nimg)[:n_c]
        c_flake_names = [str(c) + ',' + c_names[j].strip() for j in c_flake_ids]
        sampled_flakes.extend(c_flake_names)

    # write to file
    with open(anno_file_name, 'w') as f:
        f.write('\n'.join(sampled_flakes))


# average probabily for each cluster
def gen_filelist2(subcluster_rslt_path, anno_file_name, total_annos=3000):
    cluster_names_all = os.listdir(subcluster_rslt_path)
    cluster_names = []
    for cname in cluster_names_all:
        if os.path.isdir(os.path.join(subcluster_rslt_path, cname)):
            cluster_names.append(cname)
    num_clusters = len(cluster_names)
    # first sample 1 images from each clusters
    sampled_cluster_ids = []

    num_rep = int(np.ceil(total_annos * 1.0/ num_clusters))
    # the distribution of cluster, beginning clusters have more weights
    cluster_bag = np.arange(0, num_clusters)
    cluster_bag = np.tile(np.expand_dims(cluster_bag, 1), [1, num_rep]).reshape([-1])
    # alpha = 10/num_clusters;
    # cluster_dis = 1/(1+np.exp(alpha*(cluster_bag-num_clusters/2)))
    # cluster_dis = cluster_dis / cluster_dis.sum()
    # for i in range(total_annos - num_clusters):
    #     sampled_cluster_ids.append(np.random.choice(cluster_bag, p=cluster_dis))

    sampled_cluster_ids.extend(list(cluster_bag))
    # sampled_cluster_ids.sort()
    sampled_cluster_ids = np.array(sampled_cluster_ids)
    # np.random.shuffle(sampled_cluster_ids)

    # sample within each clusters
    sampled_flakes = []

    for c in range(num_clusters):
        n_c = (sampled_cluster_ids == c).sum()
        assert n_c >= 1

        c_fp = open(os.path.join(subcluster_rslt_path, str(c), 'names.txt'))
        c_names = c_fp.readlines()
        c_nimg = len(c_names)
        c_flake_ids = np.random.permutation(c_nimg)[:n_c]
        c_flake_names = [str(c) + ',' + c_names[j].strip() for j in c_flake_ids]
        sampled_flakes.extend(c_flake_names)

    random.shuffle(sampled_flakes)

    # write to file
    with open(anno_file_name, 'w') as f:
        f.write('\n'.join(sampled_flakes))





