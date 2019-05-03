"""
Convert the cluster annotation to detailed flake annotation:
    ori_imgname, ori_flakeid, cluster_id, flakecenter_x, flakecenter_y, flakesize, withincluster_id

By: Boyu Wang (boywang@cs.stonybrook.edu)
Created Data: 28 Apr 2019
Last Modified Date: 28 Apr 2019
"""

from PIL import Image
import os
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from skimage.morphology import disk
import pickle
import sqlite3
from multiprocessing import Pool
from joblib import Parallel, delayed


def load_one_cluster(cluster_name_label, cluster_save_path, result_path):
    cluster_name = cluster_name_label[0]
    thicklabel = cluster_name_label[1]
    cluster_id = int(cluster_name.split('-')[1])
    batch_id = int(cluster_name.split('.')[0].split('-')[2])
    num_row = 5
    flake_start_id = num_row * num_row * batch_id
    # cluster_save_path = os.path.join(cluster_save_path, str(cluster_id))
    names_path = os.path.join(cluster_save_path, str(cluster_id), 'names.txt')
    with open(names_path, 'r') as f:
        names = f.readlines()
    num_flakes = len(names)
    flake_end_id = min(num_row * num_row * (batch_id+1), num_flakes)

    info = []
    for fid in range(flake_start_id, flake_end_id):
        # find the original image name, and flake id
        img_name = names[fid].split('-')[1]
        # img_id = [ nn for nn in range(len(all_img_names)) if img_name in all_img_names[nn]]
        # assert len(img_id) == 1
        # img_id = img_id[0]
        flake_id = int(names[fid].split('-')[2].split('.')[0])
        # ids.append([img_id, flake_id])
        oriname_oriflakeid = img_name + '-' + str(flake_id)

        oriname = img_name.split('.')[0] + '.p'
        flake_info = pickle.load(open(os.path.join(result_path, oriname), 'rb'))['flakes']
        flake_size = int(flake_info[flake_id-1]['flake_size'])
        flakecenterx, flakecentery = flake_info[flake_id-1]['flake_center']
        info.append([oriname_oriflakeid, thicklabel, cluster_id, fid, int(flakecenterx), int(flakecentery), flake_size])

    return info



# read the cluster level annotation file
def readdb(cluster_dbname):
    conn = sqlite3.connect(cluster_dbname)
    c = conn.cursor()
    c.execute('PRAGMA TABLE_INFO({})'.format('annotab'))
    info = c.fetchall()
    col_dict = dict()
    for col in info:
        col_dict[col[1]] = 0

    c.execute('SELECT * FROM annotab')
    db = c.fetchall()

    clustername_labels = []
    for i in range(len(db)):
        clustername_labels.append([db[i][0], db[i][1]])

    return clustername_labels


def main():
    anno_file = '../data/data_jan2019_anno/anno_cluster_YoungJaeShinSamples_4_useryoungjae.db'
    detailed_anno_file = '../data/data_jan2019_anno/anno_cluster_detailed_YoungJaeShinSamples_4_useryoungjae.db'
    cluster_save_path = '../results/data_jan2019_script/cluster_sort_784/YoungJaeShinSamples/4/all_affinity'
    result_path = '../results/data_jan2019_script/mat/YoungJaeShinSamples/4/'
    clustername_labels = readdb(anno_file)
    n_total_clusters = len(clustername_labels)

    # img_names = os.listdir(subexp_dir)
    # img_names.sort()
    
    cluster_imgflake_ids = Parallel(n_jobs=8)(delayed(load_one_cluster)(clustername_labels[i], cluster_save_path, result_path) for i in range(len(clustername_labels)))
    print('loading done')
    conn = sqlite3.connect(detailed_anno_file)
    for i in range(n_total_clusters):
        cluster_info = cluster_imgflake_ids[i]
        n_exp = len(cluster_info)
        # print(i)
        for j in range(n_exp):
            # oriname = cluster_info[j].split('-')[0].split('.')[0] + '.p'
            # oriflakeid = int(cluster_info[j].split('-')[1])
            # print(oriflakeid)
            # print(os.path.join(result_path, oriname))
            # flake_info = pickle.load(open(os.path.join(result_path, oriname), 'rb'))['flakes']
            # print(len(flake_info))
            # flake_size = flake_info[oriflakeid-1]['flake_size']
            # flakecenterx, flakecentery = flake_info[oriflakeid-1]['flake_center']
            # print(cluster_info[j])
            oriname_oriflakeid, thicklabel, clusterid, withinclusterid, flakecenterx, flakecentery, flake_size = cluster_info[j]
            # print(oriname_oriflakeid, flakecenterx, flakecentery, flake_size)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS annotab (oriname_oriflakeid STRING PRIMARY KEY, thicklabel STRING, clusterid INTEGER, withinclusterid INTEGER, flakecenterrow INTEGER, flakecentercolumn INTEGER, flakesize INTEGER)''')
            t = (oriname_oriflakeid, thicklabel, clusterid, withinclusterid, flakecenterx, flakecentery, flake_size)
            c.execute("INSERT OR REPLACE INTO annotab(oriname_oriflakeid, thicklabel, clusterid, withinclusterid, flakecenterrow, flakecentercolumn, flakesize) VALUES (?, ?, ?, ?, ?, ?, ?)", t)
            conn.commit()


if __name__ == '__main__':
    main()
