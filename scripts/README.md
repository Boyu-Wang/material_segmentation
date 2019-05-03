# Prerequirements

The scripts are writeen in python 3.6, you need the following package to run the code:

+ opencv3
+ skimage
+ sklearn
+ pickle
+ joblib
+ multiprocessing 

to install opencv3:
> conda install -c anaconda opencv

export PYTHONPATH=/home/boyu/anaconda2/envs/pytorch1.0/lib/python3.6/site-packages/:$PYTHONPATH


# Region Segmentation

The method use robustfit to detect foreground regions (flakes or glue). Given a image, the algorithm tries to fit a function as background. Everything doesn't fit the background function well are identified as outlier (flake).

For each detected flake, save all related informations, e.g:

+ flake size, center
+ bbox
+ contour pixel coordinates (sorted clockwise)
+ shape feature: length area ratio, fract dimention, center to contour distance distribution
+ color stats: mean, std, entropy. for both flake region and inner flake region

To run the code:

```
python flake_segmentaion.py --exp_sid 0 --exp_eid 1
```

# Region clustering
cluster the detected regions based on different features: shape, color, and both.

Use different clustering method: kMeans, spectral clustering. Spectral clustering tends to generate more clusters, and the flakes within the same cluster are more similar.


To run the code:

``` 
python flake_clustering.py --exp_sid 0 --exp_eid 1
```


# Flake/Glue classification

Given cluster level annotation, which is stored at ```
../data/data_jan2019_anno/anno_cluster_YoungJaeShinSamples_4_useryoungjae.db```

the initial classifier can be trained using:

```
python flake_classify_round0_clusterinfo.py
```

After the initial classifier, an online learning based annotation tool is used to suggest and label for each flake. The collected annotation is stored at ```
../data/data_jan2019_anno/anno_flakeglue_useryoungjae.db```

To train the classifier using accurate annotation:

```
python flake_classify_round1_flakeinfo.py
```

# Autoencoder for region features
Instead of using handcrafted features for each region, using autoencoder to learn features for each region. Put each region in the center of a black 256*256 image. The autoencoder tries to reconstruct such image. 

Learn autoencoder:

```
python flake_glue_AE.py --gpuid 0 
```

Use the pretrained autoencoder to extract features and learn flake/glue classifier:

```
python flake_glue_AE.py --gpuid 0 --evaluate 1 --handfea 1 --C 2
```


# Annoation:
1. annotation tool for cluster level annotation:

Check out ```annotation/annotate_cluster_server.cgi```

2. annotation tool for region level annotation (use 1st annotation to learn a initial classifier and online learning for suggestions):

Check out ```annotation/annotate_online_server.cgi```

3. anntation each image separately with strike drawing:

Check out ```annotation/flake_labeling.py``




