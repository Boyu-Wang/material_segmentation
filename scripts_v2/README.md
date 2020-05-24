# Prerequirements

The scripts are writeen in python 3.6, you need the following package to run the code:

+ opencv3
+ skimage
+ sklearn
+ pickle
+ joblib
+ multiprocessing 
+ pytorch 1.0 

to install opencv3:
> conda install -c anaconda opencv

export PYTHONPATH=/home/boyu/anaconda2/envs/pytorch1.0/lib/python3.6/site-packages/:$PYTHONPATH


# Steps to run the code on new data
You can download new data, and put it under the directory: data/data_111x_individual/



1. flake segmentation:
```
 python flake_segmentation.py --exp_sid 0 --exp_eid 1 --subexp_sid 0 --subexp_eid 5 --known_bg 0
```

If background image is unknown, set `known_bg` to 0. It will estimate the background image using robustfit on single image.

If background image is known, set  `known_bg` to 1, and provide the background image name by changing Line 311 in the file. ( set `bg_img_name`).


This will process the first 5 experiments (you can change the number in `subexp_sid` and `subexp_eid` to others). The results of this script will be saved in `results/data_jan2019_script/mat_2.0_100` (for detailed pickle file), and `results/data_jan2019_script/fig_2.0_100` (for visualization).


The flake features are saved in `flake_shape_fea`, `flake_color_fea`, `flake_contrast_color_fea`, `flake_innercontrast_color_fea`, `flake_bg_color_fea`, `subsegment_features_3`. The detailed description of each feature vector is saved in a list of names. This list is the same size as feature vector, each element represent the feature meaning. e.g: `flake_shape_fea_names`, `flake_color_fea_names`, `flake_contrast_color_fea_names`, `flake_bg_color_fea_names`.


2. estimate the background
```
 python estimate_background.py --exp_sid 0 --exp_eid 1 --subexp_sid 0 --subexp_eid 1 --bg_option single_fit
```

Given a set of images, this process will take the median of original raw images as background images.
 
3. then run the classification:
```
 python flake_classify.py --exp_sid 0 --exp_eid 1 --subexp_sid 0 --subexp_eid 1 --topk 100 --color_fea threesub-contrast-bg-shape
```

This one classifies regions segmented from the previous step. The feature type can be changed by modifying `color_fea`: `threesub-contrast-bg-shape` contains subsegment features, contrast features, background features, and shape features'; `firstcluster-contrast-bg-shape` contains the features from the clusters with small gray value within three subsegments, contrast features, background features, and shape features.

The corresponding classifier are saved in `../results/pretrained_clf/graphene_classifier_with_moreanno_v3_colorfea-threesub-contrast-bg-shape` and `../results/pretrained_clf/graphene_classifier_with_moreanno_v3_colorfea-firstcluster-contrast-bg-shape` 

For each image, it will classify the flakes in the image into 5 different types, and save the result images as follows:

    graphene: white
    junk: green
    thin: red
    thick: orange
    multi: pink

The result will be saved to `classify_colorfea-threesub-contrast-bg-shape_2.0_100`

For a set of images, it will output top 100 predicted graphenes. You could change the output number by setting `topk`. The result will be saved to  `classify_colorfea-threesub-contrast-bg-shape_2.0_100_top-100`.


4. train classifiers with different features.

```
 python classifier_training.py --annotation_path ../data/anno_graphene_v3_youngjae  --color_fea threesub-contrast-bg-shape
```
This script will train a multi class classifier using annotated files in `../data/anno_graphene_v3_youngjae`. Change this path to new annotation path. If you want to use different features, change `color_fea`. For a more customized features, change Line 110 - 153. 

5. annotation:

The annotation file is `annotate_graphene.cgi`. To annotate images:

1) change the `rslt_path` in Line 35 to be the path of predicted graphenes that you want to annotate, for example: `../result/data_111x_individual_result/classify_colorfea-threesub-contrast-bg-shape_2.0_100_top-100`.
2) move the file `annotate_graphene.cgi` to places that host the web service, e.g: `/usr/lib/cgi-bin/`
3) annotate using links like: `http://vision.cs.stonybrook.edu/cgi-bin/AnnoEx/cgi-bin/annotate_graphene.cgi?fea=threesub-contrast-bg-shape&expinfo=PDMS-QPress 60s_2&size=100-20000&id=0&user=testuser123`. Change the address accordingly. change the `fea=` to features you used, change the `expinfo` to the experiment name and subexperiment name, connectted by `_`, change `user=` to the a specific user id to avoid annotation override by others.




