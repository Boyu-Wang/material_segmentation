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
 python flake_segmentation.py --exp_sid 0 --exp_eid 1 --sub_exp_sid 0 --sub_exp_eid 5 --known_bg 0
```

If background image is unknown, set `known_bg` to 0. It will estimate the background image using robustfit on single image.

If background image is known, set  `known_bg` to 1, and provide the background image name by changing Line 311 in the file. ( set `bg_img_name`).


This will process the first 5 experiments (you can change the number in sub_exp_sid and sub_exp_eid to others). The results of this script will be saved in results/data_jan2019_script/mat (for detailed pickle file), and results/data_jan2019_script/fig (for visualization).

2. estimate the background
```
 python estimate_background.py --exp_sid 0 --exp_eid 1 --sub_exp_sid 0 --sub_exp_eid 1 --bg_option single_fit
```

Given a set of images, this process will take the median of original raw images as background images.
 
3. then run the classification:
```
 python flake_classify.py --exp_sid 0 --exp_eid 1 --sub_exp_sid 0 --sub_exp_eid 1 --topk 100
```

This one classifies regions segmented from the previous step. For each image, it will classify the flakes in the image into 5 different types, and save the result images as follows:

    graphene: white
    junk: green
    thin: red
    thick: orange
    multi: pink
For a set of images, it will output top 100 predicted graphenes. You could change the output number by setting `topk`

