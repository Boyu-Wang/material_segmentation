#!/usr/bin/env python         
# A simple example for web annotation using Python + CGI
# http://vision.cs.stonybrook.edu/cgi-bin/AnnoEx/cgi-bin/annotate2_server.cgi?expinfo=YoungJaeShinSamples_4&id=0&user=test0123
# http://vision.cs.stonybrook.edu/cgi-bin/AnnoEx/cgi-bin/annotate2_server.cgi?expinfo=EXPT2TransferPressure_HighPressure&id=0&user=test0123
import cgi, cgitb, sqlite3, os, random, glob
cgitb.enable()

print "Content-type: text/html" 
print """
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Annotation Tool</title>
</head>                                                                      
<body  id="main_body">
"""     

form = cgi.FieldStorage();
thicklabel = form.getvalue('thicklabel') # label of previously annotated image
qualitylabel = form.getvalue('qualitylabel') # label of previously annotated image
id = int(form.getvalue('id'))
user = form.getvalue('user')
expinfo = form.getvalue('expinfo')
assert user is not None
assert expinfo is not None
# cluster_id = int(form.getvalue('cluster_id'))

# expid = int(form.getvalue('expid'))
# subexpid = int(form.getvalue('subexpid'))
exp_name, subexp_name = expinfo.split('_')
# exp_name = 'EXPT2TransferPressure'
# subexp_name = 'HighPressure'

ori_img_path = '../data/data_jan2019'
rslt_path = '../rslt/data_jan2019_script/cluster_sort/'
fea_type = 'all'
cluster_method = 'affinity'

subori_img_path = os.path.join(ori_img_path, exp_name, subexp_name)
subcluster_rslt_path = os.path.join(rslt_path, exp_name, subexp_name, '%s_%s'%(fea_type, cluster_method))
cluster_names_all = os.listdir(subcluster_rslt_path)
cluster_names = []
for cname in cluster_names_all:
	if os.path.isdir(os.path.join(subcluster_rslt_path, cname)):
		cluster_names.append(cname)

num_clusters = len(cluster_names)

subannos_db_path = os.path.join('../data/anno/', exp_name, subexp_name, '%s_%s'%(fea_type, cluster_method))
if not os.path.exists(subannos_db_path):
	os.makedirs(subannos_db_path)
annos_db_name = os.path.join(subannos_db_path, 'anno_user%s.db'%(user))

anno_file = os.path.join(subannos_db_path, 'anno_file.txt')
total_annos = 3000
if not os.path.exists(anno_file):
	# generate the file
	# from gen_filelist import gen_filelist as gen_filelist
	from gen_filelist import gen_filelist2 as gen_filelist
	gen_filelist(subcluster_rslt_path, anno_file, total_annos)

fp = open(anno_file)

for i, line in enumerate(fp):
	if (thicklabel is not None or qualitylabel is not None) and i == id -1: # add or update the annotation database, for the one annotated from previous step
		flake_name_db = line.strip()
		
		conn = sqlite3.connect(annos_db_name)
		c = conn.cursor()
		c.execute('''CREATE TABLE IF NOT EXISTS annotab (imid STRING PRIMARY KEY, thicklabel STRING, qualitylabel STRING)''')
		t = (flake_name_db, thicklabel, qualitylabel)
		c.execute("INSERT OR REPLACE INTO annotab(imid, thicklabel, qualitylabel) VALUES (?, ?, ?)", t)
		conn.commit()

		print """<h2> You selected: '%s, %s' for the previous image. 
					<a href="annotate2_server.cgi?expinfo=%s&id=%d&user=%s">Click here to relabel it</a> </h2>""" % (thicklabel, qualitylabel, expinfo, id - 1, user)

	elif i == id:
		next_im_id = line.strip()
		break			
fp.close()


if (next_im_id is None) or (not next_im_id):
	print "<h2>Great. No more image to label!</h2>"
else: # display the next image for annotation
	cluster_id, flake_name = next_im_id.split(',')
	flake_bw_name = flake_name[:-4] + '_bw.png'
        flake_bbox_name = flake_name[:-4] + '_bbox.png'
	ori_img_name = flake_name.split('-')[1] + '.tif'
	ori_img_fullpath = os.path.join(subori_img_path, ori_img_name)
	flake_name_fullpath = os.path.join(subcluster_rslt_path, str(cluster_id), flake_name)
	flake_bw_name_fullpath = os.path.join(subcluster_rslt_path, str(cluster_id), flake_bw_name)
        flake_bbox_name_fullpath = os.path.join(subcluster_rslt_path, str(cluster_id), flake_bbox_name)

	print """ 
		<div id="form_container">      
			<form id="form_390674" class="appnitro"  method="post" action="annotate2_server.cgi?expinfo=%s&id=%d&user=%s">      
			<h2>Annotate the below image</h2>
                        <img src="http://vision.cs.stonybrook.edu/~boyu/material_segmentation/%s" height=320 alt="">
                        <img src="http://vision.cs.stonybrook.edu/~boyu/material_segmentation/%s" height=320 alt="">
                        <img src="http://vision.cs.stonybrook.edu/~boyu/material_segmentation/%s" height=180 alt="">
                        <img src="http://vision.cs.stonybrook.edu/~boyu/material_segmentation/%s" height=180 alt="">
			<br>
			<p>What is the Thickness of this flake? </p>
			<input type="radio" name="thicklabel" value="1"  > 1<br>
			<input type="radio" name="thicklabel" value="2" > 2 <br>
			<input type="radio" name="thicklabel" value="3" > 3 <br>
			<input type="radio" name="thicklabel" value="thick" > thick <br>
			<input type="radio" name="thicklabel" value="verythick" > verythick <br>
			<input type="radio" name="thicklabel" value="multithick" > multithick <br>
			<input type="radio" name="thicklabel" value="not sure" > not sure <br>
			<p>What is the Quality of this flake? </p>
			<input type="radio" name="qualitylabel" value="no crack no fold" onclick="this.form.submit()" > no crack no fold<br>
			<input type="radio" name="qualitylabel" value="slightly crack fold" onclick="this.form.submit()"> slightly crack fold<br>
			<input type="radio" name="qualitylabel" value="unusable" onclick="this.form.submit()"> unusable<br>
			<input type="radio" name="qualitylabel" value="glue junk" onclick="this.form.submit()"> glue junk<br>
			<input type="radio" name="qualitylabel" value="segmentation error" onclick="this.form.submit()"> segmentation error <br>
			<input type="radio" name="qualitylabel" value="others" onclick="this.form.submit()"> others <br>
			
                </form> """ % (expinfo, id+1, user, ori_img_fullpath[3:], flake_bbox_name_fullpath[3:], flake_name_fullpath[3:], flake_bw_name_fullpath[3:])

print """</body></html>""" 



# if n_annos < total_annos:
# 	if (thicklabel is not None or qualitylabel is not None) and i == id -1: # add or update the annotation database, for the one annotated from previous step
# 	# random sample one cluster
# 	cluster_id = random.randint(0, num_clusters-1)
# 	# random sample one image from the cluster
# 	c_fp = open(os.path.join(subcluster_rslt_path, str(cluster_id), 'names.txt'))
# 	c_names = c_fp.readlines()
# 	c_nimg = len(c_names)
# 	flake_id = random.randint(0, c_nimg-1)
# 	flake_name = c_names[c_nimg].strip()
# 	flake_name_db = '%d'%(cluster_id) + ',' + flake_name

# 	conn = sqlite3.connect(annos_db_name)
# 	c = conn.cursor()
# 	c.execute('''CREATE TABLE IF NOT EXISTS annotab (imid STRING PRIMARY KEY, thicklabel STRING, qualitylabel STRING)''')
# 	t = (flake_name_db, thicklabel, qualitylabel)
# 	c.execute("INSERT OR REPLACE INTO annotab(imid, thicklabel, qualitylabel) VALUES (?, ?, ?)", t)
# 	conn.commit()

# 	flake_bw_name = flake_name[:-4] + '_bw.png'
# 	ori_img_name = flake_name.split('-')[1] + '.tif'
# 	ori_img_fullpath = os.path.join(subori_img_path, ori_img_name)
# 	flake_name_fullpath = os.path.join(subcluster_rslt_path, str(cluster_id), flake_name)
# 	flake_bw_name_fullpath = os.path.join(subcluster_rslt_path, str(cluster_id), flake_bw_name)

# 	# display the next image for annotation
# 	print """ 
# 		<div id="form_container">      
# 			<form id="form_390674" class="appnitro"  method="post" action="annotate.cgi?id=%d">      
# 			<h2>Annotate the below image</h2>
# 			<img src="../data/%s" height=480 alt="">
# 			<br>
# 			<p>Is it what you are looking for? </p>
# 			<input type="radio" name="label" value="Yes" onclick="this.form.submit()" > Yes<br>
# 			<input type="radio" name="label" value="No" onclick="this.form.submit()"> No <br>
# 			<input type="radio" name="label" value="Maybe" onclick="this.form.submit()"> Maybe <br>
# 		</form> """ % (id+1, next_im_id)

# else:
# 	print "<h2>Great. No more image to label!</h2>"


# print """</body></html>""" 


# fp = open("../data/ids.txt") # file contain list of image ids for annotation
# for i, line in enumerate(fp):
# 	if label is not None and i == id -1: # add or update the annotation database, for the one annotated from previous step
# 		im_id = line.strip()
		
# 		conn = sqlite3.connect('../data/anno.db')
# 		c = conn.cursor()
# 		c.execute('''CREATE TABLE IF NOT EXISTS annotab (imid STRING PRIMARY KEY, label STRING)''')
# 		t = (im_id, label)
# 		c.execute("INSERT OR REPLACE INTO annotab(imid, label) VALUES (?, ?)", t)
# 		conn.commit()

# 		print """<h2> You selected: '%s' for the previous image. 
# 					<a href="annotate.cgi?id=%d">Click here to relabel it</a> </h2>""" % (label, id - 1)

# 	elif i == id:
# 		next_im_id = line.strip()
# 		break			
# fp.close()

# if (next_im_id is None) or (not next_im_id):
# 	print "<h2>Great. No more image to label!</h2>"
# else: # display the next image for annotation
# 	print """ 
# 		<div id="form_container">      
# 			<form id="form_390674" class="appnitro"  method="post" action="annotate.cgi?id=%d">      
# 			<h2>Annotate the below image</h2>
# 			<img src="../data/%s" height=480 alt="">
# 			<br>
# 			<p>Is it what you are looking for? </p>
# 			<input type="radio" name="label" value="Yes" onclick="this.form.submit()" > Yes<br>
# 			<input type="radio" name="label" value="No" onclick="this.form.submit()"> No <br>
# 			<input type="radio" name="label" value="Maybe" onclick="this.form.submit()"> Maybe <br>
# 		</form> """ % (id+1, next_im_id)

# print """</body></html>""" 
