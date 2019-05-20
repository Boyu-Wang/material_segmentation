#!/usr/bin/env python3         
# A simple example for web annotation using Python + CGI
# without online learning for flake thickness annotation, just read from prediction
# http://vision.cs.stonybrook.edu/cgi-bin/AnnoEx/cgi-bin/annotate_online_thicknessv2_server.cgi?expinfo=YoungJaeShinSamples_4&sizethre=784&id=0&user=test0123
import cgi, cgitb, sqlite3, os, random, glob, pickle
import numpy as np
cgitb.enable()

print("Content-type: text/html")
print("""
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Annotation Tool</title>
</head>                                                                      
<body  id="main_body">
""")

form = cgi.FieldStorage();
# thicklabel = form.getvalue('thicklabel') # label of previously annotated image
# qualitylabel = form.getvalue('qualitylabel') # label of previously annotated image
id = int(form.getvalue('id'))
sizethre = int(form.getvalue('sizethre'))
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
if sizethre == 784:
	rslt_path = '../rslt/data_jan2019_script/cluster_sort_784_thickness_online_v2/'
# elif sizethre == 100:
# 	rslt_path = '../rslt/data_jan2019_script/cluster_sort/'
else:
	raise NotImplementedError

n_row = 2
n_column = 5

# fea_type = 'all'
# cluster_method = 'affinity'
clf_method = 'linearsvm'
subori_img_path = os.path.join(ori_img_path, exp_name, subexp_name)
# subcluster_rslt_path = os.path.join(rslt_path, exp_name, subexp_name, '%s_%s'%(fea_type, cluster_method))
subonline_rslt_path = os.path.join(rslt_path, exp_name, subexp_name + '_' + clf_method)
with open(os.path.join(subonline_rslt_path, 'all_test_names.txt'), 'r') as f:
	all_test_names_predlabels = f.readlines()
with open(os.path.join(subonline_rslt_path, 'glue_test_names.txt'), 'r') as f:
	all_glue_test_names_predlabels = f.readlines()
with open(os.path.join(subonline_rslt_path, 'thin_test_names.txt'), 'r') as f:
	all_thin_test_names_predlabels = f.readlines()
with open(os.path.join(subonline_rslt_path, 'thick_test_names.txt'), 'r') as f:
	all_thick_test_names_predlabels = f.readlines()

all_test_names = [n.split(',')[0] for n in all_test_names_predlabels]
all_predlabels = [n.split(',')[1] for n in all_test_names_predlabels]
all_glue_test_names = [n.split(',')[0] for n in all_glue_test_names_predlabels]
all_thin_test_names = [n.split(',')[0] for n in all_thin_test_names_predlabels]
all_thick_test_names = [n.split(',')[0] for n in all_thick_test_names_predlabels]
num_glue = len(all_glue_test_names)
num_thin = len(all_thin_test_names)
num_thick = len(all_thick_test_names)
# all_test_feas = pickle.load(open(os.path.join(subonline_rslt_path, 'all_test_feas.p'), 'rb'))

subannos_db_path = os.path.join('../data/anno_thicknessv2_online%d/'%(sizethre), exp_name, subexp_name+ '_' + clf_method, user)
if not os.path.exists(subannos_db_path):
	os.makedirs(subannos_db_path)
annos_db_name = os.path.join(subannos_db_path, 'anno_user%s.db'%(user))

# if not os.path.exists(os.path.join(subannos_db_path, 'thickness_classifier-%s_b-%d.p'%(clf_method, -1))):
# 	src_path = os.path.join(subonline_rslt_path, 'thickness_classifier-%s_b-%d.p'%(clf_method, -1))
# 	des_path = os.path.join(subannos_db_path, 'thickness_classifier-%s_b-%d.p'%(clf_method, -1))
# 	os.system("cp %s %s" %(src_path, des_path))

# labeled_test_names_file = os.path.join(subannos_db_path, 'labeled_test_names.txt')
# labeled_batch_file = os.path.join(subannos_db_path, 'labeled_batch.p')
# fb = open(labeled_batch_file, 'a+')
# last_labeled_test_names_file = os.path.join(subannos_db_path, 'last_labeled_test_names.txt')

labeled_test_names = []
last_labeled_test_names = []
# read all labeled files
for bi in range(id):
	labeled_name_batch_file = os.path.join(subannos_db_path, 'labeled_test_names_b-%d.txt'%(bi))
	with open(labeled_name_batch_file, 'r') as f:
		batch_names = f.readlines()
	labeled_test_names = labeled_test_names + [n.strip() for n in batch_names]
	if bi == id - 1:
		last_labeled_test_names = [n.strip() for n in batch_names]

# if os.path.exists(labeled_test_names_file):
# 	with open(labeled_test_names_file, 'r') as f:
# 		labeled_test_names = f.readlines()
# 	labeled_test_names = [n.strip() for n in labeled_test_names]
# else:
# 	labeled_test_names = []

if len(last_labeled_test_names) > 0:
	# update the annoation database, for the one annotated from previous step, and update the classifier
	conn = sqlite3.connect(annos_db_name)
	c = conn.cursor()
	# last_clf = pickle.load(open(os.path.join(subannos_db_path, 'thickness_classifier-%s_b-%d.p'%(clf_method, id-2)), 'rb'))
	# Cn_inv = last_clf['Cn_inv'] 
	# last_coef = last_clf['coef'] # [dim, 1]
	selected_feas = []
	selected_labels = []
	for name_i in last_labeled_test_names:
		anno_i = form.getvalue(name_i)
		c.execute('''CREATE TABLE IF NOT EXISTS annotab (imflakeid STRING PRIMARY KEY, thicklabel STRING)''')
		t = (name_i, anno_i)
		c.execute("INSERT OR REPLACE INTO annotab(imflakeid, thicklabel) VALUES (?, ?)", t)
		conn.commit()
		# if anno_i != 'others':
		# 	selected_feas.append(all_test_feas[name_i])
		# 	# xn = np.expand_dims(all_test_feas[name_i], 1)
		# 	# Cn_inv = Cn_inv - np.matmul(np.matmul(np.matmul(Cn_inv, xn), xn.transpose()), Cn_inv) / (1+ np.matmul(np.matmul(xn.transpose(), Cn_inv), xn))
		# 	if anno_i == 'thin':
		# 		yn = 1
		# 	elif anno_i == 'thick':
		# 		yn = -1
		# 	# last_coef = last_coef + np.matmul(Cn_inv, xn) * (yn - np.matmul(xn.transpose(), last_coef))
		# 	selected_labels.append(yn)
	# Xk = np.stack(selected_feas, axis=1) # [dim, K]
	# Yk = np.expand_dims(np.array(selected_labels), 1)
	# denom = np.matmul(np.matmul(Xk.transpose(), Cn_inv), Xk) + np.eye(Xk.shape[1])
	# denom = np.linalg.inv(denom)
	# Cn_inv = Cn_inv - np.matmul(np.matmul(np.matmul(np.matmul(Cn_inv, Xk), denom), Xk.transpose()), Cn_inv)
	# last_coef = last_coef + np.matmul(Cn_inv, np.matmul(Xk, Yk) - np.matmul(np.matmul(Xk, Xk.transpose()), last_coef))

	# to_save = dict()
	# to_save['coef'] = last_coef
	# to_save['Cn_inv'] = Cn_inv
	# pickle.dump(to_save, open(os.path.join(subannos_db_path, 'thickness_classifier-%s_b-%d.p'%(clf_method, id-1)), 'wb'))

	# print("""<h2> Go back to correct annotation error. 
 # 					<a href="annotate_online_server.cgi?expinfo=%s&sizethre=%d&id=%d&user=%s">Click here to relabel them </a> </h2>""" % (expinfo, sizethre, id - 1, user))

# else:
# 	last_clf = pickle.load(open(os.path.join(subannos_db_path, 'thickness_classifier-%s_b-%d.p'%(clf_method, id-1)), 'rb'))
# 	Cn_inv = last_clf['Cn_inv'] 
# 	last_coef = last_clf['coef'] # [dim, 1]

# find what needs to be label
tolabel_names = list(set(all_test_names) - set(labeled_test_names))
num_tolabel = len(tolabel_names)
if num_tolabel > 0:
	# tolabel_feas = [all_test_feas[k] for k in tolabel_names]
	# tolabel_feas = np.stack(tolabel_feas, axis=0)
	# run the classifier, find the most confident thin(>0) or flake(<0) candidates
	# print(tolabel_feas.shape)
	# print(all_test_feas[0].shape)
	# tolabel_scores = np.matmul(tolabel_feas, last_coef) # [num_exp, 1]
	# tolabel_scores = tolabel_scores[:, 0] # [num_exp]
	# find top thin(positive) and top flake(negative)
	# tolabel_sort_idxs = np.argsort(tolabel_scores) # from small to large
	
	range_idxs = list(range( id * n_column, (id+1) * n_column))

	# for thin, first use thin, then select from glue (top)
	if range_idxs[-1] < num_thin:
		batch_thin_names = [all_thin_test_names[_] for _ in range_idxs]
	if range_idxs[0] >= num_thin:
		batch_thin_names = [all_glue_test_names[_-num_thin] for _ in range_idxs]
	# print(range_idxs)
	if range_idxs[0] < num_thin and range_idxs[-1] >= num_thin:
		batch_thin_names = all_thin_test_names[range_idxs[0]:] + [all_glue_test_names[_-num_thin] for _ in range_idxs if _-num_thin >=0]
		# print([all_glue_test_names[_-num_thin] for _ in range_idxs if _-num_thin >=0])
	# print(batch_thin_names)

	# for thick, first use thick, then select from glue (bottom)
	if range_idxs[-1] < num_thick:
		batch_thick_names = [all_thick_test_names[_] for _ in range_idxs]
	if range_idxs[0] >= num_thick:
		batch_thick_names = [all_glue_test_names[num_thick-_-1] for _ in range_idxs]
	if range_idxs[0] < num_thick and range_idxs[-1] >= num_thick:
		batch_thick_names = all_thick_test_names[range_idxs[0]:] + [all_glue_test_names[num_thick-_-1] for _ in range_idxs if _-num_thick >=0]

	# find their location in all_names
	top_thin_idxs = [all_test_names.index(xx) for xx in batch_thin_names]
	top_thick_idxs = [all_test_names.index(xx) for xx in batch_thick_names]

	# assert len(set(top_thin_idxs)) + len(set(top_thick_idxs)) == len(set(top_thin_idxs+top_thick_idxs))

	# top_thin_idxs = tolabel_sort_idxs[-min(num_tolabel/2, n_column):]
	# top_thick_idxs = tolabel_sort_idxs[:min(num_tolabel - num_tolabel/2, n_column)]

	top_thin_labels = [all_predlabels[xx] for xx in top_thin_idxs]
	top_thick_labels = [all_predlabels[xx] for xx in top_thick_idxs]

	# top_thin_labels = [tolabel_scores[x]>-0.5 for x in top_thin_idxs]
	# top_thick_labels = [tolabel_scores[x]>0 for x in top_thick_idxs]

	# print(tolabel_scores)
	# print(tolabel_sort_idxs)
	# print(num_tolabel)
	# print(top_thin_idxs)
	# batch_names = [tolabel_names[kk] for kk in top_thin_idxs ] + [tolabel_names[kk] for kk in top_thick_idxs]
	batch_names = batch_thin_names + batch_thick_names
	
	labeled_name_batch_file = os.path.join(subannos_db_path, 'labeled_test_names_b-%d.txt'%(id))
	with open(labeled_name_batch_file, 'w') as f:
		f.write('\n'.join(batch_names))
	# with open(labeled_test_names_file, 'a+') as f:
	# 	f.write('\n'.join(batch_names))

	# display with default selection
	print(""" 
		<div id="form_container">      
			<form id="form_390674" class="appnitro"  method="post" action="annotate_online_thicknessv2_server.cgi?expinfo=%s&sizethre=%d&id=%d&user=%s">      
			<h2>Annotate the below regions</h2> """ % (expinfo, sizethre, id+1, user) )

	print("""
	<table>
	    <tr> """)
	for c_i in range(n_column):
		print("""
		<th>%d</th> """%(c_i))
	print("""</tr> """)
	### thin
	# the image part
	print("""<tr> """)
	for c_i in range(len(top_thin_idxs)):
		img_path = os.path.join(subonline_rslt_path[2:], "img-%s.png"%(all_test_names[top_thin_idxs[c_i]]))
		print("""
		<td><img src="http://vision.cs.stonybrook.edu/~boyu/material_segmentation/%s" """ % (img_path))
		print(""" style="width:100%" alt=""> </td>""")
	print("""</tr> """)
	# the label part, thin
	print("""<tr> """)
	for c_i in range(len(top_thin_idxs)):
		ck_str = "checked" if top_thin_labels[c_i] == 'thin' else ""
		print("""
	    <td align="center"><input type="radio" name=%s data-col=%d value="thin" %s > thin </td>""" %(all_test_names[top_thin_idxs[c_i]], c_i, ck_str))
	print("""</tr> """)
	# the label part, thick
	print("""<tr> """)
	for c_i in range(len(top_thin_idxs)):
		ck_str = "checked" if top_thin_labels[c_i] == 'thick' else ""
		print("""
	    <td align="center"><input type="radio" name=%s data-col=%d value="thick" %s > thick </td>""" %(all_test_names[top_thin_idxs[c_i]], c_i, ck_str))
	print("""</tr> """)
	# label part, glue
	print("""<tr> """)
	for c_i in range(len(top_thin_idxs)):
		ck_str = "checked" if top_thin_labels[c_i] == 'glue' else ""
		print("""
	    <td align="center"><input type="radio" name=%s data-col=%d value="glue" %s > glue </td>""" %(all_test_names[top_thin_idxs[c_i]], c_i, ck_str))
	print("""</tr> """)
	# the label part, others
	print("""<tr> """)
	for c_i in range(len(top_thin_idxs)):
		print("""
	    <td align="center"><input type="radio" name=%s data-col=%d value="others" > others </td>""" %(all_test_names[top_thin_idxs[c_i]], c_i))
	print("""</tr> """)

	### thick
	# the image part
	print("""<tr> """)
	for c_i in range(len(top_thick_idxs)):
		img_path = os.path.join(subonline_rslt_path[2:], "img-%s.png"%(all_test_names[top_thick_idxs[c_i]]))
		print("""
		<td align="center"><img src="http://vision.cs.stonybrook.edu/~boyu/material_segmentation/%s" """ % (img_path))
		print(""" style="width:100%" alt=""> </td>""")
	print("""</tr> """)
	# the label part, thin
	print("""<tr> """)
	for c_i in range(len(top_thick_idxs)):
		ck_str = "checked" if top_thick_labels[c_i] =='thin' else ""
		print("""
	    <td align="center"><input type="radio" name=%s data-col=%d value="thin" %s > thin </td>""" %(all_test_names[top_thick_idxs[c_i]], c_i, ck_str))
	print("""</tr> """)
	# the label part, thick
	print("""<tr> """)
	for c_i in range(len(top_thick_idxs)):
		ck_str = "checked" if top_thick_labels[c_i] == 'thick' else ""
		print("""
	    <td align="center"><input type="radio" name=%s data-col=%d value="thick" %s > thick </td>""" %(all_test_names[top_thick_idxs[c_i]], c_i, ck_str))
	print("""</tr> """)
	# the label part, glue
	print("""<tr> """)
	for c_i in range(len(top_thick_idxs)):
		ck_str = "checked" if top_thick_labels[c_i] == 'glue' else ""
		print("""
	    <td align="center"><input type="radio" name=%s data-col=%d value="glue" %s > glue </td>""" %(all_test_names[top_thick_idxs[c_i]], c_i, ck_str))
	print("""</tr> """)
	# the label part, others
	print("""<tr> """)
	for c_i in range(len(top_thick_idxs)):
		print("""
	    <td align="center"><input type="radio" name=%s data-col=%d value="others" > others </td>""" %(all_test_names[top_thick_idxs[c_i]], c_i))
	print("""</tr> """)

	print("""</table>""")

	print("""
	<div style="text-align: center;">
		<input type="submit" value="Submit" onclick="this.form.submit()" >
	</div>""")
	# print(""" <input type="submit" value="Submit" onclick="this.form.submit()" > """)
	print(""" </form> """)
	print("""</body></html>""")

else:
	print("<h2>Great. No more image to label!</h2>")


# for i, line in enumerate(fp):
# 	if thicklabel is not None and i == id -1: # add or update the annotation database, for the one annotated from previous step
# 		flake_name_db = line.strip()
		
# 		conn = sqlite3.connect(annos_db_name)
# 		c = conn.cursor()
# 		c.execute('''CREATE TABLE IF NOT EXISTS annotab (imid STRING PRIMARY KEY, thicklabel STRING)''')
# 		t = (flake_name_db, thicklabel)
# 		c.execute("INSERT OR REPLACE INTO annotab(imid, thicklabel) VALUES (?, ?)", t)
# 		conn.commit()

# 		print """<h2> You selected: '%s' for the previous image. 
# 					<a href="annotate_cluster_server.cgi?expinfo=%s&sizethre=%d&id=%d&user=%s">Click here to relabel it</a> </h2>""" % (thicklabel, expinfo, sizethre, id - 1, user)

# 	elif i == id:
# 		next_im_id = line.strip()
# 		break			
# fp.close()


# if (next_im_id is None) or (not next_im_id):
# 	print "<h2>Great. No more image to label!</h2>"
# else: # display the next image for annotation
# 	# cluster_id, flake_name = next_im_id.split(',')
# 	# flake_bw_name = flake_name[:-4] + '_bw.png'
#  #        flake_bbox_name = flake_name[:-4] + '_bbox.png'
# 	# ori_img_name = flake_name.split('-')[1] + '.tif'
# 	# ori_img_fullpath = os.path.join(subori_img_path, ori_img_name)
# 	# flake_name_fullpath = os.path.join(subcluster_rslt_path, str(cluster_id), flake_name)
# 	# flake_bw_name_fullpath = os.path.join(subcluster_rslt_path, str(cluster_id), flake_bw_name)
#  #        flake_bbox_name_fullpath = os.path.join(subcluster_rslt_path, str(cluster_id), flake_bbox_name)
#  	cluster_img_name = os.path.join(subcluster_rslt_path[2:], next_im_id)

# 	print """ 
# 		<div id="form_container">      
# 			<form id="form_390674" class="appnitro"  method="post" action="annotate_cluster_server.cgi?expinfo=%s&sizethre=%d&id=%d&user=%s">      
# 			<h2>Annotate the below clusters</h2>
#                         <img src="http://vision.cs.stonybrook.edu/~boyu/material_segmentation/%s" height=600 alt="">
# 			<br>
# 			<p>What is this flake? </p>
# 			<input type="radio" name="thicklabel" value="thin" onclick="this.form.submit()" > thin <br>
# 			<input type="radio" name="thicklabel" value="thick" onclick="this.form.submit()" > thick <br>
# 			<input type="radio" name="thicklabel" value="thin" onclick="this.form.submit()" > thin <br>
# 			<input type="radio" name="thicklabel" value="mixed cluster" onclick="this.form.submit()" > mixed cluster <br>
# 			<input type="radio" name="thicklabel" value="others" onclick="this.form.submit()" > others <br>

#                 </form> """ % (expinfo, sizethre, id+1, user, cluster_img_name)

# print """</body></html>""" 
