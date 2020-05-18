#!/usr/bin/env python3         
# A simple example for web annotation using Python + CGI
# annotate graphene, just read from predicted graphene
# http://vision.cs.stonybrook.edu/cgi-bin/AnnoEx/cgi-bin/annotate_graphene_server.cgi?fea=contrast-bg&expinfo=laminator_home-pdms&size=100-2000&id=0&user=test0123
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
size = form.getvalue('size')
size_lower, size_upper = size.split('-')
size_lower = int(size_lower)
size_upper = int(size_upper)
user = form.getvalue('user')
expinfo = form.getvalue('expinfo')
fea_type = form.getvalue('fea')
assert user is not None
assert expinfo is not None
# cluster_id = int(form.getvalue('cluster_id'))

# expid = int(form.getvalue('expid'))
# subexpid = int(form.getvalue('subexpid'))
exp_name, subexp_name = expinfo.split('_', 1)
# exp_name = 'EXPT2TransferPressure'
# subexp_name = 'HighPressure'

ori_img_path = '../data/data_111x_individual'
# rslt_path = '../rslt/data_111x_individual_script/test_classify_graphene_colorfea-%s_clf-linear_data-sep-oct_topvisualize-100_patch/'%(fea_type)
# rslt_path = '../rslt/data_111x_individual_script/test_classify_graphene_colorfea-%s_clf-linear_data-sep-oct_topvisualize-100_patch-withori/'%(fea_type)
rslt_path = '../rslt/data_111x_individual_script/test_classify_graphene_colorfea-%s_clf-linear_data-cleananno_topvisualize-50_patch-withori/'%(fea_type)

n_row = 4
n_column = 3

# fea_type = 'all'
# cluster_method = 'affinity'
# clf_method = 'linearsvm'
subori_img_path = os.path.join(ori_img_path, exp_name, subexp_name)
subrslt_path = os.path.join(rslt_path, exp_name, subexp_name)
rslt_names = os.listdir(subrslt_path)
rslt_names = [rname for rname in rslt_names if rname[0] not in ['.', '_']]
rslt_names.sort()
num_flakes = len(rslt_names)
# find the size of the results
flake_sizes = []
distances = []
for rname in rslt_names:
    tmp_distance = float(rname.split('_',5)[2])
    tmp_size = int(rname.split('_', 5)[4].split('-')[1])
    flake_sizes.append(tmp_size)
    distances.append(tmp_distance)

# filter graphene based on size
idxes = [i for i in range(num_flakes) if flake_sizes[i] >= size_lower and flake_sizes[i] <= size_upper]
rslt_names = [rslt_names[i] for i in idxes]
flake_sizes = [flake_sizes[i] for i in idxes]
distances = [distances[i] for i in idxes]
num_flakes = len(rslt_names)

subannos_db_path = os.path.join('../data/anno_graphene_%s/'%user, exp_name, subexp_name)
if not os.path.exists(subannos_db_path):
	os.makedirs(subannos_db_path)
annos_db_name = os.path.join(subannos_db_path, 'anno_user-%s.db'%(user))

# labeled_names_file = os.path.join(subannos_db_path, 'labeled_names.txt')
# labeled_batch_file = os.path.join(subannos_db_path, 'labeled_batch.p')
# fb = open(labeled_batch_file, 'a+')

# labeled_test_names = []
last_labeled_test_names = []
last_labeled_file = os.path.join(subannos_db_path, 'last_labeled_names.txt')
if os.path.exists(last_labeled_file):
    with open(last_labeled_file, 'r') as f:
    	batch_names = f.readlines()
    last_labeled_test_names = [n.strip() for n in batch_names]

num_group = (num_flakes - 1) // (n_row * n_column) + 1


if len(last_labeled_test_names) > 0:
	# update the annoation database, for the one annotated from previous step, and update the classifier
	conn = sqlite3.connect(annos_db_name)
	c = conn.cursor()
	for name_i in last_labeled_test_names:
		anno_i = form.getvalue(name_i)
		c.execute('''CREATE TABLE IF NOT EXISTS annotab (imflakeid STRING PRIMARY KEY, thicklabel STRING)''')
		t = (name_i, anno_i)
		c.execute("INSERT OR REPLACE INTO annotab(imflakeid, thicklabel) VALUES (?, ?)", t)
		conn.commit()


if (id >= num_group):
    print("<h2>Great. No more image to label!</h2>")
else: 
    # find what needs to be label
    tolabel_names = rslt_names[id*n_row*n_column: min((id+1)*n_row*n_column, num_flakes)]
    num_tolabel = len(tolabel_names)
    tolabel_distances = distances[id*n_row*n_column: min((id+1)*n_row*n_column, num_flakes)]
    tolabel_sizes = flake_sizes[id*n_row*n_column: min((id+1)*n_row*n_column, num_flakes)]

    with open(last_labeled_file, 'w') as f:
        f.write('\n'.join(tolabel_names))

	# display with default selection
    print(""" 
		<div id="form_container">      
			<form id="form_390674" class="appnitro"  method="post" action="annotate_graphene_server.cgi?fea=%s&expinfo=%s&size=%s&id=%d&user=%s">      
			<h2>Annotate the below regions</h2> """ % (fea_type, expinfo, size, id+1, user) )

    print("""
	<table>
	    <tr> """)

    for r_i in range(n_row):
        row_names = tolabel_names[r_i*n_column: min((r_i+1)*n_column, num_tolabel)]
        row_distances = tolabel_distances[r_i*n_column: min((r_i+1)*n_column, num_tolabel)]
        row_sizes = tolabel_sizes[r_i*n_column: min((r_i+1)*n_column, num_tolabel)]
        for c_i in range(len(row_names)):
            print("""
    		<th>size: %d, distance: %f</th> """%(row_sizes[c_i], row_distances[c_i]))
        print("""</tr> """)
    	# the image part
        print("""<tr> """)
        for c_i in range(len(row_names)):
            img_path = os.path.join(subrslt_path[3:], row_names[c_i])
            print("""
    		<td><img src="http://vision.cs.stonybrook.edu/~boyu/material_segmentation/%s" """ % (img_path))
            print(""" width="400"px alt=""> </td>""")
        print("""</tr> """)
    	# the label part, graphene
        print("""<tr> """)
        for c_i in range(len(row_names)):
            ck_str = "checked" if row_distances[c_i] >=0 else ""
            print("""
    	    <td align="center"><input type="radio" name=%s data-col=%d value="graphene" %s > graphene </td>""" %(row_names[c_i], c_i, ck_str))
        print("""</tr> """)
    	# the label part, others
        print("""<tr> """)
        for c_i in range(len(row_names)):
            ck_str = "checked" if row_distances[c_i] <0 else ""
            print("""
    	    <td align="center"><input type="radio" name=%s data-col=%d value="others" %s > others </td>""" %(row_names[c_i], c_i, ck_str))
        print("""</tr> """)


    print("""</table>""")
    print("""
	<div style="text-align: center;">
		<input type="submit" value="Submit" onclick="this.form.submit()" >
	</div>""")
	# print(""" <input type="submit" value="Submit" onclick="this.form.submit()" > """)
    print(""" </form> """)
    print("""</body></html>""")
