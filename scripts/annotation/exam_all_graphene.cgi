#!/usr/bin/env python3         
# A simple example for web annotation using Python + CGI
# annotate graphene, just read from predicted graphene
# http://vision.cs.stonybrook.edu/cgi-bin/AnnoEx/cgi-bin/exam_all_graphene.cgi?id=0&user=test0123
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
id = int(form.getvalue('id'))
user = form.getvalue('user')
assert user is not None

ori_img_path = '../data/data_111x_individual'
existing_img_path = '../rslt/data_111x_individual_script/test_classify_graphene_colorfea-%s_clf-linear_data-sep-oct_topvisualize-100_patch-withori/'%'contrast-bg'
existing_annos_db_path = '../data/anno_graphene_youngjae/'
examed_annos_db_path = '../data/double_check_anno_graphene_%s/'%user
if not os.path.exists(examed_annos_db_path):
    os.makedirs(examed_annos_db_path)
examed_annos_db_name = os.path.join(examed_annos_db_path, 'anno_user-%s.db'%(user))


n_row = 25
n_column = 4


# read from the annotation
def readdb(all_dbname, exp_name, sname):
    conn = sqlite3.connect(all_dbname)
    c = conn.cursor()
    c.execute('SELECT imflakeid, thicklabel FROM annotab')
    db = c.fetchall()

    num_graphene = 0
    num_others = 0
    oriname_flakeids = []
    for i in range(len(db)):
        imflakeid = db[i][0]
        # flake_id = int(imflakeid.split('_')[3].split('-')[1])
        # flake_oriname = imflakeid.split('_', 5)[5]
        label = db[i][1]
        label_id = None
        if label == 'graphene':
            num_graphene += 1
            label_id = 1
        elif label == 'others':
            num_others += 1
            label_id = 0
        # else:
        #     raise NotImplementedError
        if label_id is not None:
            oriname_flakeids.append([exp_name, sname, imflakeid, label_id])    
        # else:
        #     print(imflakeid, label)
        
    return oriname_flakeids


# merge existing all annotation
exp_names = ['laminator', 'PDMS-QPress 6s']
exp_names.sort()

all_anno_data = []
for exp_name in exp_names:
    subexp_names = os.listdir(os.path.join(existing_annos_db_path, exp_name))
    subexp_names = [sname for sname in subexp_names if sname[0] not in ['.', '_']]
    subexp_names = [sname for sname in subexp_names if os.path.isdir(os.path.join(existing_annos_db_path, exp_name, sname))]
    subexp_names.sort()
    for sname in subexp_names:
        sub_anno_path = os.path.join(existing_annos_db_path, exp_name, sname, 'anno_user-youngjae.db')
        if not os.path.exists(sub_anno_path):
            print('not exist annotation! ', sub_anno_path)
        sub_anno_data = readdb(sub_anno_path, exp_name, sname)
        if len(sub_anno_data) > 0:
            all_anno_data.extend(sub_anno_data)
        else:
            print(exp_name, sname)

num_anno = len(all_anno_data)
# group graphene together
all_graphene_ids = [im_i for im_i in range(num_anno) if all_anno_data[im_i][3] == 1]
all_other_ids = [im_i for im_i in range(num_anno) if all_anno_data[im_i][3] == 0]

num_graphene = len(all_graphene_ids)
num_other = len(all_other_ids)

print(num_anno, num_graphene, num_other)
# print(all_anno_data[0])

all_anno_data_sorted = [all_anno_data[im_i] for im_i in all_graphene_ids] + [all_anno_data[im_i] for im_i in all_other_ids]
# print(all_anno_data_sorted[0])
last_labeled_test_names = []
last_labeled_file = os.path.join(examed_annos_db_path, 'last_labeled_names.txt')
if os.path.exists(last_labeled_file):
    with open(last_labeled_file, 'r') as f:
    	batch_names = f.readlines()
    last_labeled_test_names = [n.strip() for n in batch_names]

num_group = (num_anno - 1) // (n_row * n_column) + 1


if len(last_labeled_test_names) > 0:
	# update the annoation database, for the one annotated from previous step, and update the classifier
	conn = sqlite3.connect(examed_annos_db_name)
	c = conn.cursor()
	for name_i in last_labeled_test_names:
		anno_i = form.getvalue(name_i)
		c.execute('''CREATE TABLE IF NOT EXISTS annotab (expname_subexpname_imflakeid STRING PRIMARY KEY, thicklabel STRING)''')
		t = (name_i, anno_i)
		c.execute("INSERT OR REPLACE INTO annotab(expname_subexpname_imflakeid, thicklabel) VALUES (?, ?)", t)
		conn.commit()


if (id >= num_group):
    print("<h2>Great. No more image to label!</h2>")
else: 
    # find what needs to be label
    tolabel_data = all_anno_data_sorted[id*n_row*n_column: min((id+1)*n_row*n_column, num_anno)]
    # print(tolabel_data[0])
    tolabel_names = [d[0] + '+' + d[1] + '+' + d[2] for d in tolabel_data]
    num_tolabel = len(tolabel_names)

    with open(last_labeled_file, 'w') as f:
        f.write('\n'.join(tolabel_names))

	# display with default selection
    print(""" 
		<div id="form_container">      
			<form id="form_390674" class="appnitro"  method="post" action="exam_all_graphene.cgi?id=%d&user=%s">
			<h2>Annotate the below regions</h2> """ % (id+1, user) )

    print("""
	<table>
	    <tr> """)

    for r_i in range(n_row):
        row_data = tolabel_data[r_i*n_column: min((r_i+1)*n_column, num_tolabel)]
        # the image part
        print("""<tr> """)
        for c_i in range(len(row_data)):
            img_path = os.path.join(existing_img_path[3:], row_data[c_i][0], row_data[c_i][1], row_data[c_i][2])
            print("""
    		<td><img src="http://vision.cs.stonybrook.edu/~boyu/material_segmentation/%s" """ % (img_path))
            print(""" width="400"px alt=""> </td>""")
        print("""</tr> """)
    	# the label part, graphene
        print("""<tr> """)
        for c_i in range(len(row_data)):
            # print(row_data[c_i][3])
            ck_str = "checked" if row_data[c_i][3] ==1 else ""
            # print(ck_str)
            entry_name = row_data[c_i][0] + '+' + row_data[c_i][1] + '+' + row_data[c_i][2]
            print("""
    	    <td align="center"><input type="radio" name="%s" data-col=%d value="graphene" %s > graphene </td>""" %(entry_name, c_i, ck_str))
        print("""</tr> """)
    	# the label part, others
        print("""<tr> """)
        for c_i in range(len(row_data)):
            ck_str = "checked" if row_data[c_i][3] == 0 else ""
            entry_name = row_data[c_i][0] + '+' + row_data[c_i][1] + '+' + row_data[c_i][2]
            print("""
    	    <td align="center"><input type="radio" name="%s" data-col=%d value="others" %s > others </td>""" %(entry_name, c_i, ck_str))
        print("""</tr> """)
        

    print("""</table>""")
    print("""
	<div style="text-align: center;">
		<input type="submit" value="Submit" onclick="this.form.submit()" >
	</div>""")
    print(""" </form> """)
    print("""</body></html>""")
