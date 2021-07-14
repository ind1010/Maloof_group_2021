'''
-----------------------------------------------------------------------
combine_categories.py

Author: Indu Panigrahi
Description: Condenses categories in COCO format annotation file into 2
categories: one for the class of interest (i.e., Archaeocyathids) and one 
as a general class for everything else.

Run in a Python virtual environment: python combine_categories.py
-----------------------------------------------------------------------
'''

import json
import os

# ---------------------------------------------------------------------
def main():
	with open('put/the/path/to/json/to/edit') as fp:
		data_dict = json.load(fp)

	'''
	You will probably need to edit the following code depending 
	on the structure of your json file.
	In my case, a category_id of 1 was Archaeocyathid, and I
	combined the rest into 'Red Mud' which had category_id 2.
	'''
	data_list = data_dict.get("annotations")
	for i in range(len(data_list)):
		old_id = data_list[i].get('category_id')
		# change label for annotations that are not Archaeocyathid to Red Mud
		if int(old_id) > 1:
			data_list[i]['category_id'] = 2
	data_dict["annotations"] = data_list

	data_list = data_dict.get("categories")
	# remove the rest of the classes from list of categories
	data_dict["categories"] = data_dict["categories"][0:2]

	# json with combined classes will be in combined.json in cwd
	with open('combined.json', 'w', encoding='utf-8') as f:
		json.dump(data_dict, f, ensure_ascii=False, indent=4)

# ---------------------------------------------------------------------
if __name__ == '__main__':
	main()