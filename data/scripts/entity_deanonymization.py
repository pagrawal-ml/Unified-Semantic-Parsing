import re
import os
import sys
from itertools import izip

if len (sys.argv) != 5 :
    print("Usage: python entity_deanonymization.py parseFile.txt parse_transformation.txt output.txt beam_width")
    sys.exit (1)


source_file = sys.argv[1]
transformation_file = sys.argv[2]
output_file = sys.argv[3]
bw = int(sys.argv[4])

fout = open(output_file, 'w')
transl = open(transformation_file).readlines()
srcl  =  open(source_file).readlines()
#with open(source_file) as textfile1:#, open(transformation_file) as textfile2: 
	#for x, y in izip(textfile1, textfile2):
for i in range(len(srcl)):
		x = srcl[i]
		y = transl[int(i/bw)]
		parse = x.split('<EOS>')[0].strip()
		#parse = x.split('\t')[1].strip()
		transformation = y.strip()
	
		transformation = transformation.replace("{","")
		transformation = transformation.replace("}","")
	
		if not transformation:
			fout.write(parse+"\n")
			continue

	
		transformation_parts = transformation.split(",")
		for transformation_part in transformation_parts:
			transformation_dict = transformation_part.split("=")
			key = transformation_dict[0].strip()
			value = transformation_dict[1].strip()

			pattern = '\\b' + key + '\\b'
			parse = re.sub(pattern,value,parse.strip())

	
		fout.write(parse+ "\n")
fout.close()
