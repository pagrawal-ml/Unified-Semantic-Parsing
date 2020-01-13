import re
import os
import sys
'''
dir = 'overnight-lf/'
files = []
files.append(os.path.join(dir,'recipes_train.tsv'))
files.append(os.path.join(dir,'recipes_test.tsv'))
'''


dir = sys.argv[1]
domain = sys.argv[2]
outdir = sys.argv[3]
files = []
files.append(os.path.join(dir, domain + '_train.tsv'))
files.append(os.path.join(dir, domain + '_test.tsv'))

for filename in files:
  f = open(filename)
  outfilename = os.path.basename(filename)
  fout = open(os.path.join(outdir, outfilename + '.prune.txt'), 'w')
  line = f.readline()
  while line:
    line = (re.sub('\( call SW.listValue \( ','',line)).strip()
    line = line[:-4].strip()
    line = re.sub(r'\( call SW.getProperty \( call SW.singleton (.*?) \) \( string ! type \) \)',r'entity-\1',line.strip())
    line = re.sub(r'call SW', 'SW', line)
    #line = re.sub(r'\( number 2 \)', 'num2', line)
    line = re.sub(r'\( number (.*?) \)', r'number\1', line)
    
    ####-------------- prune 2
    line = re.sub(r'\( string (.*?) \)',r'string-\1',line.strip())
    line = re.sub(r'call \.size','.size',line.strip())
    #line = re.sub(r'\( date (.*?) -1 -1 \)',r'date-\1',line.strip())
    line = re.sub(r'\( date (.*?) (.*?) (.*?) \)',r'date:\1:\2:\3',line.strip())
    line = re.sub(r'! ',r'!',line.strip())
    line = re.sub(r'SW.aggregate ','',line.strip())       
    line = re.sub(r'\( SW\.reverse string\-(.*?) \)',r'SW.reverse string-\1',line.strip())   
    line = re.sub(r'\( SW\.ensureNumericProperty string\-(.*?) \)',r'SW.ensureNumericProperty string-\1 ',line)
    line = re.sub(r'\( SW.getProperty en(.*?) string\-(.*?) \)',r'SW.getProperty en\1 string-\2 ',line)
      
    fout.write(line+'\n')
    line = f.readline()
  fout.close()
  f.close()
