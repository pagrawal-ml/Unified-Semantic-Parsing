import re
import os, sys
import shutil
'''
dir = '../../data/overnight/recipes/'
files = []
files.append(os.path.join(dir,'recipes_train.tsv'))
files.append(os.path.join(dir,'recipes_test.tsv'))
dir = '../../data/overnight/housing/'
files = []
files.append(os.path.join(dir,'housing_train.tsv'))
files.append(os.path.join(dir,'housing_test.tsv'))

'''
dir = sys.argv[1]
outdir = sys.argv[2]
files = []
'''
files.append('publications_train.tsv.entity.prune.txt')
files.append('publications_test.tsv.entity.prune.txt')
files.append('recipes_train.tsv.entity.prune.txt')
files.append('recipes_test.tsv.entity.prune.txt')
files.append('restaurants_train.tsv.entity.prune.txt')
files.append('restaurants_test.tsv.entity.prune.txt')
files.append('calendar_train.tsv.entity.prune.txt')
files.append('calendar_test.tsv.entity.prune.txt')
files.append('blocks_train.tsv.entity.prune.txt')
files.append('blocks_test.tsv.entity.prune.txt')
files.append('housing_train.tsv.entity.prune.txt')
files.append('housing_test.tsv.entity.prune.txt')
'''
#files.append('basketball_train.tsv.entity.prune.txt')
#files.append('basketball_test.tsv.entity.prune.txt')
#files.append('socialnetwork_train.tsv.entity.prune.txt')
#files.append('socialnetwork_test.tsv.entity.prune.txt')

'''
dir = 'overnight-lf/'
files = os.listdir(dir)
'''

dir = sys.argv[1]
domain = sys.argv[2]
outdir = sys.argv[3]
files = []
files.append(os.path.join(dir, domain + '_train.tsv.entity.prune.txt'))
files.append(os.path.join(dir, domain + '_test.tsv.entity.prune.txt'))


for filepath in files:
  outfilepath = os.path.join(outdir, os.path.basename(filepath)+'.prune2.txt')
  if domain not in ['basketball', 'socialnetwork']:
    shutil.copyfile(filepath, outfilepath)
  
  f = open(filepath)
  fout = open(outfilepath, 'w')
  line = f.readline()
  while line:
    line = (re.sub(r'\( SW\.concat \(.*?\) \(.*?\) \)',r'SW.concat',line))
    line = (re.sub(r'\( SW\.concat (.*?) (.*?) \)',r'SW.concat',line))
    subs = line.split('\t')[1].strip()
    if subs.startswith('SW.concat'):
    	line = (re.sub(r'SW\.concat (.*?) (.*?)\n','SW.concat',line)).strip()
    
    #line = re.sub(r' SW.getProperty \( \( lambda s \( SW.filter \( var s \) ',r' ( ( var-s ',line)
    line = re.sub(r'SW.getProperty \( \( lambda s \( (.*?) \( var s \) (.*?) \) \)',r'\1 var-s \2',line)
    line = re.sub(r'(.*?) SW.getProperty e0 SW.reverse',r'\1',line)
    # line = re.sub(r'(.*?) SW.getProperty e0 SW.reverse',r'\1',line)
    '''
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
    '''
    count = len(subs.split(' '))
    #fout.write(str(count) + '\t' + line.strip()+'\n')
    fout.write(line.strip()+'\n')
    line = f.readline()
  fout.close()
  f.close()
