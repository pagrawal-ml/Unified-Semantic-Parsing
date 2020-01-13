import re
import os

def preprocess1(dir, files):
  for filename in files:
    f = open(filename)
    outfilename = os.path.basename(filename)
    fout = open('prune1' + outfilename + '.prune', 'w')
    line = f.readline()
    while line:
      line = (re.sub('\( call SW.listValue \( ','',line)).strip()
      line = line[:-4].strip()
      line = re.sub(r'\( call SW.getProperty \( call SW.singleton (.*?) \) \( string ! type \) \)',r'entity-\1',line.strip())
      line = re.sub(r'call SW', 'SW', line)
      #line = re.sub(r'\( number 2 \)', 'num2', line)
      line = re.sub(r'\( number (.*?) \)', r'num\1', line)
      
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

def preprocess_2(dir, files):
  for filename in files:

    f = open(filename)
    outfilename = os.path.basename(filename)
    fout = open('prune4/' + outfilename+'.prune', 'w')
    line = f.readline()
    while line:
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


def preprocess(dir, domains):
  for domain in domains:
    print('Running for: ', domain)
    files = []
    files.append(os.path.join(dir, domain + '_train.tsv'))
    files.append(os.path.join(dir,domain + '_test.tsv'))
    preprocess1(dir, files)
    if domain in ['basketball', 'socialnetwork']:
    	preprocess_2(dir, files)


def main():
  dir = 'overnight-lf/'
  domains = ['recipes', 'housing', 'basketball', 'blocks', 'calendar', 'publications', 'restaurants', 'socialnetwork']
  preprocess(dir, domains)
  

if __name__ == '__main__':
	main()