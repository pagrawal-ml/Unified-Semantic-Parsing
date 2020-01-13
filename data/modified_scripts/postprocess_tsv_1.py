import re
import os, sys

'''
dir = '../../data/overnight/recipes/'
files = []
files.append(os.path.join(dir,'recipes_train.tsv.prune'))
files.append(os.path.join(dir,'recipes_test.tsv.prune'))

dir = '../../data/overnight/housing/'
files = []
files.append(os.path.join(dir,'housing_train.tsv.prune'))
files.append(os.path.join(dir,'housing_test.tsv.prune'))

dir = '../../data/overnight/restaurant/'
files = []
files.append(os.path.join(dir,'restaurants_train.tsv.prune'))
files.append(os.path.join(dir,'restaurants_test.tsv.prune'))
'''

import collections
edict = collections.OrderedDict()
edict['bobs']= ['e0 en.person.bob']
edict['male']= ['en.gender.male en.gender.female']
edict['female']= ['en.gender.male en.gender.female']
edict['gender']= ['en.gender.male en.gender.female']
edict['supper']= ['e0 en.meal.dinner']
edict['forwards'] = ['en.position.point_guard en.position.forward']
edict['cleveland'] = ['en.team.lakers en.team.cavaliers']
edict[' cavs'] = ['e0 en.team.cavaliers']
edict['morning']=['( time 10 0 ) ( time 15 0 )']
edict['300pm']=['( time 10 0 ) ( time 15 0 )']
edict[' square feet'] = ['number800 en.square_feet number1000 en.square_feet']
edict['dollar sign'] = ['number2 en.dollar_sign number3 en.dollar_sign']
edict['dollar price rating'] = ['number2 en.dollar_sign number3 en.dollar_sign']
edict[' stars'] = ['number3 en.star number5 en.star']
edict[' dollars'] = ['number1500 en.dollar number2000 en.dollar']
edict[' 2000'] = ['number1500 en.dollar number2000 en.dollar']
edict[' 1500'] = ['number1500 en.dollar number2000 en.dollar']
edict[ 'jan'] = ['date:2015:1:2 date:2015:2:3']
edict['inch'] = ['number3 en.inch number6 en.inch']
edict[' hour'] = ['number3 en.hour number1 en.hour']


def post2(line):
  count = len(line.split(' '))
  line = line.strip() + ' '
  if True: # line.startswith('SW.concat')
    if not line.startswith("SW.concat"):
      line = (re.sub(r'SW\.concat ',r'( SW.concat ) ',line)).strip()
   
    if 'e0' in subs[0] and ('e1' in subs[0]):
      line = (re.sub(r'SW\.concat','SW.concat e0 e1',line))
    elif any(key in subs[0] for key in edict.keys()):
      for key in edict.keys():
        if key in subs[0]: 
           break
      rep = 'SW.concat '+edict[key][0]   
      line = (re.sub(r'SW\.concat', rep, line))
    elif ('blocks' in filename) and (' 2' in subs[0]):
      line = re.sub(r'SW\.concat','SW.concat en.block.block1 en.block.block2',line)
    else:
    
      if 'd1' not in subs[0]:
        if ('calendar' in filename) and ('d0' in subs[0]):
          line = (re.sub(r'SW\.concat','SW.concat date:2015:1:2 d0',line))
        elif 'before d0' in subs[0]:
          line = (re.sub(r'SW\.concat','SW.concat date:2004:-1:-1 d0',line))
        else:
          line = (re.sub(r'SW\.concat','SW.concat date:2004:-1:-1 date:2010:-1:-1',line))
          #line = (re.sub(r'SW\.concat','SW.concat d0 date:2010:-1:-1',line))
      else:
        line = (re.sub(r'SW\.concat','SW.concat d0 d1',line))
    '''
    elif any(key in subs[0] for key in edict.keys()):
      for key in edict.keys():
        if key in subs[0]: 
           break
      rep = 'SW.concat '+edict[key][0]   
      line = (re.sub(r'SW\.concat', rep, line))    elif 'bobs' in subs[0]:
      line = (re.sub(r'SW\.concat','SW.concat e0 en.person.bob',line))
    elif any(c in subs[0] for c in ['male', 'female' , 'gender']):
      line = (re.sub(r'SW\.concat','SW.concat en.gender.male en.gender.female',line))
    '''
  #if not line.startswith("SW.concat"):
  #  line = (re.sub(r'SW\.concat (.*?) (.*?) ',r'( SW.concat \1 \2 ) ',line)).strip()
    #else:
    #  line = (re.sub(r'SW\.concat','( SW.concat d0 d1 )',line)).strip()
  #if e1 not in susb[0]:
  
  #fout.write(str(count) + '\t' + line.strip()+ '\n')  
  return line


dir = sys.argv[1]
outdir = sys.argv[2]

files = os.listdir(dir)

for filename in files:
  if 'transout' not in filename:
    continue
  f = open(os.path.join(dir, filename))
  print('input' , filename)
  fout = open(os.path.join(outdir, filename+'.orig2'), 'w')
  #fout = open('outputexec2.txt', 'w')
  line = f.readline()
  while line:
    subs = line.split('\t')
    line = line.strip()
    if line == '':
      continue
    print(subs)
    line = subs[1]
    

    count = len(subs[1].split(' '))
    line = re.sub(r' SW.getProperty en(.*?) string\-(.*?)[ \n]',r' ( SW.getProperty en\1 string-\2 )',line)
    line = re.sub(r'SW\.reverse string\-(.*?) ',r'( SW.reverse string-\1 ) ',line)   
    line = re.sub(r'SW\.ensureNumericProperty string\-(.*?)[ \n]',r'( SW.ensureNumericProperty string-\1 )',line)
    line = re.sub(r'string\-sum ', r'SW.aggregate string-sum ', line)
    line = re.sub(r'string\-avg ', r'SW.aggregate string-avg ', line)
    
    line = re.sub(r'SW\.', 'call SW.', line)
    
    line = '( call SW.listValue ( ' + line.strip() + ' ) )' 
    line = re.sub(r'string\-(.*?) ', r'( string \1 ) ', line)
    line = re.sub(r'!',r'! ',line)
    line = re.sub(r'entity-(.*?) ',r'( call SW.getProperty ( call SW.singleton \1 ) ( string ! type ) ) ',line)
    
    #line = re.sub('num2', r'number 2 ', line)
    #print(line + '\n')
    try: 
      if 'basketball' in filename:
        line = re.sub(r'number3 ((.*?) )?', r'( number 3 \1) ', line)
        line = re.sub('number2 ', r'( number 2 ) ', line)
      else:
        line = re.sub(r'number(.*?) (en(.*?) )?', r'( number \1 \2) ', line)
    except:
      #if 'basketball' not in filename:
      line = re.sub('number(.*?) ', r'( number \1 ) ', line)
    # parag
    #line = re.sub('num(.*?) (en(.*?) )?', r'( number \1 \2) ', line)
    
    ####--------------prune 2
    line = re.sub(r'\.size','call .size',line)
    line = re.sub(r'date:(.*?):(.*?):(.*?) ' , r'( date \1 \2 \3 ) ',line)
    
    #fout.write(str(count) + '\t' + line.strip()+ '\n')  
    fout.write(subs[0] + '\t' + line.strip()+'\n')
    line = f.readline()
  fout.close()
  f.close()
