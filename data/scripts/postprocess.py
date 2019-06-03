import re
import os

def process(line,filename):

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
    
    ####--------------prune 2
    line = re.sub(r'\.size','call .size',line)
    line = re.sub(r'date:(.*?):(.*?):(.*?) ' , r'( date \1 \2 \3 ) ',line)
    line = re.sub(r'\(_time_(.*?)_(.*?)_\) ' , r'( time \1 \2 ) ',line)

    return line

import sys
domain = sys.argv[1]

filename= 'output.txt'

lines1 = open(filename).readlines()

j=0
fout = open('output_postprocessed.txt', 'w')
for i in range(0,len(lines1)):

    line = process(lines1[i],domain)
    fout.write(line.strip()+"\n")
    j+=1

fout.close()
