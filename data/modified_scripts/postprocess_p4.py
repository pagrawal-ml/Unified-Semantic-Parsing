import re
import os

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
import sys
def postprocess(file, outdir, edict, bw):
    #srcf = '../data/prune_entity_scripts/prune3_entity_2502/recipes/recipes_test.tgt'
    #tgtf = '../data/prune_entity_scripts/test_8231.out'
    #bw = 10
    
    

    #source_lines = open(srcf).readlines()
    #target_lines = open(tgtf).readlines()
    lines = open(file, 'r').readlines()
    source_lines = []
    target_lines = []
    for line in lines:
        s, t = line.split('\t')
        source_lines.append(s)
        target_lines.append(t)

    #fout = open(tgtf+'.orig', 'w')
    filename = os.path.basename(file)
    fout = open(os.path.join(outdir, filename+'.orig1'), 'w')
    for i in range(len(target_lines)):  
        subs = []
        subs.append(source_lines[int(i/bw)])
        line = target_lines[i]
        line = line.strip() + ' '
        line = re.sub(r'SW.filter var-s (.*?) \( SW.domain',r'SW.getProperty ( ( lambda s ( SW.filter ( var s ) \1 ) ) ( SW.domain',line)
        line = re.sub(r'SW.getProperty var-s (.*?) \( SW.domain',r'SW.getProperty ( ( lambda s ( SW.getProperty ( var s ) \1 ) ) ( SW.domain',line)
        line = re.sub(r'SW.(.*?) var-s (.*?) \( SW.domain',r'SW.getProperty ( ( lambda s ( SW.\1 ( var s ) \2 ) ) ( SW.domain',line)
        line = re.sub(r'SW.getProperty string',r'SW.getProperty SW.getProperty e0 SW.reverse string',line)
        line = re.sub(r'SW.filter string',r'SW.filter SW.getProperty e0 SW.reverse string',line)
        if True: # line.startswith('SW.concat'):
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
          elif ('blocks' in file) and (' 2' in subs[0]):
            line = re.sub(r'SW\.concat','SW.concat en.block.block1 en.block.block2',line)
          else:

            if 'd1' not in subs[0]:
              if ('calendar' in file) and ('d0' in subs[0]):
                line = (re.sub(r'SW\.concat','SW.concat date:2015:1:2 d0',line))
              elif 'before d0' in subs[0]:
                line = (re.sub(r'SW\.concat','SW.concat date:2004:-1:-1 d0',line))
              else:
                line = (re.sub(r'SW\.concat','SW.concat date:2004:-1:-1 date:2010:-1:-1',line))
                #line = (re.sub(r'SW\.concat','SW.concat d0 date:2010:-1:-1',line))
            else:
                line = (re.sub(r'SW\.concat','SW.concat d0 d1',line))

        #fout.write(str(count) + '\t' + line.strip()+ '\n')  
        #fout.write(subs[0] + '\t' + line.strip()+'\n')
        fout.write(source_lines[i] + '\t' + line.strip()+'\n')
    fout.close()

def main():
    #gfile = sys.argv[1]
    #rfile = sys.argv[2]
    #beam_width = int(sys.argv[3])
    beam_width = 1
    dir = sys.argv[1]
    outdir = sys.argv[2]

    files = os.listdir(dir)

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
    edict['300 pm']=['( time 10 0 ) ( time 15 0 )']
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
    for file in files:
        postprocess(os.path.join(dir, file),outdir, edict, beam_width)

if __name__ == '__main__':
  main()
