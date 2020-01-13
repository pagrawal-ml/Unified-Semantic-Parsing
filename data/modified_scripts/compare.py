import os

import difflib

def print_diff(case_a, case_b):
    output_list = [li for li in difflib.ndiff(case_a, case_b) if li[0] != ' ']
    print(output_list)

domains = ["recipes", "restaurants", "publications", "housing", "calendar","blocks", "basketball"] # socialnetwork

generatedFolder = "outputs/postprocessed_final";

def compare_reverse(domain):    
    gTrainFile = os.path.join(generatedFolder, domain +  "_train.transout.txt.orig2")
    gTestFile = os.path.join(generatedFolder, domain + "_test.transout.txt.orig2")  
      

    actualFolder = 'overnight-lf' + '/'
    aTrainFile = os.path.join(actualFolder , domain +     "_train.tsv")
    aTestFile = os.path.join(actualFolder,   domain + "_test.tsv")    

    generatedLines = open(gTestFile, 'r').readlines()
    generatedDict = {}    

    for line in generatedLines:
        l, o = line.split('\t')
        generatedDict[l.rstrip('\n')] = o.rstrip('\n')    

    #print(generatedDict)    

    actualLines = open(aTestFile, 'r').readlines()
    actualDict = {}    

    for line in actualLines:
        l, o = line.split('\t')
        actualDict[l.rstrip('\n')] = o.rstrip('\n')  
      

    print(len(actualDict))
    print(len(generatedDict))  
      
    counter = 0
    for al, gl in zip(actualLines, generatedLines):
        #print(al)
        #print(gl)
        al = al.rstrip('\n')
        gl = gl.rstrip('\n')
        ai, ao = al.split('\t')
        gi, go = gl.split('\t')
        counter += 1
        if ai != gi or ao != go:
            print(ai == gi, ao == go)
            
            lendiff = len(go) - len(ao)
            #print('lendiff', lendiff)
            if abs(lendiff) > 1:
                print('$$$$ Actual: ', al)
                print('$$$$: Generated ', gl)
                print('lendiff', lendiff)
                print('#examples ', counter)   
            #print_diff(al, gl)
        else:
            pass
            #print('######')
            #print(al)
            #print(gl)
 
for domain in domains:
    print('###### domain: ', domain)
    compare_reverse(domain)
