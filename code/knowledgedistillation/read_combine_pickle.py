import pickle
import numpy as np

with open('domain1.pkl', 'rb') as f:
   d1 = np.asarray(pickle.load(f))
with open('domain2.pkl', 'rb') as f:
   d2 = np.asarray(pickle.load(f))

d_comb = np.concatenate((d1,d2,),axis=0)
with open('d_combined.pkl', 'wb') as f:
   pickle.dump(d_comb,f)
