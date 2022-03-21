import numpy as np
import pandas as pd
import math

names=['RLBHE_BICYCLE.csv','RLBHE_BOAT.csv','RLBHE_BOTTLE.csv','RLBHE_BUS.csv','RLBHE_CAR.csv',
       'RLBHE_CAT.csv','RLBHE_CHAIR.csv','RLBHE_CUP.csv','RLBHE_DOG.csv','RLBHE_MOTORBIKE.csv',
       'RLBHE_PEOPLE.csv','RLBHE_TABLE.csv']
for name in names:
    df=pd.read_csv(name)
    val1=np.array(df['Entropy'])
    val2=np.array(df['BRISQUE'])
    val3=np.array(df['PSNR'])
    x=np.mean(val1,axis=0)
    y=np.mean(val2,axis=0)
    z=np.mean(val3,axis=0)
    print(name, "{:.2f}".format(x), "{:.2f}".format(y), "{:.2f}".format(z))
    print('\n')