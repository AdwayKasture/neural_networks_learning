#scipt to extract data from ASL signlanguage dataset from kaggle
#pandas to read  csv , make directories with os.makedir()
#cv2 to convert to 3 channel numpy array and save jpg ONLY got error with png
#link for dataset https://www.kaggle.com/datamunge/sign-language-mnist

import pandas as pd
import numpy as np
import  cv2
import os 
import shutil
import random
import glob
import cv2



#make directories to store images
'''
#uncomment below line during first run
#os.makedirs('./organised')
for i in range(0,27):
    #replace test with train and valid while creating those directories
    os.makedirs('./organised/test/'+str(i))
'''
#read csv replace with sign_mnist_train for train and validation dir
df = pd.read_csv("sign_mnist_test.csv")

#read image from csv
def convert_to_images(df,path,r):
    img = np.ones((28,28), dtype = np.uint8)
    for i in range(0,28) :
        for j in range(0,28):
            img[i,j] =  df.iloc[r,28*i + j+1] 
    backtorgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    file_name = './organised/'+ str(path) + '/' + str(df.iloc[r,0])+ str(r)+ ".png"
    cv2.imwrite(file_name,backtorgb)
    

#iterate through df
#train range 0- 25000
#validation range 25001 - 27000
for i in range(1,1000):
    #replace test with train and validation correspondingly
    convert_to_images(df,"test",i)
    print("done",i)

