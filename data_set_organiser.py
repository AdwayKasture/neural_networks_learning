#simple script paste in data_set
#replace cat and got with your choice of name , and count in for loop to select samples 
#stores data in previous directory in an "organised folder"



import os 
import shutil
import random
import glob
os.makedirs('../organised')
os.makedirs('../organised/train/dog')
os.makedirs('../organised/train/cat')
os.makedirs('../organised/valid/dog')
os.makedirs('../organised/valid/cat')
os.makedirs('../organised/test/dog')
os.makedirs('../organised/test/cat')


r = 0
#to selesct rando smaples 
for i in random.sample(glob.glob('cat*'), 500):
    r = r + 1
    print(i,r)
    shutil.move(i, '../organised/train/cat')      
for j in random.sample(glob.glob('dog*'), 500):
    shutil.move(j, '../organised/train/dog')
for i in random.sample(glob.glob('cat*'), 100):
    shutil.move(i, '../organised/valid/cat')        
for i in random.sample(glob.glob('dog*'), 100):
    shutil.move(i, '../organised/valid/dog')
for i in random.sample(glob.glob('cat*'), 50):
    shutil.move(i, '../organised/test/cat')      
for i in random.sample(glob.glob('dog*'), 50):
    shutil.move(i, '../organised/test/dog')
