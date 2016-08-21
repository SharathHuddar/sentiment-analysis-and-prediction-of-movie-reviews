import pyprind, os, shutil, urllib.request, sys, tarfile
import pandas as pd
import numpy as np


basepath = './aclImdb'

URL = "http://ai.stanford.edu/~amaas/data/sentiment/"
filename = "aclImdb_v1.tar.gz"


def dlProgress(count, blockSize, totalSize):
    percent = int(count*blockSize*100/totalSize)
    sys.stdout.write("\r" + URL + filename + "...%d%%" % percent)
    sys.stdout.flush()

print ("Downloading " + filename + "...")
tpl = urllib.request.urlretrieve(URL + filename, filename, reporthook=dlProgress)
print("\nDownload complete")
print("Extracting " + filename + "...")
tar = tarfile.open(filename)
tar.extractall()
tar.close()
print ("Done extracting " + filename + " to " + filename[:-10])
os.remove(filename)

print ("Loading data into a pandas DataFrame...")

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
print ("Done loading.")
shutil.rmtree("aclImdb")
df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

print ("Dumping data to movie_data.csv...")
df.to_csv('./movie_data.csv', index=False)
print ("Done")
