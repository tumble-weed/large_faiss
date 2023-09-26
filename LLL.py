import numpy as np
import torch 
import dutils
import inspect
import faiss
import importlib
def npuq(*args,**kwargs):
    return np.unique(*args,**kwargs)

def reload(mod):
    # L = inspect.f_back.f_locals['']
    importlib.reload(mod)
def create_index(data):
    # pass
    index = faiss.IndexFlatL2(data.shape[-1])
    index.add(data)
    return index    

def create_index_hnsw(data):
    # pass
    index = faiss.IndexHNSWFlat(data.shape[-1],32)
    index.add(data)
    return index    
# hi = 1
'''
import faiss
vecs = faiss.IndexFlatL2(patches_.shape[-1])
index = faiss.IndexIVFPQ(vecs, patches_.shape[-1], 2048, 7, 8)
# index = faiss.IndexIVFPQ()
v = np.random.randn(324*10000,147)
'''