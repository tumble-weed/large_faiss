import numpy as np
import faiss
import glob
import tqdm
def build_index(
        embeddings=None,
        index_path=None,
        # index_infos_path=index_infos_path,
        # max_index_memory_usage="10G",
        # current_memory_available=f"{current_memory_available}G",
        ):
    pass

def create_embeddings():
    fname = 'tmp.mmap'
    nimages= 10000
    npatches_per_im= 48*48
    patch_dim= 147
    npatches = npatches_per_im*nimages
    data = np.memmap(fname,dtype='float32',mode='w+',shape=(nimages*npatches_per_im,patch_dim))
    # pass

    batch_size = 1000
    nbatches = npatches//batch_size
    for bi in tqdm.tqdm(range(nbatches)):
        
        data[bi*batch_size:(bi+1)*batch_size] = np.random.randn(batch_size,patch_dim)
    del data
def create_index():
    fname = 'tmp.mmap'
    nimages= 10000
    npatches_per_im= 48*48
    patch_dim= 147
    nlist = 2048
    npatches = npatches_per_im*nimages
    data = np.memmap(fname, dtype='float32', mode='r',shape=(npatches,patch_dim))
    print(data.shape)
    # Quantizer and Index
    quantizer = faiss.IndexFlatL2(patch_dim) 
    index = faiss.IndexIVFPQ(quantizer, patch_dim, nlist, 7, 8)  # 8 bytes per vector,
    import time
    tic = time.time()
    print("Training the index...")
    index.train(data)         
    toc = time.time()
    print('time taken to create',toc-tic)
    print("Adding data")
    tic = time.time()
    index.add(data)
    toc = time.time()
    print('time taken to add',toc-tic)
    import ipdb;ipdb.set_trace()
    del data
# create_embeddings()
create_index()