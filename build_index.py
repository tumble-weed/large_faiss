from autofaiss import build_index
import argparse
import os
import time
import json
import faiss
# def create_imagenet_index():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--embeddings-folder',default='embeddings/tiny-imagenet-200')
#     parser.add_argument('--index-folder',default='index')
#     parser.add_argument('--split',default='val')
#     parser.add_argument('--current-memory-available',default=20,type=int)

#     # parser.add_argument('--batch-size',default=1000,type=int)
#     args = parser.parse_args()

#     embeddings_folder = args.embeddings_folder
#     index_folder = args.index_folder
#     split = args.split
#     current_memory_available = args.current_memory_available
#     '''
#     current_memory_available is the parameter controlling the maximum amount of
#     ram to use, it's that one you need to put at 16GB if you want to use only
#     16GB while creating the index

#     max_index_memory_usage controls the size of the built index, which is
#     different.

#     '''
#     build_index(
#     embeddings=f"{embeddings_folder}/{split}",
#     index_path=f"{index_folder}/{split}.index",
#     index_infos_path="{index_folder}/{split}_infos.json",
#     max_index_memory_usage="10G",
#     current_memory_available=f"{current_memory_available}G",
#     )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='cifar')
    parser.add_argument('--split',default='test')
    parser.add_argument('--embeddings-root-folder',default='embeddings')
    parser.add_argument('--index-folder',default='index')
    parser.add_argument('--current-memory-available',default=20,type=int)

    # parser.add_argument('--batch-size',default=1000,type=int)
    args = parser.parse_args()
    dataset = args.dataset
    split = args.split
    embeddings_root_folder = args.embeddings_root_folder
    embeddings_folder = os.path.join(embeddings_root_folder,dataset,split)
    index_folder = args.index_folder
    split = args.split
    current_memory_available = args.current_memory_available
    create_index_autofaiss(dataset,split,
current_memory_available,embeddings_folder,index_folder)

def create_index_autofaiss(dataset,split,size,
current_memory_available,embeddings_folder,index_folder):
    '''
    current_memory_available is the parameter controlling the maximum amount of
    ram to use, it's that one you need to put at 16GB if you want to use only
    16GB while creating the index

    max_index_memory_usage controls the size of the built index, which is
    different.

    '''
    # import ipdb;ipdb.set_trace()
    index_path = f"{index_folder}/{dataset}_{split}_{size}.index"
    index_infos_path = f"{index_folder}/{dataset}_{split}_{size}_infos.json"
    if True:
        build_index(
        embeddings=embeddings_folder,
        index_path=index_path,
        index_infos_path=index_infos_path,
        max_index_memory_usage="10G",
        current_memory_available=f"{current_memory_available}G",
        )
    return index_path,index_infos_path

def create_index_mine(dataset,split,size,
# current_memory_available,
# patch_dim,
embeddings_fname,
# npatches,
index_folder,
nlist = 2048,
m = 7,
nbits = 8,
):
    '''
    current_memory_available is the parameter controlling the maximum amount of
    ram to use, it's that one you need to put at 16GB if you want to use only
    16GB while creating the index

    max_index_memory_usage controls the size of the built index, which is
    different.

    '''
    # import ipdb;ipdb.set_trace()
    index_path = f"{index_folder}/{dataset}_{split}_{size}.index"
    index_infos_path = f"{index_folder}/{dataset}_{split}_{size}_infos.json"
    
    if True:

        with open(embeddings_fname.replace('.memmap','.info'),'r') as f:
            info = json.load(f)
        data = np.memmap(embeddings_fname, dtype='float32', mode='r',shape=(info['total_len'],info['patch_dim']))
        quantizer = faiss.IndexFlatL2(info['patch_dim']) 
        
        index = faiss.IndexIVFPQ(quantizer, info['patch_dim'], nlist, m, nbits)  # 8 
        
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
        faiss.write_index(index, index_path)


    return index_path,index_infos_path


if __name__ == '__main__':
    main()