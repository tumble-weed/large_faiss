# from autofaiss import build_index
# import argparse
# import os
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

def create_index():
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
    '''
    current_memory_available is the parameter controlling the maximum amount of
    ram to use, it's that one you need to put at 16GB if you want to use only
    16GB while creating the index

    max_index_memory_usage controls the size of the built index, which is
    different.

    '''
    build_index(
    embeddings=embeddings_folder,
    index_path=f"{index_folder}/{dataset}_{split}.index",
    index_infos_path="{index_folder}/{dataset}_{split}_infos.json",
    max_index_memory_usage="10G",
    current_memory_available=f"{current_memory_available}G",
    )
if __name__ == '__main__':
    create_index()