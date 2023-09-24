import numpy as np
import argparse
import torch
import os
import build_index
import create_embeddings
def main():
    parser = argparse.ArgumentParser()
    #========================================================
    parser.add_argument('--batch-size',default=1000,type=int)
    parser.add_argument('--coarse-dim',default=14,type=int)
    parser.add_argument('--pyramid-ratio',default=4/3,type=int)
    #========================================================
    parser.add_argument('--dataset-folder',default='tiny-imagenet-200')
    parser.add_argument('--split',default='val')
    parser.add_argument('--embeddings-root-folder',default='embeddings',type=str)
    parser.add_argument('--index-folder',default='index')
    parser.add_argument('--current-memory-available',default=20,type=int)
    #========================================================
    # parser.add_argument('--max-pyramid-depth',default='embeddings',type=str)
    
    args = parser.parse_args()
    
    # dataset_folder = args.dataset_folder
    dataset = os.path.basename((args.dataset_folder).rstrip(os.path.sep))
    if 'cifar' in dataset:
        dataset = 'cifar'
    if dataset == 'cifar':
        size0 = 32
    elif dataset == 'tiny-imagenet-200':
        size0 = 64
    max_pyramid_depth =  np.log(size0 / (args.coarse_dim)) / np.log(args.pyramid_ratio)
    add_base_level = True if np.ceil(max_pyramid_depth) > max_pyramid_depth else False
    sizes = [ int(size0 * (args.pyramid_ratio**(-i))) for i in range(int(np.ceil(max_pyramid_depth)))]
    if add_base_level is True:
        sizes[-1] = args.coarse_dim
    # assert False
    for size in sizes:
        # assert False
        # split = args.split
        # batch_size = args.batch_size
        # embeddings_root_folder = args.embeddings_root_folder
        if dataset == 'tiny-imagenet-200':
            create_embeddings.create_tiny_imagenet_embeddings(args.dataset_folder,
            args.split,
            args.batch_size,
            args.embeddings_root_folder,
            size=size)  
        elif dataset == 'cifar':

            embeddings_folder = create_embeddings.create_cifar_embeddings(args.dataset_folder,
            args.split,
            args.batch_size,
            args.embeddings_root_folder,
            size=size)  
            build_index.create_index(
                dataset,
                args.split,
                20,
                embeddings_folder,
                args.index_folder
            )
            cleanup(embeddings_folder)
            assert False
def cleanup(folder):
    cmd = f'rm -rf {folder}'
    import ipdb;ipdb.set_trace()
    os.system(cmd)
if __name__ == '__main__':
    main()