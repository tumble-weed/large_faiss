import glob
import os
import torch
import numpy as np
import torchvision.transforms
from PIL import Image
import skimage.io
import tqdm
import argparse
import dutils
import debugger
import embedding_savers
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
#====================================================== 
# import blosc2
def convert_to_relative_index(I,patches_per_image):
    image_ix = I//patches_per_image
    rel_patch_ix = I%patches_per_image
    return image_ix,rel_patch_ix

def extract_patches(t,patch_size,stride):
    patches =  t.unfold(2,patch_size,stride).unfold(3,patch_size,stride)
    patches_per_image = np.prod(patches.shape[2:4])
    patches = patches.permute(0,2,3,1,4,5)
    patches = patches.flatten(start_dim=0,end_dim=2)
    patches = patches.view(patches.shape[0],-1)
    patches_ = tensor_to_numpy(patches)
    patches_ = patches_.astype(np.float16)
    
    patches_ = np.reshape(patches_,(patches_.shape[0],-1))
    # import ipdb;ipdb.set_trace()
    return patches_,patches_per_image

def extract_patches_for_imagenet_split(filepaths,save_pattern,patch_size=7,stride=1,batch_size = 100,size=64):
    nbatches = (len(filepaths) + batch_size - 1)//batch_size
    ndigits = len(str(nbatches - 1))
    # print(filepaths)
    for b in tqdm.tqdm(range(nbatches)):
        bpaths = filepaths[b*batch_size:(b+1)*batch_size]
        tbatch =[]
        for p in tqdm.tqdm(bpaths):
            im = skimage.io.imread(p)
            if im.ndim == 2:
                im = np.stack([im,im,im],axis=-1)
            if size != 64:
                im = skimage.transform.resize(im,(size,size,3))
                assert False
            im_pil = Image.fromarray(im)
            t = torchvision.transforms.ToTensor()(im_pil)
            tbatch.append(t)
        tbatch = torch.stack(tbatch,0)
        patches_ = extract_patches(tbatch,patch_size,stride)
        
        strb = "{:0{width}d}".format(b, width=ndigits)
        np.save(save_pattern.format(size,strb), patches_)
        # if b ==1:
        #     break


def read_cifar_file(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    d[b'data'] = d[b'data']/255.
    # d[b'data'] = d[b'data'] - 0.5
    return d



def create_tiny_imagenet_embeddings(dataset_folder,
split,
batch_size,
embeddings_root_folder,
size = 64,
):
    dataset = 'tiny-imagenet-200'
    save_dir = os.path.join(embeddings_root_folder,dataset,split,str(size))
    save_pattern = os.path.join(save_dir,f'{dataset}_{split}')+'_{}_{}'
    os.makedirs(save_dir,exist_ok=True)
    if split == 'train':
        pattern = os.path.join(dataset_folder,split,'*','images','*.JPEG')
    elif split == 'val':
        pattern = os.path.join(dataset_folder,split,'images','*.JPEG')
    print(pattern)
    filepaths = glob.glob(pattern)

    extract_patches_for_imagenet_split(filepaths,
    save_pattern,batch_size = batch_size,size=size)
    return save_dir


def create_cifar_embeddings(dataset_folder,
split,
batch_size,
embeddings_root_folder,
size = 32,
):
    dataset = 'cifar'
    save_dir = os.path.join(embeddings_root_folder,dataset,split,str(size))
    save_pattern = os.path.join(save_dir,f'{dataset}_{split}')+'_{}_{}'
    os.makedirs(save_dir,exist_ok=True)
    if split == 'train':
        raise NotImplementedError
        pattern = os.path.join(dataset_folder,split,'*','images','*.JPEG')
    elif split == 'test':
        data_dict = read_cifar_file(os.path.join(dataset_folder,'test_batch'))

    # import ipdb;ipdb.set_trace()
    # print(pattern)
    # filepaths = glob.glob(pattern)

    extract_patches_for_cifar_split(data_dict,
    save_pattern,batch_size = batch_size,size=size)
    return save_dir

def extract_patches_for_cifar_split(data_dict,save_pattern,patch_size=7,stride=1,batch_size = 100,size=32):
    
    # print(filepaths)
    data = data_dict[b'data']
    # assert False
    nbatches = (data.shape[0] + batch_size - 1)//batch_size
    ndigits = len(str(nbatches - 1))
    # saver = embedding_savers.npy_saver(n_batches,save_pattern.format(size,'{}'))
    saver = embedding_savers.memmap_saver(None,save_pattern.format(size,'.memmap'))
    n_images_done = 0
    for b in tqdm.tqdm(range(nbatches)):

        tbatch_ = data[b*batch_size:(b+1)*batch_size:]
        n_images_in_batch = tbatch_.shape[0]
        tbatch_ = np.reshape(tbatch_,(-1,3,32,32))

        
        # assert False
        if size != 32:
            tbatch_bhwc = np.transpose(tbatch_,(0,2,3,1))
            tbatch_bhwc1 = skimage.transform.resize(tbatch_bhwc,(tbatch_bhwc.shape[0],size,size,3))
            tbatch_ = np.transpose(tbatch_bhwc1,(0,3,1,2))

        tbatch = torch.tensor(tbatch_)
        # assert False
        patches_,n_patches_per_image = extract_patches(tbatch,patch_size,stride)
        if b ==0:
            saver.total_len= data.shape[0] * n_patches_per_image
        

        strb = "{:0{width}d}".format(b, width=ndigits)
        # import ipdb;ipdb.set_trace()
        # np.save(    save_pattern.format(size,strb), patches_)
        saver.add(patches_)
        # if b ==0:
        #     debugger.patches_ = patches_
        #     break
        n_images_done += n_images_in_batch
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-folder',default='tiny-imagenet-200')
    parser.add_argument('--split',default='val')
    parser.add_argument('--batch-size',default=1000,type=int)
    parser.add_argument('--embeddings-root-folder',default='embeddings',type=str)
    args = parser.parse_args()
    # dataset_folder = args.dataset_folder
    dataset = os.path.basename((args.dataset_folder).rstrip(os.path.sep))
    if 'cifar' in dataset:
        dataset = 'cifar'
    # assert False
    # split = args.split
    # batch_size = args.batch_size
    # embeddings_root_folder = args.embeddings_root_folder
    if dataset == 'tiny-imagenet-200':
        create_tiny_imagenet_embeddings(args.dataset_folder,
        args.split,
        args.batch_size,
        args.embeddings_root_folder)  
    elif dataset == 'cifar':
        create_cifar_embeddings(args.dataset_folder,
        args.split,
        args.batch_size,
        args.embeddings_root_folder)  

if __name__ == '__main__':
    main()