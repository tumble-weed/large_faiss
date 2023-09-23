import glob
import os
import torch
import numpy as np
import torchvision.transforms
from PIL import Image
import skimage.io
import tqdm
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
# import blosc2
def extract_patches(t,patch_size,stride):
    patches =  t.unfold(2,patch_size,stride).unfold(3,patch_size,stride)
    patches = patches.permute(0,2,3,1,4,5)
    patches = patches.flatten(start_dim=0,end_dim=2)
    patches = patches.view(patches.shape[0],-1)
    return patches
def extract_patches_for_split(imagenet_folder,split,embeddings_root_folder,patch_size=7,stride=1,batch_size = 100):
    save_dir = os.path.join(embeddings_root_folder,split)
    os.makedirs(save_dir,exist_ok=True)
    if split == 'train':
        pattern = os.path.join(imagenet_folder,split,'*','images','*.JPEG')
    elif split == 'val':
        pattern = os.path.join(imagenet_folder,split,'images','*.JPEG')
    print(pattern)
    filepaths = glob.glob(pattern)
    nbatches = (len(filepaths) + batch_size - 1)//batch_size
    # print(filepaths)
    for b in tqdm.tqdm(range(nbatches)):
        bpaths = filepaths[b*batch_size:(b+1)*batch_size]
        tbatch =[]
        for p in tqdm.tqdm(bpaths):
            im = skimage.io.imread(p)
            if im.ndim == 2:
                im = np.stack([im,im,im],axis=-1)
            im_pil = Image.fromarray(im)
            t = torchvision.transforms.ToTensor()(im_pil)
            tbatch.append(t)
        tbatch = torch.stack(tbatch,0)
        patches = extract_patches(tbatch,patch_size,stride)
        patches_ = tensor_to_numpy(patches)
        patches_ = patches_.astype(np.float16)
        # import ipdb;ipdb.set_trace()
        patches_ = np.reshape(patches_,(patches_.shape[0],-1))
        # np.savez(os.path.join(save_dir,f'tiny_imagenet_{split}_{b}'), patches_)
        np.save(os.path.join(save_dir,f'tiny_imagenet_{split}_{b}'), patches_)
        # if b ==1:
        #     break
        # assert False
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--imagenet-folder',default='tiny-imagenet-200')
parser.add_argument('--split',default='val')
parser.add_argument('--batch-size',default=1000,type=int)
args = parser.parse_args()

imagenet_folder = args.imagenet_folder
split = args.split
batch_size = args.batch_size
extract_patches_for_split(imagenet_folder,split,'embeddings',batch_size = batch_size)


