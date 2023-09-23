from autofaiss import build_index
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--embeddings-folder',default='tiny-imagenet-200')
parser.add_argument('--index-folder',default='index')
parser.add_argument('--split',default='val')
# parser.add_argument('--batch-size',default=1000,type=int)
args = parser.parse_args()

embeddings_folder = args.embeddings_folder
index_folder = args.index_folder
split = args.split
# batch_size = args.batch_size
# extract_patches_for_split(imagenet_folder,split,'embeddings',batch_size = batch_size)

# split = 'val'
build_index(
embeddings=f"{embeddings_folder}/{split}",
index_path=f"{index_folder}/{split}.index",
index_infos_path="{index_folder}/{split}_infos.json",
max_index_memory_usage="8G",
current_memory_available="10G",
)
