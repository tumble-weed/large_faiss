from autofaiss import build_index
if False:
    build_index(
    embeddings="embeddings",
    index_path="knn.index",
    index_infos_path="infos.json",
    max_index_memory_usage="4G",
    current_memory_available="5G",
)

from autofaiss import build_index
import numpy as np

# embeddings = np.float32(np.random.rand(100000, 512))
def create_large_array():
    nimages = 10
    channels = 3
    image_shape = (64,64)
    patch_size = (7,7)
    dtype = np.float32
    shape = nimages,channels,image_shape[0],image_shape[1],patch_size[0],patch_size[1]
    mmapped_array = np.memmap('dummy_array.npy', dtype=dtype, mode='w+', shape=shape)

    # Now you can work with the mmapped_array just like a regular NumPy array
    # For example, you can assign values to it
    mmapped_array[:] = np.random.rand(*shape)

    # Remember to flush and close the memory-mapped array when you're done
    mmapped_array.flush()
    del mmapped_array  # This deletes the reference to the memory-mapped array, closing it

    # Later, you can reopen the memory-mapped array if needed
    reopened_array = np.memmap('dummy_array.npy', dtype=dtype, mode='r', shape=shape)

    # Access and manipulate the reopened array as needed
    return reopened_array

# embeddings = create_large_array()
embeddings = np.random.rand(100,3*64*64*7*7)
index, index_infos = build_index(embeddings, save_on_disk=False)
_, I = index.search(embeddings, 1)
print(I)