import numpy as np
import os
class npy_saver():
    def __init__(self,n_batches,save_pattern):
        self.b = 0
        self.n_batches = n_batches
        self.ndigits = len(str(nbatches - 1))
        pass
    def add(self,patches_):
        strb = "{:0{width}d}".format(self.b, width=self.ndigits)
        np.save(save_pattern.format(strb), patches_)
        self.b += 1
        pass
    pass

class memmap_saver():
    def __init__(self,total_len,filename):
        # self.b = 0

        self.total_len = total_len
        self.f = filename
        self.pointer = 0
        self.d = os.path.dirname(filename)
        os.makedirs(self.d,exist_ok=True)
        self.total_len = None
        pass
    def add(self,patches_):
        if self.pointer == 0:
            self.patch_dim = patches_.shape[-1]
            self.arr = np.memmap(self.f, dtype='float32', mode='w+',shape=(self.total_len,self.patch_dim))            
        # strb = "{:0{width}d}".format(self.b, width=self.ndigits)
        # np.save(save_pattern.format(strb), patches_)
        delta = patches_.shape[0]
        self.arr[self.pointer:self.pointer+delta] = patches_
        # self.b += 1
        self.pointer = self.pointer + delta
        if self.pointer == self.total_len:
            del self.arr
            
            import json
            info_fname = self.f.replace('.memmap','.info')
            assert self.total_len == int(self.total_len)
            with open(info_fname,'w') as f:
                json.dump({'patch_dim':int(self.patch_dim),'total_len':int(self.total_len)},f)
    pass
