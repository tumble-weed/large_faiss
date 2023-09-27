import pickle
def read_cifar_file(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    d[b'data'] = d[b'data']/255.
    # d[b'data'] = d[b'data'] - 0.5
    return d

class CIFARdata():
    def __init__():
        pass
    def get_size():
        pass
    def get_size_blurred():
        pass
    pass