import struct
import numpy as np
import cv2

def read_images(filename):
    with open(filename, 'rb') as fin:
        magic = struct.unpack('>i', fin.read(4))
        n = struct.unpack('>i', fin.read(4))[0]
        sz = struct.unpack('>ii', fin.read(8))
        data = []
        for t in range(n):
            img = np.fromfile(fin, dtype=np.uint8, count=sz[0]*sz[1]).reshape(sz)
            data.append(img)
    return data


def read_labels(filename):
    with open(filename, 'rb') as fin:
        magic = struct.unpack('>i', fin.read(4))
        n = struct.unpack('>i', fin.read(4))[0]
        return np.fromfile(fin, dtype=np.uint8, count=n)


if __name__ == '__main__':
    data = read_images('data/train-images-idx3-ubyte')
    labels = read_labels('data/train-labels-idx1-ubyte')
    print len(data), len(labels)
