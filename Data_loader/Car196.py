from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""

import os

from Data_loader.CUB200 import MyData
# from CUB200 import MyData, default_loader, Generate_transform_Dict
# share the same MyData with CUB200


class Car196:
    def __init__(self, root=None, origin_width=256, width=224, ratio=0.16, transform=None):
        # Data loading code
        if root is None:
            root = '/data/DML/Car196/'
        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')
        self.train = MyData(root, label_txt=train_txt, mode='rand_crop')
        self.gallery = MyData(root, label_txt=test_txt, mode='center_crop')
        self.query = MyData(root, label_txt=test_txt, mode='center_crop')


def testCar196():
    data = Car196()
    print(len(data.gallery))
    print(len(data.train))
    print(data.train[1])


if __name__ == "__main__":
    testCar196()


