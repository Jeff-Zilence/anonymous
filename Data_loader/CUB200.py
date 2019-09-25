from __future__ import absolute_import, print_function

"""
CUB-200-2011 data-set for Pytorch
"""
import os
from collections import defaultdict
from .image_preprocessing import image_converter
debug = False


class MyData:
    def __init__(self, root=None, label_txt=None,
                 mode=''):

        # Initialization data path and train(gallery or query) txt path
        if root is None:
            self.root = "/data/DML/CUB_200_2011/"
        self.root = root

        if label_txt is None:
            label_txt = os.path.join(root, 'train.txt')

        # read txt get image path and labels
        file = open(label_txt)
        images_anon = file.readlines()
        file.close()

        images = []
        labels = []

        for img_anon in images_anon:
            # img_anon = img_anon.replace(' ', '\t')

            [img, label] = img_anon.split(' ')
            images.append(img)
            labels.append(int(label))

        classes = list(set(labels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels):
            Index[label].append(i)

        # Initialization Done
        self.root = root
        self.images = images
        self.labels = labels
        self.classes = classes
        self.mode = mode
        self.Index = Index
        self.loader = image_converter(mode)
        self.current = 0
        self.size = self.__len__()
        if debug:
            print(images, labels, classes, Index)

    def __getitem__(self, index):
        fn, label = self.images[index], self.labels[index]
        fn = os.path.join(self.root, fn)
        img = self.loader.get_input(path = fn, mode = self.mode)
        return img, label, fn

    def __getimage__(self, index):
        fn, label = self.images[index], self.labels[index]
        fn = os.path.join(self.root, fn)
        img = self.loader.read(path = fn)
        return img[:, :, ::-1], label

    def __len__(self):
        return len(self.images)


class CUB_200_2011:
    def __init__(self, width=224, origin_width=256, ratio=0.16, root=None, transform=None):
        # Data loading code
        # print('ratio is {}'.format(ratio))
        if root is None:
            root = "/data/DML/CUB_200_2011/"

        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')

        self.train = MyData(root, label_txt=train_txt, mode ='rand_crop')
        self.gallery = MyData(root, label_txt=test_txt, mode = 'center_crop')
        self.query = MyData(root, label_txt=test_txt, mode = 'center_crop')



def testCUB_200_2011():
    print(CUB_200_2011.__name__)
    data = CUB_200_2011()
    print(data.gallery.__getitem__(0))
    print(data.train.__getitem__(0))


if __name__ == "__main__":
    testCUB_200_2011()
