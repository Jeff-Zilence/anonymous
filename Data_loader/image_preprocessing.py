import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math


class image_converter:
    def __init__(self, mode = ''):
        self.mode = mode

    def read(self, path = ''):
        if len(path)==0:
            path = '/data/DML/CUB_200_2011/train/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
        image = cv2.imread(path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image)
        return image / 255.

    def norm(self, input):
        input[:, :, 0] -= (104 / 255.)
        input[:, :, 1] -= (117 / 255.)
        input[:, :, 2] -= (128 / 255.)
        return input * 255.

    def resize(self, input, size = [256, 256]):
        output = cv2.resize(input, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        return output

    def center_crop(self, input, size = [224, 224]):
        start_0 = (input.shape[0] - size[0]) // 2
        end_0 = start_0 + size[0]
        start_1 = (input.shape[1] - size[1]) // 2
        end_1 = start_1 + size[1]
        output = input[start_0 : end_0, start_1 : end_1, :]
        return output

    def horizontalflip(self, input):
        rand = random.random()
        if rand > 0.5:
            return input[:, ::-1, :]
        else:
            return input

    def rand_crop(self, input, size = [224, 224], scale = [0.16, 1], ratio = [3. / 4. , 4. / 3.]):
        i, j, h, w = self.get_params(input, scale, ratio)
        output = cv2.resize(input[i:(i+h), j:(j+w), :], (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        return self.horizontalflip(output)

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.shape[0] * img.shape[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.shape[0] and h <= img.shape[1]:
                i = random.randint(0, img.shape[1] - h)
                j = random.randint(0, img.shape[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.shape[0] / img.shape[1]
        if (in_ratio < min(ratio)):
            w = img.shape[0]
            h = w / min(ratio)
        elif (in_ratio > max(ratio)):
            h = img.shape[1]
            w = h * max(ratio)
        else:  # whole image
            w = img.shape[0]
            h = img.shape[1]
        i = (img.shape[1] - h) // 2
        j = (img.shape[0] - w) // 2
        return i, j, h, w

    def get_input(self, path = '', mode = ''):
        image = self.read(path)
        if mode == 'rand_crop':
            image = self.resize(image)
            image = self.rand_crop(image)
            image = self.norm(image)
        elif mode == 'center_crop':
            image = self.resize(image)
            image = self.center_crop(image)
            image = self.norm(image)
        elif mode == 'origin':
            #image = self.resize(image, size=[224,224])
            image = self.norm(image)
        return image


def test_image_converter():
    while True:
        converter = image_converter()
        image = converter.read()
        print(converter.get_input( mode = 'center_crop'))
        print(converter.get_input( mode = 'rand_crop'))
        plt.figure()
        image = converter.resize(image)
        plt.imshow(image)
        plt.figure()
        image_center = converter.center_crop(image)
        plt.imshow(image_center)
        plt.figure()
        image_rand = converter.rand_crop(image)
        plt.imshow(image_rand)
        plt.show()

if __name__=='__main__':
    test_image_converter()