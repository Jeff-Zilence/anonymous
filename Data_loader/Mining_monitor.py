import numpy as np
import torch
import random
import matplotlib.pyplot as plt

debug = False

class Mining_monitor:
    def __init__(self, mode = 'hardest', class_num = 16, instance = 10, data_set = None):
        self.__factory = {
            'scan': self.gen_scan,
            'mining': self.gen_mining,
            'rand': self.gen_rand,
        }
        self.mode = mode
        self.class_num = class_num
        self.instance = instance
        self.mining_start = 5
        if data_set is not None:
            self.classes = list(data_set.Index.keys())
            self.classes_random = random.sample(self.classes, len(self.classes)//2)
            print(len(self.classes), self.classes)
            self.mining_counter = 0
            self.cur_list = []
            self.cur_scan = 0
            self.mining_pool = mining_pool(data_set = data_set)
            self.data_set = data_set
            self.data_len = data_set.__len__()
            self.epoch_steps = self.data_len // (class_num * instance) + 1

    def gen_mining(self, class_num, instance):
        if self.mining_pool.ready:
            if 'hardest' in self.mode:
                classes = self.mining_pool.gen_hardest_class(class_num)
            else:
                classes = self.mining_pool.gen_hard_class(class_num)
            if 'positive' in self.mode:
                batch_list = self.mining_pool.gen_hard_positive(classes, instance)
            else:
                batch_list = []
                for index in classes:
                    batch_list.extend(random.sample(self.data_set.Index[index], instance))
        else:
            batch_list = self.gen_rand(class_num, instance)
        return batch_list

    def gen_scan(self, class_num, instance):
        batch_list = np.arange(self.cur_scan, self.cur_scan + class_num * instance)
        self.cur_scan += class_num * instance
        return batch_list

    def gen_rand(self, class_num, instance):
        batch_list = []
        classes = random.sample(self.classes_random, class_num)
        for index in classes:
            batch_list.extend(random.sample(self.data_set.Index[index], instance))
        if debug:
            print(self.data_set.Index)
            print(classes)
            print(batch_list)
        return batch_list

    def get_batch(self, mode = 'rand', class_num = None, instance = None):
        if class_num is None:
            class_num = self.class_num
        if instance is None:
            instance = self.instance
        batch_list = self.__factory[mode](class_num = class_num, instance = instance)
        self.cur_list = batch_list
        size = [224, 224]
        image_batch = np.zeros([class_num*instance, 3, size[0], size[1]])
        label_batch = np.zeros([class_num*instance])
        for i,index in enumerate(batch_list):
            image, label = self.data_set.__getitem__(index)
            if debug:
                print(image, label)
                plt.imshow(image/255.+0.5)
                plt.show()
            image_batch[i, :, :, :] = np.transpose(image, (2, 0, 1))
            label_batch[i] = label
        #print(image_batch, label_batch)
        return torch.from_numpy(image_batch.astype(np.float32)), torch.from_numpy(label_batch.astype(np.int))

    def update(self, embed_feat):
        self.mining_pool.refresh(self.cur_list, embed_feat)
        '''
        if self.mining_counter % 1000 == 0:
            self.mining_pool.sort_product()
        '''
        if self.mining_counter % 20 == 0:
            self.mining_pool.refresh_product()
            self.mining_pool.refresh_centers()
            self.mining_pool.sort_class_product()
            self.mining_pool.cal_mean()
        if self.mining_counter == (self.epoch_steps * self.mining_start):
            self.mining_pool.ready = True
        self.mining_counter += 1
        #raise Exception


class mining_pool:
    def __init__(self, mode = '', data_set = None, dim = 512):
        self.pool = np.zeros([data_set.__len__(), dim])
        self.data_set = data_set
        self.classes = list(data_set.Index.keys())
        self.class_centers = np.zeros([len(self.classes), dim])
        self.center_product = np.zeros([len(self.classes), len(self.classes)])
        self.class_ranking = np.zeros([len(self.classes), len(self.classes)])
        self.hardest_product = np.zeros([len(self.classes)])
        self.hardest_ranking = np.zeros([len(self.classes)])
        self.product = np.zeros([data_set.__len__(), data_set.__len__()])
        #self.ranking = np.zeros([data_set.__len__(), data_set.__len__()])
        self.intra_ranking = {}
        self.class_product = {}
        self.positive_mean = 0
        self.negative_mean = 0
        self.positive_mean_all = []
        self.between_class = 0
        self.intra_class = 0
        self.ready = False
        self.mining_count = 0

    def refresh(self, batch_list, embed_feat):
        self.pool[batch_list, :] = embed_feat.cpu().detach().numpy()

    def gen_hard_class(self, class_num, k = 8):
        if k is None:
            k = class_num//2
        rand_class = random.sample(self.classes, class_num - k)
        class_list = rand_class.copy()
        product_list = []
        for i in rand_class[:k]:
            start = -1
            while self.class_ranking[i, start] in class_list:
                start -= 1
            class_list.append(self.class_ranking[i, start])
            product_list.append(self.center_product[i][self.class_ranking[i, start]])
        if self.mining_count % 20 == 0:
            print('hard negative class mining: {}'.format(product_list))
        return class_list

    def gen_hardest_class(self, class_num, pool_size = 16, k = None):
        if k is None:
            k = class_num//2
        class_list = []
        product_list = []
        for i in range(k):
            if i < k//2:
                rand_class = random.choice(self.classes)
                while rand_class in class_list:
                    rand_class = random.choice(self.classes)
            else:
                rand_class = random.choice(self.hardest_ranking[-pool_size:])
                count = 0
                while rand_class in class_list:
                    count += 1
                    if count < 10:
                        rand_class = random.choice(self.hardest_ranking[-pool_size:])
                    else:
                        rand_class = random.choice(self.classes)

            class_list.append(rand_class)
            start = -1
            while self.class_ranking[rand_class, start] in class_list:
                start -= 1
            class_list.append(self.class_ranking[rand_class, start])
            if self.center_product[rand_class][self.class_ranking[rand_class, start]] > 0.9999:
                print('error:{},start:{}'.format(class_list, start))
            product_list.append(self.center_product[rand_class][self.class_ranking[rand_class, start]])
        if self.mining_count % 20 == 0:
            print('hardest negative class mining: {}'.format(product_list))
        return class_list

    def gen_hard_positive(self, classes, instance, k = 1):
        batch_list = []
        product_list = []
        for i in classes:
            batch_list.extend(random.sample(self.data_set.Index[i], instance - k))
            hit = 0
            start = 0
            while hit < k:
                if self.intra_ranking[i][start] not in batch_list:
                    hit += 1
                    batch_list.append(self.data_set.Index[i][self.intra_ranking[i][start]])
                    product_list.append(self.class_product[i][self.intra_ranking[i][start]])
                start += 1
        if self.mining_count % 20 == 0:
            print('hard positive mining:{}'.format(product_list))
        self.mining_count += 1
        return batch_list

    def refresh_product(self):
        self.product = np.matmul(self.pool, np.transpose(self.pool))
    '''
    def sort_product(self):
        self.ranking = np.argsort(self.product, axis = 1)

    def resort_product(self, resort_size = 1000):
        sorted_index = self.ranking[:, -resort_size:]
        to_be_sorted = np.zeros(sorted_index.shape, dtype = np.int)
        for i in range(sorted_index.shape[0]):
            to_be_sorted[i] = self.product[i, sorted_index[i, :]]
        sort_result = np.argsort(to_be_sorted, axis = 1)
        for i in range(sorted_index.shape[0]):
            self.ranking[i, -resort_size:] = sorted_index[i, sort_result[i, :]]
    '''
    def sort_class_product(self):
        for i in self.classes:
            self.intra_ranking[i] = np.argsort(self.class_product[i])

    def refresh_centers(self):
        for i in self.classes:
            self.class_centers[i] = np.mean(self.pool[self.data_set.Index[i], :], axis = 0)
            if np.sum(self.class_centers[i]**2) != 0:
                self.class_centers[i] =  self.class_centers[i] / np.sqrt(np.sum(self.class_centers[i]**2))
            self.class_product[i] = np.matmul(self.class_centers[i], np.transpose(self.pool[self.data_set.Index[i], :]))
        self.center_product = np.matmul(self.class_centers, np.transpose(self.class_centers))
        self.class_ranking = np.argsort(self.center_product, axis = 1)
        for i in self.classes:
            self.hardest_product[i] = self.center_product[i, self.class_ranking[i, -2]]
        self.hardest_ranking = np.argsort(self.hardest_product)

    def cal_mean(self):
        positive_mean = []
        negative_mean = []
        intra_class = []
        for i in self.classes:
            index_len = self.data_set.__len__()
            index = np.array(self.data_set.Index[i])
            non_index = np.delete(np.arange(index_len), index)
            positive_mean.append(np.sum(self.product[index[0]: (index[-1]+1), index[0]: (index[-1]+1)] * (1 - np.eye(index.shape[0]))) / index.shape[0] / (index.shape[0] - 1))
            negative_mean.append(np.mean(self.product[index[0]: (index[-1]+1), non_index[0]: (non_index[-1]+1)]))
            intra_class.append(np.mean(self.class_product[i]))
        self.positive_mean_all = np.array(positive_mean)
        self.positive_mean = np.mean(positive_mean)
        self.negative_mean = np.mean(negative_mean)
        self.intra_class = np.mean(intra_class)
        self.between_class = np.sum(self.center_product*(1-np.eye(self.class_centers.shape[0])))/(self.class_centers.shape[0]-1)/self.class_centers.shape[0]
        print('positive mean:{}, negative mean:{}, intra class:{}, between class:{}'.format(self.positive_mean, self.negative_mean, self.intra_class, self.between_class))

if __name__=='__main__':
    monitor = Mining_monitor()
    print(monitor.instance, monitor.get_batch())
