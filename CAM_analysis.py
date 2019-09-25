import os,sys
from Data_loader.CUB200 import MyData
import models
from utils.serialization import save_checkpoint, load_checkpoint
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
from skimage.feature import peak_local_max
import random
from matplotlib.patches import Rectangle
import time

class CAM_analyzer:
    def __init__(self):
        root = "/data/DML/CUB_200_2011/"
        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')
        self.CAM_1 = None
        self.CAM_2 = None
        self.size_1 = [224, 224]
        self.size_2 = [224, 224]
        self.data_loader = MyData(root=root, label_txt=train_txt, mode ='origin')
        with open(os.path.join(root, 'new/CUB_200_2011/bounding_boxes.txt')) as fp:
            box_file = fp.read().splitlines()
        with open(os.path.join(root, 'new/CUB_200_2011/images.txt')) as fp:
            self.name_list = fp.read().splitlines()
        self.bounding_box = {}
        for i,row in enumerate(box_file):
            row_str = np.array(row.split(' ')).astype(np.float)
            self.bounding_box[self.name_list[i].split(' ')[-1].split('/')[-1]] = row_str[1:]
        #print(self.bounding_box)

    def set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    def imshow_convert(self, cam):
        cam2 = (cam*-1.0) + 1.0
        cam_heatmap = np.array(cv2.applyColorMap(np.uint8(255*cam2), cv2.COLORMAP_JET))
        return cam_heatmap

    def get_input(self, index1, index2, data='cub', size = (224, 224), show = False):
        #data_loader = Data_loader.create(data)
        #data_loader.train.mode = 'center_crop'
        image_1, labels_1, fn_1 = self.data_loader.__getitem__(index1)
        image_2, labels_2, fn_2 = self.data_loader.__getitem__(index2)

        labels_1 = np.array(fn_1.split('/')[-2].split('.')[0]).astype(np.int)
        labels_2 = np.array(fn_2.split('/')[-2].split('.')[0]).astype(np.int)

        bbox_1 = self.bounding_box[fn_1.split('/')[-1]]
        bbox_2 = self.bounding_box[fn_2.split('/')[-1]]

        bbox_1 = bbox_1 * np.tile(self.size_1[::-1], [2]) / np.tile(image_1.shape[:2][::-1], [2])
        bbox_2 = bbox_2 * np.tile(self.size_2[::-1], [2]) / np.tile(image_2.shape[:2][::-1], [2])
        bbox_1[2] = bbox_1[0] + bbox_1[2]
        bbox_1[3] = bbox_1[1] + bbox_1[3]
        bbox_2[2] = bbox_2[0] + bbox_2[2]
        bbox_2[3] = bbox_2[1] + bbox_2[3]

        image_1 = cv2.resize(image_1, (self.size_1[1], self.size_1[0]), interpolation=cv2.INTER_LINEAR)
        image_2 = cv2.resize(image_2, (self.size_2[1], self.size_2[0]), interpolation=cv2.INTER_LINEAR)

        inputs_1 = np.transpose(image_1, (2, 0, 1))
        inputs_2 = np.transpose(image_2, (2, 0, 1))
        # wrap them in Variable
        inputs_1 = Variable(torch.from_numpy(np.expand_dims(inputs_1.astype(np.float32), axis=0))).cuda()
        inputs_2 = Variable(torch.from_numpy(np.expand_dims(inputs_2.astype(np.float32), axis=0))).cuda()


        if show:
            plt.figure()
            plt.imshow(image_1[:,:,::-1] / 255. + 0.5)
            ax = plt.gca()
            rect = Rectangle((bbox_1[0], bbox_1[1]), bbox_1[2] - bbox_1[0], bbox_1[3] - bbox_1[1], linewidth=1, edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)
            plt.title(str(labels_1))
            plt.figure()
            plt.imshow(image_2[:,:,::-1] / 255. + 0.5)
            ax = plt.gca()
            rect = Rectangle((bbox_2[0], bbox_2[1]), bbox_2[2] - bbox_2[0], bbox_2[3] - bbox_2[1], linewidth=1, edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)
            plt.title(str(labels_2))
        return inputs_1, image_1, labels_1, bbox_1, inputs_2, image_2, labels_2, bbox_2

    def get_embed(self, inputs_1, inputs_2, data):
        model_1 = models.create('BN-Inception', pretrained=True, dim=512)
        model_2 = models.create('BN-Inception', pretrained=True, dim=512)
        #resume = '/data/DML/model/Weight/' + data + '/BN-Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80/ckp_ep400.pth.tar'
        resume = '/data/DML/model/HardMining/' + data + '/BN-Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80/ckp_ep400.pth.tar'

        # resume model
        print('load model from {}'.format(resume))
        chk_pt = load_checkpoint(resume)
        weight = chk_pt['state_dict']
        start = chk_pt['epoch']
        model_1.load_state_dict(weight)
        model_2.load_state_dict(weight)

        # model = torch.nn.DataParallel(model)
        model_1 = model_1.cuda()
        model_2 = model_2.cuda()

        embed_1, map_1, ori_1 = model_1(inputs_1, True)
        embed_2, map_2, ori_2 = model_2(inputs_2, True)

        fc_1 = model_1.classifier.linear.cpu()
        fc_2 = model_2.classifier.linear.cpu()  # .weight.data.numpy()

        return embed_1, map_1, ori_1, fc_1, embed_2, map_2, ori_2, fc_2

    def generate_CAM(self, index1, index2, counter = False, data = 'car',top = 1, size = (224,224)):
        inputs_1, image_1, labels_1, bbox_1, inputs_2, image_2, labels_2, bbox_2 = self.get_input(index1 = index1, index2=index2, data=data, size=size)

        embed_1, map_1, ori_1, fc_1, embed_2, map_2, ori_2, fc_2 = self.get_embed(inputs_1=inputs_1, inputs_2=inputs_2, data=data)

        map_1.requires_grad_(True)
        map_2.requires_grad_(True)
        map_1.retain_grad()
        map_2.retain_grad()

        #product_vector = torch.mul(embed_1, embed_2)
        product_vector = torch.mul(ori_1, ori_2)
        product = torch.sum(product_vector)

        product_vector_max = torch.max(product_vector)
        sorted_product_vector, indices = torch.sort(product_vector, 0)
        sorted_product_vector = torch.squeeze(sorted_product_vector)
        print(sorted_product_vector.shape)
        if top > 0:
            sorted_product_vector[-top].backward(torch.tensor(1.).cuda())
        else:
            product.backward(torch.tensor(1.).cuda())
        print(product)

        image_1 = image_1[:, :, ::-1] / 255. + 0.5
        image_2 = image_2[:, :, ::-1] / 255. + 0.5

        # GradCAM
        cam_1 = self.GradCAM(map_1, counter=counter)
        cam_2 = self.GradCAM(map_2, counter=counter)

        image_cam_1 = image_1 * 0.7 + self.imshow_convert(cam_1) / 255.0 * 0.3
        image_cam_2 = image_2 * 0.7 + self.imshow_convert(cam_2) / 255.0 * 0.3

        plt.figure()
        plt.imshow(cam_1)
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(self.imshow_convert(cam_1))
        plt.title(str(labels_1))
        plt.subplot(2,2,2)
        plt.imshow(self.imshow_convert(cam_2))
        plt.title(str(labels_2))
        plt.subplot(2,2,3)
        plt.imshow(image_cam_1)
        plt.subplot(2,2,4)
        plt.imshow(image_cam_2)
        #plt.show()

        #EGradCAM
        cam_1 = self.EGradCAM(map_1, counter=counter)
        cam_2 = self.EGradCAM(map_2, counter=counter)

        image_cam_1 = image_1 * 0.7 + self.imshow_convert(cam_1) / 255.0 * 0.3
        image_cam_2 = image_2 * 0.7 + self.imshow_convert(cam_2) / 255.0 * 0.3

        plt.figure()
        plt.imshow(cam_1)
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(self.imshow_convert(cam_1))
        plt.title(str(labels_1))
        plt.subplot(2, 2, 2)
        plt.imshow(self.imshow_convert(cam_2))
        plt.title(str(labels_2))
        plt.subplot(2, 2, 3)
        plt.imshow(image_cam_1)
        plt.subplot(2, 2, 4)
        plt.imshow(image_cam_2)

        # MatchingCAM
        CAM = self.MatchingCAM(map_1 = map_1, map_2 = map_2, fc_1 = fc_1, fc_2 = fc_2, mode = 'GMP')

        cam_1, cam_2, points = self.QueryCAM(CAM, top=top, size=size)
        #points = [86, 143, 114, 130]
        #points = [133, 130, 121, 130]
        # points = [78, 143, 88, 128] # head

        #points = [133, 130, 121, 130] # middle

        image_cam_1 = image_1 * 0.7 + self.imshow_convert(cam_1) / 255.0 * 0.3
        image_cam_2 = image_2 * 0.7 + self.imshow_convert(cam_2) / 255.0 * 0.3

        plt.figure()
        plt.imshow(cam_1)
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(image_1)
        if points is not None:
            plt.plot(points[1], points[0], 'dr')
        plt.subplot(2, 3, 2)
        plt.imshow(self.imshow_convert(cam_1))
        plt.title(str(labels_1))
        plt.subplot(2, 3, 3)
        plt.imshow(image_cam_1)
        plt.subplot(2, 3, 4)
        plt.imshow(image_2)
        if points is not None:
            plt.plot(points[3], points[2], 'dr')
        plt.subplot(2, 3, 5)
        plt.imshow(self.imshow_convert(cam_2))
        plt.title(str(labels_2))
        plt.subplot(2, 3, 6)
        plt.imshow(image_cam_2)
        plt.show()

    def GradCAM(self, map, size = (224, 224), counter = False):
        gradient = map.grad.cpu().numpy()
        map = map.detach().cpu().numpy()
        #print(map_1.shape, map_2.shape, gradient_1.shape, gradient_2.shape)
        weights = np.mean(gradient[0], axis=(1, 2), keepdims=True)
        if counter:
            weights = -weights
        print(weights.shape, map.shape)
        # print deep_linearization_weights
        #grad_CAM_map = np.sum(weights * map[0], axis = 0)
        grad_CAM_map = np.sum(np.tile(weights, [1, map.shape[-2], map.shape[-1]]) * map[0], axis=0)
        print(grad_CAM_map.shape)
        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0

        # cam = resize(cam, size)
        cam = cv2.resize(cam, size)

        return cam

    def EGradCAM(self, map, size = (224, 224), counter = False):
        gradient = map.grad.cpu().numpy()
        map = map.detach().cpu().numpy()
        #print(map_1.shape, map_2.shape, gradient_1.shape, gradient_2.shape)
        weights = gradient[0]#
        if counter:
            weights = -weights
        print(weights.shape, map.shape)
        # print deep_linearization_weights
        grad_CAM_map = np.sum(weights * map[0], axis = 0)
        #grad_CAM_map = np.sum(np.tile(weights, [1, map.shape[-2], map.shape[-1]]) * map[0], axis=0)
        print(grad_CAM_map.shape)
        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0

        # cam = resize(cam, size)
        cam = cv2.resize(cam, size)

        return cam

    def MatchingCAM(self, map_1, map_2, fc_1 = None, fc_2 = None, size = (224, 224), counter = False, mode = 'GAP'):
        print('fc_1:', fc_1.weight.data.numpy().shape)
        if mode == 'GAP':
            return 0
        elif mode == 'GMP':
            map_1 = np.transpose(map_1.detach().cpu().numpy(),(0,2,3,1))
            map_2 = np.transpose(map_2.detach().cpu().numpy(),(0,2,3,1))
            for k in range(map_1.shape[-1]):
                map_1[:,:,:,k] = map_1[:,:,:,k] * (map_1[:,:,:,k] == np.max(map_1[:,:,:,k]))
            for k in range(map_2.shape[-1]):
                map_2[:,:,:,k] = map_2[:,:,:,k] * (map_2[:,:,:,k] == np.max(map_2[:,:,:,k]))
            map_1_reshape = np.reshape(map_1, [-1, map_1.shape[-1]])
            print('map1_reshape:',map_1_reshape.shape)
            map_2_reshape = np.reshape(map_2, [-1, map_2.shape[-1]])
            map_1_embed = np.matmul(map_1_reshape, np.transpose(fc_1.weight.data.numpy())) #+ fc_1.bias.data.numpy() / map_1_reshape.shape[0]
            map_2_embed = np.matmul(map_2_reshape, np.transpose(fc_2.weight.data.numpy())) #+ fc_2.bias.data.numpy() / map_2_reshape.shape[0]
            map_1_embed = np.reshape(map_1_embed, [map_1.shape[1], map_1.shape[2],-1])
            map_2_embed = np.reshape(map_2_embed, [map_2.shape[1], map_2.shape[2],-1])
            CAM = np.zeros([map_1.shape[1],map_1.shape[2],map_2.shape[1],map_2.shape[2]])
            for i in range(map_1.shape[1]):
                for j in range(map_1.shape[2]):
                    for x in range(map_2.shape[1]):
                        for y in range(map_2.shape[2]):
                            CAM[i,j,x,y] = np.sum(map_1_embed[i,j]*map_2_embed[x,y])
            CAM = CAM / np.max(CAM)
            CAM = np.maximum(CAM, 0)
            return CAM

    def QueryCAM(self, CAM, top = 0, size = (224,224), points = None):
        cam_1 = resize(np.sum(CAM, axis=(2, 3)), size, mode='edge')
        cam_1 = cam_1 / np.max(cam_1)
        cam_2 = resize(np.sum(CAM, axis=(0, 1)), size, mode='edge')
        cam_2 = cam_2 / np.max(cam_2)

        if top == 0:
            return cam_1, cam_2, points

        if points is None:
            points = np.zeros(4).astype(np.int)
            coordinates_1 = peak_local_max(cam_1, min_distance=20)
            index_1 = np.argsort(cam_1[coordinates_1[:, 0], coordinates_1[:, 1]])
            print(coordinates_1[index_1])
            coordinates_2 = peak_local_max(cam_2, min_distance=20)
            index_2 = np.argsort(cam_2[coordinates_2[:, 0], coordinates_2[:, 1]])
            print(coordinates_2[index_2])
            if top <= index_2.shape[0]:
                points[0] = coordinates_1[index_1[-top], 0]
                points[1] = coordinates_1[index_1[-top], 1]
                points[2] = coordinates_2[index_2[-top], 0]
                points[3] = coordinates_2[index_2[-top], 1]
                #points = [86, 143, 114, 130]
            else:
                points[0] = random.randint(0, self.size_1[0])
                points[1] = random.randint(0, self.size_1[1])
                points[2] = random.randint(0, self.size_2[0])
                points[3] = random.randint(0, self.size_2[1])

        cam_1 = resize(self.query_cam(cam=CAM, point=points[2:], view=2), self.size_1, mode='edge')
        cam_2 = resize(self.query_cam(cam=CAM, point=points[:2], view=1), self.size_2, mode='edge')
        cam_1 = cam_1 / np.max(cam_1)
        cam_2 = cam_2 / np.max(cam_2)

        return cam_1, cam_2, points

    def query_cam(self, cam, point = [0,0],view = 1, lib = True, same_image=False):
        shape = cam.shape
        if lib:
            if view == 1:
                if self.CAM_1 is None or not same_image:
                    self.CAM_1 = np.zeros([self.size_1[0],self.size_1[1],shape[2],shape[3]])
                    for i in range(shape[2]):
                        for j in range(shape[3]):
                            self.CAM_1[:,:,i,j] = resize(cam[:,:,i,j], self.size_1, mode = 'edge')
                return self.CAM_1[point[0],point[1]]
            elif view == 2:
                if self.CAM_2 is None or not same_image:
                    self.CAM_2 = np.zeros([shape[0],shape[1],self.size_2[0],self.size_2[1]])
                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            self.CAM_2[i,j,:,:] = resize(cam[i,j,:,:], self.size_2, mode = 'edge')
                return self.CAM_2[:,:,point[0],point[1]]
        else:
            if view == 0:
                pass
            elif view == 1:
                pass
            return 0

    def generate_bbox(self, x, show=False):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(x.astype('uint8'), connectivity=4)
        #print(nb_components, output, stats, centroids)
        num_back = np.sum(x == 0)
        if len(stats)>1 and stats[0][-1]==num_back:
            area = np.array(stats)[1:, -1]
            order = np.argsort(area)
            bbox = stats[1+order[-1]][0:4].copy()
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            '''
            bbox[0] = int(bbox[0] * self.image.shape[1] / self.size[0])
            bbox[2] = int(bbox[2] * self.image.shape[1] / self.size[0])
            bbox[1] = int(bbox[1] * self.image.shape[0] / self.size[1])
            bbox[3] = int(bbox[3] * self.image.shape[0] / self.size[1])
            '''
            #print(stats, area, order, bbox)
            if show:
                plt.figure()
                plt.imshow(x)
                plt.figure()
                plt.imshow(output)
                plt.figure()
                plt.imshow(self.image)
                for box in self.boxes:
                    box = np.array(box)
                    box = box[1:].astype(np.int)
                    ax = plt.gca()
                    rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                     facecolor='none')
                    ax.add_patch(rect)
                rect = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 linewidth=2, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
                plt.title(self.label)
            return bbox
        else:
            # 2013
            print('loc error!')
            print(nb_components, output, stats, centroids)
            return [0,0,self.size[0],self.size[1]]

    def localization(self, cam, threshold=0.2, box=None, show=False):
        bi_map = (cam / np.max(cam) > threshold) * 1.
        bbox = self.generate_bbox(bi_map, show=show)
        #IOU_list = []
        box = np.array(box)
        box = box.astype(np.int)
        #IOU_list.append()
        # print(IOU)
        return self.IOU(bbox, box), bbox

    def IOU(self, bbox1, bbox2):
        #print(bbox1, bbox2)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        if np.maximum(bbox1[0], bbox2[0]) < np.minimum(bbox1[2], bbox2[2]) and np.maximum(bbox1[1],
                                                                                          bbox2[1]) < np.minimum(
                bbox1[3], bbox2[3]):
            I = (np.minimum(bbox1[2], bbox2[2]) - np.maximum(bbox1[0], bbox2[0])) * (
                        np.minimum(bbox1[3], bbox2[3]) - np.maximum(bbox1[1], bbox2[1]))
            U = area1 + area2 - I
            return I * 1. / U
        else:
            return 0

    def generate_loc(self, index1=1, index2=2, data='cub', size = (224,224), show=False, threshold=0.2):
        inputs_1, image_1, labels_1, bbox_1, inputs_2, image_2, labels_2, bbox_2 = self.get_input(index1=index1,
                                                                                                  index2=index2,
                                                                                                  data=data, size=size)

        embed_1, map_1, ori_1, fc_1, embed_2, map_2, ori_2, fc_2 = self.get_embed(inputs_1=inputs_1, inputs_2=inputs_2,
                                                                                  data=data)
        loc_result_1 = []
        loc_result_2 = []

        map_1.requires_grad_(True)
        map_2.requires_grad_(True)
        map_1.retain_grad()
        map_2.retain_grad()

        #product_vector = torch.mul(embed_1, embed_2)
        product_vector = torch.mul(ori_1, ori_2)
        product = torch.sum(product_vector)

        product_vector_max = torch.max(product_vector)
        sorted_product_vector, indices = torch.sort(product_vector, 0)
        sorted_product_vector = torch.squeeze(sorted_product_vector)
        print(sorted_product_vector.shape)
        print(product)
        product.backward(torch.tensor(1.).cuda())

        image_1 = image_1[:, :, ::-1] / 255. + 0.5
        image_2 = image_2[:, :, ::-1] / 255. + 0.5

        # GradCAM
        cam_1 = self.GradCAM(map_1, counter=False)
        cam_2 = self.GradCAM(map_2, counter=False)
        IOU_1, box_1 = self.localization(cam=cam_1, threshold=threshold, box=bbox_1, show=False)
        IOU_2, box_2 = self.localization(cam=cam_2, threshold=threshold, box=bbox_2, show=False)
        loc_result_1.append(IOU_1)
        loc_result_2.append(IOU_2)
        if show:
            image_cam_1 = image_1 * 0.7 + self.imshow_convert(cam_1) / 255.0 * 0.3
            image_cam_2 = image_2 * 0.7 + self.imshow_convert(cam_2) / 255.0 * 0.3

            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(self.imshow_convert(cam_1))
            plt.title(str(labels_1))
            plt.subplot(2, 2, 2)
            plt.imshow(self.imshow_convert(cam_2))
            plt.title(str(labels_2))
            plt.subplot(2, 2, 3)
            plt.imshow(image_cam_1)
            ax = plt.gca()
            rect = Rectangle((bbox_1[0], bbox_1[1]), bbox_1[2] - bbox_1[0], bbox_1[3] - bbox_1[1], linewidth=1,
                             edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)
            ax = plt.gca()
            rect = Rectangle((box_1[0], box_1[1]), box_1[2] - box_1[0], box_1[3] - box_1[1], linewidth=2,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)
            plt.subplot(2, 2, 4)
            plt.imshow(image_cam_2)
            ax = plt.gca()
            rect = Rectangle((bbox_2[0], bbox_2[1]), bbox_2[2] - bbox_2[0], bbox_2[3] - bbox_2[1], linewidth=1,
                             edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)
            ax = plt.gca()
            rect = Rectangle((box_2[0], box_2[1]), box_2[2] - box_2[0], box_2[3] - box_2[1], linewidth=2,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)

            # plt.show()

        # EGradCAM
        cam_1 = self.EGradCAM(map_1, counter=False)
        cam_2 = self.EGradCAM(map_2, counter=False)
        IOU_1, box_1 = self.localization(cam=cam_1, threshold=threshold, box=bbox_1, show=False)
        IOU_2, box_2 = self.localization(cam=cam_2, threshold=threshold, box=bbox_2, show=False)
        loc_result_1.append(IOU_1)
        loc_result_2.append(IOU_2)

        if show:
            image_cam_1 = image_1 * 0.7 + self.imshow_convert(cam_1) / 255.0 * 0.3
            image_cam_2 = image_2 * 0.7 + self.imshow_convert(cam_2) / 255.0 * 0.3

            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(self.imshow_convert(cam_1))
            plt.title(str(labels_1))
            plt.subplot(2, 2, 2)
            plt.imshow(self.imshow_convert(cam_2))
            plt.title(str(labels_2))
            plt.subplot(2, 2, 3)
            plt.imshow(image_cam_1)
            ax = plt.gca()
            rect = Rectangle((bbox_1[0], bbox_1[1]), bbox_1[2] - bbox_1[0], bbox_1[3] - bbox_1[1], linewidth=1,
                             edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)
            ax = plt.gca()
            rect = Rectangle((box_1[0], box_1[1]), box_1[2] - box_1[0], box_1[3] - box_1[1], linewidth=2,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)
            plt.subplot(2, 2, 4)
            plt.imshow(image_cam_2)
            ax = plt.gca()
            rect = Rectangle((bbox_2[0], bbox_2[1]), bbox_2[2] - bbox_2[0], bbox_2[3] - bbox_2[1], linewidth=1,
                             edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)
            ax = plt.gca()
            rect = Rectangle((box_2[0], box_2[1]), box_2[2] - box_2[0], box_2[3] - box_2[1], linewidth=2,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)

        #############################33 MatchingCAM
        CAM = self.MatchingCAM(map_1=map_1, map_2=map_2, fc_1=fc_1, fc_2=fc_2, mode='GMP')

        cam_1, cam_2, points = self.QueryCAM(CAM, top=0, size=size)

        IOU_1, box_1 = self.localization(cam=cam_1, threshold=threshold, box=bbox_1, show=False)
        IOU_2, box_2 = self.localization(cam=cam_2, threshold=threshold, box=bbox_2, show=False)
        loc_result_1.append(IOU_1)
        loc_result_2.append(IOU_2)
        # points = [86, 143, 114, 130]
        # points = [133, 130, 121, 130]
        # points = [78, 143, 88, 128] # head

        # points = [133, 130, 121, 130] # middle
        if show:
            image_cam_1 = image_1 * 0.7 + self.imshow_convert(cam_1) / 255.0 * 0.3
            image_cam_2 = image_2 * 0.7 + self.imshow_convert(cam_2) / 255.0 * 0.3

            plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(image_1)
            if points is not None:
                plt.plot(points[1], points[0], 'dr')
            plt.subplot(2, 3, 2)
            plt.imshow(self.imshow_convert(cam_1))
            plt.title(str(labels_1))
            plt.subplot(2, 3, 3)
            plt.imshow(image_cam_1)
            ax = plt.gca()
            rect = Rectangle((bbox_1[0], bbox_1[1]), bbox_1[2] - bbox_1[0], bbox_1[3] - bbox_1[1], linewidth=1,
                             edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)
            ax = plt.gca()
            rect = Rectangle((box_1[0], box_1[1]), box_1[2] - box_1[0], box_1[3] - box_1[1], linewidth=2,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)
            plt.subplot(2, 3, 4)
            plt.imshow(image_2)
            if points is not None:
                plt.plot(points[3], points[2], 'dr')
            plt.subplot(2, 3, 5)
            plt.imshow(self.imshow_convert(cam_2))
            plt.title(str(labels_2))
            plt.subplot(2, 3, 6)
            plt.imshow(image_cam_2)
            ax = plt.gca()
            rect = Rectangle((bbox_2[0], bbox_2[1]), bbox_2[2] - bbox_2[0], bbox_2[3] - bbox_2[1], linewidth=1,
                             edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)
            ax = plt.gca()
            rect = Rectangle((box_2[0], box_2[1]), box_2[2] - box_2[0], box_2[3] - box_2[1], linewidth=2,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)

        ################### MatchingCAM top1
        CAM = self.MatchingCAM(map_1=map_1, map_2=map_2, fc_1=fc_1, fc_2=fc_2, mode='GMP')
        points = np.array([(bbox_1[0]+bbox_1[2])//2, (bbox_1[1]+bbox_1[3])//2, (bbox_2[0]+bbox_2[2])//2, (bbox_2[1]+bbox_2[3])//2]).astype(np.int)
        cam_1, cam_2, points = self.QueryCAM(CAM, top=1, size=size, points=points)
        IOU_1, box_1 = self.localization(cam=cam_1, threshold=threshold, box=bbox_1, show=False)
        IOU_2, box_2 = self.localization(cam=cam_2, threshold=threshold, box=bbox_2, show=False)
        loc_result_1.append(IOU_1)
        loc_result_2.append(IOU_2)

        if show:
            image_cam_1 = image_1 * 0.7 + self.imshow_convert(cam_1) / 255.0 * 0.3
            image_cam_2 = image_2 * 0.7 + self.imshow_convert(cam_2) / 255.0 * 0.3

            plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(image_1)
            if points is not None:
                plt.plot(points[1], points[0], 'dr')
            plt.subplot(2, 3, 2)
            plt.imshow(self.imshow_convert(cam_1))
            plt.title(str(labels_1))
            plt.subplot(2, 3, 3)
            plt.imshow(image_cam_1)
            ax = plt.gca()
            rect = Rectangle((bbox_1[0], bbox_1[1]), bbox_1[2] - bbox_1[0], bbox_1[3] - bbox_1[1], linewidth=1,
                             edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)
            ax = plt.gca()
            rect = Rectangle((box_1[0], box_1[1]), box_1[2] - box_1[0], box_1[3] - box_1[1], linewidth=2,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)
            plt.subplot(2, 3, 4)
            plt.imshow(image_2)
            if points is not None:
                plt.plot(points[3], points[2], 'dr')
            plt.subplot(2, 3, 5)
            plt.imshow(self.imshow_convert(cam_2))
            plt.title(str(labels_2))
            plt.subplot(2, 3, 6)
            plt.imshow(image_cam_2)
            ax = plt.gca()
            rect = Rectangle((bbox_2[0], bbox_2[1]), bbox_2[2] - bbox_2[0], bbox_2[3] - bbox_2[1], linewidth=1,
                             edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)
            ax = plt.gca()
            rect = Rectangle((box_2[0], box_2[1]), box_2[2] - box_2[0], box_2[3] - box_2[1], linewidth=2,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)

        ################ normalization
        embed_1, map_1, ori_1, fc_1, embed_2, map_2, ori_2, fc_2 = self.get_embed(inputs_1=inputs_1, inputs_2=inputs_2,
                                                                                  data=data)
        map_1.requires_grad_(True)
        map_2.requires_grad_(True)
        map_1.retain_grad()
        map_2.retain_grad()

        product_vector = torch.mul(embed_1, embed_2)
        #product_vector = torch.mul(ori_1, ori_2)
        product = torch.sum(product_vector)

        product_vector_max = torch.max(product_vector)
        sorted_product_vector, indices = torch.sort(product_vector, 0)
        sorted_product_vector = torch.squeeze(sorted_product_vector)
        print(sorted_product_vector.shape)
        print(product)
        product.backward(torch.tensor(1.).cuda())

        # GradCAM
        cam_1 = self.GradCAM(map_1, counter=False)
        cam_2 = self.GradCAM(map_2, counter=False)
        IOU_1, box_1 = self.localization(cam=cam_1, threshold=threshold, box=bbox_1, show=False)
        IOU_2, box_2 = self.localization(cam=cam_2, threshold=threshold, box=bbox_2, show=False)
        loc_result_1.append(IOU_1)
        loc_result_2.append(IOU_2)
        if show:
            image_cam_1 = image_1 * 0.7 + self.imshow_convert(cam_1) / 255.0 * 0.3
            image_cam_2 = image_2 * 0.7 + self.imshow_convert(cam_2) / 255.0 * 0.3

            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(self.imshow_convert(cam_1))
            plt.title(str(labels_1))
            plt.subplot(2, 2, 2)
            plt.imshow(self.imshow_convert(cam_2))
            plt.title(str(labels_2))
            plt.subplot(2, 2, 3)
            plt.imshow(image_cam_1)
            ax = plt.gca()
            rect = Rectangle((bbox_1[0], bbox_1[1]), bbox_1[2] - bbox_1[0], bbox_1[3] - bbox_1[1], linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
            ax.add_patch(rect)
            ax = plt.gca()
            rect = Rectangle((box_1[0], box_1[1]), box_1[2] - box_1[0], box_1[3] - box_1[1], linewidth=2,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)
            plt.subplot(2, 2, 4)
            plt.imshow(image_cam_2)
            ax = plt.gca()
            rect = Rectangle((bbox_2[0], bbox_2[1]), bbox_2[2] - bbox_2[0], bbox_2[3] - bbox_2[1], linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
            ax.add_patch(rect)
            ax = plt.gca()
            rect = Rectangle((box_2[0], box_2[1]), box_2[2] - box_2[0], box_2[3] - box_2[1], linewidth=2,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)
            # plt.show()

        # EGradCAM
        cam_1 = self.EGradCAM(map_1, counter=False)
        cam_2 = self.EGradCAM(map_2, counter=False)
        IOU_1, box_1 = self.localization(cam=cam_1, threshold=threshold, box=bbox_1, show=False)
        IOU_2, box_2 = self.localization(cam=cam_2, threshold=threshold, box=bbox_2, show=False)
        loc_result_1.append(IOU_1)
        loc_result_2.append(IOU_2)

        if show:
            image_cam_1 = image_1 * 0.7 + self.imshow_convert(cam_1) / 255.0 * 0.3
            image_cam_2 = image_2 * 0.7 + self.imshow_convert(cam_2) / 255.0 * 0.3

            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(self.imshow_convert(cam_1))
            plt.title(str(labels_1))
            plt.subplot(2, 2, 2)
            plt.imshow(self.imshow_convert(cam_2))
            plt.title(str(labels_2))
            plt.subplot(2, 2, 3)
            plt.imshow(image_cam_1)
            ax = plt.gca()
            rect = Rectangle((bbox_1[0], bbox_1[1]), bbox_1[2] - bbox_1[0], bbox_1[3] - bbox_1[1], linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
            ax.add_patch(rect)
            ax = plt.gca()
            rect = Rectangle((box_1[0], box_1[1]), box_1[2] - box_1[0], box_1[3] - box_1[1], linewidth=2,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)
            plt.subplot(2, 2, 4)
            plt.imshow(image_cam_2)
            ax = plt.gca()
            rect = Rectangle((bbox_2[0], bbox_2[1]), bbox_2[2] - bbox_2[0], bbox_2[3] - bbox_2[1], linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
            ax.add_patch(rect)
            ax = plt.gca()
            rect = Rectangle((box_2[0], box_2[1]), box_2[2] - box_2[0], box_2[3] - box_2[1], linewidth=2,
                             edgecolor='g',
                             facecolor='none')
            ax.add_patch(rect)
            plt.show()
        print('localization:',loc_result_1, loc_result_2)
        return loc_result_1, loc_result_2


def test_cam():
    analyzer = CAM_analyzer()
    #analyzer.get_input(index1=1, index2=2, show=True)
    #analyzer.generate_CAM(1, 2, counter=False, data='car')
    analyzer.generate_CAM(30, 31, counter=False,top=0, data='cub')
    '''
    
    result = []
    th = 0.6
    for key in analyzer.data_loader.Index:
        print('key:', key)
        for i in analyzer.data_loader.Index[key]:
            j = random.choice(analyzer.data_loader.Index[key])
            while j == i:
                j = random.choice(analyzer.data_loader.Index[key])
            t1 = time.time()
            loc_1, loc_2 = analyzer.generate_loc(index1=i, index2=j, threshold=th, show=False)
            result.append(loc_1.copy())
            result.append(loc_2.copy())
            t2 = time.time()
            print('time:', t2 - t1)
    result = np.array(result)
    #np.save('Lifted_loc_positive_'+str(th)+'.npy', result)
    #result = np.load('Binomial_loc_positive_0.2.npy')
    threshold = 0.5
    GradCAM_count = np.sum((result[:, 0] > threshold) * 1.) / result.shape[0]
    EGradCAM_count = np.sum((result[:, 1] > threshold) * 1.) / result.shape[0]
    ECAM_count = np.sum((result[:, 2] > threshold) * 1.) / result.shape[0]
    ECAM_top_count = np.sum((result[:, 3] > threshold) * 1.) / result.shape[0]
    GradCAM_norm_count = np.sum((result[:, 4] > threshold) * 1.) / result.shape[0]
    EGradCAM_norm_count = np.sum((result[:, 5] > threshold) * 1.) / result.shape[0]
    print(result.shape)
    print('Grad_CAM:{}, EGradCAM:{}, ECAM:{}, ECAM_TOP:{}, GradCAM_norm:{}, EGradCAM_norm:{}'.format(GradCAM_count,
                                EGradCAM_count, ECAM_count, ECAM_top_count, GradCAM_norm_count, EGradCAM_norm_count))
    '''

if __name__=='__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    test_cam()
