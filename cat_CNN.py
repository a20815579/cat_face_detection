import os
import torch
import torchvision
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

import cv2

import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim
from torchsummary import summary

from math import atan2, degrees

class CatPicture():

    def __init__(self, filename):

        self.which_p = 1
        self.which_d = 1
        self.wear_list = [0, 0, 0]

        self.ori_filename = filename

        tmp_idx = filename.rfind(".")
        self.short_filename = filename[:tmp_idx]

        def resize_img(im):
            old_size = im.shape[:2]  # old_size is in (height, width) format
            ratio = float(img_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            # new_size should be in (width, height) format
            im = cv2.resize(im, (new_size[1], new_size[0]))
            delta_w = img_size - new_size[1]
            delta_h = img_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=[0, 0, 0])
            return new_im, ratio, top, left

        class CatDataset(Dataset):

            def __init__(self, images, labels, transform):

                self.imgs = images
                self.labels = labels
                self.transform = transform

            def __len__(self):

                return len(self.imgs)  # return DataSet 長度

            def __getitem__(self, idx):

                image = self.imgs[idx]
                image = image[..., ::-1].copy()
                image = self.transform(image)
                label = np.array(self.labels[idx])

                return image, label  # return 模型訓練所需的資訊

        def Catface_dataloader(img):

            test_inputs = []
            test_inputs.append(img)
            test_labels = [0 for i in range(4)]

            test_dataloader = DataLoader(CatDataset(test_inputs, test_labels, test_transformer),
                                         batch_size=1, shuffle=False)
            return test_dataloader

        def Lmks_dataloader(img):

            test_inputs = []
            test_inputs.append(img)
            test_labels = [0 for i in range(18)]
            test_dataloader = DataLoader(CatDataset(test_inputs, test_labels, test_transformer),
                                         batch_size=1, shuffle=False)
            return test_dataloader

        class CatFaceModule(nn.Module):

            def __init__(self):
                super(CatFaceModule, self).__init__()
                v = torch.hub.load('pytorch/vision:v0.6.0',
                                   'mobilenet_v2', pretrained=True)
                v.classifier[1] = nn.Linear(v.last_channel, 4)
                self.layer1 = v

            def forward(self, x):
                out = self.layer1(x)
                return out

        class LmksModule(nn.Module):
            def __init__(self):
                super(LmksModule, self).__init__()
                v = torch.hub.load('pytorch/vision:v0.6.0',
                                   'mobilenet_v2', pretrained=True)
                v.classifier[1] = nn.Linear(v.last_channel, 18)
                self.layer1 = v

            def forward(self, x):
                out = self.layer1(x)
                return out

        # main program

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_size = 224
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        # the path of models
        cat_face_model_path = "mobilenet_RMSELoss500.ph"
        lmks_model_path = "mobilenet_RMSELoss100_36.ph"

        # load the models
        cat_model = CatFaceModule().to(device)
        cat_model.load_state_dict(torch.load(cat_face_model_path))
        cat_model.eval()

        lmks_model = LmksModule().to(device)
        lmks_model.load_state_dict(torch.load(lmks_model_path))
        lmks_model.eval()

        # the path of the image you want to test
        img_path = filename
        # read the image
        img = cv2.imread(img_path)
        ori_img = img.copy()
        result_img = img.copy()
        img, ratio, top, left = resize_img(img)
        # plt.figure()
        # plt.imshow(img)
        predicted = []

        # catface predicted
        catface_dataloader = Catface_dataloader(img)
        for i, (x, label) in enumerate(catface_dataloader):
            with torch.no_grad():
                x, label = x.to(device), label.to(device)
                output = cat_model(x)
                # loss = criterion(output, label.long())
                predicted = output.data[0].reshape((-1, 2))

        # the position of the cat face box
        pre_bb = predicted.cpu().numpy()
        # print(pre_bb)

        # the positoin of the cat face box when it at the origin image
        ori_bb = ((pre_bb - np.array([left, top])) / ratio).astype(np.int)
        # print(ori_bb)

        # cut the face image
        center = np.mean(ori_bb, axis=0)
        face_size = max(np.abs(ori_bb[1] - ori_bb[0]))
        new_bb = np.array([
            center - face_size * 0.6,
            center + face_size * 0.6
        ]).astype(np.int)
        new_bb = np.clip(new_bb, 0, 99999)
        face_img = ori_img[new_bb[0][1]:new_bb[1]
                           [1], new_bb[0][0]:new_bb[1][0]]
        # plt.figure()
        # plt.imshow(face_img)

        face_img, face_ratio, face_top, face_left = resize_img(face_img)

        # landmark prediction
        lmks_dataloader = Lmks_dataloader(face_img)
        for i, (x, label) in enumerate(lmks_dataloader):
            with torch.no_grad():
                x, label = x.to(device), label.to(device)
                output = lmks_model(x)
                # loss = criterion(output, label.long()) # 計算測試資料的準確度
                predicted = output.data[0].reshape((-1, 2))

        pred_lmks = predicted.cpu().numpy()
        # print(pred_lmks)

        new_lmks = (
            (pred_lmks - np.array([face_left, face_top])) / face_ratio).astype(np.int)
        self.ori_lmks = new_lmks + new_bb[0]

        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        # initial cat
        self.ori_pic = result_img

        tmp_filename = self.short_filename + "_000.jpg"
        cv2.imwrite(tmp_filename, result_img)

        # main end

    def angle_between(self, p1, p2):
        xDiff = p2[0] - p1[0]
        yDiff = p2[1] - p1[1]
        return degrees(atan2(yDiff, xDiff))

    def overlay_transparent(self, background_img, img_to_overlay_t, x, y, overlay_size=None):
        bg_img = background_img.copy()
        # convert 3 channels to 4 channels
        if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(
                img_to_overlay_t.copy(), overlay_size)

        b, g, r, a = cv2.split(img_to_overlay_t)

        for i in range(len(a)):
            for j in range(len(a[0])):
                if a[i][j] < 200:
                    a[i][j] = 0

        #mask = cv2.medianBlur(a, 5)
        mask = a

        h, w, _ = img_to_overlay_t.shape
        roi = bg_img[int(y - h / 2):int(y + h / 2),
                     int(x - w / 2):int(x + w / 2)]

        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(),
                                  mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(
            img_to_overlay_t, img_to_overlay_t, mask=mask)

        bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2)               :int(x + w / 2)] = cv2.add(img1_bg, img2_fg)

        # convert 4 channels to 4 channels
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

        return bg_img

    def setWearList(self, which_pattern):
        self.wear_list[self.which_d] = which_pattern

    def CreateBasePic(self):
        tmp_list = self.wear_list[:]
        tmp_list[self.which_d] = 0
        base_filepath = self.short_filename + "_" + \
            str(tmp_list[0]) + str(tmp_list[1]) + str(tmp_list[2]) + ".jpg"
        if os.path.isfile(base_filepath):
            return
        self.WearHat(tmp_list[0], 0, 0)
        self.WearBow(tmp_list[0], tmp_list[1], 0)
        self.WearGlasses(tmp_list[0], tmp_list[1], tmp_list[2])

    def WearDecorate(self):
        self.CreateBasePic()
        if self.which_d == 0:
            self.WearHat(self.wear_list[0],
                         self.wear_list[1], self.wear_list[2])
        elif self.which_d == 1:
            self.WearBow(self.wear_list[0],
                         self.wear_list[1], self.wear_list[2])
        else:
            self.WearGlasses(self.wear_list[0],
                             self.wear_list[1], self.wear_list[2])
        new_name = self.short_filename + "_" + \
            str(self.wear_list[0]) + str(self.wear_list[1]
                                         ) + str(self.wear_list[2]) + ".jpg"
        return new_name

    def WearHat(self, h_n, b_n, g_n):
        if h_n == 0:
            return
        # add hat
        hat_name = "hat" + str(h_n) + ".png"
        hat = cv2.imread(hat_name, cv2.IMREAD_UNCHANGED)
        hat_center = np.mean([self.ori_lmks[5], self.ori_lmks[6]], axis=0)
        hat_size = np.linalg.norm(self.ori_lmks[5] - self.ori_lmks[6]) * 3
        angle = -self.angle_between(self.ori_lmks[5], self.ori_lmks[6])
        M = cv2.getRotationMatrix2D(
            (hat.shape[1] / 2, hat.shape[0] / 2), angle, 1)
        rotated_hat = cv2.warpAffine(hat, M, (hat.shape[1], hat.shape[0]))

        base_name = self.short_filename + "_" + \
            "0" + str(b_n) + str(g_n) + ".jpg"
        new_name = self.short_filename + "_" + \
            str(h_n) + str(b_n) + str(g_n) + ".jpg"
        base_pic = cv2.imread(base_name)

        try:
            cat = self.overlay_transparent(base_pic, rotated_hat, hat_center[0], hat_center[1], overlay_size=(
                int(hat_size), int(hat.shape[0] * hat_size / hat.shape[1])))
        except:
            print('failed overlay image')

        cv2.imwrite(new_name, cat)

    def WearBow(self, h_n, b_n, g_n):
        if b_n == 0:
            return
        # add bow
        bow_name = "bow" + str(b_n) + ".png"
        bow = cv2.imread(bow_name, cv2.IMREAD_UNCHANGED)
        bow_center = np.mean([self.ori_lmks[3], self.ori_lmks[5]], axis=0)
        bow_size = np.linalg.norm(self.ori_lmks[3] - self.ori_lmks[5]) * 1.5
        angle = -self.angle_between(self.ori_lmks[3], self.ori_lmks[5])
        M = cv2.getRotationMatrix2D(
            (bow.shape[1] / 2, bow.shape[0] / 2), angle, 1)
        rotated_bow = cv2.warpAffine(bow, M, (bow.shape[1], bow.shape[0]))

        base_name = self.short_filename + "_" + \
            str(h_n) + "0" + str(g_n) + ".jpg"
        new_name = self.short_filename + "_" + \
            str(h_n) + str(b_n) + str(g_n) + ".jpg"
        base_pic = cv2.imread(base_name)

        cat = self.overlay_transparent(base_pic, rotated_bow, bow_center[0], bow_center[1], overlay_size=(
            int(bow_size), int(bow.shape[0] * bow_size / bow.shape[1])))

        cv2.imwrite(new_name, cat)

    def WearGlasses(self, h_n, b_n, g_n):
        # add glasses
        if g_n == 0:
            return
        glasses_name = "glasses" + str(g_n) + ".png"
        glasses = cv2.imread(glasses_name, cv2.IMREAD_UNCHANGED)
        glasses_center = np.mean([self.ori_lmks[0], self.ori_lmks[1]], axis=0)
        glasses_size = np.linalg.norm(
            self.ori_lmks[0] - self.ori_lmks[1]) * 2.5
        angle = -self.angle_between(self.ori_lmks[0], self.ori_lmks[1])
        M = cv2.getRotationMatrix2D(
            (glasses.shape[1] / 2, glasses.shape[0] / 2), angle, 1)
        rotated_glasses = cv2.warpAffine(
            glasses, M, (glasses.shape[1], glasses.shape[0]))

        base_name = self.short_filename + "_" + \
            str(h_n) + str(b_n) + "0" + ".jpg"
        new_name = self.short_filename + "_" + \
            str(h_n) + str(b_n) + str(g_n) + ".jpg"
        base_pic = cv2.imread(base_name)

        cat = self.overlay_transparent(base_pic, rotated_glasses, glasses_center[0], glasses_center[1], overlay_size=(
            int(glasses_size), int(glasses.shape[0] * glasses_size / glasses.shape[1])))

        cv2.imwrite(new_name, cat)
