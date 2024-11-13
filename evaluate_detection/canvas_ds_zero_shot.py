import torch.utils.data as data
import sys
import os
# print(sys.path)
import argparse

relative_path = './evaluate_detection'
abs_path = os.path.abspath(relative_path)

# 将绝对路径添加到sys.path中
sys.path.append(abs_path)

# print("?")

# print(sys.path)
from .voc_orig import VOCDetection4Val, VOCDetection4Train, make_transforms
import cv2
from PIL import Image
from .voc import make_transforms, create_grid_from_images
import torch
import numpy as np
import torchvision.transforms as T
import json
import torchvision.transforms.functional as TTTT
import h5py


def box_to_img(mask, target, border_width=4):
    if mask is None:
        mask = np.zeros((112, 112, 3))
    h, w, _ = mask.shape
    for box in target['boxes']:
        x_min, y_min, x_max, y_max = list((box * (h - 1)).round().int().numpy())
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), border_width)
    return Image.fromarray(mask.astype('uint8'))


def get_annotated_image(img, boxes, border_width=3, mode='draw', bgcolor='white', fg='image'):
    if mode == 'draw':
        image_copy = np.array(img.copy())
        for box in boxes:
            box = box.numpy().astype('int')
            cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), border_width)
    elif mode == 'keep':
        image_copy = np.array(Image.new('RGB', (img.shape[1], img.shape[0]), color=bgcolor))

        for box in boxes:
            box = box.numpy().astype('int')
            if fg == 'image':
                image_copy[box[1]:box[3], box[0]:box[2]] = img[box[1]:box[3], box[0]:box[2]]
            elif fg == 'white':
                image_copy[box[1]:box[3], box[0]:box[2]] = 255

    return image_copy


class CanvasDataset4Val(data.Dataset):
    def __init__(self, pascal_path='pascal-5i',args = None, years=("2012",),simidx = 1, random=False, **kwargs):
        self.train_ds = VOCDetection4Val(pascal_path, years, image_sets=['train'], transforms=None,
                                         keep_single_objs_only=1, filter_by_mask_size=1)
        self.val_ds = VOCDetection4Val(pascal_path, years, image_sets=['val'], transforms=None,
                                       keep_single_objs_only=1, filter_by_mask_size=1)
        self.pascal_pat = pascal_path
        self.background_transforms = T.Compose([
            T.Resize((224, 224)),
            T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ])
        self.transforms = make_transforms('val')
        self.random = random
        self.simidx = simidx
        self.images_top50 = self.get_top50_images()
        self.img_feature_for_train_path = './VOC2012/features_vit-laion2b_pixel-level_query_all_detection/detection_eval_query_zero_shot.h5df'
        self.img_feature_for_train_path = os.path.join(pascal_path, self.img_feature_for_train_path)
        self.support_feature_for_train_path = './VOC2012/features_vit-laion2b_pixel-level_support_all_detection/detection_eval_support_zero_shot.h5df'
        self.support_feature_for_train_path = os.path.join(pascal_path, self.support_feature_for_train_path)
        self.args = args
        self.cache = {}

    def get_top50_images(self):
        with open(f'{self.pascal_pat}/VOC2012/features_vit-laion2b_pixel-level_val_all_detection/new_top_50-similarity.json') as f:
            images_top50 = json.load(f)
        return images_top50
    
    def __len__(self):
        return len(self.val_ds)
    
    def __getitem__(self, idx):
        if idx in self.cache and self.cache[idx]['valid']:
            # print("Cache hit for index:", idx)
            return self.cache[idx]['batch']
        # else:
        #     print("Cache miss for index:", idx)
        grids = torch.tensor([]) 
        support_imgs = torch.tensor([]) 
        support_masks = torch.tensor([]) 
        query_img_features = torch.tensor([]) 
        support_features = torch.tensor([]) 
        for sim_idx in range(self.simidx):
            if sim_idx == 0:
                query_image, query_target = self.val_ds[idx]

                query_image_name = self.val_ds.images[idx].split('/')[-1][:-4]
                # print(self.val_ds.nameToNum[query_image_name])
                # print(query_image_name)
                # print(query_image_name)
                label = query_target['labels'].numpy()[0]
                # print(self.images_top50[query_image_name][sim_idx].split(' '))
                support_image_name = self.images_top50[query_image_name][sim_idx]
            

                support_image, support_target = self.train_ds[self.train_ds.nameToNum[support_image_name]]
                support_label = support_target['labels'].numpy()[0]
                # print(query_image_name,support_image_name)
                boxes = support_target['boxes'][torch.where(support_target['labels'] == support_label)[0]]
                support_image_copy = get_annotated_image(np.array(support_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                support_image_copy_pil = Image.fromarray(support_image_copy)

                boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
                query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                query_image_copy_pil = Image.fromarray(query_image_copy)

                query_image_ten = self.transforms(query_image, None)[0]
                query_target_ten = self.transforms(query_image_copy_pil, None)[0]
                support_target_ten = self.transforms(support_image_copy_pil, None)[0]
                support_image_ten = self.transforms(support_image, None)[0]

                background_image = Image.new('RGB', (224, 224), color='white')
                background_image = self.background_transforms(background_image)
                grid = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                                query_target_ten)
                query_feature, support_feature = self.load_feature(query_image_name,support_image_name)

                query_img_features = torch.tensor(query_feature).unsqueeze(0)
                support_features = torch.tensor(support_feature).unsqueeze(0)
                support_imgs = support_image_ten.unsqueeze(0)
                support_masks = support_target_ten.unsqueeze(0)
                grids = grid.unsqueeze(0)

            else:
                support_image_name = self.images_top50[query_image_name][sim_idx]

                support_image, support_target = self.train_ds[self.train_ds.nameToNum[support_image_name]]
                support_label = support_target['labels'].numpy()[0]
                # print(query_image_name,support_image_name)
                boxes = support_target['boxes'][torch.where(support_target['labels'] == support_label)[0]]
                support_image_copy = get_annotated_image(np.array(support_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                support_image_copy_pil = Image.fromarray(support_image_copy)

                boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
                query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                query_image_copy_pil = Image.fromarray(query_image_copy)

                query_image_ten = self.transforms(query_image, None)[0]
                query_target_ten = self.transforms(query_image_copy_pil, None)[0]
                support_target_ten = self.transforms(support_image_copy_pil, None)[0]
                support_image_ten = self.transforms(support_image, None)[0]

                background_image = Image.new('RGB', (224, 224), color='white')
                background_image = self.background_transforms(background_image)
                grid = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                                query_target_ten)
                support_img = support_image_ten.unsqueeze(0)
                support_mask = support_target_ten.unsqueeze(0)
                grid = grid.unsqueeze(0)
                _,  support_feature = self.load_feature(query_image_name,support_image_name)
                support_feature = torch.tensor(support_feature).unsqueeze(0)
                support_features = torch.cat((support_features,support_feature))
                support_imgs = torch.cat((support_imgs,support_img))
                support_masks = torch.cat((support_masks,support_mask))
                grids = torch.cat((grids,grid))

        # trans = T.ToTensor()
        # image = TTTT.to_pil_image(trans(query_image))

        #     # # 保存图像
        # image.save("query_image.jpg")

        # image = TTTT.to_pil_image(trans(support_image))

        #     # # 保存图像
        # image.save("support_image.jpg")
        # image = TTTT.to_pil_image(grid_stack)

        #     # # 保存图像
        # image.save("result.jpg")
        batch = {'query_img': query_image_ten,
            'query_mask': query_target_ten,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'grids': grids,
        ##my code
            'query_img_features': query_img_features,
            'support_features': support_features
        ##end my code
        }
        self.cache[idx] = {'valid': True, 'batch': batch}

        return batch
    def load_feature(self,query_name, support_name):
        with h5py.File(self.img_feature_for_train_path, "r") as f:
            query_img_feature = f[query_name][...]
        with h5py.File(self.support_feature_for_train_path, "r") as f:
            support_feature = f[support_name][...]
        #print("debug        ")
        #print(support_img_feature-support_mask_feature)
        return query_img_feature,support_feature

class CanvasDataset4Train(data.Dataset):
    def __init__(self, pascal_path='pascal-5i', args = None,years=("2012",), simidx = 1, random=False, **kwargs):
        self.train_ds = VOCDetection4Train(pascal_path, years, image_sets=['train'], transforms=None, keep_single_objs_only=1, filter_by_mask_size=1)
        self.val_ds = VOCDetection4Train(pascal_path, years, image_sets=['val'], transforms=None, keep_single_objs_only=1, filter_by_mask_size=1)
        self.pascal_pat = pascal_path
        self.background_transforms = T.Compose([
            T.Resize((224, 224)),
            T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ])
        self.transforms = make_transforms('val')
        self.random = random
        self.simidx = simidx
        self.images_top50 = self.get_top50_images()
        self.img_feature_for_train_path = './VOC2012/features_vit-laion2b_pixel-level_query_all_detection/detection_train_query_zero_shot.h5df'
        self.img_feature_for_train_path = os.path.join(pascal_path, self.img_feature_for_train_path)
        self.support_feature_for_train_path = './VOC2012/features_vit-laion2b_pixel-level_support_all_detection/detection_train_support_zero_shot.h5df'
        self.support_feature_for_train_path = os.path.join(pascal_path, self.support_feature_for_train_path)
        self.args = args
        self.cache = {}
    def get_top50_images(self):
        with open(f'{self.pascal_pat}/VOC2012/features_vit-laion2b_pixel-level_train_all_detection/new_top_50-similarity.json') as f:
            images_top50 = json.load(f)
        return images_top50
    
    def __len__(self):
        return len(self.val_ds)
    
    def load_feature(self,query_name, support_name):
        with h5py.File(self.img_feature_for_train_path, "r") as f:
            query_img_feature = f[query_name][...]
        with h5py.File(self.support_feature_for_train_path, "r") as f:
            support_feature = f[support_name][...]
        #print("debug        ")
        #print(support_img_feature-support_mask_feature)
        return query_img_feature,support_feature
    
    def __getitem__(self, idx):
        if idx in self.cache and self.cache[idx]['valid']:
            # print("Cache hit for index:", idx)
            return self.cache[idx]['batch']
        # else:
        #     print("Cache miss for index:", idx)
        grids = torch.tensor([]) 
        support_imgs = torch.tensor([]) 
        support_masks = torch.tensor([]) 
        query_img_features = torch.tensor([]) 
        support_features = torch.tensor([]) 
        for sim_idx in range(self.simidx):
            if sim_idx == 0:
                query_image, query_target = self.val_ds[idx]

                query_image_name = self.val_ds.images[idx].split('/')[-1][:-4]
                # print(self.val_ds.nameToNum[query_image_name])
                # print(query_image_name)
                label = query_target['labels'].numpy()[0]
                # print(self.images_top50[query_image_name][sim_idx].split(' '))
                support_image_name = self.images_top50[query_image_name][sim_idx]
            

                support_image, support_target = self.train_ds[self.train_ds.nameToNum[support_image_name]]
                support_label = support_target['labels'].numpy()[0]
                # print(query_image_name,support_image_name)
                boxes = support_target['boxes'][torch.where(support_target['labels'] == support_label)[0]]
                support_image_copy = get_annotated_image(np.array(support_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                support_image_copy_pil = Image.fromarray(support_image_copy)

                boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
                query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                query_image_copy_pil = Image.fromarray(query_image_copy)

                query_image_ten = self.transforms(query_image, None)[0]
                query_target_ten = self.transforms(query_image_copy_pil, None)[0]
                support_target_ten = self.transforms(support_image_copy_pil, None)[0]
                support_image_ten = self.transforms(support_image, None)[0]

                background_image = Image.new('RGB', (224, 224), color='white')
                background_image = self.background_transforms(background_image)
                grid = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                                query_target_ten)
                query_feature, support_feature = self.load_feature(query_image_name,support_image_name)
                # print("query_feature   ",query_feature.shape)
                query_img_features = torch.tensor(query_feature).unsqueeze(0)
                support_features = torch.tensor(support_feature).unsqueeze(0)
                support_imgs = support_image_ten.unsqueeze(0)
                support_masks = support_target_ten.unsqueeze(0)
                grids = grid.unsqueeze(0)

            else:
                support_image_name = self.images_top50[query_image_name][sim_idx]

                support_image, support_target = self.train_ds[self.train_ds.nameToNum[support_image_name]]
                support_label = support_target['labels'].numpy()[0]
                # print(query_image_name,support_image_name)
                boxes = support_target['boxes'][torch.where(support_target['labels'] == support_label)[0]]
                support_image_copy = get_annotated_image(np.array(support_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                support_image_copy_pil = Image.fromarray(support_image_copy)

                boxes = query_target['boxes'][torch.where(query_target['labels'] == label)[0]]
                query_image_copy = get_annotated_image(np.array(query_image), boxes, border_width=-1, mode='keep', bgcolor='black', fg='white')
                query_image_copy_pil = Image.fromarray(query_image_copy)

                query_image_ten = self.transforms(query_image, None)[0]
                query_target_ten = self.transforms(query_image_copy_pil, None)[0]
                support_target_ten = self.transforms(support_image_copy_pil, None)[0]
                support_image_ten = self.transforms(support_image, None)[0]

                background_image = Image.new('RGB', (224, 224), color='white')
                background_image = self.background_transforms(background_image)
                grid = create_grid_from_images(background_image, support_image_ten, support_target_ten, query_image_ten,
                                                query_target_ten)
                support_img = support_image_ten.unsqueeze(0)
                support_mask = support_target_ten.unsqueeze(0)
                grid = grid.unsqueeze(0)
                _,  support_feature = self.load_feature(query_image_name,support_image_name)
                support_feature = torch.tensor(support_feature).unsqueeze(0)
                support_features = torch.cat((support_features,support_feature))
                support_imgs = torch.cat((support_imgs,support_img))
                support_masks = torch.cat((support_masks,support_mask))
                grids = torch.cat((grids,grid))

        # trans = T.ToTensor()
        # image = TTTT.to_pil_image(trans(query_image))

        #     # # 保存图像
        # image.save("query_image.jpg")

        # image = TTTT.to_pil_image(trans(support_image))

        #     # # 保存图像
        # image.save("support_image.jpg")
        # image = TTTT.to_pil_image(grids[0])

        #     # # 保存图像
        # image.save("result.jpg")
        batch = {'query_img': query_image_ten,
            'query_mask': query_target_ten,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'grids': grids,
        ##my code
            'query_img_features': query_img_features,
            'support_features': support_features
        ##end my code
        }
        # print("hit?",1 if idx in self.cache else 0)
        self.cache[idx] = {'valid': True, 'batch': batch}
        # print("idx",idx)
        # print("hit?",1 if idx in self.cache else 0)
        return batch

# def get_args():
#     parser = argparse.ArgumentParser('InMeMo training for detection', add_help=False)

#     return parser

# if __name__ == "__main__":
#     # model = prepare_model('/shared/amir/Deployment/arxiv_mae/logs_dir/pretrain_small_arxiv2/checkpoint-799.pth',
#     #                       arch='mae_vit_small_patch16')
#     args = get_args()

#     args = args.parse_args()
#     args.device = 'cpu'
#     canvas_ds = CanvasDataset4Val('VisualICL/pascal-5i',args=args)

#     canvas = canvas_ds[0]
#     canvas = canvas_ds[0]