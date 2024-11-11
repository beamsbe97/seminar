"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
import sys
current_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# print(os.path.dirname(current_path))
if current_path not in sys.path:
    sys.path.append(current_path)

print(sys.path)
import numpy as np
import torch
from torch.utils.data import Dataset
from evaluate.mae_utils import PURPLE, YELLOW
import json
import h5py

mapped_dict = {
    '0': {'05': '05', '02': '02', '16': '15', '09': '09', '44': '40'},
    '1': {'06': '06', '03': '03', '17': '16', '62': '57', '21': '20'},
    '2': {'67': '61', '18': '17', '19': '18', '04': '04', '01': '01'},
    '3': {'64': '59', '20': '19', '63': '58', '07': '07', '72': '63'}
}
class DatasetMSCOCO(Dataset):
    def __init__(self, datapath, args, fold, split, image_transform, mask_transform, padding: bool = 1,
                 use_original_imgsize: bool = False, flipped_order: bool = False,have_tokens: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False, ensemble: bool = False,
                 purple: bool = False, cluster: bool = False, feature_name: str = 'features_vit-laion2b_no_cls_trn',
                 percentage: str = '', seed: int = 0, mode: str = '', arr: str = 'a1', cls_base: bool = False,
                 selected_label: int = -1, simidx: int = 1):
        self.fold = fold
        self.args = args
        self.mscoco_pat = datapath
        self.split = split
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20  # 20
        self.ncluster = 200
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.cluster = cluster
        self.use_original_imgsize = use_original_imgsize
        self.cls_base = cls_base
        self.selected_label = selected_label

        self.img_path = os.path.join(args.save_base_dir,'pascal-5i', 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(args.save_base_dir,'pascal-5i', 'VOC2012/SegmentationClassAug/')
        self.coco_img_path = os.path.join(datapath, 'trn2014')
        self.coco_ann_path = os.path.join(datapath, 'Coco_Trainlabel')
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform

        # self.class_ids = self.build_class_ids()
        self.img_metadata_val = self.build_img_metadata('val')
        self.all_img_metadata_trn = self.build_all_img_metadata('trn')
        self.feature_name = feature_name
        self.seed = seed
        self.percentage = percentage
        ##my code
        self.img_feature_for_train_path = os.path.join(datapath, f'{self.feature_name}/folder{self.fold}_features_by_vqgan_encoder.h5df')
        self.img_feature_for_val_path = os.path.join(args.save_base_dir,'pascal-5i', f'VOC2012/features_vit-laion2b_pixel-level_val/folder{self.fold}_query_features_by_vqgan_encoder.h5df')
        ##end my code
        self.images_top50_val = self.get_top50_images_for_validation()
        self.images_top50_trn = self.get_top50_images_trn()
        self.mode = mode
        self.arr = arr
        self.simidx = simidx
        # self.cache = {}
        self.have_tokens = have_tokens
    def __len__(self):
        # return 1000
        # if self.cls_base:
        return len(self.img_metadata_val)
        # else:
        #     return 1000

    def get_top50_images_for_validation(self):
        print('feature name for val: ', self.feature_name[:-4] + '_val')
        with open(f"{self.mscoco_pat}/{self.feature_name[:-4]}_val/folder{self.fold}_top_50-similarity.json") as f:
            images_top50 = json.load(f)

        images_top50_new = {}
        for img_name, img_class in self.img_metadata_val:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}
            images_top50_new[img_name]['top50'] = images_top50[img_name]
            images_top50_new[img_name]['class'] = img_class

        return images_top50_new

    def get_top50_images_trn(self):
        images_top50_new = {}
        for img_name, img_class in self.all_img_metadata_trn:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}

            images_top50_new[img_name]['class'] = img_class

        return images_top50_new

    def create_gradiant_grid_images(self, support_img, support_mask, query_img, query_mask, arr):
        # create grid image for suppot images and query image.
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))

        content_list = [support_img, support_mask, query_img, query_mask]

        if arr == 'a1':
            support_img = content_list[0]
            support_mask = content_list[1]
            query_img = content_list[2]
            query_mask = content_list[3]

        elif arr == 'a2':
            support_img = content_list[1]
            support_mask = content_list[0]
            query_img = content_list[3]
            query_mask = content_list[2]

        elif arr == 'a3':
            support_img = content_list[3]
            support_mask = content_list[2]
            query_img = content_list[1]
            query_mask = content_list[0]

        elif arr == 'a4':
            support_img = content_list[2]
            support_mask = content_list[3]
            query_img = content_list[0]
            query_mask = content_list[1]

        elif arr == 'a5':
            support_img = content_list[1]
            support_mask = content_list[3]
            query_img = content_list[0]
            query_mask = content_list[2]

        elif arr == 'a6':
            support_img = content_list[3]
            support_mask = content_list[1]
            query_img = content_list[2]
            query_mask = content_list[0]

        elif arr == 'a7':
            support_img = content_list[2]
            support_mask = content_list[0]
            query_img = content_list[3]
            query_mask = content_list[1]

        elif arr == 'a8':
            support_img = content_list[0]
            support_mask = content_list[2]
            query_img = content_list[1]
            query_mask = content_list[3]

        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
        canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
        canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def create_arr_grid_from_images(self, support_img, support_mask, query_img, query_mask, positions):
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))
        if positions == 'a2':
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_mask
        elif positions == 'a3':
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_mask
        elif positions == 'a4':
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_mask
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_mask
        elif positions == 'a5':
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_mask
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_mask
        elif positions == 'a6':
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_mask
        elif positions == 'a7':
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_img
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_mask
        elif positions == 'a8':
            canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def create_all_grids(self, support_img, support_mask, query_img, query_mask):
        canvas_list = []
        # List of all possible arrangements
        arrangements = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']

        for arr in arrangements:
            canvas = self.create_ensemble_grid_from_images(support_img, support_mask, query_img, query_mask, arr)
            canvas_list.append(canvas)

        return canvas_list
    
    def get_tokens(self,query_name,support_name):
        with h5py.File(f'{self.mscoco_pat}/features_vit-laion2b_pixel-level_val/folder_{self.fold}_gt_tokens.h5','r') as f:
            group = f.require_group(query_name)
            # print(234,query_name,support_name)
            if support_name not in group:
                print(123,query_name,support_name)
                print('very bad! ')
                assert False
            return torch.tensor(f[query_name][support_name][:],dtype=torch.long)
        
    def __getitem__(self, idx):
        # if idx in self.cache and self.cache[idx]['valid']:
            # print("Cache hit for index:", idx)
            # return self.cache[idx]['batch']
        # idx %= len(self.img_metadata_val)  # for testing, as n_images < 1000
        # grids = [] 
        # support_imgs = torch.tensor([]) 
        # support_masks = torch.tensor([]) 
        query_img_features = [] 
        support_features = []
        support_image_names = []
        vq_tokens = []
        #end my code
        query_name, _, class_sample_query, _ = self.sample_episode(idx, sim_idx=0)
        # query_img = self.read_img(query_name)
        query_cmask = self.read_mask(query_name)
            # query_img = self.image_transform(query_img)
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask, class_sample_query,
                                                            purple=self.purple)
        if self.mask_transform:
            query_mask = self.mask_transform(query_mask)
        grid = self.create_gradiant_grid_images(query_mask, query_mask, query_mask, query_mask, self.arr)


        for sim_idx in range(self.simidx):
            _, support_name, _, class_sample_support = self.sample_episode(idx, sim_idx=sim_idx)
            support_image_names.append(support_name)
            if self.have_tokens:
                vq_tokens.append(self.get_tokens(query_name=query_name,support_name=support_name))
            query_img_feature, support_feature = self.load_feature(query_name,support_name)
            support_features.append(torch.tensor(support_feature))
            # grids.append(grid)
        #end my annation
        query_img_features.append(torch.tensor(query_img_feature))

        if self.have_tokens:
            vq_tokens = torch.stack(vq_tokens,dim = 0)
        # grids = torch.stack(grids,dim = 0)
        query_img_features = torch.stack(query_img_features,dim = 0)
        support_features = torch.stack(support_features,dim = 0)
        #my annation
#        else:
#            grid_stack = torch.cat((grid_stack, grid))
        #end my annation
        # print('grid stack: ', grid_stack.shape)
        batch = {'query_img': torch.tensor([]),
                 'query_mask': query_mask,
                 'support_img': torch.tensor([]),
                 'support_mask': torch.tensor([]),
                 'grid_stack': grid,
                 'query_img_features': query_img_features,
                 'support_features': support_features,
                 'query_image_name': query_name,
                 'support_image_names': support_image_names,
                 'vq_tokens': vq_tokens,
                 'grids': grid
                }
        # self.cache[idx] = {'valid': True, 'batch': batch}

        return batch

    def extract_ignore_idx(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        if not purple:
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 255
            return Image.fromarray(mask), boundary
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x, y] != class_id + 1:
                    color_mask[x, y] = np.array(PURPLE)
                else:
                    color_mask[x, y] = np.array(YELLOW)
        return Image.fromarray(color_mask), boundary

    def load_frame(self, query_name, support_name):
        # import pdb;pdb.set_trace()
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_img = self.read_coco_img(support_name)
        support_mask = self.read_coco_mask(support_name)
        org_qry_imsize = query_img.size

        return query_img, query_mask, support_img, support_mask, org_qry_imsize
    #my code
    def load_feature(self,query_name, support_name):
        with h5py.File(self.img_feature_for_train_path, "r") as f:
            support_feature = f[str(support_name)+'.jpg'][...]
        with h5py.File(self.img_feature_for_val_path, "r") as f:
            query_img_feature = f[query_name][...]
        return query_img_feature,support_feature
    #end my code
    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')
    def read_coco_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.coco_ann_path, img_name) + '.png')
        return mask

    def read_coco_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.coco_img_path, img_name) + '.jpg')
    def sample_episode(self, idx, sim_idx):
        """Returns the index of the query, support and class."""
        query_name, class_sample = self.img_metadata_val[idx]

        if self.cls_base:
            support_name = self.images_top50_val[query_name]['top50'][sim_idx]
            support_class = self.images_top50_trn[support_name]['class']
            while support_class != class_sample:
                sim_idx += 1
                if sim_idx >= len(self.images_top50_val[query_name]['top50']):
                    break
                support_name = self.images_top50_val[query_name]['top50'][sim_idx]
                support_class = self.images_top50_trn[support_name]['class']
        else:
            support_name = self.images_top50_val[query_name]['top50'][sim_idx]
            support_class = self.images_top50_trn[support_name]['class']

        if support_name == query_name:
            print('support_name = query_name ' + support_name)
            return self.sample_episode(idx, sim_idx + 1)

        if sim_idx >= len(self.images_top50_val[query_name]['top50']):
            print('query name: ', query_name)
            sim_idx = 0
            return self.sample_episode(idx + 1, sim_idx)

        return query_name, support_name, class_sample, support_class

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val

    def build_img_metadata(self, split):

        def read_metadata(split, fold_id):
            # cwd = os.path.dirname(os.path.abspath(__file__))
            cwd = self.args.save_base_dir

            fold_n_metadata_path = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))

            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]

            if self.cls_base:
                new_fold_n_metadata = []
                for data in fold_n_metadata:
                    label = int(data.split('__')[1]) - 1
                    if label + 1 == self.selected_label:
                        element = [data.split('__')[0], label]
                        # print('element: ',  element)
                        new_fold_n_metadata.append(element)
            else:
                new_fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) -1] for data in fold_n_metadata]

            return new_fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata(split, self.fold)

        print('Total (%s) images are : %d' % (split, len(img_metadata)))

        return img_metadata

    def build_all_img_metadata(self, split):

        def read_metadata(split, fold_id):
            # cwd = os.path.dirname(os.path.abspath(__file__))
            cwd = self.args.save_base_dir

            fold_n_metadata_path = os.path.join(cwd, 'split/coco/%s/fold%d.txt' % (split, fold_id))

            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            # import pdb;pdb.set_trace()

            new_fold_n_metadata = [[data.split('__')[0][:-4], int(mapped_dict[str(self.fold)][(data.split('__')[1])]) -1] for data in fold_n_metadata]

            return new_fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata(split, self.fold)

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        if len(self.img_metadata[0]) != 3:
            for img_name, img_class in self.img_metadata:
                img_metadata_classwise[img_class] += [img_name]
        else:
            for img_name, img_class, _ in self.img_metadata:
                img_metadata_classwise[img_class] += [img_name]

        return img_metadata_classwise



# import argparse
# import torchvision.transforms as T

# def get_args():
#     parser = argparse.ArgumentParser('InMeMo training for segmentation', add_help=False)
#     parser.add_argument('--mae_model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
#                         help='Name of model to train')
#     parser.add_argument("--mode", type=str, default='spimg_spmask',
#                         choices=['no_vp', 'spimg_spmask', 'spimg', 'spimg_qrimg', 'qrimg', 'spimg_spmask_qrimg'],
#                         help="mode of adding vp on img.")
#     parser.add_argument('--output_dir', default=f'./output_samples')
#     parser.add_argument('--device', default='cuda:7',
#                         help='device to use for training / testing')
#     parser.add_argument('--base_dir', default='./VisualICL/coco', help='pascal base dir')
#     parser.add_argument('--seed', default=0, type=int)
#     parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
#     parser.add_argument('--task', default='segmentation', choices=['segmentation', 'detection'])
#     parser.add_argument('--ckpt', default='./weights/checkpoint-1000.pth', help='model checkpoint')
#     parser.add_argument('--save_base_dir', default='./VisualICL', help='/prefix/VisualICL/')
#     parser.add_argument('--vq_ckpt_dir', default='/data/luotianci/TO_JPSX/VisualICL/weights/vqgan', help="dir for vq-gan's config and model ckpt")
#     parser.add_argument('--simidx', default=1, type=int)
#     parser.add_argument('--dropout', default=0.3, type=float)
#     # parser.add_argument('--temperature', default=0.1, type=float)
#     parser.add_argument('--fold', default=0, type=int)
#     parser.add_argument('--split', default='trn', type=str)
#     parser.add_argument('--purple', default=0, type=int)
#     parser.add_argument('--flip', default=0, type=int)
#     parser.add_argument('--feature_name', default='features_vit-laion2b_pixel-level_trn', type=str)
#     parser.add_argument('--percentage', default='', type=str)
#     parser.add_argument('--cluster', action='store_true')
#     parser.add_argument('--random', action='store_true')
#     parser.add_argument('--G_pre_mean', action='store_true')
#     parser.add_argument('--G_copy_another', action='store_true')
#     parser.add_argument('--G_only_div', action='store_true')
#     parser.add_argument('--ensemble', action='store_true')
#     parser.add_argument('--aug', action='store_true')
#     parser.add_argument('--fsl', action='store_true')
#     parser.add_argument('--save_examples', action='store_true', help='whether save the example in val')
#     # parser.add_argument('--sigma', default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0], type=float, nargs=8, help='A list of four float numbers')
#     parser.add_argument('--sigma', default=0.1, type=float)

#     # training settings
#     parser.add_argument("--batch-size", type=int, default=32,
#                         help="Number of images sent to the network in one step.")
#     parser.add_argument("--lr", type=float, default=40,
#                         help="Base learning rate for training with polynomial decay.")
#     parser.add_argument("--epoch", type=int, default=100,
#                         help="Number of training steps.")
#     parser.add_argument("--loss-function", type=str, default='CrossEntropy',
#                         help="loss function for training")
#     parser.add_argument("--scheduler", type=str, default='cosinewarm',
#                         help="scheduler for training")
#     parser.add_argument("--optimizer", type=str, default='Adam',
#                         help="optimizer for training")
#     parser.add_argument("--arr", type=str, default='a1',
#                         help="the setting of arrangements of canvas")
#     parser.add_argument("--p-eps", type=int, default=1,
#                         help="Number of pad weight hyperparameter [0, 1].")
#     parser.add_argument("--vp-model", type=str, default='pad',
#                         help="type of the VP Prompter.")
#     parser.add_argument("--loss_mean",type=int, default=1)
#     # Number of images for few-shot training
#     parser.add_argument("--n-shot", type=int, default=16,
#                         help="Number of images for fsl.")
#     parser.add_argument("--choice", type=str, default='Zero',
#                         help="choose prompt composer")
#     parser.add_argument('--align_s',type=int, default=1)
#     parser.add_argument('--align_q',type=int, default=1)
#     return parser
# import torchvision.transforms.functional as TF

# if __name__ == "__main__":
#     # model = prepare_model('/shared/amir/Deployment/arxiv_mae/logs_dir/pretrain_small_arxiv2/checkpoint-799.pth',
#     #                       arch='mae_vit_small_patch16')
#     args = get_args()

#     args = args.parse_args()
#     args.device = 'cpu'
#     padding = 1
#     image_transform = T.Compose(
#         [T.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
#          T.ToTensor()])
#     mask_transform = T.Compose(
#         [T.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
#          T.ToTensor()])
#     train_dataset = DatasetMSCOCO(args.base_dir, args=args, fold=args.fold, split=args.split, image_transform=image_transform,
#                          mask_transform=mask_transform,have_tokens = False,
#                          flipped_order=args.flip, purple=args.purple, random=args.random, cluster=args.cluster,
#                          feature_name=args.feature_name, percentage=args.percentage, seed=args.seed, mode=args.mode,
#                          arr=args.arr,simidx=args.simidx)
#     bat = train_dataset[0]
#     image = TF.to_pil_image(bat['grid_stack'])
#     # print(img_name)
#     # # 保存图像
#     image.save("cat.jpg")

#     # canvas = canvas_ds[0]

