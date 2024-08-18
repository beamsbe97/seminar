import torch.nn as nn
from evaluate.mae_utils import *
from evaluate.segmentation_utils import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.prompt_generator import PromptGenerator,PromptGeneratorlimzero
from PIL import Image
import torchvision.transforms.functional as TF

class Scheduler(object):
    def __init__(self, name, num_epoch):
        self.name = name
        self.num_epoch = num_epoch

    def select_scheduler(self, optimizer_ft):
        scheduler = ''
        if 'multistep' == self.name:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer_ft, milestones=[30, 60], gamma=0.1)
        elif 'cosine' == self.name:
            print("This is the scheduler of CosineAnnealingLR.")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=self.num_epoch, eta_min=0)
        elif 'cosinewarm' == self.name:
            print("This is the scheduler of CosineAnnealingWarmRestarts.")
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, T_0=10, T_mult=1, eta_min=0)
        elif 'reducelr' == self.name:
            print('the scheduler is reduceLR.')
            scheduler = ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=5, verbose=True)
        elif 'normal' == self.name:
            scheduler = None

        return scheduler


class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return F.interpolate(img, size=self.size, mode='bilinear', align_corners=False)


def _generate_result_for_canvas(args, model, canvas_pred_tokens , canvas_label, arr):
    """canvas is already in the right range."""
    ids_shuffle, len_keep = generate_arr_mask_for_evaluation(arr)
    batch_size = canvas_pred_tokens.shape[0]
    original_image_list = []
    generated_result_list = []

    for i in range(batch_size):
        _, im_paste, _ = generate_image(canvas_pred_tokens[i].unsqueeze(0).to(args.device), model, ids_shuffle.to(args.device),
                                        canvas_label[i].unsqueeze(0).to(args.device), len_keep, device=args.device)
        canvas_ = torch.einsum('chw->hwc', canvas_label[i])
        canvas_ = torch.clip((canvas_.cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
        assert canvas_.shape == im_paste.shape, (canvas_.shape, im_paste.shape)

        original_image_list.append(np.uint8(canvas_))
        generated_result_list.append(np.uint8(im_paste))


    return original_image_list, generated_result_list


def round_image(img, options=(WHITE, BLACK, RED, GREEN, BLUE), outputs=None, t=(0, 0, 0)):
    # img.shape == [224, 224, 3], img.dtype == torch.int32
    img = torch.tensor(img)
    t = torch.tensor((t)).to(img)
    options = torch.tensor(options)
    opts = options.view(len(options), 1, 1, 3).permute(1, 2, 3, 0).to(img)
    nn = (((img + t).unsqueeze(-1) - opts) ** 2).float().mean(dim=2)
    nn_indices = torch.argmin(nn, dim=-1)
    if outputs is None:
        outputs = options
    ##修改
    res_img = img + (torch.tensor(outputs)[nn_indices]-img).detach()
    return res_img


class PGVP(nn.Module):
    """editted for visual prompting"""
    def __init__(self, args, vqgan, mode, arr):
        super().__init__()
        self.args = args
        self.device = self.args.device
        self.padding = 1
        self.vqgan = vqgan
        self.arr = arr
        # print('????????',args.sigma)
        if args.choice == 'Zero':
            self.PromptGenerator = PromptGeneratorlimzero(dropout=args.dropout,args=args)
        else:
            self.PromptGenerator = PromptGenerator(dropout=args.dropout,sigma=args.sigma,device=args.device)
        self.transform224 = ResizeTransform((224, 224))
        self.transform111 = ResizeTransform((111, 111))
        self.mode = mode
        self.register_buffer('imagenet_mean',torch.tensor([0.485, 0.456, 0.406])) 
        self.register_buffer('imagenet_std',torch.tensor([0.229, 0.224, 0.225]))
        
    def create_grid_from_images(self, prompt_img, support_mask, query_img, query_mask):
        canvas = torch.ones((prompt_img.shape[1], 2 * prompt_img.shape[2] + 2 * self.padding,
                             2 * prompt_img.shape[3] + 2 * self.padding))
        canvas[:, :prompt_img.shape[2], :prompt_img.shape[3]] = prompt_img

        canvas[:, -query_img.shape[2]:, :query_img.shape[3]] = query_img
        canvas[:, :prompt_img.shape[2], -prompt_img.shape[3]:] = support_mask
        canvas[:, -query_img.shape[2]:, -prompt_img.shape[3]:] = query_mask
        canvas = (canvas.detach().numpy() - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

        return torch.from_numpy(canvas)

    def create_gradiant_grid_images(self, support_img, support_mask, query_img, query_mask, grid, arr):
        # create grid image for suppot images and query image.
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

        img_size = 111
        grid[:, :, :img_size, :img_size] = support_img

        grid[:, :, -img_size:, :img_size] = query_img
        grid[:, :, :img_size, -img_size:] = support_mask
        grid[:, :, -img_size:, -img_size:] = query_mask

        return grid

    def create_gradiant_grid_label_images(self, support_img, support_mask, query_img, query_mask, grid):
        # create grid image for suppot images and query image.
        grid[:, :, :support_img.shape[2], :support_img.shape[3]] = support_img

        grid[:, :, -query_img.shape[2]:, :query_img.shape[3]] = query_img
        grid[:, :, :support_img.shape[2], -support_img.shape[3]:] = support_mask
        grid[:, :, -query_img.shape[2]:, -support_img.shape[3]:] = query_mask

        return grid

    def _generate_raw_prediction(self, canvas_tokens, arr):
        """canvas is already in the right range."""
        ids_shuffle, len_keep = generate_arr_mask_for_evaluation(arr)
        # print(ids_shuffle,ids_shuffle.shape,len_keep,len_keep.shape)
        # assert False
        y_pred, mask = generate_raw_pred_for_train(canvas_tokens, self.vqgan,
                                                   ids_shuffle.to(self.device),
                                                   len_keep, device=self.device)

        return y_pred, mask

    def forward(self, support_img, support_mask, query_img, query_mask, grid, query_features, support_features):
        canvas_label = grid.clone()
        canvas_return_label = grid.clone()
        if self.args.dataset_type != 'pascal_det':
            canvas_return_label = (canvas_return_label - self.imagenet_mean[:, None, None]) / self.imagenet_std[:, None, None]

        canvas_return_label = canvas_return_label.permute(1,0,2,3,4)
        canvas_return_label = canvas_return_label[0]

        # print("support_features min:", support_features.min().item(), "support_features max:", support_features.max().item())        
        # print("query_features min:", query_features.min().item(), "query_features max:", query_features.max().item())        


        canvas_pred_tokens = self.PromptGenerator(support_features,query_features)

        grid = grid.permute(1,0,2,3,4)
        grid = grid[0]
        # print("canvas_pred_tokens min:", canvas_pred_tokens.min().item(), "canvas_pred_tokens max:", canvas_pred_tokens.max().item())        
        y_pred, mask = self._generate_raw_prediction(canvas_pred_tokens, self.arr)
        canvas_label = canvas_label.permute(1,0,2,3,4)
        if self.args.dataset_type != 'pascal_det':
            canvas_label = (canvas_label - self.imagenet_mean[:, None, None]) / self.imagenet_std[:, None, None]
        N = canvas_label.shape[0]
        loss_ce = 0
        # print("y_pred min:", y_pred.min().item(), "y_pred max:", y_pred.max().item())
        if self.args.G_pre_mean:
            list = []
            for sub_label in canvas_label:
                list.append(self.vqgan.forward_loss(sub_label, y_pred, mask))
            target = torch.mean(torch.stack(list, dim=0), dim=0)
            loss = nn.CrossEntropyLoss(reduction='none')(input=y_pred.permute(0, 2, 1), target=target)
            loss = (loss * mask).sum() / mask.sum()
            loss_ce = loss/canvas_label[0].shape[0]
            return loss_ce, canvas_pred_tokens, canvas_return_label
        if self.args.loss_mean:
            for sub_label in canvas_label:
                loss_ce += self.vqgan.forward_loss(sub_label, y_pred, mask)
            loss_ce /= N
        else :
            loss_ce = self.vqgan.forward_loss(canvas_label[0],y_pred,mask)

        return loss_ce, canvas_pred_tokens, canvas_return_label