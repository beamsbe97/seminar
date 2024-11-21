import os.path
from tqdm import tqdm
from trainer import val_pascal_dataloader
from evaluate.reasoning_dataloader import *
import torchvision
from evaluate.mae_utils import *
import argparse
from pathlib import Path
from evaluate.segmentation_utils import *
from PIL import Image
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from models.train_models import _generate_result_for_canvas, PGVP
from evaluate_detection.voc_orig import CLASS_NAMES
import global_var
import torchvision.transforms.functional as TF
import numpy as np

import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

def normalize_to_01(matrix):
    """Normalize a matrix to the range [0, 1]."""
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

def resize_matrix(matrix, target_size):
    """Resize the matrix to target size using interpolation."""
    return cv2.resize(matrix, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

def overlay_heatmap_on_image(image, heatmap, alpha=0.5):
    """Overlay a heatmap on an image with given transparency."""
    # Convert heatmap to BGR format
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    # Overlay heatmap onto the image
    overlay = cv2.addWeighted(heatmap_colored, alpha, image, 1 - alpha, 0)
    return overlay
def get_args():
    parser = argparse.ArgumentParser('InMeMo training for segmentation', add_help=False)
    parser.add_argument('--mae_model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument("--mode", type=str, default='spimg_spmask',
                        choices=['no_vp', 'spimg_spmask', 'spimg', 'spimg_qrimg', 'qrimg', 'spimg_spmask_qrimg'],
                        help="mode of adding vp on img.")
    parser.add_argument('--output_dir', default=f'./vis_output_samples')
    parser.add_argument('--device', default='cuda:7',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--task', default='segmentation', choices=['segmentation', 'detection'])
    parser.add_argument('--ckpt', default='./weights/checkpoint-1000.pth', help='model checkpoint')
    parser.add_argument('--vq_ckpt_dir', default='/data/luotianci/TO_JPSX/VisualICL/weights/vqgan', help="dir for vq-gan's config and model ckpt")
    parser.add_argument('--dataset_type', default='pascal')
    parser.add_argument('--simidx', default=2, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    # parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--G_pre_mean', action='store_true')
    parser.add_argument('--G_copy_another', action='store_true')
    parser.add_argument('--G_only_div', action='store_true')
    # parser.add_argument('--sigma', default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0], type=float, nargs=8, help='A list of four float numbers')
    # training settings
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Number of training steps.")
    parser.add_argument("--scheduler", type=str, default='cosinewarm',
                        help="scheduler for training")
    parser.add_argument("--arr", type=str, default='a1',
                        help="the setting of arrangements of canvas")
    parser.add_argument("--vp-model", type=str, default='pad',
                        help="type of the VP Prompter.")
    # Number of images for few-shot training
    parser.add_argument("--choice", type=str, default='Conv',
                        help="choose prompt composer")
    parser.add_argument('--align_s',type=int, default=1)
    parser.add_argument('--align_q',type=int, default=0)
    parser.add_argument('--kernel_size', default=7, type=int)
    parser.add_argument("--loss_choice", type=str, default='cos',
                        help="choose prompt composer")
    parser.add_argument("--lamba", type=float, default='0.6',
                        help="choose prompt composer")
    parser.add_argument("--pos", type=str, default='after',
                        help="choose prompt composer")
    parser.add_argument('--save_model_path',
                        help='model checkpoint')

    return parser


def create_gradiant_grid_images(support_img, support_mask, query_img, query_mask, arr):
    # create grid image for suppot images and query image.
    canvas = torch.ones((3,224,224),dtype=torch.float32)

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

def convert_to_rgb(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def get_bw(mask):
    mask = np.array(mask)
    mask[mask != 255] = 0
    mask[mask == 255] = 255
    return Image.fromarray(mask)

def test_for_generate_results(args):
    padding = 1
    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         convert_to_rgb,
         torchvision.transforms.ToTensor()])
    mask_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         convert_to_rgb,
         torchvision.transforms.ToTensor()])

    print("load data over")

    # MAE_VQGAN model
    vqgan = prepare_model(args.ckpt, arch=args.mae_model, vq_ckpt_dir=args.vq_ckpt_dir)

    if args.vp_model == 'Prompt':
        print('load pad prompter.')
        VP = PGVP(args=args, vqgan=vqgan.to(args.device), mode=args.mode, arr=args.arr)
    else:
        raise ValueError("No VP model is loaded, Please use 'pad'")

    if args.mode != 'no_vp':
        checkpoint = torch.load(args.save_model_path,map_location=args.device)
        VP.PromptGenerator.load_state_dict(checkpoint["visual_prompt_dict"])

    VP.eval()
    VP.to(args.device)

    setting = f'{args.mode}_{args.task}_{args.arr}'
    eg_save_path = f'{args.output_dir}/{args.vp_model}_output_examples/{args.mode}'
    os.makedirs(eg_save_path, exist_ok=True)

    print(f'This is the mode of {args.mode}.')
    print(f'This is the arrangement of {args.arr}.')

    eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
    examples_save_path = eg_save_path + f'/{setting}/'
    os.makedirs(examples_save_path, exist_ok=True)

    with open(os.path.join(examples_save_path, 'log.txt'), 'w') as log:
        log.write(str(args) + '\n')

    image_number = 0
    brid_root = ''
    support_img_1 = image_transform(Image.open(os.path.join(brid_root,'s1.png')))
    support_mask_1 = mask_transform(get_bw(Image.open(os.path.join(brid_root,'m1.png'))))
    support_img_2 = image_transform(Image.open(os.path.join(brid_root,'s2.png')))
    support_mask_2 = mask_transform(get_bw(Image.open(os.path.join(brid_root,'m2.png'))))
    query_img = image_transform(Image.open(os.path.join(brid_root,'q1.png')))
    query_mask = mask_transform(get_bw(Image.open(os.path.join(brid_root,'l1.png'))))
    print(support_img_1.shape,support_mask_1.shape,query_img.shape,query_mask.shape)
    grid_1 = create_gradiant_grid_images(support_img_1, support_mask_1, query_img, query_mask, args.arr)
    grid_2 = create_gradiant_grid_images(support_img_2, support_mask_2, query_img, query_mask, args.arr)
    grids = [grid_1,grid_2]
    grids = torch.stack(grids,dim = 0)
    grids_fe = (grids - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
    print(grids_fe.dtype)
    grids_fe = grids_fe.to(torch.float32)
    grids_fe = grids_fe.to(args.device)
    with torch.no_grad():
        print(grids_fe.shape)
        img_features = vqgan.patch_embed(grids_fe)
        img_features = img_features + vqgan.pos_embed[:,1:,:]

    support_features = img_features[:,:98,:].unsqueeze(0)
    query_img_features = img_features[:1,98:,:].unsqueeze(0)

    support_features = support_features.to(args.device, dtype=torch.float32)
    query_img_features = query_img_features.to(args.device, dtype=torch.float32)
    ##end my code
    grid_stack = grids
    support_img_1 = support_img_1.to(args.device, dtype=torch.float32)
    support_mask_1 = support_mask_1.to(args.device, dtype=torch.float32)

    support_img_2 = support_img_2.to(args.device, dtype=torch.float32)
    support_mask_2 = support_mask_2.to(args.device, dtype=torch.float32)
    query_img = query_img.to(args.device, dtype=torch.float32)
    query_mask = query_mask.to(args.device, dtype=torch.float32)
    grid_stack = grid_stack.to(args.device, dtype=torch.float32)

    _, canvas_pred_tokens, canvas_label = VP(support_img_1.unsqueeze(1), support_mask_1.unsqueeze(1), query_img, query_mask, grid_stack.unsqueeze(1), 
                        query_img_features, support_features)


    original_image_list, generated_result_list = _generate_result_for_canvas(args, vqgan.to(args.device),
                                                                                canvas_pred_tokens, canvas_label, args.arr)
    for index in range(len(original_image_list)):

        # Image.fromarray(original_image_list[index]).save(
        #     examples_save_path + f'original_image_{image_number}.png')
        Image.fromarray(generated_result_list[index]).save(
            examples_save_path + f'generated_image_{image_number}.png')

        original_image = round_image(original_image_list[index], [WHITE, BLACK])
        generated_result = round_image(generated_result_list[index], [WHITE, BLACK], t=args.t)

        current_metric = calculate_metric(args, original_image, generated_result, fg_color=WHITE, bg_color=BLACK)
        # print('current_metric: ', current_metric)
        if index == 0:
            image = TF.to_pil_image(generated_result_list[0])
                # # 保存图像
            image.save(os.path.join(brid_root,"result.jpg"))
            image = TF.to_pil_image((generated_result/255).permute(2,0,1))
            image.save(os.path.join(brid_root,"final_result.jpg"))
            image = TF.to_pil_image((original_image/255).permute(2,0,1))
            image.save(os.path.join(brid_root,"original_image.jpg"))
        with open(os.path.join(examples_save_path, 'log.txt'), 'a') as log:
            # log.write(str(idx) + '\t' + str(current_metric) + '\n')
            log.write(str(image_number) + '\t' + str(current_metric) + '\n')
        image_number += 1

        for i, j in current_metric.items():
            eval_dict[i] += j

        # print('eval_dict: ', eval_dict)
    print('val metric: {}'.format(eval_dict))
    with open(os.path.join(examples_save_path, 'log.txt'), 'a') as log:
        log.write('all\t' + str(eval_dict) + '\n')
    self_attn_weight = global_var.self_attn_weight

    cross_attn_weight = global_var.cross_attn_weight
    print(cross_attn_weight.shape)
    print(self_attn_weight.shape)
    self_attn_weight = ((self_attn_weight.reshape(2,7,14,7,14))[:,:,:,:,:7]).reshape(2,98,49)
    # self_attn_weight = F.softmax(self_attn_weight,dim=-1)
    print(self_attn_weight.shape)
    self_attn_weight = torch.sum(self_attn_weight,dim=1).reshape(2,7,7)
    self_attn_weight = F.softmax(self_attn_weight,dim=-1)
    print(self_attn_weight)
    print(self_attn_weight.shape)
    cross_attn_weight = cross_attn_weight.permute(2,1,0).reshape(2,7,7)
    attn_ = cross_attn_weight * self_attn_weight
    print(attn_)


    mtx1 = attn_[0].detach().cpu().numpy()  # Example 7x7 matrix 1
    mtx2 = attn_[1].detach().cpu().numpy()  # Example 7x7 matrix 2

    img1 = (np.transpose(support_img_1.detach().cpu().numpy(),(1,2,0))*255).astype(np.uint8)  # Example 111x111 white image 1
    img2 = (np.transpose(support_img_2.detach().cpu().numpy(),(1,2,0))*255).astype(np.uint8)  # Example 111x111 white image 2
    img3 = ((np.transpose(query_img.detach().cpu().numpy(),(1,2,0))*255).clip(0,255)).astype(np.uint8)  # Example 111x111 white image 2
    # Step 2: Normalize the matrices
    mtx1_normalized = normalize_to_01(mtx1)
    mtx2_normalized = normalize_to_01(mtx2)
    image = TF.to_pil_image(img1)
        # # 保存图像
    image.save("debug.jpg")

    # Step 3: Resize the normalized matrices to 111x111
    mtx1_resized = resize_matrix(mtx1_normalized, 111)
    mtx2_resized = resize_matrix(mtx2_normalized, 111)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img3)
    axes[0].set_title("Query",fontsize=30)
    axes[0].axis('off')

    # 在第二个子图中显示原图叠加热力图1
    axes[1].imshow(img1)
    # heatmap1_colored = plt.cm.jet(mtx1_resized)[:, :, :3]
    axes[1].imshow(mtx1_resized, cmap='jet', alpha=0.5)  # 叠加热力图1
    # overlay1 = img1 * (1 - 0.5) + heatmap1_colored * 0.5  # 控制热力图透明度
    # axes[1].imshow(overlay1)

    axes[1].set_title("Attention for Prompt 1",fontsize=30)
    axes[1].axis('off')

    # 在第三个子图中显示原图叠加热力图2
    axes[2].imshow(img2)
    axes[2].imshow(mtx2_resized, cmap='jet', alpha=0.5)  # 叠加热力图2
    axes[2].set_title("Attention for Prompt 2",fontsize=30)
    axes[2].axis('off')

    # 显示结果
    plt.tight_layout()  # 调整布局
    plt.subplots_adjust(top=0.85)  # 增加布局间距

    plt.savefig('/data/luotianci/TO_JPSX/rabbit_brid/fin_attn_vis_2.pdf')

    plt.show()


    # # Step 4: Overlay heatmaps onto images
    # output_img1 = overlay_heatmap_on_image(img1, mtx1_resized, alpha=0)
    # output_img2 = overlay_heatmap_on_image(img2, mtx2_resized, alpha=0)
    # output_img1 = img1
    # # Step 5: Show and save results
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # plt.title("Image 1 with Heatmap")
    # plt.imshow(output_img1)
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.title("Image 2 with Heatmap")
    # plt.imshow(cv2.cvtColor(output_img2, cv2.COLOR_BGR2RGB))
    # plt.axis('off')

    # plt.tight_layout()
    # plt.savefig('/data/luotianci/TO_JPSX/rabbit_brid/a.png')

    # plt.show()

if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    test_for_generate_results(args)
