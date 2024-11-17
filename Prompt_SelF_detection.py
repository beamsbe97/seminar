import os.path
from tqdm import tqdm
import sys
current_path = os.path.dirname(os.path.dirname(__file__))
# print(os.path.dirname(current_path))
if current_path not in sys.path:
    sys.path.append(current_path)

print(sys.path)

from evaluate_detection.canvas_ds import CanvasDataset4Train,CanvasDataset4Val
from evaluate.reasoning_dataloader import *
import torchvision.transforms as T
from evaluate.mae_utils import *
import argparse
from pathlib import Path
from evaluate.segmentation_utils import *
from PIL import Image
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from models.prompt_generator import PromptGenerator
from models.train_models import _generate_result_for_canvas, PGVP, Scheduler
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
import torch.nn.utils.parametrize as parametrize
from trainer.Lora import linear_layer_parameterization,save_lora_state_dict,load_lora_state_dict,freeze_base_weights

from evaluate_detection.box_ops import to_rectangle

def get_args():
    parser = argparse.ArgumentParser('InMeMo training for detection', add_help=False)
    parser.add_argument('--mae_model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument("--mode", type=str, default='spimg_spmask',
                        choices=['no_vp', 'spimg_spmask', 'spimg', 'spimg_qrimg', 'qrimg', 'spimg_spmask_qrimg'],
                        help="mode of adding vp on img.")
    parser.add_argument('--output_dir', default=f'./detection')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--base_dir', default='./pascal-5i', help='pascal base dir')  # TODO: check the base dir path.
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--task', default='detection', choices=['segmentation', 'detection'])
    parser.add_argument('--ckpt', default='./weights/checkpoint-1000.pth', help='model checkpoint')
    parser.add_argument('--save_base_dir', default='./VisualICL', help='/prefix/VisualICL/')
    parser.add_argument('--vq_ckpt_dir', default='/data/luotianci/TO_JPSX/VisualICL/weights/vqgan', help="dir for vq-gan's config and model ckpt")
    parser.add_argument('--dataset_type', default='pascal_det',
                        choices=['pascal', 'pascal_det'])
    parser.add_argument('--simidx', default=1, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--split', default='trn', type=str)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--feature_name', default='features_vit-laion2b_pixel-level_trn', type=str)
    parser.add_argument('--percentage', default='', type=str)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--fast', default=0, type=int)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--save_examples', action='store_true', help='whether save the example in val')
    # parser.add_argument('--sigma', default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0], type=float, nargs=8, help='A list of four float numbers')
    # parser.add_argument('--sigma', default=[1.0], type=float, nargs=4, help='A list of four float numbers')
    # train settings
    parser.add_argument('--sigma', default=0.1, type=float)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--lr", type=float, default=40,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Number of training steps.")
    parser.add_argument("--scheduler", type=str, default='cosinewarm',
                        help="scheduler for training")
    parser.add_argument("--optimizer", type=str, default='Adam',
                        help="optimizer for training")
    parser.add_argument("--arr", type=str, default='a1',
                        help="the setting of arrangements of canvas")
    parser.add_argument("--p-eps", type=int, default=1,
                        help="Number of mae weight hyperparameter,[0, 1].")
    parser.add_argument("--vp-model", type=str, default='pad',
                        help="pad prompter.")
    parser.add_argument("--choice", type=str, default='Zero',
                        help="choose prompt composer")
    parser.add_argument('--align_s',type=int, default=1)
    parser.add_argument('--align_q',type=int, default=1)
    parser.add_argument("--loss_mean",type=int, default=1)
    parser.add_argument('--G_pre_mean', action='store_true')
    parser.add_argument('--G_copy_another', action='store_true')
    parser.add_argument('--G_only_div', action='store_true')
    parser.add_argument("--pos", type=str, default='after',
                        help="choose prompt composer")

    return parser

def _generate_raw_prediction(model, canvas_tokens, arr,args):
    """canvas is already in the right range."""
    ids_shuffle, len_keep = generate_arr_mask_for_evaluation(arr)
    # ids_shuffle, len_keep = generate_arr_mask_for_evaluation(arr)
    # print(ids_shuffle,ids_shuffle.shape,len_keep,len_keep.shape)
    # assert False
    y_pred, mask = generate_raw_pred_for_train(canvas_tokens, model,
                                                ids_shuffle.to(args.device),
                                                len_keep, device=args.device)
    return y_pred, mask

def model_forward(grid, query_features, support_features,args,imagenet_mean, imagenet_std):
        canvas_label = grid.clone()
        canvas_return_label = grid.clone()
        if args.dataset_type != 'pascal_det':
            canvas_return_label = (canvas_return_label - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

        canvas_return_label = canvas_return_label.permute(1,0,2,3,4)
        canvas_return_label = canvas_return_label[0]
        bz = support_features.shape[0]
        K = support_features.shape[1]
        query_features = query_features.expand(-1, K, -1, -1)
        # print(support_features.shape,query_features.shape)
        canvas_pred_tokens = torch.cat((support_features,query_features),dim=2).permute(1,0,2,3)
        loss_ce = 0
        return loss_ce, canvas_pred_tokens, canvas_return_label

# def forward(model, support_img, support_mask, query_img, query_mask, grid, imagenet_mean, imagenet_std):
#     # canvas_label = grid.clone()
#     canvas_label = grid.clone()
#     # print("support_features min:", support_features.min().item(), "support_features max:", support_features.max().item())        
#     # print("query_features min:", query_features.min().item(), "query_features max:", query_features.max().item())        

#     grid = grid.permute(1,0,2,3,4)
#     grid = grid[0]
#     # print("canvas_pred_tokens min:", canvas_pred_tokens.min().item(), "canvas_pred_tokens max:", canvas_pred_tokens.max().item())        
#     y_pred, mask = _generate_raw_prediction(model,grid, args.arr,args)
#     canvas_label = canvas_label.permute(1,0,2,3,4)
#     canvas_label = (canvas_label - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
#     N = canvas_label.shape[0]
#     loss_ce = 0
#     vq_tokens = vq_tokens.permute(1,0,2)
#     # print("y_pred min:", y_pred.min().item(), "y_pred max:", y_pred.max().item())
#     for sub_label in canvas_label:
#         loss_ce += model.forward_loss(sub_label, y_pred, mask)
#     loss_ce /= N
#     canvas_pred = (canvas_pred - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]

#     return loss_ce

def train(args):

    setting = f'_Prompt_self_fusion_lr_{args.lr}_task_{args.task}'

    model_save_path = f'{args.save_base_dir}/save_ours_ckpt/task_lora_{args.task}_{args.choice}_G_copy_another_{args.G_copy_another}_G_only_div_{args.G_only_div}_align_s{args.align_s}_align_q{args.align_q}_loss_mean{args.loss_mean}/fold_{args.fold}/simidx_{args.simidx}_model/sigma_{args.sigma}/{setting}'
    eg_save_path = f'{args.output_dir}/task_{args.task}_{args.choice}_G_copy_another_{args.G_copy_another}_G_only_div_{args.G_only_div}_align_s{args.align_s}_align_q{args.align_q}_loss_mean{args.loss_mean}/fold_{args.fold}/simidx_{args.simidx}/sigma_{args.sigma}/{setting}'


    padding = 1
    image_transform = T.Compose(
        [T.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         T.ToTensor()])
    mask_transform = T.Compose(
        [T.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         T.ToTensor()])
    
    val_dataset = {
        'pascal_det': CanvasDataset4Val
    }[args.dataset_type](args.base_dir,simidx=args.simidx, fold=args.fold, split=args.split, image_transform=image_transform,
                         mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, random=args.random, cluster=args.cluster,
                         feature_name=args.feature_name, percentage=args.percentage, seed=args.seed, mode=args.mode,args=args,
                         arr=args.arr)
    print('number of val demonstation',args.simidx)
    print('length of val dataset: ', len(val_dataset))

    dataloaders = {}

    # set batch size to 1/2 on val set to adapt GPU memory.修改了
    dataloaders['val'] = DataLoader(val_dataset, batch_size=args.batch_size//2, shuffle=False)

    print('val datalaoder: ', len(dataloaders['val']))

    print("load data over")
    # MAE_VQGAN model
    vqgan = prepare_model(args.ckpt, arch=args.mae_model, vq_ckpt_dir=args.vq_ckpt_dir)
    print(args.device)
    vqgan.to(args.device)
    # if args.vp_model == 'pad':
    #     print('load pad prompter.')
    #     VP = CustomVP(args=args, vqgan=vqgan.to(args.device), mode=args.mode, arr=args.arr, p_eps=args.p_eps)
    # if args.vp_model == 'Prompt':
    #     print('load prompt generator')
    #     VP = PGVP(args=args, vqgan=vqgan.to(args.device), mode=args.mode, arr=args.arr)
    # else:
    #     raise ValueError("Please check the mode of InMeMo!")
    # total_parameters_lora = 0
    # tot = 0
    scaler = GradScaler()
    # for block in vqgan.blocks:
    #     for layer in [block.attn.qkv, block.attn.proj]:
    #         parametrize.register_parametrization(layer,"weight",linear_layer_parameterization(layer,args.device))
    #         total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement() + layer.parametrizations["weight"][0].lora_B.nelement()
    #         # tot = tot +1
    #         # print('layer ',tot,' lora   ',layer.parametrizations["weight"][0].lora_A.nelement(),layer.parametrizations["weight"][0].lora_B.nelement())
    #         layer.parametrizations["weight"][0].enabled = True
    # # tot = 0
    # for block in vqgan.decoder_blocks:
    #     for layer in [block.attn.qkv, block.attn.proj]:
    #         parametrize.register_parametrization(layer,"weight",linear_layer_parameterization(layer,args.device))
    #         # tot = tot +1
    #         # print('layer ',tot,' lora   ',layer.parametrizations["weight"][0].lora_A.nelement(),layer.parametrizations["weight"][0].lora_B.nelement())
    #         total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement() + layer.parametrizations["weight"][0].lora_B.nelement()
    #         layer.parametrizations["weight"][0].enabled = True

    # print('total para meter number ',total_parameters_lora)
    # assert False
    for _, p in vqgan.named_parameters():
        p.requires_grad = False
    # freeze_base_weights(vqgan)
    best_iou = 0.
    optimizer = torch.optim.SGD(vqgan.parameters(), lr=args.lr, weight_decay=0)
    scheduler = Scheduler(args.scheduler, args.epoch).select_scheduler(optimizer)
    begin_epoch = 1
    ckpt_path = os.path.join(model_save_path, 'ckpt.pth')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(os.path.join(model_save_path, 'ckpt.pth'),map_location=args.device)
        # state_dict = torch.load(, map_location=args.device)
        load_lora_state_dict(vqgan,checkpoint['visual_prompt_dict'])
        # vqgan.load_state_dict(checkpoint["visual_prompt_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        begin_epoch = checkpoint['epoch'] + 1  # 新的 epoch 数值
        best_iou = checkpoint['best_iou']  # 加载最佳 iou
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        scaler.load_state_dict(checkpoint['scaler_dict'])
        print(begin_epoch)
        print(best_iou)

       # print(_)

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(eg_save_path, exist_ok=True)

  #  print(VP.PromptGenerator.vqgan.decoder.conv_in.weight.requires_grad)

    print(f'We use the mode of {args.mode}.')
    print(f'We adopt the arrangement of {args.arr}.')

    print("*" * 50)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(args.device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(args.device)

    lr_list = []
    val_iou_list = []
    min_loss = 100.0
    # with torch.autograd.detect_anomaly():

    for epoch in range(begin_epoch, args.epoch + 1):
        epoch_loss = 0.0

        eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
        train_eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
        print("start_training round" + str(epoch))
        print("lr_rate: ", optimizer.param_groups[0]["lr"])
        lr_list.append(optimizer.param_groups[0]["lr"])
        vqgan.train()
        
        examples_save_path = eg_save_path + f'/{setting}_{epoch}/'
        print("start_val round" + str(epoch // 1))
        vqgan.eval()
        os.makedirs(examples_save_path, exist_ok=True)
        with open(os.path.join(examples_save_path, 'log.txt'), 'w') as log:
            log.write(str(args) + '\n')
        image_number = 0

        # Validation phase
        for i, data in enumerate(tqdm(dataloaders["val"])):
            len_dataloader = len(dataloaders["val"])
            ##my code
            support_features = data['support_features']
            query_img_features = data['query_img_features']
            support_features = support_features.to(args.device, dtype=torch.float32)
            query_img_features = query_img_features.to(args.device, dtype=torch.float32)
            ##end my code
            support_img, support_mask, query_img, query_mask, grid_stack = \
                data['support_imgs'], data['support_masks'], data['query_img'], data['query_mask'], data['grids']
            # vq_tokens = data['vq_tokens']
            support_img = support_img.to(args.device, dtype=torch.float32)
            support_mask = support_mask.to(args.device, dtype=torch.float32)
            query_img = query_img.to(args.device, dtype=torch.float32)
            query_mask = query_mask.to(args.device, dtype=torch.float32)
            grid_stack = grid_stack.to(args.device, dtype=torch.float32)
            # vq_tokens = vq_tokens.to(args.device,dtype=torch.long)

            _, canvas_pred_tokens, canvas_label = model_forward( grid_stack, 
                                query_img_features, support_features,args,imagenet_mean,imagenet_std)
            original_image_list = []
            for sub_pred_tokens in canvas_pred_tokens:
                sub_image_list, generated_result_list = _generate_result_for_canvas(args, vqgan.to(args.device),
                                                                                        sub_pred_tokens, canvas_label,
                                                                                        args.arr)
                original_image_list.append(sub_image_list)

            num_sub_image_lists = len(original_image_list)  # 16
            num_images_per_sub_image_list = len(original_image_list[0])  # 8
            average_image_list = [np.zeros_like(original_image_list[0][0], dtype=np.float32) for _ in range(num_images_per_sub_image_list)]
            for sub_image_list in original_image_list:
                for i, img in enumerate(sub_image_list):
                    average_image_list[i] += img
            for i in range(num_images_per_sub_image_list):
                average_image_list[i] /= num_sub_image_lists
            average_image_list = [np.clip(img, 0, 255).astype(np.uint8) for img in average_image_list]
            tep_list = average_image_list
            original_image_list = generated_result_list
            generated_result_list = average_image_list
            
            for index in range(len(original_image_list)):
                sub_image = generated_result_list[index][113:, 113:]
                sub_image = round_image(sub_image, [WHITE, BLACK], t=args.t)
                generated_result_list[index][113:, 113:] = sub_image

                original_image = round_image(original_image_list[index], [WHITE, BLACK])
                generated_result = generated_result_list[index]
                if args.task == 'detection':
                    generated_result = to_rectangle(generated_result)
                if args.save_examples:
                    Image.fromarray((generated_result.cpu().numpy()).astype(np.uint8)).save(
                        examples_save_path + f'generated_image_{image_number}.png')
                # if index == 0:
                #     image = TF.to_pil_image(generated_result_list[0])

                #     # # 保存图像
                #     image.save("result.jpg")
                # #    print(generated_result.shape)
                #     image = TF.to_pil_image((generated_result/255).permute(2,0,1))
                #     # # 保存图像
                #     image.save("final_result.jpg")
                #     image = TF.to_pil_image((original_image/255).permute(2,0,1))
                #     # # 保存图像
                #     image.save("original_image.jpg")

                current_metric = calculate_metric(args, original_image, generated_result, fg_color=WHITE, bg_color=BLACK)
                with open(os.path.join(examples_save_path, 'log.txt'), 'a') as log:
                    log.write(str(image_number) + '\t' + str(current_metric) + '\n')
                image_number += 1

                for i, j in current_metric.items():
                    eval_dict[i] += (j / len(val_dataset))


        print('val metric: {}'.format(eval_dict))
        with open(os.path.join(examples_save_path, 'log.txt'), 'a') as log:
            log.write('all\t' + str(eval_dict) + '\n')
        
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
    train(args)