import os.path
from tqdm import trange, tqdm
from evaluate.reasoning_dataloader import *
import torchvision
from evaluate.mae_utils import *
import argparse
from pathlib import Path
from evaluate.segmentation_utils import *
from PIL import Image
from torch.utils.data import DataLoader
from evaluate_detection.random_canvas_ds import RandomCanvasDataset4Train, RandomCanvasDataset4Val
import torch.multiprocessing as mp
from models.train_models import _generate_result_for_canvas, PGVP, Scheduler
from torch.cuda.amp import autocast, GradScaler
from evaluate_detection.box_ops import to_rectangle
import torchvision.transforms.functional as TF

# matplotlib.use('TkAgg')
from utils import Monitor


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
    parser.add_argument('--task', default='random_detection', choices=['segmentation', 'detection'])
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
    
    return parser


def train(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    # args.device = 'cuda:0'
    # print(args.sigma)
    setting = f'_lr_{args.lr}_task_{args.task}'

    model_save_path = f'{args.save_base_dir}/save_ours_ckpt/task_{args.task}_{args.choice}/fold_{args.fold}/simidx_{args.simidx}_model/sigma_{args.sigma}/{setting}'
    eg_save_path = f'{args.output_dir}/task_{args.task}_{args.choice}/fold_{args.fold}/simidx_{args.simidx}/sigma_{args.sigma}/{setting}'

    padding = 1
    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])
    mask_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), 3),
         torchvision.transforms.ToTensor()])

    train_dataset = {
        'pascal_det': RandomCanvasDataset4Train
    }[args.dataset_type](args.base_dir,simidx=args.simidx, fold=args.fold, split=args.split, image_transform=image_transform,
                         mask_transform=mask_transform, flipped_order=args.flip, purple=args.purple,
                         random=args.random, cluster=args.cluster, feature_name=args.feature_name,args=args,
                         percentage=args.percentage, seed=args.seed, mode=args.mode, arr=args.arr)

    val_dataset = {
        'pascal_det': RandomCanvasDataset4Val
    }[args.dataset_type](args.base_dir,simidx=args.simidx, fold=args.fold, split=args.split, image_transform=image_transform,
                         mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, random=args.random, cluster=args.cluster,
                         feature_name=args.feature_name, percentage=args.percentage, seed=args.seed, mode=args.mode,args=args,
                         arr=args.arr)
    print('number of val demonstation',args.simidx)
    print('length of val dataset: ', len(val_dataset))

    dataloaders = {}
    dataloaders['val'] = DataLoader(val_dataset, batch_size=args.batch_size // 2, shuffle=False,num_workers=4)
    dataloaders['train'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=4)

    print('train datalaoder: ', len(dataloaders['train']))
    print('val datalaoder: ', len(dataloaders['val']))

    print("load data over")

    # MAE_VQGAN model
    vqgan = prepare_model(args.ckpt, arch=args.mae_model, vq_ckpt_dir=args.vq_ckpt_dir)
    print(args.device)

    if args.vp_model == 'Prompt':
        print('load prompt generator')
        VP = PGVP(args=args, vqgan=vqgan.to(args.device), mode=args.mode, arr=args.arr)
    else:
        raise ValueError("Please check the mode of InMeMo!")
    scaler = GradScaler()

    VP.to(args.device)

    best_iou = 0.
    optimizer = torch.optim.SGD(VP.PromptGenerator.parameters(), lr=args.lr, weight_decay=0)
    scheduler = Scheduler(args.scheduler, args.epoch).select_scheduler(optimizer)
    begin_epoch = 1
    ckpt_path = os.path.join(model_save_path, 'ckpt.pth')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(os.path.join(model_save_path, 'ckpt.pth'),map_location=args.device)
        # state_dict = torch.load(, map_location=args.device)
        VP.PromptGenerator.load_state_dict(checkpoint["visual_prompt_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        begin_epoch = checkpoint['epoch'] + 1  # 新的 epoch 数值
        best_iou = checkpoint['best_iou']  # 加载最佳 iou
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        scaler.load_state_dict(checkpoint['scaler_dict'])
        print(begin_epoch)
        print(best_iou)
    
    for _, p in VP.PromptGenerator.named_parameters():
        p.requires_grad = True


    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(eg_save_path, exist_ok=True)
    print(f'We use the mode of {args.mode}.')
    print(f'We adopt the arrangement of {args.arr}.')
    if args.aug:
        print("This is the aug dataloader.")
    else:
        print("This is no aug dataloader.")

    print("*" * 50)

    lr_list = []
    val_iou_list = []
    min_loss = 100.0

    for epoch in range(begin_epoch, args.epoch + 1):
        epoch_loss = 0.0

        eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
        train_eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
        print("start_training round" + str(epoch))
        print("lr_rate: ", optimizer.param_groups[0]["lr"])
        lr_list.append(optimizer.param_groups[0]["lr"])
        VP.train()
        for i, data in enumerate(tqdm(dataloaders['train'])):
            len_dataloader = len(dataloaders['train'])
            support_img, support_mask, query_img, query_mask, grid_stack =\
                data['support_imgs'], data['support_masks'], data['query_img'], data['query_mask'], data['grids']
            support_features = data['support_features']
            # print("pre    ",support_features[0][0])
            query_img_features = data['query_img_features']
            support_features = support_features.to(args.device, dtype=torch.float32)
            query_img_features = query_img_features.to(args.device, dtype=torch.float32)
            support_img = support_img.to(args.device, dtype=torch.float32)
            support_mask = support_mask.to(args.device, dtype=torch.float32)
            query_img = query_img.to(args.device, dtype=torch.float32)
            query_mask = query_mask.to(args.device, dtype=torch.float32)
            grid_stack = grid_stack.to(args.device, dtype=torch.float32)

            optimizer.zero_grad()
            with autocast():
                loss, canvas_pred_tokens, canvas_label = VP(support_img, support_mask, query_img, query_mask, grid_stack, 
                                query_img_features,support_features)
                scaled_loss = scaler.scale(loss)
            if torch.isnan(loss):
                raise ValueError("nan error!")

            scaled_loss.backward()
            scaler.step(optimizer)
            # torch.nn.utils.clip_grad_norm_(VP.PromptGenerator.parameters(), max_norm=0.1)
            scaler.update()

            epoch_loss += loss.detach()
            print("now sum loss and avgloss and loss",epoch_loss,epoch_loss/(i+1),loss)

            original_image_list, generated_result_list = _generate_result_for_canvas(args, vqgan.to(args.device),
                                                                                     canvas_pred_tokens, canvas_label,
                                                                                     args.arr)
            for index in range(len(original_image_list)):
                sub_image = generated_result_list[index][113:, 113:]
                sub_image = round_image(sub_image, [WHITE, BLACK], t=args.t)
                generated_result_list[index][113:, 113:] = sub_image

                original_image = round_image(original_image_list[index], [WHITE, BLACK])
                generated_result = generated_result_list[index]
                if args.task == 'detection':
                    generated_result = to_rectangle(generated_result)
                # if index == 0:
                #     image = TF.to_pil_image(generated_result_list[0])
                #      # # 保存图像
                #     image.save("result.jpg")
                #     image = TF.to_pil_image((generated_result/255).permute(2,0,1))
                #     image.save("final_result.jpg")
                #     image = TF.to_pil_image((original_image/255).permute(2,0,1))
                #     image.save("original_image.jpg")
                current_metric = calculate_metric(args, original_image, torch.tensor(generated_result), fg_color=WHITE, bg_color=BLACK)
                
                for i, j in current_metric.items():
                    train_eval_dict[i] += (j / len(train_dataset))
        print('val metric: {}'.format(train_eval_dict))
        train_eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
        scheduler.step()

        average_epoch_loss = epoch_loss / len_dataloader
        if average_epoch_loss <= min_loss:
            min_loss = average_epoch_loss

        print('epoch: {}, loss: {:.2f}'.format(epoch, average_epoch_loss))
        print('min loss: {:.2f}'.format(min_loss))

        if epoch % 1 == 0:
            examples_save_path = eg_save_path + f'/{setting}_{epoch}/'
            print("start_val round" + str(epoch // 1))
            VP.eval()
            os.makedirs(examples_save_path, exist_ok=True)
            with open(os.path.join(examples_save_path, 'log.txt'), 'w') as log:
                log.write(str(args) + '\n')

            image_number = 0
            # Validation phase
            for i, data in enumerate(tqdm(dataloaders["val"])):
                len_dataloader = len(dataloaders["val"])
                support_features = data['support_features']
                query_img_features = data['query_img_features']
                support_features = support_features.to(args.device, dtype=torch.float32)
                query_img_features = query_img_features.to(args.device, dtype=torch.float32)
                support_img, support_mask, query_img, query_mask, grid_stack =\
                    data['support_imgs'], data['support_masks'], data['query_img'], data['query_mask'], data['grids']
                support_img = support_img.to(args.device, dtype=torch.float32)
                support_mask = support_mask.to(args.device, dtype=torch.float32)
                query_img = query_img.to(args.device, dtype=torch.float32)
                query_mask = query_mask.to(args.device, dtype=torch.float32)
                grid_stack = grid_stack.to(args.device, dtype=torch.float32)

                _, canvas_pred_tokens, canvas_label = VP(support_img, support_mask, query_img, query_mask, grid_stack, 
                                query_img_features,support_features)
                

                original_image_list, generated_result_list = _generate_result_for_canvas(args, vqgan.to(args.device),
                                                                                     canvas_pred_tokens, canvas_label,
                                                                                     args.arr)
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

                    current_metric = calculate_metric(args, original_image, torch.tensor(generated_result), fg_color=WHITE, bg_color=BLACK)
                    with open(os.path.join(examples_save_path, 'log.txt'), 'a') as log:
                        log.write(str(image_number) + '\t' + str(current_metric) + '\n')
                    image_number += 1

                    for i, j in current_metric.items():
                        eval_dict[i] += (j / len(val_dataset))

            print('val metric: {}'.format(eval_dict))
            with open(os.path.join(examples_save_path, 'log.txt'), 'a') as log:
                log.write('all\t' + str(eval_dict) + '\n')
            # Save CKPT
            if args.vp_model == 'Prompt':
                state_dict = {
                    "visual_prompt_dict": VP.PromptGenerator.state_dict(),
                    "optimizer_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_iou": best_iou,
                    "scheduler_dict": scheduler.state_dict(),
                    "scaler_dict": scaler.state_dict(),
                }
            if eval_dict['iou'] > best_iou:
                best_iou = eval_dict['iou']
                state_dict['best_iou'] = best_iou
                torch.save(state_dict, os.path.join(model_save_path, 'best.pth'))
            torch.save(state_dict, os.path.join(model_save_path, 'ckpt.pth'))
            print('best iou: ', best_iou)
            val_iou_list.append(eval_dict['iou'])
            print('lr list: ', lr_list)
            print('val iou list: ', val_iou_list)
            with open(os.path.join(examples_save_path, 'log.txt'), 'a') as log:
                log.write('best\t' + str(best_iou) + '\n')

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