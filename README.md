# Embracing Collaboration Over Competition:<br>Condensing Multiple Prompts for Visual In-Context Learning

## 1.Introduction

This repository contains the **PyTorch** implementation of our work at **CVPR 2025**:

> [**Embracing Collaboration Over Competition: Condensing Multiple Prompts for Visual In-Context Learning**](http://arxiv.org/abs/2412.14518).  Jinpeng Wang<sup>\*</sup>, Tianci Luo<sup>*</sup>, Yaohua Zha, Yan Feng, Ruisheng Luo, Bin Chen, Tao Dai, Long Chen, Yaowei Wang, Shu-Tao Xia.

![main](./Figure/main.png)
We devise **CONDENSER**, a lightweight external plugin that compresses relevant fine-grained context across multiple prompts. Optimized end-to-end with the backbone and an extra pre-alignment objective, **CONDENSER** ensures stability and accurate integration of contextual cues. 

In the following, we will guide you how to use this repository step by step. 🤗

## 2.Preparation

```
git clone https://anonymous.4open.science/r/CVPR25-Condenser.git
cd CVPR25-Condenser
```

### 2.1 Environment Setup

```
conda create -n condenser python=3.8 -y
conda activate condenser
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### 2.2 Download the image datasets and organize them properly

Download the Pascal-5i Dataset, Pascal VOC 2012 Dataset, Imagenet Dataset, MSCOCO Dataset.

The working directory is expected to be organized as below:

<details><summary>CVPR25-Condenser/ </summary>
<ul>
    <li>Codes/ <span style="color: grey; font-style: italic;">Store all code files in Codes/</li>
    <ul>
    	<li>.../</li>
    </ul>
    <li>Data/</li>
    <ul>
    	<li>coco/</li>
        <ul>
            <li>Coco_Trainlabel</li>
            <li>Coco_Vallabel</li>
            <li>trn2014</li>
            <li>val2014</li>
        </ul>
        <li>imagenet/</li>
        <ul>
            <li>test_data</li>
            <li>test_label</li>
            <li>train_data</li>
            <li>train_label</li>
        </ul>
    	<li>output</li>
    	<ul>
    		<li>logs/</li>
    		<li>visual_examples/</li>
    	</ul>
    	<li>pascal-5i/</li>
    	<li>save_ours_ckpt/</li>
    	<li>ckpt/</li>
    	<li>splits/</li>
      <li>weights</li>
      <ul>
          <li>vqgan/</li>
          <ul>
              <li>last.ckpt</li>
              <li>model.yaml</li>
          </ul>
          <li>checkpoint-1000.pth</li>
      </ul>
    </ul>
</ul>
</details>

Please from the [Visual Prompting](https://github.com/amirbar/visual_prompting) to prepare the model and download the `CVF 1000 epochs` pre-train checkpoint.

**We will use Foreground Segmentation as an example to illustrate the workflow of the code.**

## 3 Preprocess

### 3.1 Prompt Retriever

We select the best samples through feature space retrieval.

First, we extract features at the pixel-level using CLIP's visual encoder, separately for the val-set and train-set.

```
python Codes/tools/feature_extractor_folderwise_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level val
python Codes/tools/feature_extractor_folderwise_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level trn
```

Then, we calculate a similarity matrix using the features, and extract the top-50 similar prompt names.

```
python Codes/tools/calculate_similariity.py features_vit-laion2b_pixel-level val trn
python Codes/tools/calculate_similariity.py features_vit-laion2b_pixel-level trn trn
```

### 3.2  Preprocessing Features

We aim to preprocess the features so that they can be directly used as embeddings for visual prompts and queries.

```
python Codes/tools/calculate_pre_feature_for_query.py
python Codes/tools/calculate_pre_feature_for_support.py
```

## 4. Training and Inference

### 4.1 Training 

```
python3 Codes/train_vp_segmentation.py \
 --mode spimg_spmask \
 --output_dir Data/output/logs/ \
 --device cuda:0 \
 --base_dir Data/pascal-5i/ \
 --batch-size 16 \
 --lr 0.03 \
 --epoch 150 \
 --scheduler cosinewarm \
 --arr a1 \
 --vp-model Prompt \
 --p-eps 1 \
 --ckpt Data/weights/checkpoint-1000.pth \
 --vq_ckpt_dir Data/weights/vqgan \
 --save_base_dir Data/ \
 --simidx 16 \
 --dropout 0.25 \
 --choice Zero \
 --loss_mean 1 \
 --align_q 0 \
 --fold 3
```
- `<fold>`: fold-id of pascal-5i and coco-5i
- `<simidx>`: number of prompt pairs

1. Replace train_vp_segmentation.py with train_vp_detection.py to train for single object detection.

2. Replace train_vp_segmentation.py with train_vp_coloring.py, then replace --base_dir Data/pascal-5i/ with --base_dir Data/imagenet/ to train for coloring.

3. Change the value of simidx to determine the number of prompt pairs used during training.

The logs, model checkpoints will be generated under the `Data/output/logs/` and `Data/save_ours_ckpt/` folders, respectively. 

### 4.2 Inference

We provide the evaluation code for model checkpoints (if exist). 
The test command is as follows:

```
python3 Codes/val_vp_segmentation.py \
 --fold 1\
 --mode spimg_spmask\
 --output_dir Data/output/logs/\ 
 --device cuda:0\ 
 --base_dir Data/pascal-5i/\ 
 --batch-size 8\
 --lr 0.03\
 --epoch 150\
 --arr a1\
 --vp-model Prompt\
 --p-eps 1\
 --ckpt Data/weights/checkpoint-1000.pth\
 --vq_ckpt_dir Data/weights/vqgan\
 --save_base_dir Data/\
 --simidx 1\
 --dropout 0.25\
 --align_q 0 \
 --save_model_path SAVE_MODEL_PATH
```

1. Replace val_vp_segmentation.py with val_vp_detection.py to inference for single object detection.

2. Replace val_vp_segmentation.py with val_vp_coloring.py, then replace --base_dir Data/pascal-5i/ with --base_dir Data/imagenet/ to inference for coloring.

3. Change the value of simidx to determine the number of prompt pairs used during inference.

To facilitate the readers' implementation of inference, we have also designed a simple bash script for inference. To run it, navigate to the root directory of CVPR25_Condenser and execute the following command:

```
bash Codes/script/run01.sh
```

This will complete the relevant inference tasks.

## 5. Results

Download the checkpoint to the `Data/ckpt` path. Run the corresponding `.sh` file to achieve one-click execution and directly obtain the results shown in the table.

<table class="tg"><thead>
  <tr>
    <th class="tg-nrix" rowspan="2">Task (Metric)</th>
    <th class="tg-nrix" colspan="2" rowspan="2">Dataset</th>
    <th class="tg-nrix" colspan="4">K=1</th>
    <th class="tg-nrix" colspan="4">K=16</th>
  </tr>
  <tr>
    <th class="tg-nrix">Performance</th>
    <th class="tg-nrix">Log</th>
    <th class="tg-nrix">Checkpoint</th>
    <th class="tg-nrix">Script</th>
    <th class="tg-nrix">Performance</th>
    <th class="tg-nrix">Log</th>
    <th class="tg-nrix">Checkpoint</th>
    <th class="tg-nrix">Script</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="4">Segmentation (mIoU↑)</td>
    <td class="tg-nrix" rowspan="4">Pascal-5i</td>
    <td class="tg-nrix">Folder 0</td>
    <td class="tg-nrix">42.13</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_Zero_align_q0/fold_0/simidx_1">Seg_K_1_Fold_0_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1Xu-9IIkvqnCMWTV5Vw9H-XZkQrqeQr-d/view?usp=drive_link" target="_blank">Seg_K_1_Fold_0.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run01.sh">run01.sh</a></td>
    <td class="tg-nrix">45.53</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_Zero_align_q0/fold_0/simidx_16">Seg_K_16_Fold_0_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1uQKYqJ_30jbxIyY1BezSgdp41GbUUaG0/view?usp=drive_link" target="_blank">Seg_K_16_Fold_0.pth</a>
    <td class="tg-nrix"><a href="Codes/script/run02.sh">run02.sh</a></td>
    </td>
  </tr>
  <tr>
    <td class="tg-nrix">Folder 1</td>
    <td class="tg-nrix">50.31</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_Zero_align_q0/fold_1/simidx_1">Seg_K_1_Fold_1_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1qhQBfA9PhdHeDP_Ov27F7cVn6uMWxnCm/view?usp=drive_link" target="_blank">Seg_K_1_Fold_1.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run03.sh">run03.sh</a></td>
    <td class="tg-nrix">52.06</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_Zero_align_q0/fold_1/simidx_16">Seg_K_16_Fold_1_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1fr-58Q_6J1c1Rwh5VtPGMvdiwx6Al5Qh/view?usp=drive_link" target="_blank">Seg_K_16_Fold_1.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run04.sh">run04.sh</a></td>
  </tr>
  <tr>
    <td class="tg-nrix">Folder 2</td>
    <td class="tg-nrix">42.20</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_Zero_align_q0/fold_2/simidx_1">Seg_K_1_Fold_2_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1J76Za3-qAnI5-3zJQDelgS5qaWSmcgWk/view?usp=drive_link" target="_blank">Seg_K_1_Fold_2.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run05.sh">run05.sh</a></td>
    <td class="tg-nrix">44.33</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_Zero_align_q0/fold_2/simidx_16">Seg_K_16_Fold_2_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1RZh1DkSHc18wSETarEHYxq3drp6Zho9z/view?usp=drive_link" target="_blank">Seg_K_16_Fold_2.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run06.sh">run06.sh</a></td>
  </tr>
  <tr>
    <td class="tg-nrix">Folder 3</td>
    <td class="tg-nrix">41.90</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_Zero_align_q0/fold_3/simidx_1">Seg_K_1_Fold_3_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1_7jXuRtflwMusH1AUEEGbZlZqYmWxx8i/view?usp=drive_link" target="_blank">Seg_K_1_Fold_3.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run07.sh">run07.sh</a></td>
    <td class="tg-nrix">44.58</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_Zero_align_q0/fold_3/simidx_16">Seg_K_16_Fold_3_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1btEcQttj4OJSBaQuUjy3Edf9FID-CgCF/view?usp=drive_link" target="_blank">Seg_K_16_Fold_3.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run08.sh">run08.sh</a></td>
  </tr>
  <tr>
    <td class="tg-nrix">Detection (mIoU↑)</td>
    <td class="tg-nrix" colspan="2">Pascal VOC 2012</td>
    <td class="tg-nrix">43.22</td>
    <td class="tg-nrix"><a href="Codes/logs/task_detection_Zero_align_q0/fold_0/simidx_1">Det_K_1_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/11ftOY6uc-TjWSiZD9DUscp_mQ-Hh2AlP/view?usp=drive_link" target="_blank">Det_K_1.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run09.sh">run09.sh</a></td>
    <td class="tg-nrix">44.64</td>
    <td class="tg-nrix"><a href="Codes/logs/task_detection_Zero_align_q0/fold_0/simidx_16">Det_K_16_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/18Nz4fQ4Rtd85SaZFnQEga6dyYT3FnXSD/view?usp=drive_link" target="_blank">Det_K_16.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run10.sh">run10.sh</a></td>
  </tr>
  <tr>
    <td class="tg-nrix">Coloring (MSE↓)</td>
    <td class="tg-nrix" colspan="2">ImageNet-1K</td>
    <td class="tg-nrix">0.56</td>
    <td class="tg-nrix"><a href="Codes/logs/task_coloring_Zero_align_q0/fold_0/simidx_1">Col_K_1_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1dm8HX4ruq0emF9xuC5xa3mgfWKDqvpA1/view?usp=drive_link" target="_blank">Col_K_1.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run11.sh">run11.sh</a></td>
    <td class="tg-nrix">0.54</td>
    <td class="tg-nrix"><a href="Codes/logs/task_coloring_Zero_align_q0/fold_0/simidx_16">Col_K_16_Log</td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1jrTLiZxNPS5DD0j4D4WtN6NsMal2OcKc/view?usp=drive_link" target="_blank">Col_K_16.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run12.sh">run12.sh</a></td>
  </tr>
</tbody></table>

We have also open-sourced the experiment logs and checkpoints for the domain adaptation experiments. The experiments were pre-trained on Coco-5i and tested on Pascal-5i.

<table class="tg"><thead>
  <tr>
    <th class="tg-nrix" rowspan="2">Task (Metric)</th>
    <th class="tg-nrix" colspan="2" rowspan="2">Dataset</th>
    <th class="tg-nrix" colspan="4">K=1</th>
    <th class="tg-nrix" colspan="4">K=16</th>
  </tr>
  <tr>
    <th class="tg-nrix">Performance</th>
    <th class="tg-nrix">Log</th>
    <th class="tg-nrix">Checkpoint</th>
    <th class="tg-nrix">Script</th>
    <th class="tg-nrix">Performance</th>
    <th class="tg-nrix">Log</th>
    <th class="tg-nrix">Checkpoint</th>
    <th class="tg-nrix">Script</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="4">Segmentation (mIoU↑)</td>
    <td class="tg-nrix" rowspan="4">Coco-5i</td>
    <td class="tg-nrix">Folder 0</td>
    <td class="tg-nrix">40.39</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_0/simidx_1">Coco_K_1_Fold_0_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1fybnfYqYvnVVgmHDoK-8ddpJaqEDAAc0/view?usp=drive_link" target="_blank">Coco_K_1_Fold_0.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run13.sh">run13.sh</a></td>
    <td class="tg-nrix">40.37</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_0/simidx_16">Coco_K_16_Fold_0_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1FMc22XHfdTgPbfdgC5kTMBHNmrqBMdsr/view?usp=drive_link" target="_blank">Coco_K_16_Fold_0.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run14.sh">run14.sh</a></td>
  </tr>
  <tr>
    <td class="tg-nrix">Folder 1</td>
    <td class="tg-nrix">44.54</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_1/simidx_1">Coco_K_1_Fold_1_Log</a></a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1nhxG7u-_iW56X6JAfzFRDMG107mqnZGd/view?usp=drive_link" target="_blank">Coco_K_1_Fold_1.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run15.sh">run15.sh</a></td>
    <td class="tg-nrix">44.85</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_1/simidx_16">Coco_K_16_Fold_1_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1c3obklnxXrL2oSqkjzuHqbB1AFK2McK9/view?usp=drive_link" target="_blank">Coco_K_16_Fold_1.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run16.sh">run16.sh</a></td>
  </tr>
  <tr>
    <td class="tg-nrix">Folder 2</td>
    <td class="tg-nrix">40.23</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_2/simidx_1">Coco_K_1_Fold_2_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1FLoyE71yWdeoPWmWSDuF4TJov3-cunE6/view?usp=drive_link" target="_blank">Coco_K_1_Fold_2.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run17.sh">run17.sh</a></td>
    <td class="tg-nrix">41.03</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_2/simidx_16">Coco_K_16_Fold_2_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1QZ4FiQ47IMENilD2GDq0qB0pKoqqci7f/view?usp=drive_link" target="_blank">Coco_K_16_Fold_2.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run18.sh">run18.sh</a></td>
  </tr>
  <tr>
    <td class="tg-nrix">Folder 3</td>
    <td class="tg-nrix">36.33</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_3/simidx_1">Coco_K_1_Fold_3_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1umj_ZNn2wXaX4iSIQTT8mH35mCBIA8Yw/view?usp=drive_link" target="_blank">Coco_K_1_Fold_3.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run19.sh">run19.sh</a></td>
    <td class="tg-nrix">35.84</td>
    <td class="tg-nrix"><a href="Codes/logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_3/simidx_16">Coco_K_16_Fold_3_Log</a></td>
    <td class="tg-nrix">  <a href="https://drive.google.com/file/d/1EZHUx-TJhqTk94Pl8fJG8JyOt3uMlq2H/view?usp=drive_link" target="_blank">Coco_K_16_Fold_3.pth</a>
    </td>
    <td class="tg-nrix"><a href="Codes/script/run20.sh">run20.sh</a></td>
  </tr>
</tbody></table>


## 6. Visual Examples

We have provided visualized results for some test cases to help readers with an intuitive understanding.

![Seg_Examples](./Figure/Seg_Examples.png)

## 7. Acknowledgments

This code is based on our previous work [InMeMo](https://github.com/Jackieam/InMeMo). 

We are also grateful for other teams for open-sourcing codes that inspire our work, including [Visual Prompting](https://github.com/amirbar/visual_prompting), [visual_prompt_retrieval](https://github.com/ZhangYuanhan-AI/visual_prompt_retrieval), [timm](https://github.com/huggingface/pytorch-image-models), [ILM-VP](https://github.com/OPTML-Group/ILM-VP).

## 8. Contact

If you have any question, you can raise an issue or email Jinpeng Wang (wjp20@mails.tsinghua.edu.cn). We will reply you soon.




















