# Embracing Collaboration Over Competition:<br>Condensing Multiple Prompts for Visual In-Context Learning

## 1.Introduction

This repository contains the **PyTorch** implementation of our work at **CVPR 2025**:

> [**Embracing Collaboration Over Competition: Condensing Multiple Prompts for Visual In-Context Learning**](http://arxiv.org/abs/2412.14518).  Jinpeng Wang, Tianci Luo, Yaohua Zha, Yan Feng, Ruisheng Luo, Bin Chen, Tao Dai, Long Chen, Yaowei Wang, Shu-Tao Xia.

![main](./Figure/main.png)
We devise ${CONDENSER}$, a lightweight external plugin that compresses relevant fine-grained context across multiple prompts. Optimized end-to-end with the backbone and an extra pre-alignment objective, ${CONDENSER}$ ensures stability and accurate integration of contextual cues. 

In the following, we will guide you how to use this repository step by step. 🤗

## 2.Preparation

```
git clone https://anonymous.4open.science/r/VICL-Condenser.git
cd VICL-Condenser
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

<details><summary>VICL-Condenser/</summary>
<ul>
    <li>Codes/</li>
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

```
python Codes/tools/feature_extractor_folderwise_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level val

python Codes/tools/feature_extractor_folderwise_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level trn

python Codes/tools/calculate_similariity.py features_vit-laion2b_pixel-level val trn
python Codes/tools/calculate_similariity.py features_vit-laion2b_pixel-level trn trn
```

### 3.2  Preprocessing Features and Codebook 

```
python Codes/tools/calculate_pre_feature_for_query.py
python Codes/tools/calculate_pre_feature_for_support.py

python Codes/tools/feature_extractor_folderwise_segmentation.py
```

## 4. Training and Inference

### 4.1 Training 

```
python3 Codes/train_vp_segmentation.py --mode spimg_spmask --output_dir data/output/logs/ --device cuda:0 --base_dir data/pascal-5i/ --batch-size 16 --lr 0.03 --epoch 150 --scheduler cosinewarm --arr a1 --vp-model Prompt --p-eps 1 --ckpt data/weights/checkpoint-1000.pth --vq_ckpt_dir data/weights/vqgan --save_base_dir data/ --simidx 16 --dropout 0.25 --choice Zero --loss_mean 1 --align_q 0 --fold 3
```

- `<simidx>`: number of prompt pairs

The logs, model checkpoints will be generated under the `data/output/logs/` and `data/save_ours_ckpt/` folders, respectively. 

### 4.2 Inference

We provide the evaluation code for model checkpoints (if exist). 
The test command is as follows:

```
python3 Codes/val_vp_segmentation.py --fold 1 --mode spimg_spmask --output_dir data/output/logs/ --device cuda:0 --base_dir data/pascal-5i/ --batch-size 8 --lr 0.03 --epoch 150 --arr a1 --vp-model Prompt --p-eps 1 --ckpt VisualICL/weights/checkpoint-1000.pth --vq_ckpt_dir VisualICL/weights/vqgan --save_base_dir VisualICL/  --simidx 1  --dropout 0.25 --save_model_path SAVE_MODEL_PATH
```

## 5. Results

<style type="text/css">
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
    padding: 8px;
  }
  th, td {
    text-align: center;
  }
</style>
<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th rowspan="2">K</th>
      <th rowspan="2">Type</th>
      <th colspan="4">Fold</th>
      <th rowspan="2">Mean</th>
    </tr>
    <tr>
      <th>Fold-0</th>
      <th>Fold-1</th>
      <th>Fold-2</th>
      <th>Fold-3</th>
    </tr>
  </thead>
  <tbody>
    <!-- Pascal-5i -->
    <tr>
      <td rowspan="6">Pascal-5i</td>
      <td>K=1</td>
      <td>Metric(IoU)</td>
      <td>42.13</td>
      <td>50.31</td>
      <td>42.20</td>
      <td>41.90</td>
      <td>44.14</td>
    </tr>
    <tr>
      <td>-</td>
      <td>CKPT</td>
      <td>ckpt</td>
      <td>ckpt</td>
      <td>ckpt</td>
      <td>ckpt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>LOG</td>
      <td><a href="logs/task_segmentation_Zero_align_q0/fold_0/simidx_1">Seg_K_1_Fold_0_Log</a></td>
      <td><a href="logs/task_segmentation_Zero_align_q0/fold_1/simidx_1">Seg_K_1_Fold_1_Log</a></td>
      <td><a href="logs/task_segmentation_Zero_align_q0/fold_2/simidx_1">Seg_K_1_Fold_2_Log</a></td>
      <td><a href="logs/task_segmentation_Zero_align_q0/fold_3/simidx_1">Seg_K_1_Fold_3_Log</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>K=16</td>
      <td>Metric(IoU)</td>
      <td>45.53</td>
      <td>52.06</td>
      <td>44.33</td>
      <td>44.58</td>
      <td>46.63</td>
    </tr>
    <tr>
      <td>-</td>
      <td>CKPT</td>
      <td>ckpt</td>
      <td>ckpt</td>
      <td>ckpt</td>
      <td>ckpt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>-</td>
      <td>LOG</td>
      <td><a href="logs/task_segmentation_Zero_align_q0/fold_0/simidx_16">Seg_K_16_Fold_0_Log</a></td>
      <td><a href="logs/task_segmentation_Zero_align_q0/fold_1/simidx_16">Seg_K_16_Fold_1_Log</a></td>
      <td><a href="logs/task_segmentation_Zero_align_q0/fold_2/simidx_16">Seg_K_16_Fold_2_Log</a></td>
      <td><a href="logs/task_segmentation_Zero_align_q0/fold_3/simidx_16">Seg_K_16_Fold_3_Log</a></td>
      <td>-</td>
    </tr>
     <!-- MSCOCO \n (domain adaptation) -->
<tr>
  <td rowspan="6">MSCOCO</td>
  <td>K=1</td>
  <td>Metric(IoU)</td>
  <td>40.39</td>
  <td>44.54</td>
  <td>40.23</td>
  <td>36.33</td>
  <td>40.37</td>
</tr>
<tr>
  <td>-</td>
  <td>CKPT</td>
  <td>ckpt</td>
  <td>ckpt</td>
  <td>ckpt</td>
  <td>ckpt</td>
  <td>-</td>
</tr>
<tr>
  <td>-</td>
  <td>LOG</td>
  <td><a href="logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_0/simidx_1">Seg_Coco_K_1_Fold_0_Log</a></td>
      <td><a href="logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_1/simidx_1">Seg_Coco_K_1_Fold_1_Log</a></td>
      <td><a href="logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_2/simidx_1">Seg_Coco_K_1_Fold_2_Log</a></td>
      <td><a href="logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_3/simidx_1">Seg_Coco_K_1_Fold_3_Log</a></td>
      <td>-</td>
</tr>
<tr>
  <td>K=16</td>
  <td>Metric(IoU)</td>
  <td>40.37</td>
  <td>44.85</td>
  <td>41.03</td>
  <td>35.84</td>
  <td>40.52</td>
</tr>
<tr>
  <td>-</td>
  <td>CKPT</td>
  <td>ckpt</td>
  <td>ckpt</td>
  <td>ckpt</td>
  <td>ckpt</td>
  <td>-</td>
</tr>
<tr>
  <td>-</td>
  <td>LOG</td>
    <td><a href="logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_0/simidx_16">Seg_Coco_K_16_Fold_0_Log</a></td>
      <td><a href="logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_1/simidx_16">Seg_Coco_K_16_Fold_1_Log</a></td>
      <td><a href="logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_2/simidx_16">Seg_Coco_K_16_Fold_2_Log</a></td>
      <td><a href="logs/task_segmentation_coco_Zero_G_copy_another_False_G_only_div_False_align_s1_align_q0_loss_mean1/fold_3/simidx_16">Seg_Coco_K_16_Fold_3_Log</a></td>
      <td>-</td>
</tr><!-- Pascal-VOC -->
<tr>
  <td rowspan="6">Pascal-VOC</td>
  <td>K=1</td>
  <td>Metric(IoU)</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>43.22</td>
</tr>
<tr>
  <td>-</td>
  <td>CKPT</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>ckpt</td>
</tr>
<tr>
  <td>-</td>
  <td>LOG</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td><a href="logs/task_detection_Zero_align_q0/fold_0/simidx_1">Det_K_1_Log</a></td>
</tr>
<tr>
  <td>K=16</td>
  <td>Metric(IoU)</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>44.64</td>
</tr>
<tr>
  <td>-</td>
  <td>CKPT</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>ckpt</td>
</tr>
<tr>
  <td>-</td>
  <td>LOG</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td><a href="logs/task_detection_Zero_align_q0/fold_0/simidx_16">Det_K_16_Log</a></td>
</tr><!-- Imagenet -->
<tr>
  <td rowspan="6">Imagenet</td>
  <td>K=1</td>
  <td>Metric(MSE)</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>0.56</td>
</tr>
<tr>
  <td>-</td>
  <td>CKPT</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>ckpt</td>
</tr>
<tr>
  <td>-</td>
  <td>LOG</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td><a href="logs/task_coloring_Zero_align_q0/fold_0/simidx_1">Col_K_1_Log</a></td>
</tr>
<tr>
  <td>K=16</td>
  <td>Metric(MSE)</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>0.54</td>
</tr>
<tr>
  <td>-</td>
  <td>CKPT</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>ckpt</td>
</tr>
<tr>
  <td>-</td>
  <td>LOG</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td><a href="logs/task_coloring_Zero_align_q0/fold_0/simidx_16">Col_K_16_Log</a></td>
</tr>  
    </tbody>
</table>

## 6. Visual Examples

![Seg_Examples](./Figure/Seg_Examples.png)

## 7. Acknowledgments

This code is based on our previous work [InMeMo](https://github.com/Jackieam/InMeMo). 

We are also grateful for other teams for open-sourcing codes that inspire our work, including [Visual Prompting](https://github.com/amirbar/visual_prompting), [visual_prompt_retrieval](https://github.com/ZhangYuanhan-AI/visual_prompt_retrieval), [timm](https://github.com/huggingface/pytorch-image-models), [ILM-VP](https://github.com/OPTML-Group/ILM-VP).

## 8. Contact

If you have any question, you can raise an issue or email Jinpeng Wang (wjp20@mails.tsinghua.edu.cn). We will reply you soon.




















