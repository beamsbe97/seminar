
echo "running feature_extractor_segmentation.py val"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python $HOME/seminar/Codes/tools/feature_extractor_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level val
echo "FINISHED feature_extractor_segmentation.py val"

echo "running feature_extractor_segmentation.py trn"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python $HOME/seminar/Codes/tools/feature_extractor_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level trn
echo "FINISHED feature_extractor_segmentation.py trn"

echo "running calculate_similariity.py val"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python $HOME/seminar/Codes/tools/calculate_similariity.py features_vit-laion2b_pixel-level val trn
echo "FINISHED calculate_similariity.py val"

echo "running calculate_similariity.py trn"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python $HOME/seminar/Codes/tools/calculate_similariity.py features_vit-laion2b_pixel-level trn trn
echo "FINISHED calculate_similariity.py trn"


echo "running calculate_pre_feature_for_query.py trn"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python -m Codes.tools.calculate_pre_feature_for_query features_vit-laion2b_pixel-level trn
echo "FINISHED calculate_pre_feature_for_query.py trn"

echo "running calculate_pre_feature_for_query.py val"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python -m Codes.tools.calculate_pre_feature_for_query features_vit-laion2b_pixel-level val
echo "FINISHED calculate_pre_feature_for_query.py val"

echo "running calculate_pre_feature_for_support.py trn"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python -m Codes.tools.calculate_pre_feature_for_support features_vit-laion2b_pixel-level trn
echo "FINISHED calculate_pre_feature_for_support.py trn"

echo "running calculate_pre_feature_for_support.py val"
/zfsstore/user/s4561341/.conda/envs/condenser/bin/python -m Codes.tools.calculate_pre_feature_for_support features_vit-laion2b_pixel-level trn
echo "FINISHED calculate_pre_feature_for_support.py val"

/zfsstore/user/s4561341/.conda/envs/condenser/bin/python -m Codes.train_vp_segmentation \
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
 --vq_ckpt_dir . \
 --save_base_dir Data/ \
 --simidx 16 \
 --dropout 0.25 \
 --choice Zero \
 --loss_mean 1 \
 --align_q 0 \
 --fold 3