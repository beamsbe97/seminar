
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
