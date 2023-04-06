python train.py \
-n C-VTON-VITON-HD \
--dataset vitonHD \
--batch_size 1 \
--add_vgg_loss \
--lambda_vgg 10 \
--add_d_loss \
--add_cd_loss \
--add_pd_loss \
--no_labelmix \
--img_size 64 \
--segmentation densepose \
--transform_cloth \
--bpgm_id 256_26_3_viton \
--gpu_ids 0


python train.py -n C-VTON-VITON-HD --dataroot .\data\vitonHD --dataset vitonHD --batch_size 1 --add_vgg_loss --lambda_vgg 10 --add_d_loss --add_cd_loss --add_pd_loss --no_labelmix --img_size 64 --segmentation densepose --transform_cloth --bpgm_id 256_26_3_viton --gpu_ids 0