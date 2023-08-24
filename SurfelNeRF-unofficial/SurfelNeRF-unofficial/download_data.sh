## IMPORTANT  BE SURE YOU HAVE PERMISSION TO DOWNLOAD AND USE THE PROVIDED DATA
## https://github.com/Gymat/SurfelNeRF/tree/unofficial

# Pretrain
gdown https://drive.google.com/u/0/uc?id=1C4_G7UY69mR40AiawSbfS8x0OT8d5PCb
unzip preprocessed_data.zip 
# mv preprocessed_data/ /app/SurfelNeRF-unofficial/datasets/ 
mv preprocessed_data/* /app/SurfelNeRF-unofficial/datasets/preprocessed_data/

# export Scannet
gdown https://drive.google.com/u/0/uc?id=1Ci5yXQYmT-i_zadvU9Saq87ha2FB731s
unzip export_Scannet_test.zip 
mv export_Scannet_test/ /app/SurfelNeRF-unofficial/datasets/

# checkpoint
gdown https://drive.google.com/u/0/uc?id=1jTv-T2EOs7Y8iTDON3CQZ-NIPXnfIJZL
mkdir -p /app/SurfelNeRF-unofficial/checkpoints/ 
mv pretrained__checkpoints.pth /app/SurfelNeRF-unofficial/checkpoints/ 

rm -rf export_Scannet_test.zip preprocessed_data.zip 