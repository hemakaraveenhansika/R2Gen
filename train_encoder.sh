python main.py \
--image_dir /kaggle/input/iu-xray-dataset/dataset/iu_xray/images/ \
--ann_path /kaggle/input/iu-xray-dataset/dataset/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 16 \
--epochs 40 \
--save_dir results \
--step_size 50 \
--gamma 0.1 \
--seed 9225 \
--resume_contrastive_model /kaggle/input/r2gencontrastivemodel-v-1/R2Gen/results/contrastive_model_best.pth \
--mode train_encoder


#--ann_path /kaggle/input/iu-xray-dataset/dataset/iu_xray/annotation.json \
#--ann_path data/annotation.json \

#--resume_contrastive_model /kaggle/input/r2gen/R2Gen/results/contrastive_model_best.pth \
#--resume_contrastive_model /kaggle/input/r2gencontrastivemodel-v-1/R2Gen/results/contrastive_model_best.pth \