python main.py \
--image_dir /kaggle/input/iu-xray-dataset/dataset/iu_xray/images/ \
--ann_path data/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 16 \
--epochs 4 \
--save_dir results \
--step_size 50 \
--gamma 0.1 \
--seed 9223 \
--mode train_encoder


#--ann_path /kaggle/input/iu-xray-dataset/dataset/iu_xray/annotation.json \
