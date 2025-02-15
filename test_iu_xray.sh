python main.py \
--image_dir /kaggle/input/iu-xray-dataset/dataset/iu_xray/images/ \
--ann_path /kaggle/input/iu-xray-dataset/dataset/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 16 \
--epochs 2 \
--save_dir results/iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9223 \
--resume /kaggle/input/r2genbasedata/R2Gen/results/r2gen_mode_base_best.pth \
--mode test

#--ann_path /kaggle/input/iu-xray-dataset/dataset/iu_xray/annotation.json \
#--ann_path data/annotation.json \

#--resume /kaggle/input/r2gen-base/R2Gen/results/r2gen_mode_base_best.pth \
#--resume /kaggle/input/r2genbasedata/R2Gen/results/r2gen_mode_base_best.pth \
#--resume /kaggle/working/R2Gen/results/r2gen_mode_base_best.pth \