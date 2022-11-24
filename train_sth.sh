### train AF-ResNet(RT=0.5)
CUDA_VISIBLE_DEVICES=0,1 python main.py something RGB \
     --arch_file AF_ResNet \
     --arch AF_resnet50 --num_segments 12 \
     --root_dataset 'path_dataset' \
     --path_backbone 'path_backbone' \
     --batch-size 32 --lr 0.01 --lr_steps 25 45 --epochs 55 \
     --gd 20 -j 12 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb \
     --rt_begin 10 --rt_end 20 --t0 1 --t_end 50 --lambda_rt 0.5 \
     --model_path 'models' \
     --rt 0.5 --round 1;



### train AF-ResNet-TSM(RT=0.5)
CUDA_VISIBLE_DEVICES=0,1 python main.py something RGB \
     --arch_file AF_ResNet \
     --arch AF_resnet50 --num_segments 12 \
     --root_dataset 'path_dataset' \
     --path_backbone 'path_backbone' \
     --batch-size 32 --lr 0.01 --lr_steps 25 45 --epochs 55 \
     --gd 20 -j 12 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb \
     --rt_begin 10 --rt_end 20 --t0 1 --t_end 50 --lambda_rt 0.5 \
     --model_path 'models' \
     --shift \
     --rt 0.5 --round 1;



### train AF-MobileNetv3-TSM(RT=0.5)
CUDA_VISIBLE_DEVICES=0,1 python main.py something RGB \
     --arch_file AF_MobileNetv3 \
     --arch AF_mobilenetv3 --num_segments 12 \
     --root_dataset 'path_dataset' \
     --path_backbone 'path_backbone' \
     --batch-size 32 --lr 0.01 --lr_steps 25 45 --epochs 55 \
     --gd 20 -j 12 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb \
     --rt_begin 10 --rt_end 20 --t0 1 --t_end 50 --lambda_rt 0.5 \
     --model_path 'models_mobilenet' \
     --shift \
     --rt 0.5 --round 1;