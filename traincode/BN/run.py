import os



os.system('python main.py --data_train DIV2K_train --data_val DIV2K_valid_20 --resume --only_MV --save_MV \
						  --scale 2 --lr_patch_size 48 --rgb_range 1.0 \
						  --model SRMD_BN --trained_model model_dir/SRMD_BN_2_bestPSNR.pth.tar')

