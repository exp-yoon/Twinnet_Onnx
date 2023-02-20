import argparse
import os
parser = argparse.ArgumentParser()

# training options
parser.add_argument("--is_train", type=int, default=1)
parser.add_argument("--skip_type", type=str, default='twin_diff', choices=('twin_diff'))
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument("--total_epoch", type=int, default=30)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--es_patience", type=int, default=15) # for early stopping
parser.add_argument("--class_weight", type=float, default=[0.4, 0.6]) # for loss function (class imbalance)

# learning schedule options
parser.add_argument("--lr_schedule_type", type=str, default="step", choices=("step", "auto"))
parser.add_argument("--learning_rate", type=float, default=1e-4) # initial learning rate
parser.add_argument("--lr_decay_rate", type=float, default=1) # lr_decay_rate 1 : constant learning rate
parser.add_argument("--lr_decay_patience", type=int, default=30)  # decay epoch

# optimizer options
parser.add_argument("--optimizer", type=str, default='adam', choices=('sgd', 'adam'))
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--epsilon", type=float, default=1e-7)

# dataset options
parser.add_argument("--input_size", type=int, default=(120, 120))
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--out_classes', type=int, default=2)
parser.add_argument("--tr_ratio", type=float, default=0.8) # to split training data set into training / valid

# data/save directory
parser.add_argument("--test_data_name", type=str, default="LB", choices=("euv"))
parser.add_argument("--train_data_dir", type=str, default="D:/ATI/DATA/EUV/train")
parser.add_argument("--test_data_dir", type=str, default="D:/ATI/DATA/EUV/test")
parser.add_argument("--test_gtruth_dir", type=str, default="D:/ATI/DATA/EUV/test_gtruth")
parser.add_argument("--save_dir", type=str, default="./RESULTS")
# Test options
parser.add_argument("--th_softmax", type=float, default=0.45)
parser.add_argument("--th_label", type=float, default=0)
parser.add_argument("--white_bar_w", type=int, default=10) # for saving results
parser.add_argument("--closing_size", type=int, default=3) # for saving results


def get_config():
    config = parser.parse_args()
    config.save_dir = f'{config.save_dir}/B{config.batch_size}_LR{format(config.learning_rate, ".0e")}' \
                      f'_EP{config.total_epoch}_DE{config.lr_decay_patience}_DR{config.lr_decay_rate}_P{config.es_patience}_cw{config.class_weight[0]}'
    config.save_test_img_dir = f'{config.save_dir}/{config.test_data_name}_sth{config.th_softmax}_lth{config.th_label}'

    if os.path.exists(config.save_dir) == 0:
        os.makedirs(config.save_dir)
    return config
