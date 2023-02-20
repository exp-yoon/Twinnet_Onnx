from glob import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def split_train_valid_data(data_dir, split_ratio=0.9, ext='.png'):
    """
    :param data_dir: training data directory
    :param split_ratio: ratio for training set
    :param ext: file extension
    :return: trainin set glob list, validation set glob list
    """
    data_list = np.array(glob(os.path.join(data_dir, '*', f'*{ext}')))
    data_list = np.array(glob(os.path.join(data_dir, f'*{ext}'))) if len(data_list) == 0 else data_list
    indices = np.arange(len(data_list))

    np.random.seed(0)
    np.random.shuffle(indices)

    train_list = data_list[indices[:int(len(indices) * split_ratio)]]
    valid_list = data_list[indices[int(len(indices) * split_ratio):]]
    return train_list, valid_list


def get_data_list(data_dir, ext='.png'):
    """
    :param data_dir: directory to load data
    :param ext: file extension
    :return: data glob list
    """
    data_list = np.array(glob(os.path.join(data_dir, '*', f'*{ext}')))
    data_list = np.array(glob(os.path.join(data_dir, f'*{ext}'))) if len(data_list) == 0 else data_list
    return data_list


def mkdirs_to_save_test_results(save_dir):
    if os.path.exists(save_dir) == 0:
        os.makedirs(f'{save_dir}/Correct')
        os.makedirs(f'{save_dir}/FP')
        os.makedirs(f'{save_dir}/FN')


def th_label_thresholding(input, threshold_value):
    """"
    thresholding by the size of segement
    :param input: prediction of model
    :param threshold_value: the size of segment
    :return: revised prediction by thresholding
    """
    # stats : x, y, width, height, pixel_num
    output = input[:]
    for b in range(len(input)):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.array(input[b], dtype=np.uint8))
        delete_seg_pos = np.where(stats[:, -1] <= threshold_value)[0]

        for d in range(len(delete_seg_pos)):
            delete_label_pos = np.where(labels == delete_seg_pos[d])
            output[b,delete_label_pos[0], delete_label_pos[1]] = 0
    return output


def imwrite_roi_test_results(gtruth, pred_fin, bitwise_and_label_pred, combine_img, save_dir, file_name, test_stats):
    for b in range(len(combine_img)):
        if np.sum(gtruth[b]) > 0:
            test_stats['num_defect'] += 1
            if np.sum(bitwise_and_label_pred[b]) > 0:
                test_stats['correct'] += 1
                cv2.imwrite(f'{save_dir}/Correct/{file_name}_{b}.png', combine_img[b])
            else:
                test_stats['fn'] += 1
                cv2.imwrite(f'{save_dir}/FN/{file_name}_{b}.png', combine_img[b])
        else:
            test_stats['num_normal'] += 1
            if np.sum(pred_fin[b]) == 0:
                test_stats['correct'] += 1
            else:
                test_stats['fp'] += 1
                cv2.imwrite(f'{save_dir}/FP/{file_name}_{b}.png', combine_img[b])

    return test_stats


def write_args(config):
    args_keys = list(config.__dict__.keys())
    args_values = list(config.__dict__.values())
    f = open(f'{config.save_dir}/args.txt', 'a')
    for i in range(len(args_keys)):
        f.write(f'{args_keys[i]} : {args_values[i]} \n')
    f.write(f"{'-'*200}\n")
    f.close()


def history_imwrite(history, save_dir):
    plt.figure(figsize=(90, 30))
    plt.rcParams.update({'font.size': 80})
    plt.plot(history.history['loss'], linewidth=6)
    plt.plot(history.history['val_loss'], linewidth=6)
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.xlabel('Loss')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig(f'{save_dir}/training_graph.png')


def get_output_shape(input_size, num_skip_layers):
    output_shape = []
    for iter in range(num_skip_layers):
        h = int(np.ceil(input_size[0] / (2 ** iter)))
        w = int(np.ceil(input_size[1] / (2 ** iter)))
        output_shape.append([h, w])
    return output_shape


#####################################################################################################################
# 아래의 get_segment_test_statistics 는 euv 테스트 데이터 셋에만 적용 가능함

def get_segment_test_statistics(gtruth, pred, closing_size, test_statistics, insp_type='defect'):
    kernel = np.ones((closing_size, closing_size), np.uint8)
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)

    # stats : x, y, width, height, pixel_num
    glabels, _, gstats, gcentroids = cv2.connectedComponentsWithStats(gtruth)
    plabels, _, pstats, _ = cv2.connectedComponentsWithStats(pred)

    if insp_type == "defect":
        test_statistics['num_defect'] += len(gstats) - 1  # background 제외
        tmp_correct = 0
        for st in range(1, len(gstats)):
            miny, maxy, minx, maxx = gstats[st, 1], gstats[st, 1] + gstats[st, 3], gstats[st, 0], gstats[st, 0] + gstats[st, 2]
            product_img = np.uint8(gtruth[miny:maxy, minx:maxx]) * np.uint8(pred[miny:maxy, minx:maxx])
            if np.sum(product_img) > 0:
                test_statistics['num_correct'] += 1
                tmp_correct += 1
            else:
                test_statistics['num_fn'] += 1
    else:
        test_statistics['num_fp'] += plabels-1

    return test_statistics


def get_euv_test_ds_inform(config, ext='.png'):
    pred_dir = config.save_test_img_dir
    gtruth_save_dir = config.test_gtruth_dir

    if os.path.exists(gtruth_save_dir) == 0:
        os.makedirs(gtruth_save_dir)

    if len(glob(os.path.join(gtruth_save_dir, f'*.{ext}'))) > 0:
        is_write_gtruth = False
    else:
        is_write_gtruth = True

    test_chip_name = ['chip1-2_ch2', 'chip1-2_ch6', 'chip1-2_ch10', 'chip1-2_ch14', 'chip1-2_ch18', 'chip1-2_ch22']
    test_chip_hw = [[69261, 2900]]  # EUV : [69261, 2900]

    ds_inform = {'roi_h': config.input_size[0], 'roi_w': config.input_size[0],
                 'closing_size': config.closing_size, 'overlap': 25, 'bar_w': config.white_bar_w,
                 'is_write_gtruth': is_write_gtruth, 'test_chip_name': test_chip_name,
                 'test_chip_hw': test_chip_hw, 'pred_dir': pred_dir, 'gtruth_save_dir': gtruth_save_dir}

    return ds_inform


def get_full_pred_image(iter_p, pred_dir, chip_hw, is_write_gtruth, roi_h, roi_w, overlap, bar_w, root_name, insp_type):
    """
    start_y : (img_h-overlap)*rh - (img_h-overlap-1)
    end_y : (img_h-overlap)*rh - (img_h-overlap-1) + (img_h-1)
    start_x : (img_w-overlap)*rw - (img_w-overlap-1)
    end_x : (img_w-overlap)*rw - (img_w-overlap-1) + (img_w-1)
    """
    gtruth = np.zeros(((chip_hw[iter_p][0], chip_hw[iter_p][1])), dtype=np.uint8) if is_write_gtruth else None
    pred = np.zeros(((chip_hw[iter_p][0], chip_hw[iter_p][1])), dtype=np.uint8)

    for rh in range(1, np.int32(np.ceil((chip_hw[iter_p][0] - 1) / (roi_h - overlap))) + 1):
        for rw in range(1, np.int32(np.ceil((chip_hw[iter_p][1] - 1) / (roi_w - overlap))) + 1):
            name = f'{root_name}_rh{rh}_rw{rw}.png'
            img = get_pred_roi(pred_dir, name, insp_type)

            if img is None:
                continue

            gtru_roi = np.uint8(img[:, 2 * roi_w + 2 * bar_w:3 * roi_w + 2 * bar_w] >0)
            pred_roi = np.uint8(img[:, 3 * roi_w + 3 * bar_w:] > 0)

            start_y = np.minimum((rh - 1) * (roi_h - overlap), chip_hw[iter_p][0] - roi_h)
            end_y = np.minimum(start_y + roi_h, chip_hw[iter_p][0])
            start_x = np.minimum((rw - 1) * (roi_w - overlap), chip_hw[iter_p][1] - roi_w)
            end_x = np.minimum(start_x + roi_w, chip_hw[iter_p][1])

            if is_write_gtruth:
                gtru_tmp = gtruth[start_y:end_y, start_x:end_x]
                gtru_or = np.uint8(np.bitwise_or(gtru_tmp, gtru_roi))
                gtruth[start_y:end_y, start_x:end_x] = gtru_or

            pred_tmp = pred[start_y:end_y, start_x:end_x]
            pred_or = np.uint8(np.bitwise_or(pred_tmp, pred_roi))
            pred[start_y:end_y, start_x:end_x] = pred_or

    return pred, gtruth


def get_pred_roi(root_dir, name, insp_type):
    if insp_type == 'defect':
        if os.path.isfile(os.path.join(root_dir, 'Correct', name)) == 1:
            img = cv2.imread(os.path.join(root_dir, 'Correct', name), cv2.IMREAD_GRAYSCALE)
        elif os.path.isfile(os.path.join(root_dir, 'FN', name)) == 1:
            img = cv2.imread(os.path.join(root_dir, 'FN', name), cv2.IMREAD_GRAYSCALE)
        else:
            img = None
    else:
        if os.path.isfile(os.path.join(root_dir, 'FP', name)) == 1:
            img = cv2.imread(os.path.join(root_dir, 'FP', name), cv2.IMREAD_GRAYSCALE)
        else:
            img = None
    return img
