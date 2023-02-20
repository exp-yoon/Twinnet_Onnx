import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from config import get_config
from model import twinnetworks
from data import pairsetloader
from tqdm import tqdm
from customFn import *
import onnxruntime
import time
from datetime import datetime

def model_type(config):
    skip_type = config.skip_type
    if skip_type.find("twin_diff") == 0:
        my_model = twinnetworks.TwinDiffSkip(config)
    else:
        raise("check your config - skip_type")
    return my_model


def optimizer_type(config):
    opt_type = config.optimizer
    if opt_type.find("adam") >= 0:
        print("ADAM Optimizer")
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=config.betas[0],
                                             beta_2=config.betas[1], epsilon=config.epsilon, name="Adam")
    elif opt_type.find("sgd") >= 0:
        print("SGD Optimizer")
        optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=config.momentum)
    else:
        raise ("check your config - optimizer")

    return optimizer


def lr_schedule_type(lr_scheduler_type):
    if lr_scheduler_type.find("auto") >= 0:
        print("ReduceLROnPlateau lr_scheduler")
        callback_learning_schedule = callbacks_lr_ReduceLROnPlateau(config=config, monitor=config.callback_monitor)
    elif lr_scheduler_type.find("step") >= 0:
        print("Step lr_scheduler")
        callback_learning_schedule = CustomLearningRateScheduler(lr_scheduler, config)
    else:
        raise ("check your config - learning scheduler type")
    return callback_learning_schedule


def train():
    # Get dataloader
    train_list, valid_list = split_train_valid_data(config.train_data_dir, config.tr_ratio)
    train_ds = pairsetloader.PairSetDataLoader(train_list, config.batch_size, True)
    valid_ds = pairsetloader.PairSetDataLoader(valid_list, config.batch_size, False)
    print("train : ", len(train_ds), "valid : ", len(valid_ds))

    # Get Model
    my_model = model_type(config)

    # Set optimizer
    optimizer = optimizer_type(config)

    # call-backs
    callback_learning_schedule = lr_schedule_type(config.lr_schedule_type)
    callback_tensorboard = callbacks_tensorboard(config)
    callback_custom = CustomCallback(config)

    # model summary
    my_model.build(input_shape=[(None, config.input_size[0], config.input_size[1], config.in_channels),
                                (None, config.input_size[0], config.input_size[1], config.in_channels)])
    my_model.summary()

    # compile
    my_model.compile(optimizer=optimizer)  # 내부적으로 custom loss and accuracy 사용

    # training
    history = my_model.fit(train_ds, epochs=config.total_epoch, verbose=1, validation_data=valid_ds,
                           callbacks=[callback_learning_schedule, callback_tensorboard, callback_custom])

    my_model.save(f'{config.save_dir}/fin/', save_format='tf')
    history_imwrite(history, save_dir=config.save_dir)


def load_model(config):
    #trained_model = tf.keras.models.load_model(f'{config.save_dir}/fin/')
    trained_model = model_type(config)
    latest = tf.train.latest_checkpoint(f"{config.save_dir}/ckpt")
    trained_model.load_weights(latest)
    print(trained_model.inputs)
    return trained_model


def roi_based_test():
    test_data_list = get_data_list(config.test_data_dir)
    test_ds = pairsetloader.PairSetDataLoader(test_data_list, 1, False)
    trained_model = load_model(config)

    test_statistics = {'num_defect': 0, 'num_normal': 0, 'correct': 0, 'fp':0, 'fn': 0}
    mkdirs_to_save_test_results(config.save_test_img_dir)

    for idx, (test_data, test_label) in enumerate(tqdm(test_ds, desc="test process ")):
        # Load test data set
        ref_img, insp_img = np.squeeze(test_data[0][0].numpy(), axis=-1), np.squeeze(test_data[1][0].numpy(), axis=-1)
        label_img = np.uint8(np.squeeze(test_label[0].numpy(), axis=-1))
        file_name = test_ds.__get_file_name__(idx)[0]
        bar_w = np.ones((config.input_size[0], config.white_bar_w)) # to save combined image

        # Get prediction results from trained model
        model_pred = trained_model.predict(test_data)
        model_softmax = tf.nn.softmax(model_pred, axis=-1).numpy() # (1, H, W, 2)

        # Thresholding by softmax value and segment size
        model_pred_th_softmax = np.uint8(model_softmax[:, :, :, 1] >= config.th_softmax) # (1, H, W)
        model_pred_fin = np.uint8(th_label_thresholding(model_pred_th_softmax, config.th_label)) # (H, W)
        bitwise_and_label_pred = label_img * model_pred_fin

        combine_img = 255 * np.concatenate((ref_img, bar_w, insp_img, bar_w, label_img, bar_w, model_pred_fin), axis=-1)
        test_statistics = imwrite_roi_test_results(label_img, model_pred_fin, bitwise_and_label_pred, combine_img,
                                               config.save_test_img_dir, file_name, test_statistics)

    return test_statistics


def segment_based_test(ds_inform):
    test_statistics = {'num_defect': 0, 'num_correct': 0, 'num_fn': 0, 'num_fp':0}

    for te in range(len(ds_inform['test_chip_name'])):
        for iter_p in range(len(ds_inform['test_chip_hw'])):
            name = f"{ds_inform['test_chip_name'][te]}"

            pred_defect, gtruth = get_full_pred_image(iter_p, ds_inform['pred_dir'], ds_inform['test_chip_hw'],
                                                     ds_inform['is_write_gtruth'], ds_inform['roi_h'], ds_inform['roi_w'],
                                                     ds_inform['overlap'], ds_inform['bar_w'], name, insp_type="defect")

            pred_normal, gtruth_normal = get_full_pred_image(iter_p, ds_inform['pred_dir'], ds_inform['test_chip_hw'],
                                                 ds_inform['is_write_gtruth'], ds_inform['roi_h'], ds_inform['roi_w'],
                                                 ds_inform['overlap'], ds_inform['bar_w'], name, insp_type="normal")

            if ds_inform['is_write_gtruth']:
                cv2.imwrite(os.path.join(ds_inform['gtruth_save_dir'], f"{ds_inform['test_chip_name'][te]}_label.png"), gtruth * 255)
            else:
                gtruth = cv2.imread(os.path.join(ds_inform['gtruth_save_dir'], f"{ds_inform['test_chip_name'][te]}_label.png"), cv2.IMREAD_GRAYSCALE)
                gtruth = np.uint8(gtruth > 0)

            test_statistics = get_segment_test_statistics(gtruth, pred_defect, ds_inform['closing_size'], test_statistics, insp_type='defect')
            test_statistics = get_segment_test_statistics(gtruth_normal, pred_normal, ds_inform['closing_size'], test_statistics, insp_type='normal')

    return test_statistics

def test_onnx():

    #tensorrt 공급자 사용
    provider = [("TensorrtExecutionProvider",{'trt_engine_cache_enable' : True, 'trt_engine_cache_path' : './cache_newmodel'})]
    ort_model = onnxruntime.InferenceSession('model_fix.onnx',providers=provider) #사용할 Onnx 모델 파일 

    test_data_list = get_data_list(config.test_data_dir)
    test_ds = pairsetloader.PairSetDataLoader(test_data_list,8, False)

    test_statistics = {'num_defect': 0, 'num_normal': 0, 'correct': 0, 'fp':0, 'fn': 0}
    mkdirs_to_save_test_results(config.save_test_img_dir)


    for idx, (test_data, test_label) in enumerate(tqdm(test_ds, desc="test process ")):
        # # Load test data set
        #
        ref_img, insp_img = np.squeeze(test_data[0][:].numpy(), axis=-1), np.squeeze(test_data[1][:].numpy(), axis=-1)
        label_img = np.uint8(np.squeeze(test_label[:].numpy(), axis=-1))
        file_name = test_ds.__get_file_name__(idx)[0]
        bar_w = np.ones((len(ref_img),config.input_size[0], config.white_bar_w)) # to save combined image
        #Onnx model에서 출력된 prediction 결과 
        model_pred = ort_model.run(None, {'input_1:0' : test_data[0].numpy(), 'input_2:0' : test_data[1].numpy()})[0]

        # Get prediction results from trained model
        model_softmax = tf.nn.softmax(model_pred, axis=-1).numpy() # (1, H, W, 2)

        # Thresholding by softmax value and segment size
        model_pred_th_softmax = np.uint8(model_softmax[:, :, :, 1] >= config.th_softmax) # (1, H, W)
        model_pred_fin = np.uint8(th_label_thresholding(model_pred_th_softmax, config.th_label)) # (H, W)
        bitwise_and_label_pred = label_img * model_pred_fin

        combine_img = 255 * np.concatenate((ref_img, bar_w, insp_img, bar_w, label_img, bar_w, model_pred_fin), axis=-1)
        test_statistics = imwrite_roi_test_results(label_img, model_pred_fin, bitwise_and_label_pred, combine_img,
                                                   config.save_test_img_dir, file_name, test_statistics)

    return test_statistics

def test():
    if config.test_data_name.find("euv") >= 0:
        roi_test_statistics = roi_based_test()
        ds_inform = get_euv_test_ds_inform(config)
        segment_test_statistics = segment_based_test(ds_inform)
        seg_print = f'Segment 기준 : \ngtruth_defects = {segment_test_statistics["num_defect"]}, ' \
                    f'num_correct = {segment_test_statistics["num_correct"]}, ' \
                    f'num_fn = {segment_test_statistics["num_fn"]}, num_fp = {segment_test_statistics["num_fp"]}\n' \
                    f'정상검출률 = {100 * segment_test_statistics["num_correct"] / segment_test_statistics["num_defect"]} %\n' \
                    f'미검률 = {100 * segment_test_statistics["num_fn"] / segment_test_statistics["num_defect"]} %\n' \
                    f'과검률 = {100 * segment_test_statistics["num_fp"] / (segment_test_statistics["num_fp"] + segment_test_statistics["num_correct"])} %\n'
    else:
        #roi_test_statistics = roi_based_test()  #Onnx 사용X일 때 Test 함수
        roi_test_statistics = test_onnx()
        print("ROI 기준", roi_test_statistics)
        seg_print = ''

    f = open(f'{config.save_test_img_dir}/statistics.txt', 'a')
    print("ROI 기준", roi_test_statistics)
    print(seg_print)
    f.write(f"ROI 기준 : {roi_test_statistics}\n")
    f.write(seg_print), f.close()


if __name__ == '__main__':
    config = get_config()
    if os.path.isfile(f'{config.save_dir}/args.txt') == 0:
        write_args(config)

    gpu = tf.config.list_physical_devices('GPU')
    print(gpu)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    if config.is_train:
        train(
        test()
    else:
        test()

