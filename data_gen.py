import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


cropSize = 256
overlapSize = 0

stack_output_dir = './twinnnet_data'
if os.path.exists(stack_output_dir) == 0:
    os.makedirs(stack_output_dir)

#
# defect_file = os.listdir('D:\\Public\\jsyoon\\AE\\venv\\wind_diff\\chip')
#
# defect_list = [
#     (os.sep.join(['D:\\Public\\jsyoon\\AE\\venv\\wind_diff\\chip', filename]))
#     for filename in defect_file]
#
# golden_file = os.listdir('D:\\Public\\jsyoon\\AE\\venv\\wind_diff\\golden')
#
# golden_list = [
#     (os.sep.join(['D:\\Public\\jsyoon\AE\\venv\\wind_diff\\golden', filename]))
#     for filename in golden_file]
#
# #
chip_file = os.listdir('D:\\wind_diff\\chip')
chip_list = [
    (os.sep.join(['D:\\wind_diff\\chip', filename]))
    for filename in chip_file]



#큰 이미지 crop해주는 함수 -> dreamer 코드랑 동일
def split_img(npimg,file_name):

    x_list = []

    w_len = npimg.shape[0] // (cropSize - overlapSize)
    if npimg.shape[0] % (cropSize - overlapSize) != 0: w_len += 1
    h_len = npimg.shape[1] // (cropSize - overlapSize)
    if npimg.shape[1] % (cropSize - overlapSize) != 0: h_len += 1


    for x_iter in range(w_len):
        img_list = []
        x = ((cropSize - overlapSize) * x_iter)
        end_x = x + cropSize

        if end_x > npimg.shape[0]:
            end_x = npimg.shape[0]
            x = end_x - cropSize

        for y_iter in range(h_len):
            y = ((cropSize - overlapSize) * y_iter)
            end_y = y + cropSize
            if end_y > npimg.shape[1]:
                end_y = npimg.shape[1]
                y = end_y - cropSize

            crop = npimg[x:end_x, y:end_y]
            outputdir = f'./crop/{file_name}'
            os.makedirs(outputdir, exist_ok=True)
            cv2.imwrite(os.path.join(outputdir, '{}_{}_{}.png'.format(file_name, x_iter, y_iter)), crop)
            img_list.append(crop)

        x_list.append((img_list))


#Twinnet용 data 생성(4장 이어붙인 이미지)
def make_dataimg(chip,golden,i,j,filename):

    gt_label = np.zeros((chip.shape[0],chip.shape[1]),dtype=np.uint8)

    stack_img = np.hstack((golden,chip,gt_label,gt_label)) #normal 일땐 빈 이미지를 label.
    cv2.imwrite(f'{stack_output_dir}/{i}_{j}_{filename}',stack_img)

if __name__ == '__main__':

    #전체 이미지 crop
    for idx,filename in enumerate(chip_file):
        img = cv2.imread(chip_list[idx],0)
        split_img(img,filename[:-4])

    #crop한 이미지 stack해서 twinnet data 만들기
    for i in range(40):
        for j in range(3):
            chip_path = f"./crop/chip({i},{j})"
            golden_path = f"./crop/golden_chip({i},{j})"

            chip_file = os.listdir(chip_path)
            chip_list = [
                (os.sep.join([chip_path, filename]))
                for filename in chip_file]

            golden_file = os.listdir(golden_path)
            golden_list = [
                (os.sep.join([golden_path, filename]))
                for filename in golden_file]

            for index, filename3 in enumerate(chip_file):
                img3 = cv2.imread(chip_list[index],0)
                img4 = cv2.imread(golden_list[index],0)
                make_dataimg(img3,img4,i,j,filename3)






