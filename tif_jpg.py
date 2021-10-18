import cv2 as cv
import cv2
import os
import shutil

def tif2jpg(tifjpgpath):
    img = cv.imread(tifjpgpath, -1)  #
    cv.imwrite(tifjpgpath.split('.')[0] + ".jpg", img)  # tif 格式转 jpg

if __name__ == '__main__':
    tifjpgpath = "imgs/000021.tif"
    tif2jpg(tifjpgpath)

# images_dir = '/home/deploy/tianchi-logic-object/data/suichang_round1_train_210120/'
# save_imgs = '/home/deploy/tianchi-logic-object/data/images'
# save_masks = '/home/deploy/tianchi-logic-object/data/masks_new'
# if not os.path.exists(save_imgs): os.makedirs(save_imgs)
# if not os.path.exists(save_masks): os.makedirs(save_masks)
# tif_list = [x for x in os.listdir(images_dir)]  # 获取目录中所有tif格式图像列表
# for num, name in enumerate(tif_list):  # 遍历列表
#     if name.endswith(".tif"):
#         img = cv.imread(os.path.join(images_dir, name), -1)  # 读取列表中的tif图像
#         cv.imwrite(os.path.join(save_imgs, name.split('.')[0] + ".jpg"), img)  # tif 格式转 jpg
#     else:
#         img = cv.imread(os.path.join(images_dir, name), cv2.IMREAD_GRAYSCALE)
#         img = img - 1
#         cv2.imwrite(os.path.join(save_masks, name), img)
#         # shutil.copy(os.path.join(images_dir, name),os.path.join(save_masks,name))
# save_test = '/home/deploy/tianchi-logic-object/data/suichang_round1_test_partA_210120/'
# save_test_dir = '/home/deploy/tianchi-logic-object/data/test_jpg'
# if not os.path.exists(save_test_dir): os.makedirs(save_test_dir)
# for name in os.listdir(save_test):
#     img = cv.imread(os.path.join(save_test, name), -1)  #
#     cv.imwrite(os.path.join(save_test_dir, name.split('.')[0] + ".jpg"), img)  # tif 格式转 jpg
