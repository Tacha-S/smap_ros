#!/usr/bin/env python3

import json
import os
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

sys.path.append("/ros/src/smap_ros/SMAP")

import dapalib
from exps.stage3_root2.config import cfg
from exps.stage3_root2.test_util import *
from lib.utils.comm import is_main_process
from model.refinenet import RefineNet
from model.smap import SMAP


def generate_3d_point_pairs(model, refine_model, data_loader, cfg, device,
                            output_dir=''):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    if refine_model is not None:
        refine_model.eval()

    result = dict()
    result['model_pattern'] = cfg.DATASET.NAME
    result['3d_pairs'] = []
    # 3d_pairs has items like{'pred_2d':[[x,y,detZ,score]...], 'gt_2d':[[x,y,Z,visual_type]...],
    #                         'pred_3d':[[X,Y,Z,score]...], 'gt_3d':[[X,Y,X]...],
    #                         'root_d': (abs depth of root (float value) pred by network),
    #                         'image_path': relative image path}

    kpt_num = cfg.DATASET.KEYPOINT.NUM
    data = tqdm(data_loader) if is_main_process() else data_loader
    for idx, batch in enumerate(data):
        if cfg.TEST_MODE == 'run_inference':
            imgs, img_path, scales = batch
            meta_data = None
        else:
            imgs, meta_data, img_path, scales = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            outputs_2d, outputs_3d, outputs_rd = model(imgs)

            outputs_3d = outputs_3d.cpu()
            outputs_rd = outputs_rd.cpu()

            if cfg.DO_FLIP:
                imgs_flip = torch.flip(imgs, [-1])
                outputs_2d_flip, outputs_3d_flip, outputs_rd_flip = model(imgs_flip)
                outputs_2d_flip = torch.flip(outputs_2d_flip, dims=[-1])
                # outputs_3d_flip = torch.flip(outputs_3d_flip, dims=[-1])
                # outputs_rd_flip = torch.flip(outputs_rd_flip, dims=[-1])
                keypoint_pair = cfg.DATASET.KEYPOINT.FLIP_ORDER
                paf_pair = cfg.DATASET.PAF.FLIP_CHANNEL
                paf_abs_pair = [x + kpt_num for x in paf_pair]
                pair = keypoint_pair + paf_abs_pair
                for i in range(len(pair)):
                    if i >= kpt_num and (i - kpt_num) % 2 == 0:
                        outputs_2d[:, i] += outputs_2d_flip[:, pair[i]] * -1
                    else:
                        outputs_2d[:, i] += outputs_2d_flip[:, pair[i]]
                outputs_2d[:, kpt_num:] *= 0.5

            for i in range(len(imgs)):
                if meta_data is not None:
                    # remove person who was blocked
                    new_gt_bodys = []
                    annotation = meta_data[i].numpy()
                    scale = scales[i]
                    for j in range(len(annotation)):
                        if annotation[j, cfg.DATASET.ROOT_IDX, 3] > 1:
                            new_gt_bodys.append(annotation[j])
                    gt_bodys = np.asarray(new_gt_bodys)
                    if len(gt_bodys) == 0:
                        continue
                    # groundtruth:[person..[keypoints..[x, y, Z, score(0:None, 1:invisible, 2:visible), X, Y, Z,
                    #                                   f_x, f_y, cx, cy]]]
                    if len(gt_bodys[0][0]) < 11:
                        scale['f_x'] = gt_bodys[0, 0, 7]
                        scale['f_y'] = gt_bodys[0, 0, 7]
                        scale['cx'] = scale['img_width'] / 2
                        scale['cy'] = scale['img_height'] / 2
                    else:
                        scale['f_x'] = gt_bodys[0, 0, 7]
                        scale['f_y'] = gt_bodys[0, 0, 8]
                        scale['cx'] = gt_bodys[0, 0, 9]
                        scale['cy'] = gt_bodys[0, 0, 10]
                else:
                    gt_bodys = None
                    # use default values
                    scale = {k: scales[k][i].numpy() for k in scales}
                    scale['f_x'] = scale['img_width']
                    scale['f_y'] = scale['img_width']
                    scale['cx'] = scale['img_width'] / 2
                    scale['cy'] = scale['img_height'] / 2

                hmsIn = outputs_2d[i]

                # if the first pair is [1, 0], uncomment the code below
                # hmsIn[cfg.DATASET.KEYPOINT.NUM:cfg.DATASET.KEYPOINT.NUM+2] *= -1
                # outputs_3d[i, 0] *= -1

                hmsIn[:cfg.DATASET.KEYPOINT.NUM] /= 255
                hmsIn[cfg.DATASET.KEYPOINT.NUM:] /= 127
                rDepth = outputs_rd[i][0]
                # no batch implementation yet
                pred_bodys_2d = dapalib.connect(hmsIn, rDepth, cfg.DATASET.ROOT_IDX, distFlag=True)
                if len(pred_bodys_2d) > 0:
                    pred_bodys_2d[:, :, :2] *= cfg.dataset.STRIDE  # resize poses to the input-net shape
                    pred_bodys_2d = pred_bodys_2d.numpy()

                pafs_3d = outputs_3d[i].numpy().transpose(1, 2, 0)
                root_d = outputs_rd[i][0].numpy()

                paf_3d_upsamp = cv2.resize(
                    pafs_3d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)
                root_d_upsamp = cv2.resize(
                    root_d, (cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0]), interpolation=cv2.INTER_NEAREST)

                # generate 3d prediction bodys
                pred_bodys_2d = register_pred(pred_bodys_2d, gt_bodys)

                if len(pred_bodys_2d) == 0:
                    continue
                pred_rdepths = generate_relZ(pred_bodys_2d, paf_3d_upsamp, root_d_upsamp, scale)
                pred_bodys_3d = gen_3d_pose(pred_bodys_2d, pred_rdepths, scale)

                if refine_model is not None:
                    new_pred_bodys_3d = lift_and_refine_3d_pose(pred_bodys_2d, pred_bodys_3d, refine_model,
                                                                device=device, root_n=cfg.DATASET.ROOT_IDX)
                else:
                    new_pred_bodys_3d = pred_bodys_3d

                if cfg.TEST_MODE == "generate_train":
                    save_result_for_train_refine(pred_bodys_2d, new_pred_bodys_3d, gt_bodys, pred_rdepths, result)
                else:
                    save_result(pred_bodys_2d, new_pred_bodys_3d, gt_bodys, pred_rdepths, img_path[i], result)

    dir_name = os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]
    pair_file_name = os.path.join(output_dir, '{}_{}_{}_{}.json'.format(dir_name, cfg.TEST_MODE,
                                                                        cfg.DATA_MODE, cfg.JSON_SUFFIX_NAME))
    with open(pair_file_name, 'w') as f:
        json.dump(result, f)
    rospy.loginfo("Pairs writed to {}".format(pair_file_name))


class Dataset(object):
    def __init__(self, img):
        self.image_list = [img]
        self.image_shape = (img.shape[1], img.shape[0])
        self.net_input_shape = (832, 512)  # (width, height)

        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.transform = transform

    def __getitem__(self, index):
        net_input_image, scale = self.aug_croppad(self.image_list[index])
        net_input_image = self.transform(net_input_image)
        image_name = 'tmp'

        return net_input_image, image_name, scale

    def __len__(self):
        return 1

    def aug_croppad(self, img):
        scale = dict()
        crop_x = self.net_input_shape[0]  # width
        crop_y = self.net_input_shape[1]  # height
        scale['scale'] = min(crop_x / self.image_shape[0], crop_y / self.image_shape[1])
        img = cv2.resize(img, (0, 0), fx=scale['scale'], fy=scale['scale'])

        scale['img_width'] = self.image_shape[0]
        scale['img_height'] = self.image_shape[1]
        scale['net_width'] = self.net_input_shape[0]
        scale['net_height'] = self.net_input_shape[1]

        center = np.array([img.shape[1] // 2, img.shape[0] // 2], dtype=np.int)

        if img.shape[1] < crop_x:    # pad left and right
            margin_l = (crop_x - img.shape[1]) // 2
            margin_r = crop_x - img.shape[1] - margin_l
            pad_l = np.ones((img.shape[0], margin_l, 3), dtype=np.uint8) * 128
            pad_r = np.ones((img.shape[0], margin_r, 3), dtype=np.uint8) * 128
            img = np.concatenate((pad_l, img, pad_r), axis=1)
        elif img.shape[0] < crop_y:  # pad up and down
            margin_u = (crop_y - img.shape[0]) // 2
            margin_d = crop_y - img.shape[0] - margin_u
            pad_u = np.ones((margin_u, img.shape[1], 3), dtype=np.uint8) * 128
            pad_d = np.ones((margin_d, img.shape[1], 3), dtype=np.uint8) * 128
            img = np.concatenate((pad_u, img, pad_d), axis=0)

        return img, scale


class PoseEstimation(object):
    def __init__(self):
        self.model = SMAP(cfg, run_efficient=cfg.RUN_EFFICIENT)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        if cfg.REFINE:
            self.refine_model = RefineNet()
            self.refine_model.to(self.device)
            refine_model_file = rospy.get_param('~refinenet_model', '/ros/src/smap_ros/resources/RefineNet.pth')
        else:
            self.refine_model = None
            refine_model_file = ""
        smap_model = rospy.get_param('~smap_model', '/ros/src/smap_ros/resources/SMAP_model.pth')

        if os.path.exists(smap_model):
            state_dict = torch.load(smap_model, map_location=lambda storage, loc: storage)
            state_dict = state_dict['model']
            self.model.load_state_dict(state_dict)
            if os.path.exists(refine_model_file):
                self.refine_model.load_state_dict(torch.load(refine_model_file))
            elif self.refine_model is not None:
                rospy.logerr("No such RefineNet checkpoint of {}".format(refine_model_file))
                return
        else:
            rospy.logerr("No such checkpoint of SMAP {}".format(smap_model))
            return

        rospy.Subscriber('~input', Image, self.callback)

        self.__pub = rospy.Publisher('~image', Image, queue_size=10)

    def callback(self, msg):
        total_now = time.time()
        try:
            image_bgr = self.__bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logwarn(e)
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        test_dataset = Dataset(image_rgb)
        data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        generate_3d_point_pairs(self.model, self.refine_model, data_loader, cfg, self.device,
                                output_dir=os.path.join(cfg.OUTPUT_DIR, "result"))

        total_then = time.time()

        text = "{:03.2f} sec".format(total_then - total_now)
        rospy.loginfo(text)

        self.__pub.publish(self.__bridge.cv2_to_imgmsg(image_debug, 'bgr8'))


def main():
    rospy.init_node('pose_estimation')
    _ = PoseEstimation()
    rospy.spin()
    return


if __name__ == '__main__':
    main()
