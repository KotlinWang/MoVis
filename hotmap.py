import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import tqdm
import argparse
import datetime
import cv2
import numpy as np

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
from lib.helpers.save_helper import load_checkpoint


parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('--checkpoint_name', default='checkpoint', type=str)
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
args = parser.parse_args()


class MakeMap(object):
    def __init__(self, cfg, checkpoint_name, model, dataloader, logger, train_cfg=None, model_name='monodetr'):
        self.cfg = cfg
        self.checkpoint_name = checkpoint_name
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name

    def test(self):
        checkpoint_path = os.path.join(self.output_dir, "{}.pth".format(self.checkpoint_name))
        assert os.path.exists(checkpoint_path)
        load_checkpoint(model=self.model,
                        optimizer=None,
                        filename=checkpoint_path,
                        map_location=self.device,
                        logger=self.logger)
        self.model.to(self.device)
        self.inference()

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)
            img_id = info['img_id']

            ###dn
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args = 0)
            ###
            for i in range(outputs.shape[0]):
                self.hotmap(img_id=img_id[i], feature_map=outputs[i])
            if batch_idx == 10:
                exit()

    def hotmap(self, img_id, feature_map):
        original_image = cv2.imread(os.path.join('/home/kotlin/DATASET/KITTI/training/image_2', '%06d.png' % img_id))  # 读取原图
        # 上采样特征图到原图大小
        heatmap = cv2.resize(feature_map.sum(dim=0).contiguous().data.cpu().numpy(), (original_image.shape[1], original_image.shape[0]))
        
        # 归一化特征图到 [0, 255]
        heatmap = heatmap.astype(np.float32)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 应用热力图
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

        # 叠加热力图与原图
        overlayed_image = cv2.addWeighted(original_image, 0.3, heatmap, 0.6, 0)

        # 创建保存结果图的文件夹
        output_folder = 'output_hotmap'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 保存结果图
        output_path = os.path.join(output_folder, '%06d.png' % img_id)
        cv2.imwrite(output_path, overlayed_image)

def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    checkpoint_name = args.checkpoint_name
    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # build dataloader
    train_loader, test_loader = build_dataloader(cfg['dataset'])

    # build model
    model, loss = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))


    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)
    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = MakeMap(cfg=cfg['tester'],
                        checkpoint_name=checkpoint_name,
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
        tester.test()
        return


if __name__ == '__main__':
    main()
