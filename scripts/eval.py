import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from light.data import get_segmentation_dataset
from light.model import get_segmentation_model
from light.utils.metric import SegmentationMetric
from light.utils.visualize import get_color_pallete
from light.utils.logger import setup_logger
from light.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler

from train import parse_args

from light.data.segdata import SegmentationData
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def mask_to_image(mask):
    h = mask.shape[0]
    w = mask.shape[1]
    mask_rgb = Image.new('RGB', (w, h))
    for j in range(0, h):
        for i in range(0, w):
            pixal = mask[j, i]
            if pixal == 0:
                mask_rgb.putpixel((i, j), (61,10,81))
            elif pixal == 1:
                mask_rgb.putpixel((i, j), (69,142,139))
            elif pixal == 2:
                mask_rgb.putpixel((i, j), (250,231,85))
            else:
                mask_rgb.putpixel((i, j), (255, 255, 255))
    return mask_rgb

def data_process(x):
    x = np.array(x, dtype='float32') / 255
    if x.ndim < 4:
        x = np.expand_dims(x, 0)
    x = np.transpose(x, (0, 3, 1, 2))
    x = torch.from_numpy(x)

    mean = [0.517446, 0.360147, 0.310427]
    std = [0.061526, 0.049087, 0.041330]
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    return x.sub_(mean[:, None, None]).div_(std[:, None, None])

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.nb_classes = 3

        valid_img_dir = '/home/wangjialei/teeth_dataset/new_data_20190621/valid_new/images'
        valid_mask_dir = '/home/wangjialei/teeth_dataset/new_data_20190621/valid_new/masks'
        # valid_transform=transforms.Compose([
        #     # transforms.ToTensor(),
        #     # transforms.Normalize([0.517446, 0.360147, 0.310427], [0.061526,0.049087, 0.041330])#R_var is 0.061526, G_var is 0.049087, B_var is 0.041330
        # ])       

        # dataset and dataloader
        valid_set = SegmentationData(images_dir=valid_img_dir, masks_dir=valid_mask_dir, nb_classes=self.nb_classes, mode='valid', transform=None)
        valid_sampler = make_data_sampler(valid_set, False, args.distributed)
        valid_batch_sampler = make_batch_data_sampler(valid_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=valid_set,
                                          batch_sampler=valid_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset,
            aux=args.aux, pretrained=True, pretrained_base=False)

        if args.distributed:
            self.model = self.model.module
        self.model.to(self.device)

        self.metric = SegmentationMetric(valid_set.num_class)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target) in enumerate(self.val_loader):
            img = data_process(image)

            img = img.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                outputs = model(img)
            self.metric.update(outputs, target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))
            if self.args.save_pred:
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()

                predict = pred.squeeze(0)

                img_show = image[0].numpy()
                img_show = img_show.astype('uint8')
                plt.subplot(1, 3, 1)
                plt.title('image')
                plt.imshow(img_show)

                mask = target.cpu().data.numpy()
                mask = mask.reshape(mask.shape[1], mask.shape[2])
                mask = mask_to_image(mask)
                plt.subplot(1, 3, 2)
                plt.title('mask')
                plt.imshow(mask)

                predict = mask_to_image(predict)
                plt.subplot(1, 3, 3)
                plt.title('pred')
                plt.imshow(predict)

                save_file = "save_fig_val"
                os.makedirs(save_file, exist_ok=True)
                plt.savefig(os.path.join(save_file, str(i) + '.png'))
        synchronize()

    def test(self):
        self.model.eval()
        if self.args.distributed:
                model = self.model.module
        else:
            model = self.model
        test_img_dir = '/home/wangjialei/projects/teeth_bad_case/'
        img_folder = os.listdir(test_img_dir)
        for iter, img_file in enumerate(img_folder):
            img_name = test_img_dir + img_file
            image = Image.open(img_name)
            print(type(image))
            img = data_process(image)
            img = img.to(self.device)
            with torch.no_grad():
                outputs = model(img)
            if self.args.save_pred:
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()

                predict = pred.squeeze(0)

                img_show = image
                plt.subplot(1, 2, 1)
                plt.title('image')
                plt.imshow(img_show)

                predict = mask_to_image(predict)
                plt.subplot(1, 2, 2)
                plt.title('pred')
                plt.imshow(predict)

                save_file = "save_fig_test"
                os.makedirs(save_file, exist_ok=True)
                plt.savefig(os.path.join(save_file, str(iter) + '.png'))

if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # TODO: optim code
    args.save_pred = True
    if args.save_pred:
        outdir = '../runs/pred_pic/{}_{}'.format(args.model, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger(args.model, args.log_dir, get_rank(),
                          filename='{}_{}_log.txt'.format(args.model, args.dataset), mode='a+')

    evaluator = Evaluator(args)
    # evaluator.eval()
    evaluator.test()

