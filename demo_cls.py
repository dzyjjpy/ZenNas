'''
This is demo for inas inference demo for classification
Author:  Alan Jia @Insta360 Research
Date: 2022.01.26
Usage:
python demo_cls.py   --arch zennet_gesture_model_size163k_flops2.59M_acc97.50_res96 --mode img --img test.jpg --res test_res.jpg --gpu 0
python demo_cls.py   --arch zennet_gesture_model_size163k_flops2.59M_acc97.50_res96 --mode dir --img_dir tmp/ --res_dir tmp_res/ --gpu 0
'''
import os, sys, argparse
import torch
import ZenNet
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# label definition for hand gesture classification
labels_gesture = {"0":"palm", "1":"ok", "2":"L", "3":"V", "4":"rock", "5":"other"}

def process_img(img_file, model):
    # get input image
    img = Image.open(img_file)
    # print("img shape: ", img.size)

    # image transforms
    transform = transforms.Compose(
        [transforms.Resize(size=(96,96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]
    )
    input = transform(img)
    # print("---->input shape: ", input.shape)
    input = input.unsqueeze(0)
    # print("---->input shape: ", input.shape)
    with torch.no_grad():
        if opt.gpu is not None:
            input = input.cuda(opt.gpu, non_blocking=True)
        output = model(input)
        prob = torch.nn.functional.softmax(output)
        pred_label = np.argmax(prob.cpu().numpy())
        score = np.max(prob.cpu().numpy())
        # print("---->pred class is ", labels_gesture[str(pred_label)])
        return pred_label, score

def visualize_result(img_file, res_file, pred_cls_name, score):
    img = cv2.imread(img_file)
    h, w, _ = img.shape
    size = (int(w/10), int(h/2))
    img = cv2.putText(img, pred_cls_name+":"+str(score), size, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # cv2.imshow("img_res.jpg", img)
    # cv2.waitKey(1000)
    cv2.imwrite(res_file, img)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default=None, help='model to be evaluated.')
    parser.add_argument('--mode', type=str, default=None, help='img or dir.')
    parser.add_argument('--img', type=str, default=None, help='image path')
    parser.add_argument('--res', type=str, default=None, help='output path')
    parser.add_argument('--img_dir', type=str, default=None, help='image folder path')
    parser.add_argument('--res_dir', type=str, default=None, help='output folder path')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID. None for CPU.')

    opt, _ = parser.parse_known_args(sys.argv)

    input_image_size = ZenNet.zennet_model_zoo[opt.arch]['resolution']
    crop_image_size = ZenNet.zennet_model_zoo[opt.arch]['crop_image_size']
    print('Predict {} at {}x{} resolution.'.format(opt.arch, input_image_size, input_image_size))
    
    # load model
    model = ZenNet.get_ZenNet(opt.arch, pretrained=True)
    if opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        torch.backends.cudnn.benchmark = True
        model = model.cuda(opt.gpu)
        print('Using GPU {}.'.format(opt.gpu))
    model.eval()

    assert opt.mode=="img" or opt.mode=="dir", "mode must be img or dir"
    if opt.mode == "img":
        pred_label, score = process_img(opt.img, model)
        cls_name = labels_gesture[str(pred_label)]
        print("predict cls_name: {} , score: {}".format(cls_name, score))
        visualize_result(opt.img, opt.res, cls_name, round(score, 4))
    elif opt.mode == "dir":
        img_dir = opt.img_dir
        res_dir = opt.res_dir
        files = os.listdir(img_dir)
        for file in tqdm(files):
            img_file = os.path.join(img_dir, file)
            res_file = os.path.join(res_dir, file)
            if not os.path.exists(res_dir):
                print("{} not exist, create res_dir".format(res_dir))
                os.makedirs(res_dir)
            if img_file.endswith(".jpg") or img_file.endswith(".png"):
                pred_label, score = process_img(img_file, model)
                cls_name = labels_gesture[str(pred_label)]
                print("predict cls_name: {} , score: {}".format(cls_name, score))
                visualize_result(img_file, res_file, cls_name, round(score, 4))     
            else:
                print("{} is not image, pls check".format(img_file))


