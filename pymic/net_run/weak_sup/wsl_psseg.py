# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from pymic.loss.seg.util import get_soft_label
from pymic.loss.seg.util import reshape_prediction_and_ground_truth
from pymic.loss.seg.util import get_classwise_dice
from torch.nn.modules.loss import CrossEntropyLoss as TorchCELoss
from pymic.net_run.weak_sup import WSLSegAgent


class WSLPSSEG(WSLSegAgent):
    """
    PS-Seg: Learning from Partial Scribbles for 3D Multiple Abdominal Organ Segmentation

    :param config: (dict) A dictionary containing the configuration.
    :param stage: (str) One of the stage in `train` (default), `inference` or `test`.

    .. note::

        In the configuration dictionary, in addition to the four sections (`dataset`,
        `network`, `training` and `inference`) used in fully supervised learning, an
        extra section `weakly_supervised_learning` is needed. See :doc:`usage.wsl` for details.
    """

    def __init__(self, config, stage="train"):
        net_type = config["network"]["net_type"]
        # if net_type not in ['UNet2D_DualBranch', 'UNet3D_DualBranch']:
        #     raise ValueError("""For WSL_DMPLS, a dual branch network is expected. \
        #         It only supports UNet2D_DualBranch and UNet3D_DualBranch currently.""")
        super(WSLPSSEG, self).__init__(config, stage)
        self.consistency = config["training"].get("consistencty", 1.0)
        self.consistency_rampup = config["training"].get("consistency_rampup", 60.0)

    @staticmethod
    def sigmoid_rampup(current, rampup_length):  #
        """Exponential rampup from https://arxiv.org/abs/1610.02242 指数增长"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(
                current, 0.0, rampup_length
            )  # np.clip()将current数组里的数限定在0和t_max之间
            # rampup_length即t_max，指迭代的最大次数
            phase = 1.0 - current / rampup_length
            return float(
                np.exp(-5.0 * phase * phase)
            )  # 返回的是，exp[-5*(1-t/t_max)^2]

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    @staticmethod
    def weight_with_EM(p, C=8):
        y1 = (
            -1
            * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
            / torch.tensor(np.log(C)).cuda()
        )
        y1 = torch.exp(-y1)
        return y1

    def wpseudoSoft_meanKL_consistency(self, outputs_list):
        probs = [F.softmax(logits, dim=1) for i, logits in enumerate(outputs_list)]
        mixture_label = (torch.stack(probs)).mean(axis=0)
        weight = self.weight_with_EM(mixture_label)
        weight_mixture_label = mixture_label * weight
        logp_mixture = weight_mixture_label.log()
        log_probs = [
            torch.sum(F.kl_div(logp_mixture, prob, reduction="none"), dim=1)
            for prob in probs
        ]
        consistency = sum(log_probs) / len(log_probs)
        loss_KL = torch.mean(consistency)
        return loss_KL

    @staticmethod
    def affinity_cross_consistencyMSE_six(outputsSoft_list, batch_size, num_classes):
        outputsSoft_list_r = [
            item.reshape(batch_size, num_classes, -1) for item in outputsSoft_list
        ]
        outputsSoft_list_rNorm = [
            F.normalize(item, p=2, dim=2) for item in outputsSoft_list_r
        ]
        outputsSoft_list_rNormT = [
            item.permute(0, 2, 1) for item in outputsSoft_list_rNorm
        ]

        affinity_list = []
        for i in range(len(outputsSoft_list)):
            outputsSoft_list_rNormT_other = (
                outputsSoft_list_rNormT[:i] + outputsSoft_list_rNormT[i + 1 :]
            )
            for j in range(len(outputsSoft_list_rNormT_other)):
                affinity_ori = torch.bmm(
                    outputsSoft_list_rNorm[i], outputsSoft_list_rNormT_other[j]
                )
                affinity_list.append(affinity_ori)

        maskI = (
            torch.eye(affinity_list[0].shape[1])
            .repeat(affinity_list[1].shape[0], 1, 1)
            .cuda()
        )

        consistency_loss = 0
        for i in range(len(affinity_list)):
            consistency_loss = consistency_loss + torch.mean(
                (affinity_list[i] - maskI) ** 2
            )
        consistency_loss = consistency_loss / len(affinity_list)
        return consistency_loss

    def training(self):
        class_num = self.config["network"]["class_num"]
        iter_valid = self.config["training"]["iter_valid"]
        iter_max = self.config["training"]["iter_max"]
        train_loss = 0
        train_dice_list = []
        data_time, gpu_time, loss_time, back_time = 0, 0, 0, 0

        ce_loss = TorchCELoss(ignore_index=class_num)
        consistency_rampepoch = self.config["training"].get(
            "consistency_rampepoch", 1000
        )
        batch_size = self.config["training"].get("train_batch_size", 2)
        alpha = self.config["training"].get("alpha", 10.0)
        gamma = self.config["training"].get("gamma", 0.01)
        self.net.train()
        iter_num = 0
        for it in range(iter_valid):
            t0 = time.time()
            try:
                data = next(self.trainIter)
            except StopIteration:
                self.trainIter = iter(self.train_loader)
                data = next(self.trainIter)
            t1 = time.time()
            # get the inputs
            inputs = self.convert_tensor_type(data["image"])
            y = self.convert_tensor_type(data["label"])

            inputs, y = inputs.to(self.device), y.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs_list, embeddings1_list, embeddings2_list, embeddings3_list = self.net(inputs)
            t2 = time.time()

            # supervised
            loss_ce_list = [
                ce_loss(pred, y[:, 0, ...].long()) for i, pred in enumerate(outputs_list)
            ]
            loss_ce = sum(loss_ce_list) / len(loss_ce_list)

            # unsupervised loss
            un_weight = self.get_current_consistency_weight(iter_num // consistency_rampepoch)

            ## L_USPC
            loss_KL = self.wpseudoSoft_meanKL_consistency(outputs_list)

            ## L_CCAC
            loss_aff0 = self.affinity_cross_consistencyMSE_six(outputs_list, batch_size, class_num)
            loss_aff1 = self.affinity_cross_consistencyMSE_six(embeddings1_list, batch_size, class_num)
            loss_aff2 = self.affinity_cross_consistencyMSE_six(embeddings2_list, batch_size, class_num)
            loss_aff3 = self.affinity_cross_consistencyMSE_six(embeddings3_list, batch_size, class_num)

            loss_aff = (loss_aff0 + loss_aff1 + loss_aff2 + loss_aff3) / 4.0


            loss = loss_ce + alpha * un_weight * loss_KL + gamma * un_weight * loss_aff

            t3 = time.time()
            loss.backward()
            t4 = time.time()
            self.optimizer.step()

            train_loss = train_loss + loss.item()
            # get dice evaluation for each class in annotated images
            outputs = outputs_list[0]
            y_cal_dice = y[:, 0, ...].long()
            y_cal_dice[y_cal_dice >= class_num] = 0
            y = F.one_hot(y_cal_dice, num_classes=class_num).permute(0, 4, 1, 2, 3).float()
            p_argmax = torch.argmax(outputs, dim=1, keepdim=True)
            p_soft = get_soft_label(p_argmax, class_num, self.tensor_type)
            p_soft, y = reshape_prediction_and_ground_truth(p_soft, y)
            dice_list = get_classwise_dice(p_soft, y)
            train_dice_list.append(dice_list.cpu().numpy())

            data_time = data_time + t1 - t0
            gpu_time = gpu_time + t2 - t1
            loss_time = loss_time + t3 - t2
            back_time = back_time + t4 - t3
        train_avg_loss = train_loss / iter_valid
        train_cls_dice = np.asarray(train_dice_list).mean(axis=0)
        train_avg_dice = train_cls_dice[1:].mean()

        iter_num += 1
        train_scalers = {
            "loss": train_avg_loss,
            "avg_fg_dice": train_avg_dice,
            "class_dice": train_cls_dice,
            "data_time": data_time,
            "forward_time": gpu_time,
            "loss_time": loss_time,
            "backward_time": back_time,
        }
        return train_scalers


    def validation(self):
        class_num = self.config['network']['class_num']

        infer_cfg = {}
        infer_cfg['class_num'] = class_num
        infer_cfg['sliding_window_enable'] = self.config['testing'].get('sliding_window_enable', True)
        if(infer_cfg['sliding_window_enable']):
            patch_size = self.config['dataset'].get('patch_size', None)
            if(patch_size is None):
                patch_size = self.config['testing']['sliding_window_size']
            sliding_window_size   = patch_size
            sliding_window_stride = [i//2 for i in patch_size]

        else: 
            raise NotImplementedError("Only sliding window inference is supported in WSL PS-Seg currently.")


        ce_loss = TorchCELoss()
        
        valid_loss_list = []
        valid_dice_list = []
        validIter  = iter(self.valid_loader)
        with torch.no_grad():
            self.net.eval()
            for data in validIter:
                inputs      = self.convert_tensor_type(data['image'])
                
                labels_prob = self.convert_tensor_type(data['label'])
                inputs, labels_prob  = inputs.to(self.device), labels_prob.to(self.device)
                batch_n = inputs.shape[0]

                outputs = self.inference(inputs, sliding_window_size, sliding_window_stride, class_num)


                loss = ce_loss(outputs, labels_prob[:, 0, ...].long())
                valid_loss_list.append(loss.item())

                outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                soft_out  = get_soft_label(outputs_argmax, class_num, self.tensor_type)
                
                y_cal_dice = labels_prob[:, 0, ...].long()
                y = F.one_hot(y_cal_dice, num_classes=class_num).permute(0, 4, 1, 2, 3).float()
                for i in range(batch_n):
                    soft_out_i, y_i = reshape_prediction_and_ground_truth(\
                        soft_out[i:i+1], y[i:i+1])
                    temp_dice = get_classwise_dice(soft_out_i, y_i)
                    valid_dice_list.append(temp_dice.cpu().numpy())

        valid_avg_loss = np.asarray(valid_loss_list).mean()
        valid_cls_dice = np.asarray(valid_dice_list).mean(axis = 0)
        valid_avg_dice = valid_cls_dice[1:].mean()
        valid_scalers = {'loss': valid_avg_loss, 'avg_fg_dice': valid_avg_dice,\
            'class_dice': valid_cls_dice}
        return valid_scalers


    def inference(self, inputs, patch_size, sliding_window_stride, class_num):
        d, w, h = inputs.shape[2:]

        d_pad = max(0, patch_size[0] - d)
        w_pad = max(0, patch_size[1] - w)
        h_pad = max(0, patch_size[2] - h)

        dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
        wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
        hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2

        # 3. Apply padding to Tensor (torch order: last dim to first dim)
        add_pad = False
        if d_pad + w_pad + h_pad > 0:
            add_pad = True
            # pad shape: (W_left, W_right, H_top, H_bottom, D_front, D_back)
            inputs = F.pad(inputs, (hl_pad, hr_pad, wl_pad, wr_pad, dl_pad, dr_pad), 
                        mode='constant', value=0)

        # Get new dimensions after padding
        _, _, dd, ww, hh = inputs.shape

        # deal the input image into patches to dit the net's input size
        sz = math.ceil((dd - patch_size[0]) / sliding_window_stride[0]) + 1
        sx = math.ceil((ww - patch_size[1]) / sliding_window_stride[1]) + 1 #向上舍入最接近的函数， step_x
        sy = math.ceil((hh - patch_size[2]) / sliding_window_stride[2]) + 1
        # print("{}, {}, {}".format(sx, sy, sz))
        score_map = torch.zeros((1, class_num, dd, ww, hh), device=inputs.device)
        cnt = torch.zeros((1, 1, dd, ww, hh), device=inputs.device)

        # 6. Sliding window inference
        for x in range(0, sx):
            xs = min(sliding_window_stride[1] * x, ww - patch_size[1])
            for y in range(0, sy):
                ys = min(sliding_window_stride[2] * y, hh - patch_size[2])
                for z in range(0, sz):
                    zs = min(sliding_window_stride[0] * z, dd - patch_size[0])
                    
                    # 7. Extract test_patch using Torch slicing
                    # inputs is [1, 1, D, H, W], so we slice on dims 2, 3, 4
                    test_patch = inputs[:, :, zs:zs+patch_size[0], xs:xs+patch_size[1], ys:ys+patch_size[2]]
                    
                    # Now test_patch is already a 5D Tensor on GPU, no need for expand_dims or from_numpy
                    # You can directly feed it to your model here:
                    # output_patch = self.model(test_patch)
                    pred_list, _, _, _ = self.net(test_patch)
                    pred = pred_list[0]
                    score_map[:, :, zs:zs+patch_size[0], xs:xs+patch_size[1], ys:ys+patch_size[2]] += pred
                    cnt[:, :, zs:zs+patch_size[0], xs:xs+patch_size[1], ys:ys+patch_size[2]] += 1

        outputs = score_map / cnt
        if add_pad:
            # 原始尺寸是 d, w, h，对应的起始索引是 dl_pad, wl_pad, hl_pad
            outputs = outputs[:, :, dl_pad:dl_pad+d, wl_pad:wl_pad+w, hl_pad:hl_pad+h]

        return outputs

    def infer(self):
        device_ids = self.config['testing']['gpus']
        device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(device)

        if(self.config['testing'].get('evaluation_mode', True)):
            self.net.eval()
            if(self.config['testing'].get('test_time_dropout', False)):
                def test_time_dropout(m):
                    if(type(m) == nn.Dropout):
                        logging.info('dropout layer')
                        m.train()
                self.net.apply(test_time_dropout)

        ckpt_mode = self.config['testing']['ckpt_mode']
        ckpt_name = self.get_checkpoint_name()
        if(ckpt_mode == 3):
            assert(isinstance(ckpt_name, (tuple, list)))
            self.infer_with_multiple_checkpoints()
            return 
        else:
            if(isinstance(ckpt_name, (tuple, list))):
                raise ValueError("ckpt_mode should be 3 if ckpt_name is a list")

        # load network parameters and set the network as evaluation mode
        print("ckpt name", ckpt_name)
        checkpoint = torch.load(ckpt_name, map_location = device, weights_only=False)
        self.net.load_state_dict(checkpoint['model_state_dict'])


        patch_size = self.config['dataset'].get('patch_size', None)
        if(patch_size is None):
            patch_size = self.config['testing']['sliding_window_size']
        sliding_window_size   = patch_size
        sliding_window_stride = [i//2 for i in patch_size]
        class_num = self.config['network']['class_num']

        infer_time_list = []
        with torch.no_grad():
            for data in self.test_loader:
                images = self.convert_tensor_type(data['image'])
                images = images.to(device)
    
                # for debug
                # for i in range(images.shape[0]):
                #     image_i = images[i][0]
                #     label_i = images[i][0]
                #     image_name = "temp/{0:}_image.nii.gz".format(names[0])
                #     label_name = "temp/{0:}_label.nii.gz".format(names[0])
                #     save_nd_array_as_image(image_i, image_name, reference_name = None)
                #     save_nd_array_as_image(label_i, label_name, reference_name = None)
                # continue
                start_time = time.time()
                outputs = self.inference(images, sliding_window_size, sliding_window_stride, class_num)
                
                pred = outputs.cpu().numpy()
                
                data['predict'] = pred
                # inverse transform
                for transform in self.test_transforms[::-1]:
                    if (transform.inverse):
                        data = transform.inverse_transform_for_prediction(data) 

                infer_time = time.time() - start_time
                infer_time_list.append(infer_time)
                self.save_outputs(data)
        infer_time_list = np.asarray(infer_time_list)
        time_avg, time_std = infer_time_list.mean(), infer_time_list.std()
        logging.info("testing time {0:} +/- {1:}".format(time_avg, time_std))


    def write_scalars(self, train_scalars, valid_scalars, lr_value, glob_it):
        loss_scalar ={'train':train_scalars['loss'], 
                      'valid':valid_scalars['loss']}
        dice_scalar ={'valid':valid_scalars['avg_fg_dice']}
        self.summ_writer.add_scalars('loss', loss_scalar, glob_it)
        self.summ_writer.add_scalars('lr', {"lr": lr_value}, glob_it)
        self.summ_writer.add_scalars('dice', dice_scalar, glob_it)
        class_num = self.config['network']['class_num']
        for c in range(class_num):
            cls_dice_scalar = {'train':train_scalars['class_dice'][c], \
                'valid':valid_scalars['class_dice'][c]}
            self.summ_writer.add_scalars('class_{0:}_dice'.format(c), cls_dice_scalar, glob_it)
        logging.info('train loss {0:.4f}'.format(train_scalars['loss']))        
        logging.info('valid loss {0:.4f}, avg foreground dice {1:.4f} '.format(
            valid_scalars['loss'], valid_scalars['avg_fg_dice']) + "[" + \
            ' '.join("{0:.4f}".format(x) for x in valid_scalars['class_dice']) + "]")  
        logging.info("data: {0:.2f}s, forward: {1:.2f}s, loss: {2:.2f}s, backward: {3:.2f}s".format(
            train_scalars['data_time'], train_scalars['forward_time'], 
            train_scalars['loss_time'], train_scalars['backward_time']))

