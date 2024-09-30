# 3D version of TransUNet; Copyright Johns Hopkins University
# Modified from nnUNet



import torch
import numpy as np
import torch.nn.functional
import torch.nn.functional as F

from copy import deepcopy
from torch import nn
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment

from ..networks.neural_network import SegmentationNetwork
from .vit_modeling import Transformer
from .vit_modeling import CONFIGS as CONFIGS_ViT

softmax_helper = lambda x: F.softmax(x, 1)

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)

class Generic_TransUNet_max_ppbp(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False, # TODO default False
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False,
                 patch_size=None, is_vit_pretrain=False, 
                 vit_depth=12, vit_hidden_size=768, vit_mlp_dim=3072, vit_num_heads=12,
                 max_msda='', is_max_ms=True, is_max_ms_fpn=False, max_n_fpn=4, max_ms_idxs=[-4,-3,-2], max_ss_idx=0,
                 is_max_bottleneck_transformer=False, max_seg_weight=1.0, max_hidden_dim=256, max_dec_layers=10,
                 mw = 0.5,
                 is_max=True, is_masked_attn=False, is_max_ds=False, is_masking=False, is_masking_argmax=False,
                 is_fam=False, fam_k=5, fam_reduct_ratio=8,
                 is_max_hungarian=False, num_queries=None, is_max_cls=False,
                 point_rend=False, num_point_rend=None, no_object_weight=None, is_mhsa_float32=False, no_max_hw_pe=False,
                 max_infer=None, cost_weight=[2.0, 5.0, 5.0], vit_layer_scale=False, decoder_layer_scale=False):

        super(Generic_TransUNet_max_ppbp, self).__init__()

        # newly added
        self.is_fam = is_fam
        self.is_max, self.max_msda, self.is_max_ms, self.is_max_ms_fpn, self.max_n_fpn, self.max_ss_idx, self.mw = is_max, max_msda, is_max_ms, is_max_ms_fpn, max_n_fpn, max_ss_idx, mw
        self.max_ms_idxs = max_ms_idxs

        self.is_max_cls = is_max_cls
        self.is_masked_attn, self.is_max_ds = is_masked_attn, is_max_ds
        self.is_max_bottleneck_transformer = is_max_bottleneck_transformer

        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []


        self.fams = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))



        if self.is_fam:
            self.fams = nn.ModuleList(self.fams)

        if self.do_ds:
            self.seg_outputs = []
            for ds in range(len(self.conv_blocks_localization)):
                self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                                1, 1, 0, 1, 1, seg_output_use_bias))
            self.seg_outputs = nn.ModuleList(self.seg_outputs)

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)

        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

        # Transformer configuration
        if self.is_max_bottleneck_transformer:
            self.patch_size = patch_size  # e.g. [48, 192, 192]
            config_vit = CONFIGS_ViT['R50-ViT-B_16']
            config_vit.transformer.num_layers = vit_depth
            config_vit.hidden_size = vit_hidden_size # 768
            config_vit.transformer.mlp_dim = vit_mlp_dim # 3072
            config_vit.transformer.num_heads = vit_num_heads # 12
            self.conv_more = nn.Conv3d(config_vit.hidden_size, output_features, 1)
            num_pool_per_axis = np.prod(np.array(pool_op_kernel_sizes), axis=0)
            num_pool_per_axis = np.log2(num_pool_per_axis).astype(np.uint8)
            feat_size = [int(self.patch_size[0]/2**num_pool_per_axis[0]), int(self.patch_size[1]/2**num_pool_per_axis[1]), int(self.patch_size[2]/2**num_pool_per_axis[2])]
            self.transformer = Transformer(config_vit, feat_size=feat_size, vis=False, feat_channels=output_features, use_layer_scale=vit_layer_scale)
            if is_vit_pretrain:
                self.transformer.load_from(weights=np.load(config_vit.pretrained_path))


        if self.is_max:
            # Max PPB+ configuration (i.e. MultiScaleStandardTransformerDecoder)
            cfg = {
                    "num_classes": num_classes,
                    "hidden_dim": max_hidden_dim,
                    "num_queries": num_classes if num_queries is None else num_queries, # N=K if 'fixed matching', else default=100,
                    "nheads": 8,
                    "dim_feedforward": max_hidden_dim * 8, # 2048,
                    "dec_layers": max_dec_layers, # 9 decoder layers, add one for the loss on learnable query?
                    "pre_norm": False,
                    "enforce_input_project": False,
                    "mask_dim": max_hidden_dim, # input feat of segm head?
                    "non_object": False,
                    "use_layer_scale": decoder_layer_scale,
                }
            cfg['non_object'] = is_max_cls
            input_proj_list = [] # from low resolution to high resolution (res4 -> res1), [1, 1024, 14, 14], [1, 512, 28, 28], 1, 256, 56, 56], [1, 64, 112, 112]
            decoder_channels = [320, 320, 256, 128, 64, 32]
            if self.is_max_ms: # use multi-scale feature as Transformer decoder input
                if self.is_max_ms_fpn:
                    for idx, in_channels in enumerate(decoder_channels[:max_n_fpn]): # max_n_fpn=4: 1/32, 1/16, 1/8, 1/4
                        input_proj_list.append(nn.Sequential(
                            nn.Conv3d(in_channels, max_hidden_dim, kernel_size=1),
                            nn.GroupNorm(32, max_hidden_dim),
                            nn.Upsample(size=(int(patch_size[0]/2), int(patch_size[1]/4), int(patch_size[2]/4)), mode='trilinear')
                        )) # proj to scale (1, 1/2, 1/2), TODO: init
                    self.input_proj = nn.ModuleList(input_proj_list)
                    self.linear_encoder_feature = nn.Conv3d(max_hidden_dim * max_n_fpn, max_hidden_dim, 1, 1) # concat four-level feature
                else:
                    for idx, in_channels in enumerate([decoder_channels[i] for i in self.max_ms_idxs]):
                        input_proj_list.append(nn.Sequential(
                            nn.Conv3d(in_channels, max_hidden_dim, kernel_size=1),
                            nn.GroupNorm(32, max_hidden_dim),
                        ))
                    self.input_proj = nn.ModuleList(input_proj_list)

                # self.linear_mask_features =nn.Conv3d(decoder_channels[max_n_fpn-1], cfg["mask_dim"], kernel_size=1, stride=1, padding=0,) # low-level feat, dot product Trans-feat
                self.linear_mask_features =nn.Conv3d(decoder_channels[-1], cfg["mask_dim"], kernel_size=1, stride=1, padding=0,) # following SingleScale, high-level feat, obtain seg_map
            else:
                self.linear_encoder_feature = nn.Conv3d(decoder_channels[max_ss_idx], cfg["mask_dim"], kernel_size=1)
                self.linear_mask_features = nn.Conv3d(decoder_channels[-1], cfg["mask_dim"], kernel_size=1, stride=1, padding=0,) # low-level feat, dot product Trans-feat

            if self.is_masked_attn:
                from .mask2former_modeling.transformer_decoder.mask2former_transformer_decoder3d import MultiScaleMaskedTransformerDecoder3d
                cfg['num_feature_levels'] = 1 if not self.is_max_ms or self.is_max_ms_fpn else 3
                cfg["is_masking"] = True if is_masking else False
                cfg["is_masking_argmax"] = True if is_masking_argmax else False
                cfg["is_mhsa_float32"] = True if is_mhsa_float32 else False
                cfg["no_max_hw_pe"] = True if no_max_hw_pe else False
                self.predictor = MultiScaleMaskedTransformerDecoder3d(in_channels=max_hidden_dim, mask_classification=is_max_cls, **cfg)
            else:
                from .mask2former_modeling.transformer_decoder.maskformer_transformer_decoder3d import StandardTransformerDecoder
                cfg["dropout"], cfg["enc_layers"], cfg["deep_supervision"] = 0.1, 0, False
                self.predictor = StandardTransformerDecoder(in_channels=max_hidden_dim, mask_classification=is_max_cls, **cfg)

    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        
        x = self.conv_blocks_context[-1](x)
        ######### TransUNet #########
        if self.is_max_bottleneck_transformer:
            x, attn = self.transformer(x) # [b, hidden, d/8, h/16, w/16]
            x = self.conv_more(x)
        #############################

        ds_feats = [] # obtain multi-scale feature
        ds_feats.append(x)
        for u in range(len(self.tu)):
            if  u<len(self.tu)-1 and isinstance(self.is_fam, str) and self.is_fam.startswith('fam_down'):
                skip_down = nn.Upsample(size=x.shape[2:])(skips[-(u + 1)]) if x.shape[2:]!=skips[-(u + 1)].shape[2:] else skips[-(u + 1)]
                x_align = self.fams[u](x, x_l=skip_down)
                x = x + x_align

            x = self.tu[u](x) # merely an upsampling or transposeconv operation

            if isinstance(self.is_fam, bool) and self.is_fam:
                x_align = self.fams[u](x, x_l=skips[-(u + 1)])
                x = x + x_align
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            if self.do_ds:
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            ds_feats.append(x)

        ######### Max PPB+ #########
        if self.is_max:
            if self.is_max_ms: # is_max_ms_fpn
                multi_scale_features = []
                ms_pixel_feats = ds_feats[:self.max_n_fpn] if self.is_max_ms_fpn else [ds_feats[i] for i in self.max_ms_idxs]
                
                for idx, f in enumerate(ms_pixel_feats): 

                    f = self.input_proj[idx](f) # proj into same spatial/channel dim , but transformer_decoder also project to same mask_dim 
                    multi_scale_features.append(f)
                transformer_decoder_in_feature = self.linear_encoder_feature(torch.cat(multi_scale_features, dim=1)) if self.is_max_ms_fpn else multi_scale_features  # feature pyramid
                mask_features = self.linear_mask_features(ds_feats[-1]) # following SingleScale
            else:
                transformer_decoder_in_feature = self.linear_encoder_feature(ds_feats[self.max_ss_idx])
                mask_features = self.linear_mask_features(ds_feats[-1])
            
            predictions = self.predictor(transformer_decoder_in_feature, mask_features, mask=None)

            if self.is_max_cls and self.is_max_ds:
                if self._deep_supervision and self.do_ds:
                    return [predictions] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]
                return predictions

            elif self.is_max_ds and not self.is_max_ms and self.mw==1.0: # aux output of max decoder
                aux_out = [p['pred_masks'] for p in predictions['aux_outputs']] # ascending order
                all_out =  [predictions["pred_masks"]] + aux_out[::-1] # reverse order, w/o sigmoid activation
                return tuple(all_out)
            elif not self.is_max_ds and self.mw==1.0:
                raise NotImplementedError
            else:
                raise NotImplementedError

        #############################

        if self._deep_supervision and self.do_ds: # assuming turn off ds
            return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp

default_dict = {
    "base_num_features": 32,
    "conv_per_stage": 2,
    "initial_lr": 0.01,
    "lr_scheduler": None,
    "lr_scheduler_eps": 0.001,
    "lr_scheduler_patience": 30,
    "lr_threshold": 1e-06,
    "max_num_epochs": 1000,
    "net_conv_kernel_sizes": [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    "net_num_pool_op_kernel_sizes": [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    "net_pool_per_axis": [4, 5, 5],
    "num_batches_per_epoch": 250,
    "num_classes": 17,
    "num_input_channels": 1,
    "transpose_backward": [0, 1, 2],
    "transpose_forward": [0, 1, 2],
}


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule



def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher3D(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    def compute_cls_loss(self, inputs, targets):
        """ Classification loss (NLL)
            implemented in compute_loss()
        """
        raise NotImplementedError 

        
    def compute_dice_loss(self, inputs, targets):
        """ mask dice loss
            inputs (B*K, C, H, W)
            target (B*K, D, H, W)
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        num_masks = len(inputs)

        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks


    def compute_ce_loss(self, inputs, targets):
        """mask ce loss"""
        num_masks = len(inputs)
        loss = F.binary_cross_entropy_with_logits(inputs.flatten(1), targets.flatten(1), reduction="none")
        loss = loss.mean(1).sum() / num_masks
        return loss

    def compute_dice(self, inputs, targets):
        """ output (N_q, C, H, W)
            target (K, D, H, W)
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss # [N_q, K]


    def compute_ce(self, inputs, targets):
        """ output (N_q, C, H, W)
            target (K, D, H, W)
            return (N_q, K)
        """
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        hw = inputs.shape[1]

        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )

        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )

        loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
            "nc,mc->nm", neg, (1 - targets)
        )

        return loss / hw

        # target_onehot = torch.zeros_like(output, device=output.device)
        # target_onehot.scatter_(1, target.long(), 1)
        # assert (torch.argmax(target_onehot, dim=1) == target[:, 0].long()).all()
        # ce_loss = F.binary_cross_entropy_with_logits(output, target_onehot)
        # return ce_loss


    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching for single aux, outputs: (b, q, d, h, w)"""
        """suppose each crop must contain foreground class"""
        bs, num_queries =  outputs["pred_logits"].shape[:2]
        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1) # [num_queries, num_classes+1]
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]

            tgt_ids = targets[b]["labels"]
            tgt_mask = targets[b]["masks"].to(out_mask) # [K, D, H, W], K is number of classes shown in this image, and K < n_class

            # target_onehot = torch.zeros_like(tgt_mask, device=out_mask.device)
            # target_onehot.scatter_(1, targets.long(), 1)

            cost_class = -out_prob[:, tgt_ids] # [num_queries, K]

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                cost_dice = self.compute_dice(out_mask, tgt_mask)
                cost_mask = self.compute_ce(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )

            C = C.reshape(num_queries, -1).cpu() # (num_queries, K)

            # linear_sum_assignment return a tuple of two arrays: row_ind, col_ind, the length of array is min(N_q, K)
            # The cost of the assignment can be computed as cost_matrix[row_ind, col_ind].sum()

            indices.append(linear_sum_assignment(C))

        final_indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

        return final_indices

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


def compute_loss_hungarian(outputs, targets, idx, matcher, num_classes, point_rend=False, num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75, no_object_weight=None, cost_weight=[2,5,5]):
    """output is a dict only contain keys ['pred_masks', 'pred_logits'] """
    # outputs_without_aux = {k: v for k, v in output.items() if k != "aux_outputs"}

    indices = matcher(outputs, targets) 
    src_idx = matcher._get_src_permutation_idx(indices) # return a tuple of (batch_idx, src_idx)
    tgt_idx = matcher._get_tgt_permutation_idx(indices) # return a tuple of (batch_idx, tgt_idx)
    assert len(tgt_idx[0]) ==  sum([len(t["masks"]) for t in targets]) # verify that all masks of (K1, K2, ..) are used
    
    # step2 : compute mask loss
    src_masks = outputs["pred_masks"]
    src_masks = src_masks[src_idx] # [len(src_idx[0]), D, H, W] -> (K1+K2+..., D, H, W) 
    target_masks = torch.cat([t["masks"] for t in targets], dim=0) # (K1+K2+..., D, H, W) actually
    src_masks = src_masks[:, None] # [K..., 1, D, H, W]
    target_masks = target_masks[:, None]

    if point_rend: # only calculate hard example
        with torch.no_grad():
            # num_points=12544 config in cityscapes

            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.float(),
                lambda logits: calculate_uncertainty(logits),
                num_points,
                oversample_ratio,
                importance_sample_ratio,
            ) # [K, num_points=12544, 3]

            point_labels = point_sample_3d(
                target_masks.float(),
                point_coords.float(),
                align_corners=False,
            ).squeeze(1) # [K, 12544]

        point_logits = point_sample_3d(
                src_masks.float(),
                point_coords.float(),
                align_corners=False,
        ).squeeze(1) # [K, 12544]

        src_masks, target_masks = point_logits, point_labels

    loss_mask_ce = matcher.compute_ce_loss(src_masks, target_masks) 
    loss_mask_dice = matcher.compute_dice_loss(src_masks, target_masks)
    
    # step3: compute class loss
    src_logits = outputs["pred_logits"].float() # (B, num_query, num_class+1)
    target_classes_o = torch.cat([t["labels"] for t in targets], dim=0) # (K1+K2+, )
    target_classes = torch.full(
            src_logits.shape[:2], num_classes, dtype=torch.int64, device=src_logits.device
        ) # (B, num_query, num_class+1)
    target_classes[src_idx] = target_classes_o


    if no_object_weight is not None:
        empty_weight = torch.ones(num_classes + 1).to(src_logits.device)
        empty_weight[-1] = no_object_weight
        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
    else:
        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes)

    loss = (cost_weight[0]/10)*loss_cls + (cost_weight[1]/10)*loss_mask_ce + (cost_weight[2]/10)*loss_mask_dice # 2:5:5, like hungarian matching
    # print("idx {}, loss {}, loss_cls {}, loss_mask_ce {}, loss_mask_dice {}".format(idx, loss, loss_cls, loss_mask_ce, loss_mask_dice))
    return loss


def point_sample_3d(input, point_coords, **kwargs):
    """
    from detectron2.projects.point_rend.point_features
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, D, H, W) that contains features map on a D x H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 3) or (N, Dgrid, Hgrid, Wgrid, 3) that contains
        [0, 1] x [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Dgrid, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2).unsqueeze(2) # why
    
    # point_coords should be (N, D, H, W, 3)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)

    if add_dim:
        output = output.squeeze(3).squeeze(3)

    return output


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


# implemented!
def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    n_dim = 3
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio) # 12544 * 3, oversampled
    point_coords = torch.rand(num_boxes, num_sampled, n_dim, device=coarse_logits.device) # (K, 37632, 3); uniform dist [0, 1)
    point_logits = point_sample_3d(coarse_logits, point_coords, align_corners=False) # (K, 1, 37632)

    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points) # 9408
    
    num_random_points = num_points - num_uncertain_points # 3136
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None] # [K, 9408]

    point_coords = point_coords.view(-1, n_dim)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, n_dim
    )  # [K, 9408, 3]
    
    if num_random_points > 0:
        # from detectron2.layers import cat
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, n_dim, device=coarse_logits.device),
            ],
            dim=1,
        ) # [K, 12544, 3]

    return point_coords