from __future__ import print_function

import open3d as o3d # open3d needs to be imported before other packages!
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from omegaconf import OmegaConf
from utils import render_box
from eval_helper import batch_torch_destandardize_box_params,postprocess_sincos2arctan,descale_box_params

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, type=str, default="/media/ymxlzgy/Data/Dataset/FRONT", help="dataset path")
parser.add_argument('--with_CLIP', type=bool, default=True, help="Load Feats directly instead of points.")

parser.add_argument('--manipulate', default=True, type=bool)
parser.add_argument('--exp', default='../experiments/layout_test', help='experiment name')
parser.add_argument('--epoch', type=str, default='100', help='saved epoch')
parser.add_argument('--render_type', type=str, default='txt2shape', help='retrieval, txt2shape, onlybox, echoscene')
parser.add_argument('--gen_shape', default=False, type=bool, help='infer diffusion')
parser.add_argument('--visualize', default=False, type=bool)
parser.add_argument('--export_3d', default=False, type=bool, help='Export the generated shapes and boxes in json files for future use')
parser.add_argument('--room_type', default='all', help='all, bedroom, livingroom, diningroom, library')

args = parser.parse_args()

def reseed(num):
    np.random.seed(num)
    torch.manual_seed(num)
    random.seed(num)

def validate_constrains_loop(modelArgs, test_dataset, model, epoch=None, normalized_file=None, cat2objs=None, datasize='large', gen_shape=False):

    test_dataloader_no_changes = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=0)

    vocab = test_dataset.vocab

    accuracy = {}
    for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']:
        # compute validation for these relation categories
        accuracy[k] = []

    for i, data in enumerate(test_dataloader_no_changes, 0):
        print(data['scan_id'])

        try:
            dec_objs, dec_triples = data['decoder']['objs'], data['decoder']['tripltes']
            instances = data['instance_id'][0]
            scan = data['scan_id'][0]
        except Exception as e:
            print(e)
            continue

        dec_objs, dec_triples = dec_objs.cuda(), dec_triples.cuda()
        encoded_dec_text_feat, encoded_dec_rel_feat = None, None
        if modelArgs['with_CLIP']:
            encoded_dec_text_feat, encoded_dec_rel_feat = data['decoder']['text_feats'].cuda(), data['decoder']['rel_feats'].cuda()

        all_pred_boxes = []
        all_pred_angles = []

        with torch.no_grad():

            data_dict = model.sample_box_and_shape(dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, gen_shape=gen_shape)

            boxes_pred, angles_pred = torch.concat((data_dict['sizes'],data_dict['translations']),dim=-1), data_dict['angles']
            shapes_pred = None
            try:
                shapes_pred = data_dict['shapes']
            except:
                print('no shape, only run layout branch.')
            if modelArgs['bin_angle']:
                angles_pred = -180 + (torch.argmax(angles_pred, dim=1, keepdim=True) + 1)* 15.0 # angle (previously minus 1, now add it back)
                 #将标准化的边界框参数转换回原始值
                boxes_pred_den = batch_torch_destandardize_box_params(boxes_pred, file=normalized_file) # mean, std
           
            else:
                #将正余弦的弧度值转换为角度
                angles_pred = postprocess_sincos2arctan(angles_pred) / np.pi * 180
                #标准化的边界框参数转换回原始值
                boxes_pred_den = descale_box_params(boxes_pred, file=normalized_file) # min, max


        if args.visualize:
            classes = sorted(list(set(vocab['object_idx_to_name'])))
            # layout and shape visualization through open3d
            print("rendering", [classes[i].strip('\n') for i in dec_objs])
            if model.type_ == 'echolayout':
                render_box(data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, datasize=datasize,
                classes=classes, render_type=args.render_type, store_img=False, render_boxes=False, visual=False, demo=False, without_lamp=True, store_path=modelArgs['store_path'])
            
            else:
                raise NotImplementedError

        all_pred_boxes.append(boxes_pred_den.cpu().detach())
        all_pred_angles.append(angles_pred.cpu().detach())
        # compute constraints accuracy through simple geometric rules
        accuracy = validate_constrains(dec_triples, boxes_pred_den, angles_pred, None, model.vocab, accuracy)

    keys = list(accuracy.keys())
    file_path_for_output = os.path.join(modelArgs['store_path'], f'{test_dataset.eval_type}_accuracy_analysis.txt')
    with open(file_path_for_output, 'w') as file:
        for dic, typ in [(accuracy, "acc")]:
            lr_mean = np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])])
            fb_mean = np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])])
            bism_mean = np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])])
            tash_mean = np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])])
            stand_mean = np.mean(dic[keys[8]])
            close_mean = np.mean(dic[keys[9]])
            symm_mean = np.mean(dic[keys[10]])
            total_mean = np.mean(dic[keys[11]])
            means_of_mean = np.mean([lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean])
            print(
                '{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}'.format(
                    typ, lr_mean,
                    fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
            print('means of mean: {:.2f}'.format(means_of_mean))
            file.write(
                '{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}\n'.format(
                    typ, lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
            file.write('means of mean: {:.2f}\n\n'.format(means_of_mean))


def evaluate():
    random.seed(48)
    torch.manual_seed(48)

    argsJson = os.path.join(args.exp, 'args.json')
    assert os.path.exists(argsJson), 'Could not find args.json for experiment {}'.format(args.exp)
    with open(argsJson) as j:
        modelArgs = json.load(j)
    normalized_file = os.path.join(args.dataset, 'centered_bounds_{}_trainval.txt').format(modelArgs['room_type'])
    test_dataset_rels_changes = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='val_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=True,
        eval=True,
        eval_type='relationship',
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=modelArgs['with_SDF'],
        large=modelArgs['large'],
        room_type=args.room_type,
        recompute_clip=False)

    test_dataset_addition_changes = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='val_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=True,
        eval=True,
        eval_type='addition',
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=modelArgs['with_SDF'],
        large=modelArgs['large'],
        room_type=args.room_type)

    test_dataset_no_changes = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='val_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=False,
        eval=True,
        eval_type='none',
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=modelArgs['with_SDF'],
        large=modelArgs['large'],
        room_type=args.room_type)

    modeltype_ = modelArgs['network_type']
    modelArgs['store_path'] = os.path.join(args.exp, "vis", args.epoch)
    replacelatent_ = modelArgs['replace_latent'] if 'replace_latent' in modelArgs else None
    with_changes_ = modelArgs['with_changes'] if 'with_changes' in modelArgs else None
    # args.visualize = False if args.gen_shape == False else args.visualize

    # instantiate the model
    diff_opt = modelArgs['diff_yaml']
    diff_cfg = OmegaConf.load(diff_opt)
    diff_cfg.layout_branch.diffusion_kwargs.train_stats_file = test_dataset_no_changes.box_normalized_stats
    diff_cfg.layout_branch.denoiser_kwargs.using_clip = modelArgs['with_CLIP']
    model = SGDiff(type=modeltype_, diff_opt=diff_cfg, vocab=test_dataset_no_changes.vocab, replace_latent=replacelatent_,
                with_changes=with_changes_, residual=modelArgs['residual'], gconv_pooling=modelArgs['pooling'], clip=modelArgs['with_CLIP'],
                with_angles=modelArgs['with_angles'], separated=modelArgs['separated'])
    model.diff.optimizer_ini()
    model.load_networks(exp=args.exp, epoch=args.epoch, restart_optim=True, load_shape_branch=args.gen_shape)
    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()
    cat2objs = None

    print('\nEditing Mode - Additions')
    reseed(47)
    # validate_constrains_loop_w_changes(modelArgs, test_dataset_addition_changes, model, normalized_file=normalized_file, bin_angles=modelArgs['bin_angle'], cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\nEditing Mode - Relationship changes')
    # validate_constrains_loop_w_changes(modelArgs, test_dataset_rels_changes, model,  normalized_file=normalized_file, bin_angles=modelArgs['bin_angle'], cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\nGeneration Mode')
    validate_constrains_loop(modelArgs, test_dataset_no_changes, model, epoch=args.epoch, normalized_file=normalized_file, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

if __name__ == "__main__":
    print(torch.__version__)
    evaluate()