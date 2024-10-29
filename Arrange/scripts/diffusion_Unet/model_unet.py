
import sys
sys.path.append('.\Arrange')
sys.path.append('.\Arrange\scripts')
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from diffusion_Unet.diffusion_point import DiffusionPoint
from diffusion_Unet.denoise_net import Unet1D
from diffusion_Unet.denoise_net_cross_attension import UNet1DModel
from diffusion_Unet.stat_logger import StatsLogger
from graph_net import *
class DiffusionScene(Module):

    def __init__(self, config,embedding_dim=128,gconv_pooling='avg', gconv_num_layers=5):
        super().__init__()
        self.text_condition=config.get("text_condition",False)
        self.diffusion=DiffusionPoint(denoise_net = UNet1DModel(**config[" denoiser_kwargs"]),
            config = config,**config["diffusion_kwargs"]
            )   
        self.obj_embeddings_ec = nn.Embedding(num_objs + 1, gconv_dim * 2)#num_objs 是改场景下物体个数
        self.pred_embeddings_ec = nn.Embedding(num_preds, gconv_dim * 2)#num_pres是物体关系总数
        self.config=config
        gconv_dim = embedding_dim
        gconv_hidden_dim = gconv_dim * 4
        gconv_kwargs_ec = {
            'input_dim_obj': gconv_dim * 2 ,
            'input_dim_pred': gconv_dim * 2 ,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': None,
            'residual': False,
            'output_dim': gconv_dim * 2
        }
        self.gcn_conv_enc=GraphTripleConvNet(**gconv_kwargs_ec)

    def encoder(self,datum):
        objs=datum["calss_ids"]
        triples=datum["triples"]
        bbox_es=datum['bbox-es']
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_embed = self.obj_embeddings_ec(objs)
        pred_embed = self.pred_embeddings_ec(p)
        latent_obj_f, latent_pred_f = self.gconv_net_ec(obj_embed, pred_embed, edges)

        return obj_embed, pred_embed, latent_obj_f, latent_pred_f #obj_embed,latent_obj_f


    def get_loss(self,datum):

        class_ids=datum["class_ids"]
        class_ids_expanded = class_ids.unsqueeze(-1)  # 扩展一个维度，变为 [126, 7, 1]
        bbox_es=datum["bbox_es"]
        locations=datum["locations"]
        quaternion_xyzw=datum['quaternion_xyzw']  
        triples=datum['triples']
        # print("class_ids:", class_ids.shape)
        # print("bbox_es:", bbox_es.shape)
        # print("locations:", locations.shape)  # 这里可能会输出一个元组
        # print("quaternion_xyzw:", quaternion_xyzw.shape)
        layout_info=torch.cat([class_ids_expanded,bbox_es,locations,quaternion_xyzw],dim=-1).contiguous()
        
        obj_embed,_,latent_obj_f,_=self.encoder(datum)
        self.loss = self.diffusion.get_loss_iter_v2(obj_embed=obj_embed, obj_triples=triples, target_box=bbox_es, rel=latent_obj_f)
        return self.loss
        
        # loss,loss_dict=self.diffusion.get_loss_iter(
        #     layout_info,condition=None,condition_cross=None)
        
        # return loss,loss_dict
    def sample(self, room_mask, num_points, point_dim, batch_size=1, text=None, 
               partial_boxes=None, input_boxes=None, ret_traj=False, ddim=False, clip_denoised=False, freq=40, batch_seeds=None, 
                ):
        device = room_mask.device
        noise = torch.randn((batch_size, num_points, point_dim))#, device=room_mask.device)

        # get the latent feature of room_mask
        if self.room_mask_condition:
            room_layout_f = self.fc_room_f(self.feature_extractor(room_mask)) #(B, F)
            
        else:
            room_layout_f = None

        # process instance & class condition f
        if self.instance_condition:
            if self.learnable_embedding:
                instance_indices = torch.arange(self.sample_num_points).long().to(device)[None, :].repeat(room_mask.size(0), 1)
                instan_condition_f = self.positional_embedding[instance_indices, :]
            else:
                instance_label = torch.eye(self.sample_num_points).float().to(device)[None, ...].repeat(room_mask.size(0), 1, 1)
                instan_condition_f = self.fc_instance_condition(instance_label) 
        else:
            instan_condition_f = None


        # concat instance and class condition   
        # concat room_layout_f and instan_class_f
        if room_layout_f is not None and instan_condition_f is not None:
            condition = torch.cat([room_layout_f[:, None, :].repeat(1, num_points, 1), instan_condition_f], dim=-1).contiguous()
        elif room_layout_f is not None:
            condition = room_layout_f[:, None, :].repeat(1, num_points, 1)
        elif instan_condition_f is not None:
            condition = instan_condition_f
        else:
            condition = None

        # concat room_partial condition, use partial boxes as input for scene completion
        if self.room_partial_condition:
            zeros_boxes = torch.zeros((batch_size, num_points-partial_boxes.shape[1], partial_boxes.shape[2])).float().to(device)
            partial_input  =  torch.cat([partial_boxes, zeros_boxes], dim=1).contiguous()
            partial_condition_f = self.fc_partial_condition(partial_input)
            condition = torch.cat([condition, partial_condition_f], dim=-1).contiguous()

        # concat  room_arrange condition, use input boxes as input for scene completion
        if self.room_arrange_condition:
            arrange_input  = torch.cat([ input_boxes[:, :, self.translation_dim:self.translation_dim+self.size_dim], input_boxes[:, :, self.bbox_dim:] ], dim=-1).contiguous()
            arrange_condition_f = self.fc_arrange_condition(arrange_input)
            condition = torch.cat([condition, arrange_condition_f], dim=-1).contiguous()


        if self.text_condition:
            if self.text_glove_embedding:
                condition_cross = self.fc_text_f(text) #sample_params["desc_emb"]
            elif self.text_clip_embedding:
                tokenized = clip.tokenize(text).to(device)
                condition_cross = self.clip_model.encode_text(tokenized)
            else:
                tokenized = self.tokenizer(text, return_tensors='pt',padding=True).to(device)
                #print('tokenized:', tokenized.shape)
                text_f = self.bertmodel(**tokenized).last_hidden_state
                print('after bert:', text_f.shape)
                condition_cross = self.fc_text_f( text_f )
        else:
            condition_cross = None
            

        if input_boxes is not None:
            print('scene arrangement sampling')
            samples = self.diffusion.arrange_samples(noise.shape, room_mask.device, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised, input_boxes=input_boxes)

        elif partial_boxes is not None:
            print('scene completion sampling')
            samples = self.diffusion.complete_samples(noise.shape, room_mask.device, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised, partial_boxes=partial_boxes)

        else:
            print('unconditional / conditional generation sampling')
            # reverse sampling
            if ret_traj:
                samples = self.diffusion.gen_sample_traj(noise.shape, room_mask.device, freq=freq, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised)
            else:
                samples = self.diffusion.gen_samples(noise.shape, room_mask.device, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised)
            
        return samples
    

def train_on_batch(model, optimizer, sample_params, config):
    # Make sure that everything has the correct size
    optimizer.zero_grad()
    # Compute the loss
    loss, loss_dict = model.get_loss(sample_params)
    for k, v in loss_dict.items():
        StatsLogger.instance()[k].value = v.item()#遍历损失字典，记录每个损失项的值
    # Do the backpropagation
    loss.backward()
    # Compuite model norm
    grad_norm = clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])#梯度裁剪
    StatsLogger.instance()["gradnorm"].value = grad_norm.item()#记录梯度范数
    # log learning rate
    StatsLogger.instance()["lr"].value = optimizer.param_groups[0]['lr']#记录学习率
    # Do the update
    optimizer.step()
    return loss.item()


def validate_on_batch(model, sample_params, config):
    # Compute the loss
    loss, loss_dict = model.get_loss(sample_params)
    for k, v in loss_dict.items():
        StatsLogger.instance()[k].value = v.item()
    return loss.item()

