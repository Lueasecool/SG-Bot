
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
   
    def sample(self, box_dim, batch_size, obj_embed=None, obj_triples=None, text=None, rel=None, ret_traj=False, ddim=False, clip_denoised=False, freq=40, batch_seeds=None):

        noise_shape = (batch_size, box_dim)
        condition = rel if self.rel_condition else None
        condition_cross = None
        # reverse sampling
        samples = self.df.gen_samples_sg(noise_shape, obj_embed.device, obj_embed, obj_triples, condition=condition, clip_denoised=clip_denoised)
        
        return samples

    @torch.no_grad()
    def generate_layout_sg(self, box_dim, text=None, ret_traj=False, ddim=False, clip_denoised=False, batch_seeds=None):

        rel = self.rel
        obj_embed = self.uc_rel
        triples = self.preds

        samples = self.sample(box_dim, batch_size=len(obj_embed), obj_embed=obj_embed, obj_triples=triples, text=text, rel=rel, ret_traj=ret_traj, ddim=ddim, clip_denoised=clip_denoised, batch_seeds=batch_seeds)
        samples_dict = {
            "sizes": samples[:, 0:self.size_dim].contiguous(),
            "translations": samples[:, self.size_dim:self.size_dim + self.translation_dim].contiguous(),
            "angles": samples[:, self.size_dim + self.translation_dim:self.bbox_dim].contiguous(),
        }
        
        return samples_dict
            
    

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

