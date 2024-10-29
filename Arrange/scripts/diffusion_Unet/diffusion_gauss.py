import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.distributions import Normal
import torch.distributed as dist
import math
import numpy as np
import torch.distributed as dist
from tqdm.auto import tqdm
import json
import torch.nn.functional as F
from functools import partial
from collections import namedtuple
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))



def identity(t, *args, **kwargs):
    return t


class GaussianDiffusion:
    def __init__(self, config, betas, loss_type, model_mean_type, model_var_type, loss_separate, loss_iou):
        # read object property dimension
        super(GaussianDiffusion,self).__init__()
        self.objectness_dim = config.get("objectness_dim", 1)
        self.class_dim = config.get("class_dim", 21)
        self.translation_dim = config.get("translation_dim", 3)
        self.size_dim = config.get("size_dim", 3)
        self.angle_dim = config.get("angle_dim", 1)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.objfeat_dim = config.get("objfeat_dim", 0)
        self.loss_separate = loss_separate
        self.loss_iou = loss_iou
        # if self.loss_iou:
            # with open(train_stats_file, "r") as f:
            #     train_stats = json.load(f)
            # self._centroids = train_stats["bounds_translations"]
            # self._centroids = (np.array(self._centroids[:3]), np.array(self._centroids[3:]))
            # self._centroids_min, self._centroids_max = torch.from_numpy(self._centroids[0]).float(), torch.from_numpy(self._centroids[1]).float()
            # print('load centriods min {} and max {} in Gausssion Diffusion'.format(self._centroids[0], self._centroids[1]))
            
            # self._sizes = train_stats["bounds_sizes"]
            # self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
            # self._sizes_min, self._sizes_max = torch.from_numpy(self._sizes[0]).float(), torch.from_numpy(self._sizes[1]).float()
            # print('load sizes min {} and max {} in Gausssion Diffusion'.format( self._sizes[0], self._sizes[1] ))
            
            # self._angles = train_stats["bounds_angles"]
            # self._angles = (np.array(self._angles[0]), np.array(self._angles[1]))

        self.room_partial_condition = config.get("room_partial_condition", False)
        self.room_arrange_condition = config.get("room_arrange_condition", False)

        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        # calculate loss weight
        snr = alphas_cumprod / (1 - alphas_cumprod)

        if model_mean_type == 'eps':
            loss_weight = torch.ones_like(snr)
        elif model_mean_type == 'x0':
            loss_weight = snr
        elif model_mean_type == 'v':
            loss_weight = snr / (snr + 1)
        self.loss_weight = loss_weight

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )
    
    def _predict_eps_from_start(self, x_t, t, x0):
        return (
            (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - x0) / \
            self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape)
        )
        
    def _predict_v(self, x0, t, eps):
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x0.device), t, x0.shape) * eps -
            self._extract(self.sqrt_one_minus_alphas_cumprod.to(x0.device), t, x0.shape) * x0
        )

    def _predict_start_from_v(self, x_t, t, v):
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
            self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_t.device), t, x_t.shape) * v
        )
        
    def model_predictions(self, denoise_fn, x_t, t, condition, condition_cross, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False): 
        model_output = denoise_fn(x_t, t, condition, condition_cross) 
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.model_mean_type == 'eps':
            pred_noise = model_output
            x_start = self._predict_xstart_from_eps(x_t, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self._predict_eps_from_start(x_t, t, x_start)

        elif self.model_mean_type == 'x0': 
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self._predict_eps_from_start(x_t, t, x_start)

        elif self.model_mean_type == 'v':
            v = model_output
            x_start = self._predict_start_from_v(x_t, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self._predict_eps_from_start(x_t, t, x_start)

        return ModelPrediction(pred_noise, x_start)


    def q_mean_variance(self, x_start, t):  
        """
        diffusion step: q(x_t | x_{t-1})
        """
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):#根据公式计算扩散后的数据
        """
        Diffuse the data (t == 0 means diffused for 1 step)   q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, condition, condition_cross, clip_denoised: bool, return_pred_xstart: bool):

        preds = self.model_predictions(denoise_fn, data, t, condition, condition_cross, x_self_cond=None)
        x_recon = preds.pred_x_start

        if clip_denoised:
            x_recon.clamp_(-1., 1.)


        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, condition, condition_cross, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, shape, device, condition, condition_cross,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq, condition, condition_cross,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0,total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False)
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs
    
    ## from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
    @torch.no_grad()
    def ddim_sample_loop(self, denoise_fn, shape, device, condition, condition_cross, noise_fn=torch.randn, clip_denoised=True, sampling_timesteps=50, ddim_sampling_eta=0., return_all_timesteps = False):
        self.ddim_sampling_eta = ddim_sampling_eta
        self.sampling_timesteps = sampling_timesteps
        batch, total_timesteps, sampling_timesteps, eta = shape[0], self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = noise_fn(size=shape, dtype=torch.float, device=device) 
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(time)
    
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, t_, condition, condition_cross, self_cond, clip_x_start = True)
                
                
            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = noise_fn(size=shape, dtype=torch.float, device=device)  #torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else imgs

        return ret
    

    def p_sample_loop_complete(self, denoise_fn, shape, device, condition, condition_cross,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False, partial_boxes=None):
        """
        Complete samples based on partial samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)

            # diffusion clean scenes
            noise =  noise_fn(size=partial_boxes.shape, dtype=torch.float, device=device)
            partial_boxes_t = self.q_sample(x_start=partial_boxes, t=t_, noise=noise)
            num_partial = partial_boxes_t.shape[1]

            # combine noisy version of clean scenes & denoising scenes
            img_t = torch.cat([ partial_boxes_t, img_t[:, num_partial:, :] ], dim=1).contiguous()

            # reverse diffusion
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                                clip_denoised=clip_denoised, return_pred_xstart=False)
            if t == 0:
                print('last:', t, self.num_timesteps, len(self.betas))
                img_t = torch.cat([ partial_boxes, img_t[:, num_partial:, :] ], dim=1).contiguous()

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_arrange(self, denoise_fn, shape, device, condition, condition_cross,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False, input_boxes=None):
        """
        Arrangement: complete other properies based on some propeties
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=(shape[0], shape[1], self.translation_dim+self.angle_dim), dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)

            # reverse diffusion
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                                clip_denoised=clip_denoised, return_pred_xstart=False)
            if t == 0:
                print('last:', t, self.num_timesteps, len(self.betas))
                img_t_trans = img_t[:, :, 0:self.translation_dim]
                img_t_angle = img_t[:, :, self.translation_dim:] 
                
                input_boxes_trans = input_boxes[:, :, 0:self.translation_dim]
                input_boxes_size  = input_boxes[:, :, self.translation_dim:self.translation_dim+self.size_dim]  
                input_boxes_angle = input_boxes[:, :, self.translation_dim+self.size_dim:self.bbox_dim] 
                input_boxes_other = input_boxes[:, :, self.bbox_dim:] 
                img_t = torch.cat([ img_t_trans, input_boxes_size, img_t_angle, input_boxes_other ], dim=-1).contiguous()

        assert img_t.shape == shape
        return img_t


    '''losses'''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, condition, condition_cross, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=data_start, x_t=data_t, t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t, t=t, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised, return_pred_xstart=True)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(data_start.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, obj_embed, triples, t, condition_cross=None):
        """
        Training loss calculation
        """
        #B, D, N = data_start.shape
        # make it compatible for 1D 
        if len(data_start.shape) == 3:
            B, D, N = data_start.shape
        elif len(data_start.shape) == 4:
            B, D, M, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(data_start.shape, dtype=data_start.dtype, device=data_start.device)
        assert noise.shape == data_start.shape and noise.dtype == data_start.dtype

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)#在第t步扩散后的数据

        if self.loss_type == 'mse':                                 #!!mse
            if self.model_mean_type == 'eps':
                target = noise   #                            !!exp
            elif self.model_mean_type == 'x0':
                target = data_start
            elif self.model_mean_type == 'v':
                target = self._predict_v(data_start, t, noise)
            else:
                raise NotImplementedError
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            #eps_recon = denoise_fn(data_t, t, condition, condition_cross)
            #denoise_out = denoise_fn(data_t, t, condition, condition_cross)#经过u-Net输出的预测噪声
            denoise_out = denoise_fn(data_t, obj_embed, triples, t, condition_cross)
            assert data_t.shape == data_start.shape
        loss_size = ((target[:, 0:self.size_dim] - denoise_out[:, 0:self.size_dim]) ** 2).mean(
            dim=list(range(1, len(data_t.shape))))
        loss_trans = ((target[:, self.size_dim:self.size_dim + self.translation_dim] - denoise_out[:,
                                                                                       self.size_dim:self.size_dim + self.translation_dim]) ** 2).mean(
            dim=list(range(1, len(data_t.shape))))
        loss_angle = ((target[:, self.size_dim + self.translation_dim:self.bbox_dim] - denoise_out[:,
                                                                                       self.size_dim + self.translation_dim:self.bbox_dim]) ** 2).mean(
            dim=list(range(1, len(data_t.shape))))
        loss_bbox = ((target[:, 0:self.bbox_dim] - denoise_out[:, 0:self.bbox_dim]) ** 2).mean(
            dim=list(range(1, len(data_t.shape))))
        losses = ((target - denoise_out) ** 2).mean(dim=list(range(1, len(data_t.shape))))

            

            
        return losses, {
            'loss.bbox': loss_bbox.mean(),
            'loss.trans': loss_trans.mean(),
            'loss.size': loss_size.mean(),
            'loss.angle': loss_angle.mean(),
        }
           
                   
    
    def descale_to_origin(self, x, minimum, maximum):
        '''
            x shape : BxNx3
            minimum, maximum shape: 3
        '''
        x = (x + 1) / 2
        x = x * (maximum - minimum)[None, None, :] + minimum[None, None, :]
        return x

    '''debug'''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, condition, condition_cross, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_= torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):

                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn, data_start=x_start, data_t=self.q_sample(x_start=x_start, t=t_b), t=t_b, condition=condition, condition_cross=condition_cross,
                    clip_denoised=clip_denoised, return_pred_xstart=True)
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start.shape
                new_mse_b = ((pred_xstart-x_start)**2).mean(dim=list(range(1, len(x_start.shape))))
                assert new_vals_b.shape == new_mse_b.shape ==  torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None]==torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start)
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                   total_bpd_b.shape == prior_bpd_b.shape ==  torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()
def axis_aligned_bbox_overlaps_3d(bboxes1,
                                  bboxes2,
                                  mode='iou',
                                  is_aligned=False,
                                  eps=1e-6):
    """Calculate overlap between two set of axis aligned 3D bboxes. If
        ``is_aligned`` is ``False``, then calculate the overlaps between each bbox
        of bboxes1 and bboxes2, otherwise the overlaps between each aligned pair of
        bboxes1 and bboxes2.
        Args:
            bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>
                format or empty.
            bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>
                format or empty.
                B indicates the batch dim, in shape (B1, B2, ..., Bn).
                If ``is_aligned`` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or "giou" (generalized
                intersection over union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Defaults to False.
            eps (float, optional): A value added to the denominator for numerical
                stability. Defaults to 1e-6.
        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """

    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimension is 6
    assert (bboxes1.size(-1) == 6 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 6 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 3] -
             bboxes1[..., 0]) * (bboxes1[..., 4] - bboxes1[..., 1]) * (
                 bboxes1[..., 5] - bboxes1[..., 2])
    area2 = (bboxes2[..., 3] -
             bboxes2[..., 0]) * (bboxes2[..., 4] - bboxes2[..., 1]) * (
                 bboxes2[..., 5] - bboxes2[..., 2])

    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])  # [B, rows, 3]
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])  # [B, rows, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(bboxes1[..., :, None, :3],
                       bboxes2[..., None, :, :3])  # [B, rows, cols, 3]
        rb = torch.min(bboxes1[..., :, None, 3:],
                       bboxes2[..., None, :, 3:])  # [B, rows, cols, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 3]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :3],
                                    bboxes2[..., None, :, :3])
            enclosed_rb = torch.max(bboxes1[..., :, None, 3:],
                                    bboxes2[..., None, :, 3:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
