import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .vq_module import Encoder
from .qantize import VectorQuantizer2 as VectorQuantizer
from .movq_module import MOVQDecoder
from einops import rearrange
from .ema import LitEma
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
from typing import Iterable

class MOVQ(pl.LightningModule):
    
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 learning_rate=None,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 monitor=None,
                 remap=None,
                 sane_index_shape=True,  # tell vector quantizer to return indices as bhw
                 ema_decay=None
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = MOVQDecoder(zq_ch=embed_dim, **ddconfig)
        self.scale = 8
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if monitor is not None:
            self.monitor = monitor
        
        if ema_decay is not None:
            self.use_ema = True
            print('use_ema = True')
            self.ema_encoder = LitEma(self.encoder, ema_decay)
            self.ema_decoder = LitEma(self.decoder, ema_decay)
            self.ema_quantize = LitEma(self.quantize, ema_decay) 
            self.ema_quant_conv = LitEma(self.quant_conv, ema_decay) 
            self.ema_post_quant_conv = LitEma(self.post_quant_conv, ema_decay)     
        else:
            self.use_ema = False
           
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant:torch.Tensor, mask:torch.Tensor=None):
        if mask is not None:
            quant = quant*self.code_mask(mask=mask, code=quant)
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec

    def decode_code(self, code_b:torch.Tensor, mask:torch.Tensor=None):
        batch_size, h, w = code_b.shape
        quant:torch.Tensor = self.quantize.embedding(code_b.view(batch_size, -1))
        quant = quant.view((batch_size, h, w, self.quantize.e_dim))
        quant = rearrange(quant, 'b h w c -> b c h w').contiguous()
        if mask is not None:
            quant = quant*self.code_mask(mask=mask, code=quant)
        
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec
    
    def code_mask(self, mask:torch.Tensor, code:torch.Tensor)->torch.Tensor:
        assert mask.shape[0] == code.shape[0]
        assert mask.shape[-2:] == code.shape[-2:]
        if mask.ndim == 3:
            # BHW
            mask = mask.unsqueeze(1)
        return mask.repeat(1, code.shape[1], 1, 1)

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
  
    def forward_ema(self, x):
        h = self.ema_encoder(x)
        h = self.ema_quant_conv(h)
        quant, emb_loss, info = self.ema_quantize(h)
        quant2 = self.ema_post_quant_conv(quant)
        dec = self.ema_decoder(quant2, quant)
        return dec, emb_loss
  
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_encoder(self.encoder)
            self.ema_decoder(self.decoder)
            self.ema_quantize(self.quantize)
            self.ema_quant_conv(self.quant_conv)
            self.ema_post_quant_conv(self.post_quant_conv)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = batch
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @classmethod
    def from_cfg(cls, cfg_file:os.PathLike, loss=None, sane_index_shape=True, on_device=torch.device('cuda'))->"MOVQ":
        cfg = OmegaConf.load(cfg_file)
        cfg['model']['params']['lossconfig'] = loss
        cfg['model']['params']['sane_index_shape'] = sane_index_shape
        model = cls(**cfg['model']['params'])
        model.eval()
        if on_device is not None:
            model = model.to(device=on_device)
        return model

    def pad_to_scale(self, img:np.ndarray, pad_value=0)->tuple[np.ndarray,tuple[int, int, int, int]]:
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        s = np.asarray(img.shape[:2])
        pad = (self.scale -s % self.scale )%self.scale
        p0 = pad//2 
        p1 = p0 + pad%2
        paded_img = np.full((*(s+p0+p1), img.shape[-1]), pad_value, dtype=img.dtype)
        top, bottom = p0[0], paded_img.shape[0] - p1[0]
        left, right = p0[1], paded_img.shape[1] - p1[1] 
        paded_img[top:bottom, left:right, :] = img
        return paded_img, (top, bottom, left, right)

    @torch.no_grad()
    def recons_cv2(self, img:np.ndarray, degrad_levels:Iterable[int]=(0,), pad_value:int=0)->tuple[list[np.ndarray], torch.Tensor]:
        
        if isinstance(degrad_levels, int):
            degrad_levels = [degrad_levels]
        
        img, (top, bottom, left, right) = self.pad_to_scale(img=img, pad_value=pad_value)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        img:torch.Tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
        img = img.to(device=self.device)
        
        idxs:torch.Tensor = self.encode(img/127.5 - 1)[-1][0]
        idxs = idxs[..., degrad_levels].chunk(len(degrad_levels), dim=-1)
        idxs = torch.concat(idxs, dim=0).squeeze(-1)
        
        rimg = self.decode_code(code_b=idxs)
        rimg = (torch.clamp(rimg, -1, 1) + 1) * 127.5

        recons_diff = torch.square(rimg - img).detach()
        
        rimg = rimg.detach().permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        rimg = [cv2.cvtColor(r[top:bottom, left:right], cv2.COLOR_RGB2BGR) for r in rimg]
        # recons_diff = recons_diff[:, :, top:bottom, left:right]
        recons_diff[:, :, :top, :left] = 0.0
        recons_diff[:, :, bottom:, right:] = 0.0
        return rimg, recons_diff, idxs, (top, bottom, left, right)
        # return rimg, recons_diff, idxs
