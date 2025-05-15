# Copyright (c) 2023 Xiaobin-Rong.
# Adapted under MIT LICENSE.
# Source: https://github.com/Xiaobin-Rong/SEtrain

import sys
sys.path.append("models")
import os
import toml
import math
import torch
import shutil
import random
import argparse
import numpy as np
import torch.distributed as dist
import toml
from datetime import datetime
from tqdm import tqdm
from glob import glob
from pathlib import Path
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter
from distributed_utils import reduce_value
from transformers import get_cosine_schedule_with_warmup
from dataloader import LibriTTSDataset as Dataset
from wavtokenizer import WavTokenizer
from decoder.discriminator_dac import DACDiscriminator
from decoder.discriminators import (
MultiPeriodDiscriminator, 
MultiResolutionDiscriminator
)
from decoder.loss import (
DiscriminatorLoss,
GeneratorLoss,
FeatureMatchingLoss,
MelSpecReconstructionLoss,
DACGANLoss
)
from decoder.modules import safe_log
from decoder.helpers import plot_spectrogram_to_numpy
from decoder.pretrained_model import instantiate_class


seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic =True


def run(rank, config, args):
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12354'
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

    args.rank = rank
    args.device = torch.device(rank)
    
    collate_fn = Dataset.collate_fn if hasattr(Dataset, "collate_fn") else None
    # config['train_dataloader']['batch_size'] = config['train_dataloader']['batch_size'] // args.world_size
    shuffle = False if args.world_size > 1 else True

    train_dataset = Dataset(**config['train_dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.world_size > 1 else None
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    sampler=train_sampler,
                                                    **config['train_dataloader'],
                                                    shuffle=shuffle,
                                                    collate_fn=collate_fn)
    
    validation_dataset = Dataset(**config['validation_dataset'])
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset) if args.world_size > 1 else None
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                        sampler=validation_sampler,
                                                        **config['validation_dataloader'], 
                                                        shuffle=False,
                                                        collate_fn=collate_fn)
    
    generator = WavTokenizer(config).to(args.device)
    
    mpd = MultiPeriodDiscriminator(num_embeddings=4).to(args.device)
    mrd = MultiResolutionDiscriminator(num_embeddings=4).to(args.device)
    dacd = DACGANLoss(DACDiscriminator()).to(args.device)
    
    disc_loss = DiscriminatorLoss().to(args.device)
    gen_loss = GeneratorLoss().to(args.device)
    feat_loss = FeatureMatchingLoss().to(args.device)
    mel_loss = MelSpecReconstructionLoss(sample_rate=config['samplerate']).to(args.device)
    
    disc_params = [
        {'params': mpd.parameters()},
        {'params': mrd.parameters()},
        {'params': dacd.parameters()} 
    ]
   
    if args.world_size > 1:
        generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[rank], find_unused_parameters=True)
        mpd = torch.nn.parallel.DistributedDataParallel(mpd, device_ids=[rank])
        mrd = torch.nn.parallel.DistributedDataParallel(mrd, device_ids=[rank])
        dacd = torch.nn.parallel.DistributedDataParallel(dacd, device_ids=[rank])

    optimizer_g = torch.optim.AdamW(params=generator.parameters(), lr=config['optimizer']['lr'], betas=(0.8, 0.99), weight_decay=0.01)
    optimizer_d = torch.optim.AdamW(params=disc_params, lr=config['optimizer']['lr'], betas=(0.8, 0.99), weight_decay=0.01)

    scheduler_g = get_cosine_schedule_with_warmup(optimizer_g, **config['scheduler'])
    scheduler_d = get_cosine_schedule_with_warmup(optimizer_d, **config['scheduler'])

    trainer = Trainer(config=config, model=[generator, mpd, mrd, dacd],
                      optimizer=[optimizer_g, optimizer_d], 
                      scheduler=[scheduler_g, scheduler_d],
                      loss_func=[mel_loss, gen_loss, disc_loss, feat_loss],
                      train_dataloader=train_dataloader,
                      validation_dataloader=validation_dataloader, 
                      train_sampler=train_sampler, args=args)

    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()


class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_func,
                 train_dataloader, validation_dataloader, train_sampler, args):
        self.config = config
        self.generator = model[0]
        self.mpd = model[1]
        self.mrd = model[2]
        self.dacd = model[3]
        self.optimizer_g = optimizer[0]
        self.optimizer_d = optimizer[1]
        self.scheduler_g = scheduler[0]
        self.scheduler_d = scheduler[1]
        self.mel_loss = loss_func[0]
        self.gen_loss = loss_func[1]
        self.disc_loss = loss_func[2]
        self.feat_loss = loss_func[3]

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size

        self.default_fs = config['train_dataset']['default_fs']

        # training config
        config['DDP']['world_size'] = args.world_size
        
        self.lamda_mel_base = config['coeff']['mel']
        self.lamda_mel = self.lamda_mel_base
        self.lamda_comm = config['coeff']['commit']
        self.max_steps = config['scheduler']['num_training_steps']
        self.num_warmup_steps = config['scheduler']['num_warmup_steps']
        self.global_steps = 0
        
        self.bandwidths = config['network_config']['feature_extractor']['bandwidths']

        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']

        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
 
        else:
            self.exp_path = self.trainer_config['exp_path'] + '_' + self.trainer_config['resume_datetime']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')
        self.code_path = os.path.join(self.exp_path, 'codes')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.code_path, exist_ok=True)

        # save the config
        if self.rank == 0:
            shutil.copy2(__file__, self.exp_path)
            shutil.copy2(args.config, Path(self.exp_path) / 'config.yaml')
            shutil.copytree(Path(__file__).parent, self.code_path, dirs_exist_ok=True)
            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 1e8

        if self.resume:
            self._resume_checkpoint()

    def _set_train_mode(self):
        self.generator.train()
        self.mpd.train()
        self.mrd.train()
        self.dacd.train()
        
    def _set_eval_mode(self):
        self.generator.eval()
        self.mpd.eval()
        self.mrd.eval()
        self.dacd.eval()

    def _save_checkpoint(self, epoch, score):
        generator_dict = self.generator.module.state_dict() if self.world_size > 1 else self.generator.state_dict()
        mpd_dict = self.mpd.module.state_dict() if self.world_size > 1 else self.mpd.state_dict()
        mrd_dict = self.mrd.module.state_dict() if self.world_size > 1 else self.mrd.state_dict()
        dacd_dict = self.dacd.module.state_dict() if self.world_size > 1 else self.dacd.state_dict()
        state_dict = {'epoch': epoch,
                      'optimizer_g': self.optimizer_g.state_dict(),
                      'optimizer_d': self.optimizer_d.state_dict(),
                      'scheduler_g': self.scheduler_g.state_dict(),
                      'scheduler_d': self.scheduler_d.state_dict(),
                      'generator': generator_dict,
                      'mpd': mpd_dict,
                      'mrd': mrd_dict,
                      'dacd': dacd_dict}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(3)}.tar'))

        if score < self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score

    def _del_checkpoint(self, epoch, score):
        # Condition 1: epoch-1 is not a multiple of self.save_checkpoint_interval;
        # Condition 2: current score is best score.
        if (epoch - 1) % self.save_checkpoint_interval != 0 or score == self.best_score:
            prev_epoch = epoch - 1
            checkpoint_file = os.path.join(self.checkpoint_path, f'model_{str(prev_epoch).zfill(3)}.tar')
        
            if os.path.exists(checkpoint_file):
                try:
                    os.remove(checkpoint_file)
                    print(f"Deleted checkpoint: {checkpoint_file}")
                except Exception as e:
                    print(f"Failed to delete checkpoint {checkpoint_file}: {e}")

    def _resume_checkpoint(self):
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]

        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d'])
        if self.world_size > 1:
            self.generator.module.load_state_dict(checkpoint['generator'])
            self.mpd.module.load_state_dict(checkpoint['mpd'])
            self.mrd.module.load_state_dict(checkpoint['mrd'])
            self.dacd.module.load_state_dict(checkpoint['dacd'])
        
        else:
            self.generator.load_state_dict(checkpoint['generator'])
            self.mpd.load_state_dict(checkpoint['mpd'])
            self.mrd.load_state_dict(checkpoint['mrd'])
            self.dacd.load_state_dict(checkpoint['dacd'])

    def mel_loss_coeff_decay(self, current_step, num_cycles=0.5):
        if current_step < self.num_warmup_steps:
            return 1.0
        progress = float(current_step - self.num_warmup_steps) / float(
            max(1, self.max_steps - self.num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    def _train_epoch(self, epoch):
        total_loss = 0
        total_loss_comm = 0
        total_loss_adv = 0
        total_loss_feat = 0
        total_loss_dis = 0
        self.train_dataloader.dataset.sample_data_per_epoch()  ### for 2000h DNS3 dataset
        self.train_bar = tqdm(self.train_dataloader, ncols=150)

        generator_loss = self.dacd.module.generator_loss if self.world_size > 1 else self.dacd.generator_loss
        discriminator_loss = self.dacd.module.discriminator_loss if self.world_size > 1 else self.dacd.discriminator_loss

        for step, true_wav in enumerate(self.train_bar, 1):
            fs = self.default_fs
            true_wav = true_wav.to(self.device)     # (B, T) 

            # For generator
            bandwidth_id = torch.randint(low=0, high=len(self.bandwidths),
                                         size=(1,), device=self.device)
            esti_wav, commit_loss = self.generator(true_wav, bandwidth_id=bandwidth_id)
            
            loss_gen_dac, loss_fm_dac = generator_loss(esti_wav.unsqueeze(1), true_wav.unsqueeze(1))
            _, gen_score_mpd, fmap_rs_mpd, fmap_gs_mpd = self.mpd(
                y=true_wav, y_hat=esti_wav, bandwidth_id=bandwidth_id,
            )
            _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.mrd(
                y=true_wav, y_hat=esti_wav, bandwidth_id=bandwidth_id,
            )
            
            loss_gen_mpd, list_loss_gen_mpd = self.gen_loss(disc_outputs=gen_score_mpd)
            loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
            loss_gen_mpd = loss_gen_mpd / len(list_loss_gen_mpd)
            loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
            loss_fm_mpd = self.feat_loss(fmap_r=fmap_rs_mpd, fmap_g=fmap_gs_mpd) / len(fmap_rs_mpd)
            loss_fm_mrd = self.feat_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)
            
            loss_mel = self.lamda_mel * self.mel_loss(esti_wav, true_wav)
            loss_comm = self.lamda_comm * commit_loss
            loss_adv = loss_gen_mpd + loss_fm_mrd + loss_gen_dac
            loss_feat = loss_fm_mpd + loss_fm_mrd + loss_fm_dac
            
            loss = loss_mel + loss_comm + loss_adv + loss_feat
            
            if self.world_size > 1:
                loss = reduce_value(loss)
                loss_comm = reduce_value(loss_comm)
                loss_adv = reduce_value(loss_adv)
                loss_feat = reduce_value(loss_feat)
            total_loss += loss.item()
            total_loss_comm += loss_comm.item()
            total_loss_adv += loss_adv.item()
            total_loss_feat += loss_feat.item()
            
            self.optimizer_g.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clip_grad_norm_value)
            self.optimizer_g.step()

            # For discriminator
            loss_dac = discriminator_loss(esti_wav.detach().unsqueeze(1), true_wav.unsqueeze(1))
            real_score_mpd, gen_score_mpd, _, _ = self.mpd(y=true_wav, y_hat=esti_wav.detach(), bandwidth_id=bandwidth_id)
            real_score_mrd, gen_score_mrd, _, _ = self.mrd(y=true_wav, y_hat=esti_wav.detach(), bandwidth_id=bandwidth_id)
            loss_mpd, loss_mpd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mpd, disc_generated_outputs=gen_score_mpd
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mpd /= len(loss_mpd_real)
            loss_mrd /= len(loss_mrd_real)
            
            loss_dis = loss_mpd + loss_mrd + loss_dac
            
            if self.world_size > 1:
                loss_dis = reduce_value(loss_dis)
            total_loss_dis += loss_dis.item()
                
            self.optimizer_d.zero_grad()
            loss_dis.backward()
            torch.nn.utils.clip_grad_norm_(self.mpd.parameters(), self.clip_grad_norm_value)
            torch.nn.utils.clip_grad_norm_(self.mrd.parameters(), self.clip_grad_norm_value)
            torch.nn.utils.clip_grad_norm_(self.dacd.parameters(), self.clip_grad_norm_value)
            self.optimizer_d.step()
            
            self.scheduler_g.step()
            self.scheduler_d.step()
            self.global_steps += 1
            self.lamda_mel = self.lamda_mel_base * self.mel_loss_coeff_decay(self.global_steps)
            
            self.train_bar.desc = '   train[{}/{}][{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, fs, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.train_bar.postfix = 'L={:.2f}, Lc={:.2f}, Lg={:.2f}, Lf={:.2f}, Ld={:.2f}'.format(total_loss / step,
                                                                                        total_loss_comm / step,
                                                                                        total_loss_adv / step,
                                                                                        total_loss_feat / step,
                                                                                        total_loss_dis / step)
        # 等待所有进程计算完毕
        if self.world_size > 1 and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lamda_mel', {'lamda_mel': self.lamda_mel}, epoch)
            self.writer.add_scalars('lr', {'lr': self.optimizer_g.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'loss': total_loss / step,
                                                   'loss_Comm': total_loss_comm / step,
                                                   'loss_Adv': total_loss_adv / step,
                                                   'loss_Feat': total_loss_feat / step, 
                                                   'loss_Dis': total_loss_dis / step}, epoch)


    @torch.inference_mode()
    def _validation_epoch(self, epoch):
        total_loss = 0
        total_loss_comm = 0
        total_loss_adv = 0
        total_loss_feat = 0
        total_loss_dis = 0
        self.validation_bar = tqdm(self.validation_dataloader, ncols=150)
        
        generator_loss = self.dacd.module.generator_loss if self.world_size > 1 else self.dacd.generator_loss
        discriminator_loss = self.dacd.module.discriminator_loss if self.world_size > 1 else self.dacd.discriminator_loss
        
        for step, true_wav in enumerate(self.validation_bar, 1):
            fs = self.default_fs
            true_wav = true_wav.to(self.device)     # (B, 1, T) 

            # For generator
            bandwidth_id = torch.randint(low=0, high=len(self.bandwidths),
                                         size=(1,), device=self.device)
            esti_wav, commit_loss = self.generator(true_wav, bandwidth_id=bandwidth_id)
            
            loss_gen_dac, loss_fm_dac = generator_loss(esti_wav.unsqueeze(1), true_wav.unsqueeze(1))
            _, gen_score_mpd, fmap_rs_mpd, fmap_gs_mpd = self.mpd(
                y=true_wav, y_hat=esti_wav, bandwidth_id=bandwidth_id,
            )
            _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.mrd(
                y=true_wav, y_hat=esti_wav, bandwidth_id=bandwidth_id,
            )
            
            loss_gen_mpd, list_loss_gen_mpd = self.gen_loss(disc_outputs=gen_score_mpd)
            loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
            loss_gen_mpd = loss_gen_mpd / len(list_loss_gen_mpd)
            loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
            loss_fm_mpd = self.feat_loss(fmap_r=fmap_rs_mpd, fmap_g=fmap_gs_mpd) / len(fmap_rs_mpd)
            loss_fm_mrd = self.feat_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)
            
            loss_mel = self.lamda_mel * self.mel_loss(esti_wav, true_wav)
            loss_comm = self.lamda_comm * commit_loss
            loss_adv = loss_gen_mpd + loss_fm_mrd + loss_gen_dac
            loss_feat = loss_fm_mpd + loss_fm_mrd + loss_fm_dac
            
            loss = loss_mel + loss_comm + loss_adv + loss_feat
            
            if self.world_size > 1:
                loss = reduce_value(loss)
                loss_comm = reduce_value(loss_comm)
                loss_adv = reduce_value(loss_adv)
                loss_feat = reduce_value(loss_feat)
            total_loss += loss.item()
            total_loss_comm += loss_comm.item()
            total_loss_adv += loss_adv.item()
            total_loss_feat += loss_feat.item()

            # For discriminator
            loss_dac = discriminator_loss(esti_wav.detach().unsqueeze(1), true_wav.unsqueeze(1))
            real_score_mpd, gen_score_mpd, _, _ = self.mpd(y=true_wav, y_hat=esti_wav, bandwidth_id=bandwidth_id)
            real_score_mrd, gen_score_mrd, _, _ = self.mrd(y=true_wav, y_hat=esti_wav, bandwidth_id=bandwidth_id)
            loss_mpd, loss_mpd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mpd, disc_generated_outputs=gen_score_mpd
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mpd /= len(loss_mpd_real)
            loss_mrd /= len(loss_mrd_real)
            
            loss_dis = loss_mpd + loss_mrd + loss_dac
            
            if self.world_size > 1:
                loss_dis = reduce_value(loss_dis)
            total_loss_dis += loss_dis.item()

            if self.rank == 0 and (epoch==1 or epoch%10 == 0) and step <= 3:
            # if self.rank == 0 and step <= 5:
                true_path = os.path.join(self.sample_path, 'fileid_{}_true.wav'.format(step))
                esti_path = os.path.join(self.sample_path, 'fileid_{}_esti_epoch{}.wav'.format(step, str(epoch).zfill(3)))
                if not os.path.exists(true_path):
                    true_wav = true_wav.cpu().numpy().squeeze()
                    sf.write(true_path, true_wav, self.config['samplerate'])
                esti_wav = esti_wav.detach().cpu().numpy().squeeze()
                sf.write(esti_path, esti_wav, self.config['samplerate'])
                
            self.validation_bar.desc = 'validate[{}/{}][{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, fs, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.validation_bar.postfix = 'L={:.2f}, Lc={:.2f}, Lg={:.2f}, Lf={:.2f}, Ld={:.2f}'.format(total_loss / step,
                                                                                        total_loss_comm / step,
                                                                                        total_loss_adv / step,
                                                                                        total_loss_feat / step,
                                                                                        total_loss_dis / step)

        # 等待所有进程计算完毕
        if (self.world_size > 1) and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('val_loss', {'loss': total_loss / step,
                                                   'loss_Com': total_loss_comm / step,
                                                   'loss_Adv': total_loss_adv / step,
                                                   'loss_Feat': total_loss_feat / step, 
                                                   'loss_Dis': total_loss_dis / step}, epoch)

        return total_loss / step


    def train(self):
        if self.resume:
            self._resume_checkpoint()

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            valid_loss = self._validation_epoch(epoch)

            # if (self.rank == 0) and (epoch % self.save_checkpoint_interval == 0):
            if self.rank == 0:
                self._save_checkpoint(epoch, valid_loss)
                self._del_checkpoint(epoch, valid_loss)

        if self.rank == 0:
            torch.save(self.state_dict_best,
                    os.path.join(self.checkpoint_path,
                    'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(3))))

            print('------------Training for {} epochs has done!------------'.format(self.epochs))



if __name__ == '__main__':
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/cfg_train.yaml')
    parser.add_argument('-D', '--device', default='0', help='The index of the available devices, e.g. 0,1,2,3')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.world_size = len(args.device.split(','))
    config = OmegaConf.load(args.config)
    
    if args.world_size > 1:
        torch.multiprocessing.spawn(
            run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
