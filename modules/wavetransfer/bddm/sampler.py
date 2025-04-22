# Adapted from https://github.com/tencent-ailab/bddm under the Apache-2.0 license.

#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  BDDM Sampler (Supports Noise Scheduling and Sampling)
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################


from __future__ import absolute_import

import os

import torch
from torch.nn.functional import normalize
import torchaudio
import numpy as np
from scipy.io.wavfile import write as wavwrite

from modules.wavetransfer.bddm.log_utils import log
from modules.wavetransfer.bddm.diffusion_utils import compute_diffusion_params, map_noise_scale_to_time_step
from modules.wavetransfer.bddm.models import get_schedule_network

from modules.wavetransfer.model import WaveGrad
from modules.wavetransfer.params import AttrDict, get_default_params
from modules.wavetransfer.preprocess import get_spec
from modules.wavetransfer.bddm.data_loader_for_sampler import from_path_valid as dataset_from_path_valid, from_path_background as dataset_from_path_background


MAX_WAV_VALUE = 32767
d = {'0': '3', '1': '4', '2': '5', '3': '0', '4': '1', '5': '2'}

class Sampler(object):

    metric2index = {"FAD": 0}

    def __init__(self, config):
        """
        BDDM Sampler Class, implements a sampling framework in PyTorch

        Parameters:
            config (namespace): BDDM Configuration
        """
        self.config = config
        self.exp_dir = config.exp_dir
        
        # Set default command if not present (for compatibility with SimpleNamespace objects from UI)
        if not hasattr(config, 'command'):
            config.command = 'generate'
            print(f"No command specified, defaulting to 'generate'")
        
        if hasattr(config, 'N'):
            self.steps2score = {}
            self.steps2schedule = {}
            # Define diffusion parameters using a pre-specified linear schedule
            self.clip = config.grad_clip
            self.load = config.load
            print("The model to be loaded is:", self.load)
            self.model = WaveGrad(AttrDict(get_default_params())).to('cuda').eval()
            self.schedule = None
            # Initialize diffusion parameters using a pre-specified linear schedule
            noise_schedule = torch.linspace(config.beta_0, config.beta_T, config.T).cuda()
            self.diff_params = compute_diffusion_params(noise_schedule)
            if self.config.command != 'train':
                # Find schedule net, if not trained then use DDPM or DDIM sampling mode
                schedule_net_path = config.bddm_load
                if not schedule_net_path == "":
                    self.load = schedule_net_path
                    print("The schedule network is trained. The model to be loaded is changed to:", self.load)
                    self.model.schedule_net = get_schedule_network(config).cuda().eval()

            # Perform noise scheduling when noise schedule file (.schedule) is not given
            if self.config.command == 'generate' and self.config.sampling_noise_schedule != '':
                # Generation mode given pre-searched noise schedule
                # Check if safetensors version exists
                ns_path = self.config.sampling_noise_schedule
                safetensors_path = ns_path.replace('.ns', '.safetensors')
                
                if os.path.exists(safetensors_path):
                    try:
                        from safetensors.torch import load_file
                        schedule_dict = load_file(safetensors_path, device='cuda')
                        # Convert to format needed by the algorithm
                        ts_infer = schedule_dict['ts_infer']
                        a_infer = schedule_dict['a_infer']
                        b_infer = schedule_dict['b_infer']
                        s_infer = schedule_dict['s_infer']
                        self.schedule = [ts_infer, a_infer, b_infer, s_infer]
                        log(f"Loaded noise schedule from {safetensors_path}", self.config)
                    except (ImportError, RuntimeError, KeyError):
                        # Fall back to torch.load
                        self.schedule = torch.load(ns_path, map_location='cuda:0')
                        log(f"Loaded noise schedule from {ns_path}", self.config)
                else:
                    # Fall back to torch.load
                    self.schedule = torch.load(ns_path, map_location='cuda:0')
                    log(f"Loaded noise schedule from {ns_path}", self.config)

            self.reset()

    def reset(self):
        """
        Reset sampling environment
        """
        noise_schedule_dir = os.path.join(self.exp_dir, 'noise_schedules')
        os.makedirs(noise_schedule_dir, exist_ok=True)
        if self.config.command != 'train' and self.load != '':
            # First try to load with safetensors
            safetensors_path = self.load
            # Convert .pkl extension to .safetensors if needed
            if safetensors_path.endswith('.pkl'):
                safetensors_path = safetensors_path.replace('.pkl', '.safetensors')
                
            try:
                # Try to load with safetensors first
                from safetensors.torch import load_file
                if os.path.exists(safetensors_path):
                    model_state = load_file(safetensors_path, device='cuda')
                    
                    # Check if this is a schedule-network-only checkpoint
                    has_schedule_prefix = any(k.startswith('schedule_net.') for k in model_state.keys())
                    has_ratio_nn = any(k.startswith('ratio_nn.') for k in model_state.keys())
                    
                    if not has_schedule_prefix and has_ratio_nn:
                        # This is a schedule network-only checkpoint
                        # First, try to load directly into the schedule_net
                        try:
                            self.model.schedule_net.load_state_dict(model_state)
                            log('Loaded schedule network weights directly: %s' % safetensors_path, self.config)
                        except RuntimeError:
                            # Maybe we need to adjust the keys
                            schedule_state = {}
                            for k, v in model_state.items():
                                schedule_state['schedule_net.' + k] = v
                            
                            # Now load the full model with updated keys
                            self.model.load_state_dict(schedule_state, strict=False)
                            log('Loaded schedule network with key fixing: %s' % safetensors_path, self.config)
                    else:
                        # Normal model loading
                        self.model.load_state_dict(model_state)
                        log('Loaded checkpoint with safetensors: %s' % safetensors_path, self.config)
                else:
                    # Fall back to torch.load
                    self._load_torch_checkpoint(self.load)
            except (ImportError, FileNotFoundError, RuntimeError) as e:
                # Fall back to torch.load
                log(f'Error loading safetensors ({str(e)}), falling back to torch', self.config)
                self._load_torch_checkpoint(self.load)

        if self.config.command == 'generate':
            self.test_dir = self.config.test_dir
            
            # Load background list with error handling
            try:
                from modules.wavetransfer.bddm.data_loader_for_sampler import from_path_background
                self.bg_list = from_path_background([self.config.background_dir], [], get_default_params())
            except Exception as e:
                log(f'Error loading background list: {str(e)}', self.config)
                # Create an empty background list as fallback
                self.bg_list = []
        else:
            # Sample a reference audio sample for noise scheduling with error handling
            try:
                from modules.wavetransfer.bddm.data_loader_for_sampler import from_path_valid, from_path_background
                self.vl_loader = from_path_valid(self.config.data_dir, self.config.validation_file, get_default_params(), 1)
                self.bg_list = from_path_background([self.config.background_dir], self.config.training_file, get_default_params())
                self.draw_reference_data_pair()
            except Exception as e:
                log(f'Error loading validation data: {str(e)}', self.config)
                
                # Create dummy data as fallback
                from torch.utils.data import DataLoader, TensorDataset
                dummy_dataset = TensorDataset(torch.zeros(1, 1), torch.zeros(1, 1))
                self.vl_loader = DataLoader(dummy_dataset, batch_size=1)
                self.bg_list = []
                
                # Create dummy reference data
                self.ref_spec = torch.zeros(1, 128, 64).cuda()
                self.ref_audio = torch.zeros(1, 1, 8192).cuda()

    def draw_reference_data_pair(self):
        """
        Draw a new input-output pair for noise scheduling
        """
        features = next(iter(self.vl_loader))
        self.ref_spec_list = [spec.cuda().unsqueeze(0) for features in self.vl_loader for spec in features['spectrogram']]
        self.ref_spec, self.ref_audio = features['spectrogram'], features['audio'].unsqueeze(1)
        self.ref_spec, self.ref_audio = self.ref_spec.cuda(), self.ref_audio.cuda()

    def generate(self):
        """
        Start the generation process
        """
        generate_dir = f"({os.path.basename(self.config.sampling_noise_schedule)})_"
        generate_dir += 'generated_DDIM' if self.config.use_ddim_steps else 'generated_DDPM'
        if self.schedule:
            generate_dir += '_BDDM'
        generate_dir = os.path.join('./generated_files', generate_dir)
        os.makedirs(generate_dir, exist_ok=True)
        scores = {metric: [] for metric in self.metric2index}
        gen_audio_list = []
        filenames = os.listdir(self.test_dir)
        for filename in filenames:
            if self.config.only_mixtures:
                if not (filename.endswith(".0.wav") or filename.endswith(".3.wav")):
                    continue
            filepath = os.path.join(self.test_dir, filename)
            gt_audio, _ = torchaudio.load(filepath)
            gt_audio = normalize(gt_audio, p=float('inf'), dim=-1, eps=1e-12) * 0.95
            mel_spec = get_spec(gt_audio, AttrDict(get_default_params()))
            mel_spec = mel_spec.cuda()

            generated_audio, n_steps = self.sampling(schedule=self.schedule,
                                                     condition=mel_spec,
                                                     ddim=self.config.use_ddim_steps)
            gen_audio_list.append(generated_audio)
            audio_key = filename.split('.')[0] + '.' + d[filename.split('.')[1]]

            model_name = 'BDDM' if self.schedule is not None else (
                'DDIM' if self.config.use_ddim_steps else 'DDPM')
            generated_file = os.path.join(generate_dir,
                '%s.wav'%(audio_key))
            wavwrite(generated_file, self.model.params.sample_rate,
                     generated_audio.squeeze().cpu().numpy())
            log('Generated '+generated_file, self.config)
        scores['FAD'] = self.assess(gen_audio_list)[0]
        log('FAD = %.3f'%(scores['FAD']), self.config)
        suffix = "_only_mixtures" if self.config.only_mixtures else ""
        new_name = generate_dir+'_FAD_%.3f'%(scores['FAD']) + suffix
        os.rename(generate_dir, new_name)

    def noise_scheduling(self, ddim=False):
        """
        Start the noise scheduling process

        Parameters:
            ddim (bool): whether to use the DDIM's p_theta for noise scheduling or not
        Returns:
            ts_infer (tensor): the step indices estimated by BDDM
            a_infer (tensor):  the alphas estimated by BDDM
            b_infer (tensor):  the betas estimated by BDDM
            s_infer (tensor):  the std. deviations estimated by BDDM
        """
        max_steps = self.diff_params["N"]
        alpha = self.diff_params["alpha"]
        alpha_param = self.diff_params["alpha_param"]
        beta_param = self.diff_params["beta_param"]
        min_beta = self.diff_params["beta"].min()
        betas = []
        x = torch.normal(0, 1, size=self.ref_audio.shape).cuda().squeeze(1)
        with torch.no_grad():
            b_cur = torch.ones(1, 1).cuda() * beta_param
            a_cur = torch.ones(1, 1).cuda() * alpha_param
            for n in range(max_steps - 1, -1, -1):
                step = map_noise_scale_to_time_step(a_cur.squeeze().item(), alpha)
                if step >= 0:
                    betas.append(b_cur.squeeze().item())
                else:
                    break
                e = self.model(x, self.ref_spec.clone(), a_cur.squeeze(1)).squeeze(1)
                a_nxt = a_cur / (1 - b_cur).sqrt()
                if ddim:
                    c1 = a_nxt / a_cur
                    c2 = -(1 - a_cur**2.).sqrt() * c1
                    x = c1 * x + c2 * e
                    c3 = (1 - a_nxt**2.).sqrt()
                    x = x + c3 * e
                else:
                    x = x - b_cur / torch.sqrt(1 - a_cur**2.) * e
                    x = x / torch.sqrt(1 - b_cur)
                    if n > 0:
                        z = torch.normal(0, 1, size=x.shape).cuda()
                        x = x + torch.sqrt((1 - a_nxt**2.) / (1 - a_cur**2.) * b_cur) * z
                a_nxt, beta_nxt = a_cur, b_cur
                a_cur = a_nxt / (1 - beta_nxt).sqrt()
                if a_cur > 1:
                    break
                b_cur = self.model.schedule_net(x,
                    (beta_nxt.view(-1, 1), (1 - a_cur**2.).view(-1, 1))).view(-1, 1)
                b_cur = b_cur[0]
                if b_cur.squeeze().item() < min_beta:
                    break
        b_infer = torch.FloatTensor(betas[::-1]).cuda()
        a_infer = 1 - b_infer
        s_infer = b_infer + 0
        for n in range(1, len(b_infer)):
            a_infer[n] *= a_infer[n-1]
            s_infer[n] *= (1 - a_infer[n-1]) / (1 - a_infer[n])
        a_infer = torch.sqrt(a_infer)
        s_infer = torch.sqrt(s_infer)

        # Mapping noise scales to time steps
        ts_infer = []
        for n in range(len(b_infer)):
            step = map_noise_scale_to_time_step(a_infer[n], alpha)
            if step >= 0:
                ts_infer.append(step)
        ts_infer = torch.FloatTensor(ts_infer)
        return ts_infer, a_infer, b_infer, s_infer

    def sampling(self, schedule=None, condition=None,
                 ddim=0, return_sequence=False, audio_size=None):
        """
        Perform the sampling algorithm

        Parameters:
            schedule (list):        the [ts_infer, a_infer, b_infer, s_infer]
                                    returned by the noise scheduling algorithm
            condition (tensor):     the condition for computing scores
            ddim (bool):            whether to use the DDIM for sampling or not
            return_sequence (bool): whether returning all steps' samples or not
            audio_size (list):      the shape of the audio to be sampled
        Returns:
            xs (list):              (if return_sequence) the list of generated audios
            x (tensor):             the generated audio(s) in shape=audio_size
            N (int):                the number of sampling steps
        """

        n_steps = self.diff_params["T"]
        if condition is None:
            condition = self.ref_spec

        if audio_size is None:
            audio_length = condition.size(-1) * self.model.params.hop_samples
            audio_size = (1, audio_length)

        if schedule is None:
            if ddim > 1:
                # Use DDIM (linear) for sampling ({ddim} steps)
                ts_infer = torch.linspace(0, n_steps - 1, ddim).long()
                a_infer = self.diff_params["alpha"].index_select(0, ts_infer.cuda())
                b_infer = self.diff_params["beta"].index_select(0, ts_infer.cuda())
                s_infer = self.diff_params["sigma"].index_select(0, ts_infer.cuda())
            else:
                # Use DDPM for sampling (complete T steps)
                # P.S. if ddim = 1, run DDIM reverse process for T steps
                ts_infer = torch.linspace(0, n_steps - 1, n_steps)
                a_infer = self.diff_params["alpha"]
                b_infer = self.diff_params["beta"]
                s_infer = self.diff_params["sigma"]
        else:
            ts_infer, a_infer, b_infer, s_infer = schedule

        sampling_steps = len(ts_infer)

        x = torch.normal(0, 1, size=audio_size).cuda()
        if return_sequence:
            xs = []
        with torch.no_grad():
            for n in range(sampling_steps - 1, -1, -1):
                if sampling_steps > 50 and (sampling_steps - n) % 50 == 0:
                    # Log progress per 50 steps when sampling_steps is large
                    log('\tComputed %d / %d steps !'%(
                        sampling_steps - n, sampling_steps), self.config)
                e = self.model(x, condition, a_infer[n].unsqueeze(0)).squeeze(1)
                if ddim:
                    if n > 0:
                        a_nxt = a_infer[n - 1]
                    else:
                        a_nxt = a_infer[n] / (1 - b_infer[n]).sqrt()
                    c1 = a_nxt / a_infer[n]
                    c2 = -(1 - a_infer[n]**2.).sqrt() * c1
                    c3 = (1 - a_nxt**2.).sqrt()
                    x = c1 * x + (c2 + c3) * e
                else:
                    x = x - b_infer[n] / torch.sqrt(1 - a_infer[n]**2.) * e
                    x = x / torch.sqrt(1 - b_infer[n])
                    if n > 0:
                        z = torch.normal(0, 1, size=audio_size).cuda()
                        x = x + s_infer[n] * z
                if return_sequence:
                    xs.append(x)
        if return_sequence:
            return xs
        return x, sampling_steps

    def noise_scheduling_with_params(self, alpha_param, beta_param):
        """
        Start the noise scheduling process with the given params

        Parameters:
            alpha_param (float): the noise level to estimate alpha
            beta_param (float):  the noise level to estimate beta
        Returns:
            best_schedule (list): the noise schedule with the best FAD score
        """
        # Prepare the diffusion parameters
        self.diff_params = {}
        self.diff_params["alpha_param"] = alpha_param
        self.diff_params["beta_param"] = beta_param
        self.diff_params["T"] = self.config.T
        self.diff_params["alpha"] = alpha = torch.cat([
            torch.FloatTensor([1. - 1e-8]), # Adding alpha_0 = 1. at index 0
            1 - torch.linspace(self.config.beta_0, self.config.beta_T, self.config.T)
        ]).cuda()
        self.diff_params["beta"] = 1 - alpha**2.
        self.diff_params["tau"] = self.config.tau
        self.diff_params["N"] = steps = self.config.N

        # Variables for BDDM calculation
        alpha = self.diff_params["alpha"]
        for i in range(1, len(alpha)):
            alpha[i] *= alpha[i-1]
        alpha = alpha.sqrt()

        # For schedule search
        schedule_dir = os.path.join(self.exp_dir, 'noise_schedules')
        schedule_name = 'BDDM_bsearch%d_Ns%d_a%.5f_b%.5f_t%d'%(
                self.config.bddm_search_bins,
                steps, alpha_param, beta_param, self.config.tau)
        
        # Create both file paths
        ns_path = os.path.join(schedule_dir, '%s.ns'%(schedule_name))
        safetensors_path = os.path.join(schedule_dir, '%s.safetensors'%(schedule_name))
        
        # Initialize ts with uniform steps first
        if steps not in self.steps2schedule:
            self.steps2schedule[steps] = []
        if os.path.exists(safetensors_path):
            # Try to load with safetensors first
            try:
                from safetensors.torch import load_file
                schedule_dict = load_file(safetensors_path, device='cuda')
                # Convert to format needed by the algorithm
                ts_infer = schedule_dict['ts_infer']
                a_infer = schedule_dict['a_infer']
                b_infer = schedule_dict['b_infer']
                s_infer = schedule_dict['s_infer']
                schedule = [ts_infer, a_infer, b_infer, s_infer]
                
                self.steps2schedule[steps].append((None, schedule))
                log('Loaded schedule from %s'%(safetensors_path), self.config)
                return schedule
            except (ImportError, RuntimeError, KeyError):
                pass
                
        # Try original .ns format as fallback
        if os.path.exists(ns_path):
            try:
                schedule = torch.load(ns_path, map_location='cuda:0')
                self.steps2schedule[steps].append((None, schedule))
                log('Loaded schedule from %s'%(ns_path), self.config)
                return schedule
            except:
                # If both loading methods fail, continue to generate new schedule
                pass

        # Perform noise scheduling
        best_score, best_schedule = float("inf"), None
        fads = []
        for n in range(self.config.noise_scheduling_attempts):
            # Perform noise scheduling via BDDM
            log('Start noise scheduling %d/%d...'%(
                n+1, self.config.noise_scheduling_attempts), self.config)
            schedule = self.noise_scheduling(ddim=self.config.use_ddim_steps)
            self.steps2schedule[steps].append((None, schedule))
            log('Getting FAD score...', self.config)
            audio_list = []
            # Sample from the noise schedule
            for i in range(10):
                # new reference sample for scheduling
                if i % 2 == 0:
                    self.draw_reference_data_pair()
                audio, n_steps = self.sampling(schedule=schedule,
                    condition=self.ref_spec.clone(), return_sequence=False)
                audio_list.append(audio)
            # Get FAD score from new samples
            fad = self.assess(audio_list)[0]
            fads.append(fad)
            log('Got schedule FAD = %.3f'%(fad), self.config)
            
            if fad < best_score:
                best_score, best_schedule = fad, schedule
                # Save both formats
                try:
                    # Try to save with safetensors
                    from safetensors.torch import save_file
                    # Create dict compatible with safetensors
                    schedule_dict = {
                        'ts_infer': best_schedule[0],
                        'a_infer': best_schedule[1],
                        'b_infer': best_schedule[2],
                        's_infer': best_schedule[3]
                    }
                    save_file(schedule_dict, safetensors_path)
                    log('Saved best schedule to %s'%(safetensors_path), self.config)
                except ImportError:
                    # Fall back to torch.save
                    torch.save(best_schedule, ns_path)
                    log('Saved best schedule to %s'%(ns_path), self.config)
        
        # Also save with original format for compatibility
        torch.save(best_schedule, ns_path)
        
        log('Average FAD = %.3f, Best FAD = %.3f'%(
            np.mean(fads), best_score), self.config)
        return best_schedule

    def noise_scheduling_without_params(self):
        """
        Search for the best noise scheduling hyperparameters: (alpha_param, beta_param)
        """
        # Noise scheduling mode, given N
        self.reverse_process = 'BDDM'
        assert 'N' in vars(self.config).keys(), 'Error: N is undefined for BDDM!'
        self.diff_params["N"] = self.config.N
        # Init search result dictionaries
        self.steps2schedule, self.steps2score = {}, {}
        search_bins = int(self.config.bddm_search_bins)
        # Define search range of alpha_param
        alpha_last = self.diff_params["alpha"][-1].squeeze().item()
        alpha_first = self.diff_params["alpha"][0].squeeze().item()
        alpha_diff = (alpha_first - alpha_last) / (search_bins + 1)
        alpha_param_list = [alpha_last + alpha_diff * (i + 1) for i in range(search_bins)]
        # Define search range of beta_param
        beta_diff = 1. / (search_bins + 1)
        beta_param_list = [beta_diff * (i + 1) for i in range(search_bins)]
        # Search for beta_param and alpha_param, take O(search_bins^2)
        for beta_param in beta_param_list:
            for alpha_param in alpha_param_list:
                if alpha_param > (1 - beta_param) ** 0.5:
                    # Invalid range
                    continue
                # Update the scores and noise schedules with (alpha_param, beta_param)
                self.noise_scheduling_with_params(alpha_param, beta_param)
        # Lastly, repeat the random starting point (x_hat_N) and choose the best schedule
        noise_schedule_dir = os.path.join(self.exp_dir, 'noise_schedules')
        steps_list = list(self.steps2score.keys())
        for steps in steps_list:
            log("-"*80, self.config)
            log("Select the best out of %d x_hat_N ~ N(0,I) for %d steps:"%(
                self.config.noise_scheduling_attempts, steps), self.config)
            # Get current best pair
            key = self.steps2score[steps][0]
            # Get back the best (alpha_param, beta_param) pair for a given steps
            alpha_param, beta_param = list(map(float, key.split(',')))
            # Repeat K times for a given number of steps
            for _ in range(self.config.noise_scheduling_attempts):
                # Random +/- 5%
                _alpha_param = alpha_param * (0.95 + np.random.rand() * 0.1)
                _beta_param = beta_param * (0.95 + np.random.rand() * 0.1)
                # Update the scores and noise schedules with (alpha_param, beta_param)
                self.noise_scheduling_with_params(_alpha_param, _beta_param)
        # Save the best searched noise schedule ({N}steps_{key}_{metric}{best_score}.ns)
        for steps in sorted(self.steps2score.keys(), reverse=True):
            filepath = os.path.join(noise_schedule_dir, '%dsteps_FAD%.4f.ns'%(
                steps, self.steps2score[steps][1]))
            torch.save(self.steps2schedule[steps], filepath)
            log("Saved searched schedule: %s" % filepath, self.config)

    def assess(self, gen_audio_list, audio_key=None):
        """
        Assess the generated audio using objective metrics: FAD.

        Parameters:
            gen_audio_list (list of tensors): the list of generated audios to be assessed
            audio_key (str):          the key of the respective audio
        Returns:
            fad_score (float):       the FAD score (the lower the better)
        """
        from frechet_audio_distance import FrechetAudioDistance

        gen_audio_list_numpy = []
        resampler = torchaudio.transforms.Resample(self.model.params.sample_rate, 16000).cuda()
        for generated_audio in gen_audio_list:
            est_audio = generated_audio.squeeze(1)
            # Resample audio to 16 kHz
            if self.model.params.sample_rate != 16000:
                est_audio_16k = resampler(est_audio.squeeze(0)).cpu().numpy()
                gen_audio_list_numpy.append(est_audio_16k)
            else:
                est_audio = est_audio.squeeze(0).cpu().numpy()
                gen_audio_list_numpy.append(est_audio)

        # Compute FAD using VGGish embeddings
        frechet = FrechetAudioDistance(
                    model_name="vggish",
                    use_pca=False,
                    use_activation=False,
                    verbose=False
                )
        fad_score = frechet.score(self.bg_list, gen_audio_list_numpy)
        # Log scores
        log('\t%sScores: FAD = %.3f'%(
            '' if audio_key is None else audio_key+' ', fad_score), self.config)
        # Return scores: the higher the better
        return [fad_score]

    def _load_torch_checkpoint(self, checkpoint_path):
        """Helper method to load PyTorch checkpoints with proper key handling"""
        package = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
        
        if 'model_state_dict' in package:
            model_state = package['model_state_dict'] 
        elif 'model' in package:
            model_state = package['model']
        else:
            model_state = package  # Assume the entire package is the state dict
        
        # Check if this is a schedule-network-only checkpoint
        has_schedule_prefix = any(k.startswith('schedule_net.') for k in model_state.keys())
        has_ratio_nn = any(k.startswith('ratio_nn.') for k in model_state.keys())
        
        if not has_schedule_prefix and has_ratio_nn:
            # This is a schedule network-only checkpoint
            # First, try to load directly into the schedule_net
            try:
                self.model.schedule_net.load_state_dict(model_state)
                log('Loaded schedule network weights directly: %s' % checkpoint_path, self.config)
            except RuntimeError:
                # Maybe we need to adjust the keys
                schedule_state = {}
                for k, v in model_state.items():
                    schedule_state['schedule_net.' + k] = v
                
                # Now load the full model with updated keys
                self.model.load_state_dict(schedule_state, strict=False)
                log('Loaded schedule network with key fixing: %s' % checkpoint_path, self.config)
        else:
            # Normal model loading
            self.model.load_state_dict(model_state)
            log('Loaded checkpoint with torch: %s' % checkpoint_path, self.config)
