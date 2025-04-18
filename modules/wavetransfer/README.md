# WaveTransfer: A Flexible End-to-end Multi-instrument Timbre Transfer with Diffusion
## Teysir Baoueb, Xiaoyu Bie, Hicham Janati, Gaël Richard

This repository contains the official implementation of the paper *WaveTransfer: A Flexible End-to-end Multi-instrument Timbre Transfer with Diffusion*.

## Training

1. **Timbre Transfer Model:** Customize the settings in `params.py`, then run the following command to train the timbre transfer model:<br/>
  ```
  python main.py \
    --model_dir <dir_to_save_model> \
    --data_dirs <dataset_dir_path> \
    --training_files <path_to_training_file> \
    --validation_files <path_to_validation_file> \
    --max_steps <training_step_number> \
    --summary_interval <summary_interval> \
    --validation_interval <validation_interval> \
    --checkpoint_interval <checkpoint_interval>
  ```
### Parameters:
- `--model_dir <dir_to_save_model>`: Directory where the model will be saved
- `--data_dirs <dataset_dir_path>`: Path to the dataset directory
- `--training_files <path_to_training_file>`: Path to the training file
- `--validation_files <path_to_validation_file>`: Path to the validation file
- `--max_steps <training_step_number>`: Number of training steps to perform
- `--summary_interval <summary_interval>`: Interval for summary logging
- `--validation_interval <validation_interval>`: Interval for running validation
- `--checkpoint_interval <checkpoint_interval>`: Interval for saving checkpoints
2. **Schedule Network:** Update the configurations in the `conf.yml` file, and subsequently run the command below to train the schedule network::<br/>
  ```
  python main_schedule_network.py --command train --config bddm/conf.yml
  ```
3. **Noise Schedule Search:** To perform the noise schedule search, add the BDDM network checkpoint path to `conf.yml` and set the desired number of sampling steps `N`. Once configured, run the search using:<br/>
  ```
  python main_schedule_network.py --command schedule --config bddm/conf.yml
  ```

## Inference
Specify the noise schedule path and the test directory path in `conf.yml`, then perform timbre transfer as follows:<br/>
  ```
  python main_schedule_network.py --command generate --config bddm/conf.yml
  ```

## References
- [BDDM: Bilateral Denoising Diffusion Models for Fast and High-Quality Speech Synthesis](https://github.com/tencent-ailab/bddm)
- [WaveGrad](https://github.com/lmnt-com/wavegrad)

## Contribution

We welcome contributions to improve the organization and computational efficiency of this code. Please feel free to contribute or report any issues or bugs.

## Citation

If this implementation is helpful in your research, please consider citing our paper with the following BibTeX entry:

```
@inproceedings{baoueb2024wavetransfer,
  title={WaveTransfer: A Flexible End-to-end Multi-instrument Timbre Transfer with Diffusion},
  author={Baoueb, Teysir and Bie, Xiaoyu and Janati, Hicham and Richard, Gaël},
  booktitle={34th {IEEE} International Workshop on Machine Learning for Signal Processing, {MLSP} 2024, London, United Kingdom, September 22-25, 2024},
  year={2024}
}
```
