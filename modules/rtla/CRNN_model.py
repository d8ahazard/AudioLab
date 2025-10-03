import torch
from torch import nn
from torchaudio import transforms as T
from nnAudio import Spectrogram as S

from .config import N_MELS


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.last_features = None

        if config.feature_type == "spec":
            self.feat = S.STFT(
                sr=config.sample_rate,
                n_fft=config.win_length,
                win_length=config.win_length,
                hop_length=config.hop_length,
                output_format="Magnitude",
            ).to(config.device)
            self.db = T.AmplitudeToDB(stype="magnitude", top_db=80)
        elif config.feature_type == "melspec":
            self.feat = S.MelSpectrogram(
                sr=config.sample_rate,
                n_fft=config.win_length,
                win_length=config.win_length,
                n_mels=config.n_mels,
                hop_length=config.hop_length,
                fmin=config.fmin,
                fmax=config.fmax,
                center=False,
            ).to(config.device)
            self.db = T.AmplitudeToDB(stype="power", top_db=80)
        elif config.feature_type == "mfcc":
            self.feat = S.MFCC(
                sr=config.sample_rate,
                n_mfcc=config.n_mfcc,
                n_fft=config.win_length,
                win_length=config.win_length,
                n_mels=config.n_mels,
                hop_length=config.hop_length,
                fmin=config.fmin,
                fmax=config.fmax,
            ).to(config.device)
            self.db = None

    def forward(self, audio):
        feature = self.feat(audio)  # [C, N, T]
        if self.db is not None:
            feature = self.db(feature)

        padding = (
            self.last_features
            if self.last_features is not None
            else torch.zeros(1, N_MELS, 2).to(feature.device)
        )
        feature = torch.cat([padding, feature], dim=2)  # [C, N, T+2]
        self.last_features = feature[:, :, -2:]
        return feature


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5),
        )

    def forward(self, data):  # [batch, 1, frames, n_mels]
        x = self.cnn(data)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class BiLSTM(nn.Module):
    inference_chunk_length = 512

    def __init__(self, input_features, recurrent_features):
        super().__init__()
        self.rnn = nn.LSTM(
            input_features, recurrent_features, batch_first=True, bidirectional=False
        )

    def forward(self, x):
        if self.training:
            return self.rnn(x)[0]
        else:
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size).to(x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size).to(x.device)
            output = torch.zeros(
                batch_size, sequence_length, num_directions * hidden_size
            ).to(x.device)

            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            return output


class CRNN(nn.Module):
    def __init__(self, config, consts):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

        self.input_features = (
            config.n_mels if config.feature_type == "melspec" else config.n_mfcc
        )
        self.output_features = config.num_lbl

        self.model = self._create_model(
            self.input_features, self.output_features, config
        )

        self.feat_ext = FeatureExtractor(config)

    def _create_model(self, input_features, output_features, config):
        modules = []
        model_complexity = config.model_complexity
        model_size = model_complexity * 16

        modules.append(ConvStack(input_features, model_size))
        modules.append(BiLSTM(model_size, model_size))
        modules.append(nn.Linear(model_size, output_features))

        return nn.Sequential(*modules)

    def forward(self, data):
        return self.model(data)

    def run_on_batch(self, batch, cal_loss=True):
        feat = (
            self.feat_ext(batch["audio"]).transpose(1, 2).unsqueeze(1)
        )  # (N, 1, T, F)
        pred = self(feat)  # (N, T, F)

        predictions = {
            "frame": pred,
        }

        if cal_loss:
            pred = pred.view(-1, self.output_features)  # (N * T, F)
            lbl = batch["label"].view(-1)
            losses = {
                "loss": self.criterion(pred, lbl),
            }

            return predictions, losses

        return predictions


