from enum import IntEnum
import numpy as np
import scipy
from tqdm import tqdm
import librosa

from .stream_processor import StreamProcessor


class Direction(IntEnum):
    REF = 1
    QUERY = 2


class OLTW:
    def __init__(
        self,
        sp: StreamProcessor,
        ref_audio_path,
        window_size,
        sample_rate,
        hop_length,
        max_run_count=3,
        metric="cosine",
        features=["chroma", "phoneme"],
    ):
        self.sp = sp
        self.ref_audio_file = ref_audio_path
        self.w = window_size
        self.max_run_count = max_run_count
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_rate = self.sample_rate / self.hop_length
        self.frame_per_seg = int(sp.chunk_size / self.hop_length)
        self.metric = metric
        self.features = features
        self.ref_pointer = 0
        self.target_pointer = 0
        self.run_count = 0
        self.previous_direction = None
        self.target_features = None
        self.acc_dist_matrix = None
        self.acc_len_matrix = None
        self.candidate = None
        self.candi_history = [[0, 0]]
        self.iteration = 0

        self.initialize_ref_audio(ref_audio_path)

    def offset(self):
        offset_x = max(self.ref_pointer - self.w, 0)
        offset_y = max(self.target_pointer - self.w, 0)
        return np.array([offset_x, offset_y])

    def initialize_ref_audio(self, audio_path):
        audio_y, _ = librosa.load(audio_path, sr=self.sample_rate)
        audio_y = np.concatenate((np.zeros(self.hop_length), audio_y))
        ref_feature = None
        for feature in self.features:
            if feature == "chroma":
                from .utils import process_chroma

                y_chroma = process_chroma(audio_y)
                ref_feature = y_chroma if ref_feature is None else np.vstack((ref_feature, y_chroma))
            elif feature == "mel":
                from .utils import process_mel

                y_mel = process_mel(audio_y)
                ref_feature = y_mel if ref_feature is None else np.vstack((ref_feature, y_mel))
            elif feature == "phoneme":
                from .utils import process_phonemes, get_crnn_model

                # reset CRNN state across files
                _ = get_crnn_model()
                y_phonemes = process_phonemes(audio_y)
                if ref_feature is not None:
                    min_len = min(ref_feature.shape[1], y_phonemes.shape[1])
                ref_feature = (
                    y_phonemes
                    if ref_feature is None
                    else np.vstack((ref_feature[:, :min_len], y_phonemes[:, :min_len]))
                )
        ref_len = ref_feature.shape[1]
        truncated_len = ((ref_len - 1) // self.frame_per_seg) * self.frame_per_seg
        self.ref_features = ref_feature[:, :truncated_len]
        self.ref_total_length = self.ref_features.shape[1]

        self.target_features = np.zeros((self.ref_features.shape[0], self.ref_total_length * 2))

    def init_matrix(self):
        x = self.ref_pointer
        y = self.target_pointer
        d = self.frame_per_seg
        wx = min(self.w, x)
        wy = min(self.w, y)
        new_acc = np.zeros((wx, wy))
        new_len_acc = np.zeros((wx, wy))
        x_seg = self.ref_features[:, x - wx : x].T
        y_seg = self.target_features[:, max(y - d, 0) : y].T
        dist = scipy.spatial.distance.cdist(x_seg, y_seg, metric=self.metric)

        for i in range(wx):
            for j in range(d):
                local_dist = dist[i, j]
                update_y0 = wy - d
                if i == 0 and j == 0:
                    new_acc[i, j] = local_dist
                elif i == 0:
                    new_acc[i, update_y0 + j] = local_dist + new_acc[i, update_y0 - 1]
                    new_len_acc[i, update_y0 + j] = 1 + new_len_acc[i, update_y0 - 1]
                elif j == 0:
                    new_acc[i, update_y0 + j] = local_dist + new_acc[i - 1, update_y0]
                    new_len_acc[i, update_y0 + j] = local_dist + new_len_acc[i - 1, update_y0]
                else:
                    compares = [
                        new_acc[i - 1, update_y0 + j],
                        new_acc[i, update_y0 + j - 1],
                        new_acc[i - 1, update_y0 + j - 1] * 0.98,
                    ]
                    len_compares = [
                        new_len_acc[i - 1, update_y0 + j],
                        new_len_acc[i, update_y0 + j - 1],
                        new_len_acc[i - 1, update_y0 + j - 1],
                    ]
                    local_direction = np.argmin(compares)
                    new_acc[i, update_y0 + j] = local_dist + compares[local_direction]
                    new_len_acc[i, update_y0 + j] = 1 + len_compares[local_direction]
        self.acc_dist_matrix = new_acc
        self.acc_len_matrix = new_len_acc
        self.select_candidate()

    def update_accumulate_matrix(self, direction):
        x = self.ref_pointer
        y = self.target_pointer
        d = self.frame_per_seg
        wx = min(self.w, x)
        wy = min(self.w, y)
        new_acc = np.zeros((wx, wy))
        new_len_acc = np.zeros((wx, wy))

        if direction is Direction.REF:
            new_acc[:-d, :] = self.acc_dist_matrix[d:]
            new_len_acc[:-d, :] = self.acc_len_matrix[d:]
            x_seg = self.ref_features[:, x - d : x].T
            y_seg = self.target_features[:, y - wy : y].T
            dist = scipy.spatial.distance.cdist(x_seg, y_seg, metric=self.metric)

            update_x0 = wx - d
            for i in range(d):
                for j in range(wy):
                    local_dist = dist[i, j]
                    if j == 0:
                        new_acc[update_x0 + i, j] = local_dist + new_acc[update_x0 + i - 1, j]
                        new_len_acc[update_x0 + i, j] = new_len_acc[update_x0 + i - 1, j] + 1
                    else:
                        compares = [
                            new_acc[update_x0 + i - 1, j],
                            new_acc[update_x0 + i, j - 1],
                            new_acc[update_x0 + i - 1, j - 1] * 0.98,
                        ]
                        len_compares = [
                            new_len_acc[update_x0 + i - 1, j],
                            new_len_acc[update_x0 + i, j - 1],
                            new_len_acc[update_x0 + i - 1, j - 1],
                        ]
                        local_direction = np.argmin(compares)
                        new_acc[update_x0 + i, j] = local_dist + compares[local_direction]
                        new_len_acc[update_x0 + i, j] = 1 + len_compares[local_direction]

        elif direction is Direction.QUERY:
            overlap_y = wy - d
            new_acc[:, :-d] = self.acc_dist_matrix[:, -overlap_y:]
            new_len_acc[:, :-d] = self.acc_len_matrix[:, -overlap_y:]
            x_seg = self.ref_features[:, x - wx : x].T
            y_seg = self.target_features[:, y - d : y].T
            dist = scipy.spatial.distance.cdist(x_seg, y_seg, metric=self.metric)

            update_y0 = wy - d
            for i in range(wx):
                for j in range(d):
                    local_dist = dist[i, j]
                    if i == 0:
                        new_acc[i, update_y0 + j] = local_dist + new_acc[i, update_y0 - 1]
                        new_len_acc[i, update_y0 + j] = 1 + new_len_acc[i, update_y0 - 1]
                    else:
                        compares = [
                            new_acc[i - 1, update_y0 + j],
                            new_acc[i, update_y0 + j - 1],
                            new_acc[i - 1, update_y0 + j - 1] * 0.98,
                        ]
                        len_compares = [
                            new_len_acc[i - 1, update_y0 + j],
                            new_len_acc[i, update_y0 + j - 1],
                            new_len_acc[i - 1, update_y0 + j - 1],
                        ]
                        local_direction = np.argmin(compares)
                        new_acc[i, update_y0 + j] = local_dist + compares[local_direction]
                        new_len_acc[i, update_y0 + j] = 1 + len_compares[local_direction]
        self.acc_dist_matrix = new_acc
        self.acc_len_matrix = new_len_acc

    def update_path_cost(self, direction):
        self.update_accumulate_matrix(direction)
        self.select_candidate()

    def select_candidate(self):
        norm_x_edge = self.acc_dist_matrix[-1, :] / np.maximum(self.acc_len_matrix[-1, :], 1e-9)
        norm_y_edge = self.acc_dist_matrix[:, -1] / np.maximum(self.acc_len_matrix[:, -1], 1e-9)
        cat = np.concatenate((norm_x_edge, norm_y_edge))
        min_idx = int(np.argmin(cat))
        offset = self.offset()
        if min_idx <= len(norm_x_edge):
            self.candidate = np.array([self.ref_pointer - offset[0], min_idx])
        else:
            self.candidate = np.array([min_idx - len(norm_x_edge), self.target_pointer - offset[1]])

    def save_history(self):
        self.candi_history.append(self.offset() + self.candidate)

    def select_next_direction(self):
        if self.target_pointer <= self.w:
            next_direction = Direction.QUERY
        elif self.run_count > self.max_run_count:
            next_direction = Direction.QUERY if self.previous_direction is Direction.REF else Direction.REF
        else:
            offset = self.offset()
            x0, y0 = offset[0], offset[1]
            if self.candidate[0] == self.ref_pointer - x0:
                next_direction = Direction.REF
            else:
                next_direction = Direction.QUERY
        return next_direction

    def get_new_input(self):
        # Use a local read pointer separate from StreamProcessor.index
        if not hasattr(self, "_read_ptr"):
            self._read_ptr = 0
        if self._read_ptr >= len(self.sp.feature_chunks):
            # guard against empty or finished buffer
            return False
        target_feature = self.sp.feature_chunks[self._read_ptr]
        q_length = self.frame_per_seg
        self.target_features[:, self.target_pointer : self.target_pointer + q_length] = target_feature
        self.target_pointer += q_length
        self._read_ptr += 1
        return True

    def _is_still_following(self):
        return self.ref_pointer <= (self.ref_total_length - self.frame_per_seg)

    def run_offline(self, target_audio_path: str):
        # Prepare mock features for the entire file
        self.sp.mock_stream(target_audio_path)

        self.ref_pointer += self.w
        if not self.get_new_input():
            raise RuntimeError("No target features available for RTLA alignment.")
        self.init_matrix()
        last_ref_checkpoint = 0
        pbar = tqdm(total=self.ref_total_length)
        while self._is_still_following():
            pbar.update(max(0, (self.candi_history[-1][0] - last_ref_checkpoint)))
            last_ref_checkpoint = self.candi_history[-1][0]
            self.save_history()
            direction = self.select_next_direction()
            if direction is Direction.QUERY:
                if not self.get_new_input():
                    break
                self.update_path_cost(direction)
            elif direction is Direction.REF:
                self.ref_pointer += self.frame_per_seg
                self.update_path_cost(direction)

            if direction == self.previous_direction:
                self.run_count += 1
            else:
                self.run_count = 1

            self.previous_direction = direction
            self.iteration += 1
        pbar.close()

        # Append final history point
        self.save_history()
        return np.array(self.candi_history).T  # shape (2, N)


