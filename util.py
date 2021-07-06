import glob
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf

import torchaudio as ta
from pydub import AudioSegment, effects

path = 'snares/'
path_processed = 'proc-snares/'


def convert_to_wav(in_path: str, out_path: str, bits=16):
    for file in glob.glob(f'{in_path}/*'):
        data, sr = ta.load(file)
        file_name = file.split('/')[1]
        ta.save(f'{out_path}/{file_name}', data, sr, bits_per_sample=bits)


def resample(path: str, resample_rate: int, bits=16):
    for file in glob.glob(f'{path}/*'):
        data, sr = sf.read(file)
        data = librosa.resample(data, sr, resample_rate)
        sf.write(file, data, resample_rate, f'PCM_{bits}')


def normalize_vol(in_path: str, out_path: str):
    for file in glob.glob(f'{in_path}/*'):
        rawsound = AudioSegment.from_file(file)
        normalizedsound = effects.normalize(rawsound)
        normalizedsound.export(f'{out_path}', format='wav')


def files_to_tensor(path: str, expected_sr: int = 44100, channels: int = 1):
    collect_data = []
    max_duration = 0

    for file in glob.glob(f'{path}/*'):
        data, sr = ta.load(file)

        if sr != expected_sr:
            print(f"{file} has a different sample rate than expected ({expected_sr}). Skipping!")
            continue

        collect_data.append(data)
        if max_duration < data.shape[1]:
            max_duration = data.shape[1]

    result = torch.zeros((len(collect_data), channels, max_duration))
    for i, data in enumerate(collect_data):
        pad_size = max_duration - data.shape[1]
        result[i] = F.pad(data, (0, pad_size))

    return result


def object_to_file(file_name: str, obj: object):
    with open(file_name, 'wb+') as file:
        pickle.dump(obj, file)


def get_longest_sample(path: str, expected_sr=44100):
    max_duration = 0
    file_name = ""
    for file in glob.glob(f'{path}/*'):
        data, sr = ta.load(file)

        if sr != expected_sr:
            continue

        if data.shape[1] > max_duration:
            max_duration = data.shape[1]
            file_name = file

    print(f"{file_name} durates the longest with {max_duration} samples on {expected_sr}.")


def num_exceed_duration(n_samples: int, path: str, expected_sr=44100):
    count = 0
    for file in glob.glob(f'{path}/*'):
        data, sr = ta.load(file)

        if sr != expected_sr:
            continue

        if data.shape[1] > n_samples:
            count += 1
            print(file)

    print(f"{count} exceeds duration of {n_samples}")


def trim_from_end(path: str, fade_out=500, bits=16):
    for file in glob.glob(f'{path}/*'):
        # Search for first window in the data that is almost empty and cut the sample there
        data, sr = ta.load(file)
        window_size = 50
        for i, sample in enumerate(data[0, :-window_size]):
            window = data[0, i:i + window_size]
            if all(np.isclose(window, np.zeros(window_size), atol=1e-02, rtol=0)) and i > 2500:
                print(f"Trimming {file} at {i}. Originally was of length {data.shape[1]}.")
                data = data[:, 0:i]
                if fade_out:
                    fade_window = torch.cat((torch.ones(data.shape[1] - fade_out), torch.linspace(1, 0, fade_out)))
                    data = data * fade_window
                break

        ta.save(file, data, sr, bits_per_sample=bits)


def trim_from_start(path: str, bits=16):
    for file in glob.glob(f'{path}/*'):
        data, sr = ta.load(file)
        i = 0

        while abs(data[0, i]) < 1e-4 and i < 1500:
            i += 1

        if i > 0:
            print(f"Trimming {file} from start up to {i}.")
            data = data[:, i - 1:]

        ta.save(file, data, sr, bits_per_sample=bits)


def convert_to_mono(path: str, bits=16):
    for file in glob.glob(f'{path}/*'):
        data, sr = ta.load(file)
        ta.save(file, data.mean(axis=0)[np.newaxis, :], sr, bits_per_sample=bits)


def check_all_mono(path: str):
    for file in glob.glob(f'{path}/*'):
        data, sr = ta.load(file)
        assert data.shape[0] == 1, f"{file} is not mono."


def hard_trim(path_processed: str, trim_at: int, bits=16):
    for file in glob.glob(f'{path_processed}/*'):
        data, sr = ta.load(file)
        ta.save(file, data[:, :trim_at], sr, bits_per_sample=bits)


def fast_downsample(path: str, expected_sr: int):
    for file in glob.glob(f'{path}/*'):
        data, sr = ta.load(file)

        if sr > expected_sr:
            print(f"Downsampled from {sr} to {expected_sr} of {file}.")
            # Fast Downsampling
            ratio = sr / expected_sr
            idxs = np.rint(np.arange(0, data.shape[1] - 1, ratio)).astype(int)
            data = data[:, idxs]
            ta.save(file, data, expected_sr)

# convert_to_wav(path, path_processed, bits=16)
# fast_downsample(path_processed, 44100)
# convert_to_mono(path_processed)
# check_all_mono(path_processed)
# trim_from_start(path_processed)
# trim_from_end(path_processed)
# hard_trim(path_processed, trim_at=44100 // 3)
# num_exceed_duration(44100 // 3, path_processed)
# object_to_file('snares2.db', files_to_tensor(path_processed))
