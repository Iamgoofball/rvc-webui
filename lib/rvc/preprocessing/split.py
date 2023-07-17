import operator
import os
from concurrent.futures import ProcessPoolExecutor
from typing import *

import librosa
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from tqdm import tqdm

from lib.rvc.utils import load_audio

from .slicer import Slicer
from multiprocessing import Process

speaker_ids_to_watch = [63, 62, 61, 60, 59]
def norm_write(
	tmp_audio: np.ndarray,
	idx0: int,
	idx1: int,
	speaker_id: int,
	outdir: str,
	outdir_16k: str,
	sampling_rate: int,
	max: float,
	alpha: float,
	is_normalize: bool,
):
	if speaker_id in speaker_ids_to_watch:
		print(f"I'm starting to write for speaker {speaker_id}")
	if is_normalize:
		tmp_audio = (tmp_audio / np.abs(tmp_audio).max() * (max * alpha)) + (
			1 - alpha
		) * tmp_audio
	else:
		# clip level to max (cause sometimes when floating point decoding)
		audio_min = np.min(tmp_audio)
		if audio_min < -max:
			tmp_audio = tmp_audio / -audio_min * max
		audio_max = np.max(tmp_audio)
		if audio_max > max:
			tmp_audio = tmp_audio / audio_max * max

	if speaker_id in speaker_ids_to_watch:
		print(f"I'm writing for speaker {speaker_id}")
	wavfile.write(
		os.path.join(outdir, f"{speaker_id:05}", f"{idx0}_{idx1}.wav"),
		sampling_rate,
		tmp_audio.astype(np.float32),
	)
	if speaker_id in speaker_ids_to_watch:
		print(f"I'm starting to resample for speaker {speaker_id}")
	tmp_audio = librosa.resample(
		tmp_audio, orig_sr=sampling_rate, target_sr=16000, res_type="soxr_vhq"
	)
	if speaker_id in speaker_ids_to_watch:
		print(f"I'm finishing writing for speaker {speaker_id}")
	wavfile.write(
		os.path.join(outdir_16k, f"{speaker_id:05}", f"{idx0}_{idx1}.wav"),
		16000,
		tmp_audio.astype(np.float32),
	)


def write_mute(
	mute_wave_filename: str,
	speaker_id: int,
	outdir: str,
	outdir_16k: str,
	sampling_rate: int,
):
	tmp_audio = load_audio(mute_wave_filename, sampling_rate)
	wavfile.write(
		os.path.join(outdir, f"{speaker_id:05}", "mute.wav"),
		sampling_rate,
		tmp_audio.astype(np.float32),
	)
	tmp_audio = librosa.resample(
		tmp_audio, orig_sr=sampling_rate, target_sr=16000, res_type="soxr_vhq"
	)
	wavfile.write(
		os.path.join(outdir_16k, f"{speaker_id:05}", "mute.wav"),
		16000,
		tmp_audio.astype(np.float32),
	)


def pipeline(
	slicer: Slicer,
	datasets: List[Tuple[str, int]],  # List[(path, speaker_id)]
	outdir: str,
	outdir_16k: str,
	sampling_rate: int,
	is_normalize: bool,
	log_all: False,
	process_id: int = 0,

):
	per = 3.7
	overlap = 0.3
	tail = per + overlap
	max = 0.95
	alpha = 0.8
	bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=sampling_rate)
	index = 0
	datacount = 0
	for data in datasets:
		if log_all and data[1] in speaker_ids_to_watch:
			print(data)
			print(f"Current count {datacount} max count {len(datasets)}")
			datacount += 1
		if data[1] in speaker_ids_to_watch:
			print(data)
			print(f"I started to load for speaker {data[1]}")
		audio = load_audio(data[0], sampling_rate)
		if isinstance(audio, str):
			print(f"Tossing {data[0]}, bad file")
			continue
		audio = signal.lfilter(bh, ah, audio)
		if data[1] in speaker_ids_to_watch:
			print(f"I loaded for speaker {data[1]}")
		idx1 = 0
		for audio in slicer.slice(audio):
			i = 0
			do_loop = True
			while do_loop:
				start = int(sampling_rate * (per - overlap) * i)
				i += 1
				if len(audio[start:]) > tail * sampling_rate:
					if data[1] in speaker_ids_to_watch:
						print(f"I'm splitting and writing for speaker {data[1]}")
					tmp_audio = audio[start : start + int(per * sampling_rate)]
					norm_write(
						tmp_audio,
						index,
						idx1,
						data[1],
						outdir,
						outdir_16k,
						sampling_rate,
						max,
						alpha,
						is_normalize,
					)
					idx1 += 1
				else:
					tmp_audio = audio[start:]
					if log_all and data[1] in speaker_ids_to_watch:
						print("I'm breaking now")
					do_loop = False
			if data[1] in speaker_ids_to_watch:
				print(f"I'm going to write for speaker {data[1]}")
			norm_write(
				tmp_audio,
				index,
				idx1,
				data[1],
				outdir,
				outdir_16k,
				sampling_rate,
				max,
				alpha,
				is_normalize,
			)
			idx1 += 1
		index += 1


def preprocess_audio(
	datasets: List[Tuple[str, int]],  # List[(path, speaker_id)]
	sampling_rate: int,
	num_processes: int,
	training_dir: str,
	is_normalize: bool,
	mute_wav_path: str,
):
	waves_dir = os.path.join(training_dir, "0_gt_wavs")
	waves16k_dir = os.path.join(training_dir, "1_16k_wavs")
	if os.path.exists(waves_dir) and os.path.exists(waves16k_dir):
		return

	for speaker_id in set([spk for _, spk in datasets]):
		os.makedirs(os.path.join(waves_dir, f"{speaker_id:05}"), exist_ok=True)
		os.makedirs(os.path.join(waves16k_dir, f"{speaker_id:05}"), exist_ok=True)
	processed = [datasets[i:i + round(len(datasets) / num_processes)] for i in range(0, len(datasets), round(len(datasets) / num_processes))]
	process_id = 0
	audio_total = 0
	procs = []
	for i in processed:
		shouldlog = False
		print(f"Number of audio files I'm handling {process_id} {len(i)}")
		for speaker_id in set([spk for _, spk in i]):
			if speaker_id in speaker_ids_to_watch:
				print(f"I'm processing speaker ID {speaker_id}")
				shouldlog = True
		audio_total += len(i)
		slicer = Slicer(
			sr=sampling_rate,
			threshold=-42,
			min_length=1500,
			min_interval=400,
			hop_size=15,
			max_sil_kept=500,
		)
		proc = Process(target=pipeline, args=(
			slicer,
			i,
			waves_dir,
			waves16k_dir,
			sampling_rate,
			is_normalize,
			process_id,
			shouldlog
		))
		procs.append(proc)
		proc.start()
		process_id += 1
	for proc in procs:
		proc.join()
	print(f"Audio total {audio_total}")
	for speaker_id in set([spk for _, spk in datasets]):
		write_mute(mute_wav_path, speaker_id, waves_dir, waves16k_dir, sampling_rate)
