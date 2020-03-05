import os
import sys
import time
import argparse
import unicodedata
import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm
from hparams import hparams


def run_prepare(args, hparams):

    def normalize_wave(wave, sample_rate):
        """normalize wave format"""
        wave = librosa.resample(wave, sample_rate, hparams.sample_rate)
        return wave

    def normalize_text(text):
        """normalize text format"""
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')
        return text.strip()

    if args.dataset == 'BIAOBEI':
        dataset_name = 'BZNSYP'
        dataset_path = os.path.join('./', dataset_name)

        if not os.path.isdir(dataset_path):
            print("BIAOBEI dataset folder doesn't exist")
            sys.exit(0)

        total_duration = 0

        text_file_path = os.path.join(dataset_path, 'ProsodyLabeling', '000001-010000.txt')
        try:
            text_file = open(text_file_path, 'r', encoding='utf8')
        except FileNotFoundError:
            print('text file no exist')
            sys.exit(0)

        data_array = np.zeros(shape=(1, 3), dtype=str)
        for index, each in tqdm(enumerate(text_file.readlines())):
            if index % 2 == 0:
                list = []
                basename = each.strip().split()[0]
                raw_text = each.strip().split()[1]
                list.append(basename)
                list.append(raw_text)
            else:
                pinyin_text = normalize_text(each)
                list.append(pinyin_text)
                data_array = np.append(data_array, np.array(list).reshape(1, 3), axis=0)
                wave_file_path = os.path.join(dataset_path, 'Wave', '{}.wav'.format(basename))
                if not os.path.exists(wave_file_path):
                    # print('wave file no exist')
                    continue
                try:
                    wave, sr = librosa.load(wave_file_path, sr=None)
                except EOFError:
                    print('wave format error at {}'.format(basename+'.wav'))
                    continue
                if not sr == hparams.sample_rate:
                    wave = normalize_wave(wave, sr)

                duration = librosa.get_duration(wave)
                total_duration += duration
                librosa.output.write_wav(wave_file_path, wave, hparams.sample_rate)

        data_frame = pd.DataFrame(data_array[1:])
        data_frame.to_csv(os.path.join(dataset_path, 'metadata.csv'), sep='|', header=False, index=False, encoding='utf8')
        text_file.close()
        print("total audio duration: %ss" % (time.strftime('%H:%M:%S', time.gmtime(total_duration))))
    elif args.dataset == 'THCHS-30':
        dataset_name = 'data_thchs30'
        dataset_path = os.path.join('./', dataset_name)

        if not os.path.isdir(dataset_path):
            print("{} dataset folder doesn't exist".format(args.dataset))
            sys.exit(0)

        total_duration = 0

        raw_dataset_path = os.path.join(dataset_path, 'wavs')

        data_array = np.zeros(shape=(1, 3), dtype=str)
        for root, dirs, files in os.walk(raw_dataset_path):
            for file in tqdm(files):
                if not file.endswith('.wav.trn'):
                    continue
                list = []
                basename = file[:-8]
                list.append(basename)
                text_file = os.path.join(raw_dataset_path, file)
                if not os.path.exists(text_file):
                    print('text file {} no exist'.format(file))
                    continue
                with open(text_file, 'r', encoding='utf8') as f:
                    lines = f.readlines()
                raw_text = lines[0].rstrip('\n')
                pinyin_text = lines[1].rstrip('\n')
                pinyin_text = normalize_text(pinyin_text)
                list.append(raw_text)
                list.append(pinyin_text)
                wave_file = os.path.join(raw_dataset_path, '{}.wav'.format(basename))
                if not os.path.exists(wave_file):
                    print('wave file {}.wav no exist'.format(basename))
                    continue
                try:
                    wave, sr = librosa.load(wave_file, sr=None)
                except EOFError:
                    print('wave file {}.wav format error'.format(basename))
                    continue
                if not sr == hparams.sample_rate:
                    print('sample rate of wave file {}.wav no match'.format(basename))
                    wave = librosa.resample(wave, sr, hparams.sample_rate)
                duration = librosa.get_duration(wave)
                if duration < 10:
                    total_duration += duration
                    librosa.output.write_wav(wave_file, wave, hparams.sample_rate)
                    data_array = np.append(data_array, np.array(list).reshape(1, 3), axis=0)
        data_frame = pd.DataFrame(data_array[1:])
        data_frame.to_csv(os.path.join(dataset_path, 'metadata.csv'), sep='|', header=False, index=False, encoding='utf8')
        print("total audio duration: %ss" % (time.strftime('%H:%M:%S', time.gmtime(total_duration))))
    elif args.dataset == 'AISHELL-2':
        dataset_name = 'aishell2_16000'
        dataset_path = os.path.join(os.getcwd(), args.dataset)

        if os.path.isdir(dataset_path):
            print("{} dataset folder already exists".format(args.dataset))
            sys.exit(0)

        os.mkdir(dataset_path)
        dataset_path = os.path.join(dataset_path, dataset_name)
        os.mkdir(dataset_path)

        sample_rate = 16000  # original sample rate
        total_duration = 0

        raw_dataset_path = os.path.join(os.getcwd(), 'aishell2', 'dataAishell2')
        wave_dir_path = os.path.join(raw_dataset_path, 'wav')
        text_file_path = os.path.join(raw_dataset_path, 'transcript', 'aishell2_transcript.txt')
        try:
            text_file = open(text_file_path, 'r', encoding='utf8')
        except FileNotFoundError:
            print('text file no exist')
            sys.exit(0)

        def normalize_text(text):
            """normalize text format"""
            text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn')
            return text.strip()

        def normalize_wave(wave):
            """normalize wave format"""
            wave = librosa.resample(wave, sample_rate, hparams.sample_rate)
            return wave

        # for index, each in tqdm(enumerate(text_file.readlines())):
        #


if __name__ == '__main__':
    print('preparing dataset..')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=['AISHELL-2', 'THCHS-30', 'BIAOBEI'], default='THCHS-30',
                        help='dataset name')
    args = parser.parse_args()

    run_prepare(args, hparams)
