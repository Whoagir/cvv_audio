import wave
import json
import soundfile as sf
from scipy.signal import wiener
import numpy as np
from vosk import Model, KaldiRecognizer
from datetime import datetime
import multiprocessing
from pyannote.audio import Pipeline

TOKEN = "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"


def remove_noise(input_audio_file, output_file):  # Функция для удаления шумов из аудиофайла.
    audio_data, sample_rate = sf.read(input_audio_file)  # Считывание аудиоданных и частоты дискретизации из файла
    if len(audio_data.shape) > 1:  # Проверка на многоканальность: если аудиофайл имеет несколько каналов, усредняем их
        audio_data = np.mean(audio_data, axis=1)
    processed_audio = wiener(audio_data)  # Применение метода Винера для удаления шумов и улучшения качества звука
    sf.write(output_file, processed_audio, sample_rate)  # Сохранение обработанного звука в новый файл


def recognize_audio(input_audio_file,
                    model_path):  # Функция для распознавания речи в аудиофайле с использованием модели.
    model = Model(model_path)  # Инициализация модели для распознавания
    with wave.open(input_audio_file, "rb") as wf:  # Открытие аудиофайла для чтения
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        results = []
        while True:  # Обработка аудиоданных по частям и запись результатов распознавания
            data = wf.readframes(16000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part_result = json.loads(rec.Result())
                # print(part_result)
                results.append(part_result)
        part_result = json.loads(rec.FinalResult())  # Получение окончательного результата распознавания
        results.append(part_result)
        print(results)
        return results


class Word:
    def __init__(self, dict):
        self.conf = dict["conf"]
        self.end = dict["end"]
        self.start = dict["start"]
        self.word = dict["word"]
        self.speaker = dict.get("speaker")

    def to_string(self):
        start_time = datetime.utcfromtimestamp(self.start).strftime('%H:%M:%S.%f')[:-3]
        end_time = datetime.utcfromtimestamp(self.end).strftime('%H:%M:%S.%f')[:-3]
        return "{} {:20} from {} to {}, confidence is {:.2f}%".format(
            self.speaker, self.word, start_time, end_time, self.conf * 100)


def process_results(results, speakers):
    def find_speaker(pos):
        for start, end, speaker in speakers:
            if pos >= start and pos <= end:
                return speaker
        return None

    recognized_strings = []
    for result in results:
        if 'result' in result:
            words = result['result']
            for word in words:
                word["speaker"] = find_speaker(word["start"])
                recognized_word = Word(word)
                recognized_strings.append(recognized_word.to_string())
                print(recognized_word.to_string())
    with open("word_data.json", "w", encoding='utf-8') as outfile:
        outfile.write('\n'.join(recognized_strings))


def process_audio_file(audio_filename, processed_audio_filename, model_path):
    remove_noise(audio_filename, processed_audio_filename)  # Удаление шума из аудиофайла
    audio_data, sample_rate = sf.read(
        processed_audio_filename)  # Чтение обработанного аудиофайла и получение данных и частоты дискретизации
    if len(audio_data.shape) > 1:  # Если аудиофайл имеет более одного канала, усредняем их
        audio_data = np.mean(audio_data, axis=1)
    pipeline = Pipeline.from_pretrained(  # Создание экземпляра pipeline для диаризации дикторов
        'pyannote/speaker-diarization-3.1',
        use_auth_token=TOKEN
    )
    diarization = pipeline(audio_filename)  # Диаризация аудиофайла для определения дикторов
    speakers = []  # Сбор информации о дикторах: начало и конец голоса, метка диктора
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append([turn.start, turn.end, speaker])
    with open('speakers.log', 'wt') as f:  # Запись информации о дикторах в файл
        for start, end, speaker in speakers:
            f.write(f'{start}\t{end}\t{speaker}\n')
    pool = multiprocessing.Pool()  # Создание пула процессов для распознавания речи
    results = pool.apply(recognize_audio, (processed_audio_filename,
                                           model_path,))  # Применение функции recognize_audio к обработанному аудиофайлу с использованием модели
    print(speakers)  # Вывод информации о дикторах на экран
    process_results(results, speakers)  # Обработка результатов распознавания речи и дикторов


if __name__ == "__main__":
    model_path = "C://model"
    audio_filename = r"C:\Users\Пользователь\PycharmProjects\test_3\hhfrelance\audio\test3.wav"
    processed_audio_filename = r"C:\Users\Пользователь\PycharmProjects\test_3\hhfrelance\audio\test3.wav"
    process_audio_file(audio_filename, processed_audio_filename, model_path)
