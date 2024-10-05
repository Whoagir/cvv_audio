import parselmouth
import numpy as np


# Откройте аудиофайл
sound = parselmouth.Sound(r"C:\Users\Пользователь\PycharmProjects\test_3\hhfrelance\audio\1.wav")
# Извлеките форманты
try:
    formants = sound.to_formant_burg()
except RuntimeError:
    print("Не удалось извлечь форманты из файла.")
    exit()
# Извлеките количество формант
num_formants = formants.get_number_of_frames()
# Проверьте, содержит ли звук достаточное количество кадров формант
if num_formants < 3:
    print("Файл не содержит достаточного количества кадров формант.")
    exit()
# Извлеките значения формант в моментах времени t = [0.5, 1.0, 1.5] секунды
times = [0.5, 1.0, 1.5]
formant_values = []
for i in range(num_formants):
    try:
        formant_values.append([formants.get_value_at_time(i + 1, time) for time in times])
    except RuntimeError:
        print("Не удалось извлечь значения формант в указанных моментах времени.")
        exit()
# Извлеките другие параметры голоса
try:
    pitch = sound.to_pitch()
    intensity = sound.to_intensity(time_step=0.01)
except RuntimeError:
    print("Не удалось извлечь другие параметры голоса.")
    exit()
# Преобразуйте объект высоты тона в список значений высоты тона
pitch_values = [frame.candidates[0].frequency for frame in pitch]
# Обрежьте массивы до минимальной длины
min_length = min(len(pitch_values), len(formant_values[0]), len(intensity))
pitch_values = pitch_values[:min_length]
formant_values = [formant_value[:min_length] for formant_value in formant_values]
intensity = intensity.values[:min_length]
# Замените пропущенные значения в массивах параметров голоса нулями
pitch_values = np.where(np.isnan(pitch_values), 0, pitch_values)
formant_values = np.where(np.isnan(formant_values), 0, formant_values)
intensity = np.where(np.isnan(intensity), 0, intensity)
# Преобразуйте список значений формант в одномерный массив NumPy
formant_values = np.array(formant_values).flatten()
# Преобразуйте массив интенсивности в одномерный массив NumPy
intensity = intensity.flatten()
# Проверьте, является ли диапазон значений высоты тона ненулевым
if np.max(pitch_values) - np.min(pitch_values) > 0:
    # Масштабируйте значения высоты тона в диапазон от 0 до 1
    pitch_values = (pitch_values - np.min(pitch_values)) / (np.max(pitch_values) - np.min(pitch_values))
else:
    # Если диапазон значений равен нулю, установите все значения высоты тона в 0
    pitch_values = np.zeros(len(pitch_values))
# Масштабируйте значения формант и интенсивности в диапазон от 0 до 1
formant_values = (formant_values - np.min(formant_values)) / (np.max(formant_values) - np.min(formant_values))
intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
# Объедините массивы значений высоты тона, формант и интенсивности
voice_fingerprint = np.concatenate((pitch_values, formant_values, intensity))
# Распечатайте длины массивов и первые 10 значений слепка голоса
print("Длины массивов:")
print(f"Высота тона: {len(pitch_values)}")
print(f"Форманты: {len(formant_values)}")
print(f"Интенсивность: {len(intensity)}")
print("Первые 10 значений слепка голоса:", voice_fingerprint[:10])
# Сохраните вектор слепка голоса
np.save("voice_fingerprint.npy", voice_fingerprint)
