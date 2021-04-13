import os, pocketsphinx, librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
import sys
import pandas as pd

STUDY_ID = 'study_id'
TIME_STAMP_INI = 'timestamp_ini'
TIME_STAMP_END = 'timestamp_end'
TRANSCRIPT_ID = 'transcript_id'
PAUSE_START = 'pause_start'
PAUSE_END = 'pause_end'
PAUSE_LENGTH = 'pause_length'
AUDIO_FILE_LENGTH = 'audio_file_length'
READING_TASK = 'Reading'
MEMORY_TASK = 'Memory'
COOKIE_THEFT_TASK = 'CookieTheft'
TASK = 'task'
PAUSE_COLUMNS = [TRANSCRIPT_ID, PAUSE_START, PAUSE_END, PAUSE_LENGTH, AUDIO_FILE_LENGTH, TASK]
PAUSE_THRESHOLD = 150
SILENCE_THRESHOLD = -32
seen = []


def get_correct_time_stamps(input_file, time_stamps):
	"""
	Finds the start and end for each activity

	:param input_file: WAV file
	:param time_stamps: DataFrame containing the timestamps for the start and end of each task
	:return: the audio segments for the cookie theft, reading and memory task
	"""
	audio = AudioSegment.from_wav(input_file)
	file_id = input_file[6:-4]
	study_times = time_stamps[time_stamps[STUDY_ID] == file_id]
	try:
		cookie_theft_start = study_times.iloc[1][TIME_STAMP_INI]
		cookie_theft_end = study_times.iloc[1][TIME_STAMP_END]
		reading_start = study_times.iloc[2][TIME_STAMP_INI]
		reading_end = study_times.iloc[2][TIME_STAMP_END]
		memory_start = study_times.iloc[3][TIME_STAMP_INI]
		memory_end = study_times.iloc[3][TIME_STAMP_END]

		cookie_theft_audio = audio[cookie_theft_start:cookie_theft_end]
		reading_audio = audio[reading_start:reading_end]
		memory_audio = audio[memory_start:memory_end]
		return cookie_theft_audio, reading_audio, memory_audio
	except IndexError:
		return 0, 0, 0


def extract_pause_features(input_file, time_stamps):
	"""
	Extracted the following information from the WAV files
	- Start of pause
	- End of pause
	- Pause length
	- Length of activity
	- Task (one of CookieTheft, Reading or Memory)

	:param input_file: WAV file
	:param time_stamps: DataFrame containing the timestamps for the start and end of each task
	:return: a CSV file containing the pause data
	"""

	cookie_theft_audio, reading_audio, memory_audio = get_correct_time_stamps(input_file, time_stamps)
	if cookie_theft_audio != 0:
		print("input: ", input_file[6:-4])

		cookie_silent_ranges = detect_silence(cookie_theft_audio, min_silence_len=PAUSE_THRESHOLD, silence_thresh=-32)
		reading_silent_ranges = detect_silence(reading_audio, min_silence_len=PAUSE_THRESHOLD, silence_thresh=-32)
		memory_silent_ranges = detect_silence(memory_audio, min_silence_len=PAUSE_THRESHOLD, silence_thresh=-32)

		pause_tracker = pd.DataFrame(columns=PAUSE_COLUMNS)

		# record all pauses in cookie theft activity
		for start, end in cookie_silent_ranges:
			pause_tracker = pause_tracker.append({
				TRANSCRIPT_ID: input_file[6:-4],
				PAUSE_START: start,
				PAUSE_END: end,
				PAUSE_LENGTH: end - start,
				AUDIO_FILE_LENGTH: cookie_theft_audio.duration_seconds*1000,
				TASK: COOKIE_THEFT_TASK
			}, ignore_index=True)

		# record all pauses in reading activity
		for start, end in reading_silent_ranges:
			pause_tracker = pause_tracker.append({
				TRANSCRIPT_ID: input_file[6:-4],
				PAUSE_START: start,
				PAUSE_END: end,
				PAUSE_LENGTH: end - start,
				AUDIO_FILE_LENGTH: reading_audio.duration_seconds*1000,
				TASK: READING_TASK
			}, ignore_index=True)

		# record all pauses in reading activity
		for start, end in memory_silent_ranges:
			pause_tracker = pause_tracker.append({
				TRANSCRIPT_ID: input_file[6:-4],
				PAUSE_START: start,
				PAUSE_END: end,
				PAUSE_LENGTH: end - start,
				AUDIO_FILE_LENGTH: memory_audio.duration_seconds*1000,
				TASK: MEMORY_TASK
			}, ignore_index=True)

		pause_tracker.to_csv('pause_csvs/' + input_file[6:-4] + '.csv')
		print('in a csv')


def extract_pause_separated_task(input_file):
	audio = AudioSegment.from_wav(input_file)
	file_id = input_file[6:-6]
	task = input_file[-5]
	if file_id in seen:
		pause_tracker = pd.read_csv('pause_csvs/' + file_id+ '.csv')
	else:
		pause_tracker = pd.DataFrame(columns=PAUSE_COLUMNS)
		seen.append(file_id)

	if task == '1':
		task = COOKIE_THEFT_TASK
	elif task == '2':
		task = READING_TASK
	elif task == '3':
		task = MEMORY_TASK
	if audio != 0:
		pauses = detect_silence(audio, min_silence_len=PAUSE_THRESHOLD, silence_thresh=-32)
		for start, end in pauses:
			pause_tracker = pause_tracker.append({
				TRANSCRIPT_ID: input_file[6:-4],
				PAUSE_START: start,
				PAUSE_END: end,
				PAUSE_LENGTH: end - start,
				AUDIO_FILE_LENGTH: audio.duration_seconds * 1000,
				TASK: task
			}, ignore_index=True)

	pause_tracker.to_csv('pause_csvs/' + file_id + '.csv')
	print('in a csv')







record = sys.argv[1]
folder = sys.argv[2]
# time_stamps = pd.read_csv(sys.argv[3])

print(record)
print(folder)

filelist=list()

if folder=='y':
	os.chdir('jan26-data')
	listdir=os.listdir()
	for i in range(len(listdir)):
		if listdir[i][-4:] in ['.wav', '.mp3', '.m4a']:
			if listdir[i][-4:] != '.wav':
				wavfile=listdir[i][0:-4]+'.wav'
				os.system('ffmpeg -i %s %s'%(listdir[i], wavfile))
				os.remove(listdir[i])
				filelist.append(wavfile)
			else:
				filelist.append(listdir[i])

	print(filelist)
	for i in range(len(filelist)):
		print(i)
		extract_pause_separated_task(filelist[i])