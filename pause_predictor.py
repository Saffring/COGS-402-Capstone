"""
This file is to use the amount of pauses in a person transcript to predict if they are a healhty control or a patient
"""
import pandas as pd
import numpy as np
import sys
import os
import re
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score

TASK_3 = [3, '3']
TASK_2 = [2, '2']
TASK_1 = [1, '1']

TRANSCRIPT_ID = 'transcript_id'
TASK = 'task'

READING_TASK = 'Reading'
MEMORY_TASK = 'Memory'
COOKIE_THEFT_TASK = 'CookieTheft'

HAS_DEMENTIA = 'has_dementia'


PAUSE_LENGTH = 'pause_length'
AUDIO_FILE_LENGTH = 'audio_file_length'

COOKIE_NUMBER_OF_PAUSES = 'cookie_number_of_pauses'
COOKIE_MAXIMUM_PAUSE_DURATION = 'cookie_maximum_pause_duration'
COOKIE_PAUSE_RATE = 'cookie_pause_rate'
COOKIE_DURATION = 'cookie_duration'
COOKIE_PHONATION_TIME ='cookie_phonation_time'
COOKIE_PROPORTION_OF_TIME_SPENT_SPEAKING = 'cookie_proportion_of_time_spent_speaking'
COOKIE_MEAN_PAUSE_LENGTH = 'cookie_mean_pause_length'
COOKIE_STD_PAUSE_LENGTH = 'cookie_sd_pause_length'

READING_NUMBER_OF_PAUSES = 'reading_number_of_pauses'
READING_MAXIMUM_PAUSE_DURATION = 'reading_maximum_pause_duration'
READING_PAUSE_RATE = 'reading_pause_rate'
READING_DURATION = 'reading_duration'
READING_PHONATION_TIME ='reading_phonation_time'
READING_PROPORTION_OF_TIME_SPENT_SPEAKING = 'reading_proportion_of_time_spent_speaking'
READING_MEAN_PAUSE_LENGTH = 'reading_mean_pause_length'
READING_STD_PAUSE_LENGTH = 'reading_sd_pause_length'

MEMORY_NUMBER_OF_PAUSES = 'memory_number_of_pauses'
MEMORY_MAXIMUM_PAUSE_DURATION = 'memory_maximum_pause_duration'
MEMORY_PAUSE_RATE = 'memory_pause_rate'
MEMORY_DURATION = 'memory_duration'
MEMORY_PHONATION_TIME ='memory_phonation_time'
MEMORY_PROPORTION_OF_TIME_SPENT_SPEAKING = 'memory_proportion_of_time_spent_speaking'
MEMORY_MEAN_PAUSE_LENGTH = 'memory_mean_pause_length'
MEMORY_STD_PAUSE_LENGTH = 'memory_sd_pause_length'

PAUSE_COLUMNS = [TRANSCRIPT_ID,
                 COOKIE_NUMBER_OF_PAUSES,
                 COOKIE_MAXIMUM_PAUSE_DURATION,
                 COOKIE_PAUSE_RATE,
                 COOKIE_DURATION,
                 COOKIE_PHONATION_TIME,
                 COOKIE_PROPORTION_OF_TIME_SPENT_SPEAKING,
                 READING_NUMBER_OF_PAUSES,
                 READING_MAXIMUM_PAUSE_DURATION,
                 READING_PAUSE_RATE,
                 READING_DURATION,
                 READING_PHONATION_TIME,
                 READING_PROPORTION_OF_TIME_SPENT_SPEAKING,
                 MEMORY_NUMBER_OF_PAUSES,
                 MEMORY_MAXIMUM_PAUSE_DURATION,
                 MEMORY_PAUSE_RATE,
                 MEMORY_DURATION,
                 MEMORY_PHONATION_TIME,
                 MEMORY_PROPORTION_OF_TIME_SPENT_SPEAKING
                 ]

COOKIE_SPEECH_RATE = 'cookie_speech_rate'
READING_SPEECH_RATE = 'reading_speech_rate'
MEMORY_SPEECH_RATE = 'memory_speech_rate'

COOKIE_AVERAGE_SYLLABLE_DURATION = 'cookie_average_syllable_duration'
READING_AVERAGE_SYLLABLE_DURATION = 'reading_average_syllable_duration'
MEMORY_AVERAGE_SYLLABLE_DURATION = 'memory_average_syllable_duration'

COOKIE_PAUSE_PER_SYLLABLE = 'cookie_pause_per_syllable'
READING_PAUSE_PER_SYLLABLE = 'reading_pause_per_syllable'
MEMORY_PAUSE_PER_SYLLABLE = 'memory_pause_per_syllable'

COOKIE_SYLLABLE_COUNT = 'cookie_syllable_count'
READING_SYLLABLE_COUNT = 'reading_syllable_count'
MEMORY_SYLLABLE_COUNT = 'memory_syllable_count'
SYLLABLE_COLUMNS = [TRANSCRIPT_ID, MEMORY_SYLLABLE_COUNT]

SPEECH_FEATURES_SET = [
    COOKIE_NUMBER_OF_PAUSES,
    COOKIE_MAXIMUM_PAUSE_DURATION,
    COOKIE_PAUSE_RATE,
    COOKIE_DURATION,
    COOKIE_PHONATION_TIME,
    COOKIE_PROPORTION_OF_TIME_SPENT_SPEAKING,
    COOKIE_SYLLABLE_COUNT,
    COOKIE_SPEECH_RATE,
    COOKIE_AVERAGE_SYLLABLE_DURATION,
    COOKIE_PAUSE_PER_SYLLABLE,

    READING_NUMBER_OF_PAUSES,
    READING_MAXIMUM_PAUSE_DURATION,
    READING_PAUSE_RATE,
    READING_DURATION,
    READING_PHONATION_TIME,
    READING_PROPORTION_OF_TIME_SPENT_SPEAKING,
    READING_SYLLABLE_COUNT,
    READING_SPEECH_RATE,
    READING_AVERAGE_SYLLABLE_DURATION,
    READING_PAUSE_PER_SYLLABLE,

    MEMORY_NUMBER_OF_PAUSES,
    MEMORY_MAXIMUM_PAUSE_DURATION,
    MEMORY_PAUSE_RATE,
    MEMORY_DURATION,
    MEMORY_PHONATION_TIME,
    MEMORY_PROPORTION_OF_TIME_SPENT_SPEAKING,
    MEMORY_SYLLABLE_COUNT,
    MEMORY_SPEECH_RATE,
    MEMORY_AVERAGE_SYLLABLE_DURATION,
    MEMORY_PAUSE_PER_SYLLABLE,
]

READING_FEATURES = [
    READING_NUMBER_OF_PAUSES,
    READING_MAXIMUM_PAUSE_DURATION,
    READING_PAUSE_RATE,
    READING_DURATION,
    READING_PHONATION_TIME,
    READING_PROPORTION_OF_TIME_SPENT_SPEAKING,
    READING_SYLLABLE_COUNT,
    READING_SPEECH_RATE,
    READING_AVERAGE_SYLLABLE_DURATION,
    READING_PAUSE_PER_SYLLABLE
]

COOKIE_FEATURES = [
    COOKIE_NUMBER_OF_PAUSES,
    COOKIE_MAXIMUM_PAUSE_DURATION,
    COOKIE_PAUSE_RATE,
    COOKIE_DURATION,
    COOKIE_PHONATION_TIME,
    COOKIE_PROPORTION_OF_TIME_SPENT_SPEAKING,
    COOKIE_SYLLABLE_COUNT,
    COOKIE_SPEECH_RATE,
    COOKIE_AVERAGE_SYLLABLE_DURATION,
    COOKIE_PAUSE_PER_SYLLABLE
]

MEMORY_FEATURES = [
    MEMORY_NUMBER_OF_PAUSES,
    MEMORY_MAXIMUM_PAUSE_DURATION,
    MEMORY_PAUSE_RATE,
    MEMORY_DURATION,
    MEMORY_PHONATION_TIME,
    MEMORY_PROPORTION_OF_TIME_SPENT_SPEAKING,
    MEMORY_SYLLABLE_COUNT,
    MEMORY_SPEECH_RATE,
    MEMORY_AVERAGE_SYLLABLE_DURATION,
    MEMORY_PAUSE_PER_SYLLABLE
]

PREDICTIONS = 'predictions'
TRAIN = 'train'
TEST = 'test'
Y_TEST = 'y_test'

F1_SCORE = 'f1_score'
ACCURACY = 'accuracy'


# Syllable stimation stuff here

subSyllables = [
    'cial',
    'tia',
    'cius',
    'cious',
    'uiet',
    'gious',
    'geous',
    'priest',
    'giu',
    'dge',
    'ion',
    'iou',
    'sia$',
    '.che$',
    '.ched$',
    '.abe$',
    '.ace$',
    '.ade$',
    '.age$',
    '.aged$',
    '.ake$',
    '.ale$',
    '.aled$',
    '.ales$',
    '.ane$',
    '.ame$',
    '.ape$',
    '.are$',
    '.ase$',
    '.ashed$',
    '.asque$',
    '.ate$',
    '.ave$',
    '.azed$',
    '.awe$',
    '.aze$',
    '.aped$',
    '.athe$',
    '.athes$',
    '.ece$',
    '.ese$',
    '.esque$',
    '.esques$',
    '.eze$',
    '.gue$',
    '.ibe$',
    '.ice$',
    '.ide$',
    '.ife$',
    '.ike$',
    '.ile$',
    '.ime$',
    '.ine$',
    '.ipe$',
    '.iped$',
    '.ire$',
    '.ise$',
    '.ished$',
    '.ite$',
    '.ive$',
    '.ize$',
    '.obe$',
    '.ode$',
    '.oke$',
    '.ole$',
    '.ome$',
    '.one$',
    '.ope$',
    '.oque$',
    '.ore$',
    '.ose$',
    '.osque$',
    '.osques$',
    '.ote$',
    '.ove$',
    '.pped$',
    '.sse$',
    '.ssed$',
    '.ste$',
    '.ube$',
    '.uce$',
    '.ude$',
    '.uge$',
    '.uke$',
    '.ule$',
    '.ules$',
    '.uled$',
    '.ume$',
    '.une$',
    '.upe$',
    '.ure$',
    '.use$',
    '.ushed$',
    '.ute$',
    '.ved$',
    '.we$',
    '.wes$',
    '.wed$',
    '.yse$',
    '.yze$',
    '.rse$',
    '.red$',
    '.rce$',
    '.rde$',
    '.ily$',
    '.ely$',
    '.des$',
    '.gged$',
    '.kes$',
    '.ced$',
    '.ked$',
    '.med$',
    '.mes$',
    '.ned$',
    '.[sz]ed$',
    '.nce$',
    '.rles$',
    '.nes$',
    '.pes$',
    '.tes$',
    '.res$',
    '.ves$',
    'ere$'
]

addSyllables = [
    'ia',
    'riet',
    'dien',
    'ien',
    'iet',
    'iu',
    'iest',
    'io',
    'ii',
    'ily',
    '.oala$',
    '.iara$',
    '.ying$',
    '.earest',
    '.arer',
    '.aress',
    '.eate$',
    '.eation$',
    '[aeiouym]bl$',
    '[aeiou]{3}',
    '^mc', 'ism',
    '^mc', 'asm',
    '([^aeiouy])1l$',
    '[^l]lien',
    '^coa[dglx].',
    '[^gq]ua[^auieo]',
    'dnt$'
]

re_subsyllables = []
for s in subSyllables:
    re_subsyllables.append(re.compile(s))

re_addsyllables = []
for s in addSyllables:
    re_addsyllables.append(re.compile(s))


def estimate(word):
    """
    Estimate the number of syllables for a word
    :param word:
    :return: integer
    """
    parts = re.split(r'[^aeiouy]+', word)
    valid_parts = []

    for part in parts:
        if part != '':
            valid_parts.append(part)

    syllables = 0

    for p in re_subsyllables:
        if p.match(word):
            syllables -= 1

    for p in re_addsyllables:
        if p.match(word):
            syllables += 1

    syllables += len(valid_parts)

    if syllables <= 0:
        syllables = 1

    return syllables


"""
The following functions are those used in the most recent feature extraction
"""


def csv_to_txt():
    """
    takes the list of transcript csv files and adds spoken words associated with task 2 to a txt file
    """
    print('csv to text')
    input_files = sys.argv[1]
    i = 0
    for filename in os.listdir(input_files):
        print(i, filename[11:-4])
        output_txt_file = ''
        current_csv_df = pd.read_csv(sys.argv[1] + filename)
        for index, row in current_csv_df.iterrows():
            if (row['task_number'] == TASK_3[0] or row['task_number'] == TASK_3[1]) and type(
                    row['spoken_word']) != float:
                output_txt_file += " " + row['spoken_word']
        txt_file = open('jan27_memory_texts/' + filename[11:-4] + '.txt', "a")
        txt_file.write(output_txt_file.lstrip(' '))
        txt_file.close()
        i+=1


def extract_syllable_features_from_txt():
    """
    This function iterates through a folder and then sums the amount of syllables in a text and then adds this a row
    in a CSV file. In order for easier merging, manual substitution of the kind of syllable count is necessary.

    exp: if you are counting the syllable count for all the memory transcript text files, it is necessary to change the
    SYLLABLE_COLUMNS constant to include MEMORY_SYLLABLE_COUNT as the second column.
    :return: a CSV with the syllable count for each text
    """
    input_files = sys.argv[1]
    csv_name = sys.argv[2]
    syllable_stats = pd.DataFrame(columns=SYLLABLE_COLUMNS)
    re_word = re.compile(r'[\w-]+')
    i = 0
    for filename in os.listdir(input_files):
        if filename != '.DS_Store':
            print(filename, i)
            syllable_count = 0
            for line in open(input_files+filename):
                for word in re_word.findall(line):
                    syllable_count += estimate(word)
            syllable_stats = syllable_stats.append({
                TRANSCRIPT_ID: filename[:-4],
                MEMORY_SYLLABLE_COUNT: syllable_count,
            }, ignore_index=True)
            i += 1
    syllable_stats = syllable_stats.set_index(TRANSCRIPT_ID)
    syllable_stats.to_csv(csv_name+'.csv')


def combine_syllable_csv():
    """
    Combines the syllable features from the cookie theft, memory and memory activities.
    :return: a CSV file with all the features combined on transcript ID
    """
    reading_csv = pd.read_csv(sys.argv[1])
    memory_csv = pd.read_csv(sys.argv[2])
    cookie_csv = pd.read_csv(sys.argv[3])
    merged = reading_csv.merge(memory_csv, on=TRANSCRIPT_ID)
    merged = merged.merge(cookie_csv, on=TRANSCRIPT_ID)
    merged.to_csv('jan27_merged_syllabus.csv', sep=',', header=True, index=False)


def extract_pause_features():
    """
    Uses the folder with all the pause CSV files as an argument. These CSV files are the output of the analysis of all
    the pauses from the WAV files for each of the three tasks: cookie theft, reading, memory.

    Extracting the pause features that were seen in the Fraser et al (2019) paper.
    - Pause Count (Number of pauses longer than 150ms)
    - Maximum pause duration
    - Duration (Total duration of the speech sample)
    - Phonation time (Total duration of time spent in speech (excludes silent pauses)
    - Proportion of time spent speaking (Phonation time divided by total duration)
    - Pause rate (Number of pauses divided by the total duration)
    - Mean Pause duration
    - Standard deviation of pause duration

    :return: a CSV file of all the extracted features
    """
    input_files = sys.argv[1]
    pause_statistics = pd.DataFrame(columns=PAUSE_COLUMNS)
    for filename in os.listdir(input_files):
        if filename != '.DS_Store':
            file_pauses = pd.read_csv(input_files+filename)
            print(filename)

            # task duration
            cookie_duration = file_pauses[file_pauses[TASK] == COOKIE_THEFT_TASK][AUDIO_FILE_LENGTH].iloc[0]
            reading_duration = file_pauses[file_pauses[TASK] == READING_TASK][AUDIO_FILE_LENGTH].iloc[0]
            memory_duration = file_pauses[file_pauses[TASK] == MEMORY_TASK][AUDIO_FILE_LENGTH].iloc[0]

            # length of pauses
            cookie_pause_lengths = file_pauses[file_pauses[TASK] == COOKIE_THEFT_TASK][PAUSE_LENGTH]
            reading_pause_lengths = file_pauses[file_pauses[TASK] == READING_TASK][PAUSE_LENGTH]
            memory_pause_lengths = file_pauses[file_pauses[TASK] == MEMORY_TASK][PAUSE_LENGTH]

            # number of pauses
            cookie_pause_number = len(file_pauses[file_pauses[TASK] == COOKIE_THEFT_TASK].index)
            reading_pause_number = len(file_pauses[file_pauses[TASK] == READING_TASK].index)
            memory_pause_number = len(file_pauses[file_pauses[TASK] == MEMORY_TASK].index)

            if cookie_duration - cookie_pause_lengths.sum() < 0:
                print("NEGATIVE COOKIE TIME ", filename)
                print(cookie_duration, cookie_pause_lengths.sum())

            if reading_duration - reading_pause_lengths.sum() < 0:
                print("NEGATIVE READING TIME ", filename)
                print(reading_duration, reading_pause_lengths.sum())
            if memory_duration - memory_pause_lengths.sum() < 0:
                print("NEGATIVE MEMORY TIME ", filename)
                print(memory_duration, memory_pause_lengths.sum())

            pause_statistics = pause_statistics.append({
                TRANSCRIPT_ID: filename[:-4],
                COOKIE_NUMBER_OF_PAUSES: cookie_pause_number,
                COOKIE_MAXIMUM_PAUSE_DURATION: cookie_pause_lengths.max(),
                COOKIE_DURATION: cookie_duration,
                COOKIE_PHONATION_TIME: cookie_duration - cookie_pause_lengths.sum(),
                COOKIE_PROPORTION_OF_TIME_SPENT_SPEAKING: (cookie_duration -
                                                           cookie_pause_lengths.sum()) / cookie_duration,
                COOKIE_PAUSE_RATE: cookie_pause_number/cookie_duration,
                COOKIE_MEAN_PAUSE_LENGTH: cookie_pause_lengths.mean(),
                COOKIE_STD_PAUSE_LENGTH: cookie_pause_lengths.std(),

                READING_NUMBER_OF_PAUSES: reading_pause_number,
                READING_MAXIMUM_PAUSE_DURATION: reading_pause_lengths.max(),
                READING_DURATION: reading_duration,
                READING_PHONATION_TIME: reading_duration - reading_pause_lengths.sum(),
                READING_PROPORTION_OF_TIME_SPENT_SPEAKING: (reading_duration -
                                                            reading_pause_lengths.sum()) / reading_duration,
                READING_PAUSE_RATE: reading_pause_number / reading_duration,
                READING_MEAN_PAUSE_LENGTH: reading_pause_lengths.mean(),
                READING_STD_PAUSE_LENGTH: reading_pause_lengths.std(),

                MEMORY_NUMBER_OF_PAUSES: memory_pause_number,
                MEMORY_MAXIMUM_PAUSE_DURATION: memory_pause_lengths.max(),
                MEMORY_DURATION: memory_duration,
                MEMORY_PHONATION_TIME: memory_duration - memory_pause_lengths.sum(),
                MEMORY_PROPORTION_OF_TIME_SPENT_SPEAKING: (memory_duration -
                                                           memory_pause_lengths.sum()) / memory_duration,
                MEMORY_PAUSE_RATE: memory_pause_number / memory_duration,
                MEMORY_MEAN_PAUSE_LENGTH: memory_pause_lengths.mean(),
                MEMORY_STD_PAUSE_LENGTH: memory_pause_lengths.std()
            }, ignore_index=True)
    pause_statistics = pause_statistics.set_index(TRANSCRIPT_ID)
    pause_statistics.to_csv('jan27_extracted_pauses.csv', sep=',', header=True)


def combine_pause_syllable():
    """
    This function takes the CSV file with the extracted pause features and the CSV with the extracted syllable features
    It then calculates the following features:
    - Speech Rate (number of syllables, divided by total duration)
    - Average syllable duration (phonation time, divided by the number of syllables)
    - Pause per syllable (number of pauses, divded by the number of syllables)

    :return: a CSV file with all the speech features from the Fraser et al 2019 paper
    """
    pause_csv = pd.read_csv(sys.argv[1])
    syllable_csv = pd.read_csv(sys.argv[2])
    merged = pause_csv.merge(syllable_csv, on=TRANSCRIPT_ID)

    # adding pause-syllable columns
    # speech rate
    merged[COOKIE_SPEECH_RATE] = merged[COOKIE_SYLLABLE_COUNT] / merged[COOKIE_DURATION]
    merged[READING_SPEECH_RATE] = merged[READING_SYLLABLE_COUNT] / merged[READING_DURATION]
    merged[MEMORY_SPEECH_RATE] = merged[MEMORY_SYLLABLE_COUNT] / merged[MEMORY_DURATION]

    # average syllable duration
    merged[COOKIE_AVERAGE_SYLLABLE_DURATION] = merged[COOKIE_PHONATION_TIME] / merged[COOKIE_SYLLABLE_COUNT]
    merged[READING_AVERAGE_SYLLABLE_DURATION] = merged[READING_PHONATION_TIME] / merged[READING_SYLLABLE_COUNT]
    merged[MEMORY_AVERAGE_SYLLABLE_DURATION] = merged[MEMORY_PHONATION_TIME] / merged[MEMORY_SYLLABLE_COUNT]

    # pause per syllable
    merged[COOKIE_PAUSE_PER_SYLLABLE] = merged[COOKIE_NUMBER_OF_PAUSES] / merged[COOKIE_SYLLABLE_COUNT]
    merged[READING_PAUSE_PER_SYLLABLE] = merged[READING_NUMBER_OF_PAUSES] / merged[READING_SYLLABLE_COUNT]
    merged[MEMORY_PAUSE_PER_SYLLABLE] = merged[MEMORY_NUMBER_OF_PAUSES] / merged[MEMORY_SYLLABLE_COUNT]

    # merged[SPEECH_RATE] = combined_syllable_count / merged[DURATION]
    # merged[AVERAGE_SYLLABLE_DURATION] = merged[PHONATION_TIME] / combined_syllable_count
    merged[HAS_DEMENTIA] = merged[TRANSCRIPT_ID].apply(lambda x: x[0] == 'E')
    merged.to_csv('jan27_language_features.csv', sep=',', header=True)


def create_logistic_regression(separate):
    """
    The goal of this logistic regression is to classify whether an individual is a healthy control or is a dementia
    patient.
    The data is stratified and undergoes 10-fold cross validation.

    :return: prints the metrics on this regression
    """

    pause_data = shuffle(pd.read_csv(sys.argv[1]))
    pause_data = pause_data.replace([np.inf, -np.inf], np.nan).dropna()
    # X = pause_data.drop([HAS_DEMENTIA, TRANSCRIPT_ID], axis=1)
    X = pause_data[MEMORY_FEATURES]
    y = pause_data[HAS_DEMENTIA]
    split_tracker = []
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=36851234)
    # n_repeats 10 too
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]
        y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]
        logmodel = LogisticRegression()
        logmodel.fit(X_train, y_train)
        predictions = logmodel.predict(X_test)
        split_tracker.append({
            TRAIN: train_index,
            TEST: test_index,
            PREDICTIONS: predictions,
            Y_TEST: y_test
        })
    accuracy = []
    f1 = []
    auc = []
    print("Predictions", split_tracker[0])
    for predictions in split_tracker:
        # print(classification_report(predictions[Y_TEST], predictions[PREDICTIONS]))
        accuracy.append(accuracy_score(predictions[Y_TEST], predictions[PREDICTIONS]))
        f1.append(f1_score(predictions[Y_TEST], predictions[PREDICTIONS]))
        auc.append(roc_auc_score(predictions[Y_TEST], predictions[PREDICTIONS]))
    print(accuracy)
    accuracy = np.array(accuracy)
    f1 = np.array(f1)
    auc = np.array(auc)
    print(len(accuracy))
    print('mean accuracy: ', accuracy.mean())
    print('mean f1 score: ',  f1.mean())
    print('mean auc: ', auc.mean())

NUMBER_OF_PAUSES = 'number_of_pauses'
MAXIMUM_PAUSE_DURATION = 'maximum_pause_duration'
PAUSE_RATE = 'pause_rate'
DURATION = 'duration'
PHONATION_TIME = 'phonation_time'
PROPORTION_OF_TIME_SPENT_SPEAKING = 'proportion_of_time_spent_speaking'
MEAN_PAUSE_LENGTH = 'mean_pause_length'
STD_PAUSE_LENGTH = 'sd_pause_length'
SPEECH_RATE = 'speech_rate'
AVERAGE_SYLLABLE_DURATION = 'average_syllable_duration'
PAUSE_PER_SYLLABLE = 'pause_per_syllable'
SYLLABLE_COUNT = 'syllable_count'

FRASER_FEATURES = [TRANSCRIPT_ID, NUMBER_OF_PAUSES, MAXIMUM_PAUSE_DURATION, PAUSE_RATE, DURATION, PHONATION_TIME,
                   PROPORTION_OF_TIME_SPENT_SPEAKING, MEAN_PAUSE_LENGTH, STD_PAUSE_LENGTH, SPEECH_RATE,
                   AVERAGE_SYLLABLE_DURATION, PAUSE_PER_SYLLABLE, SYLLABLE_COUNT, TASK]


def update_language_csv():
    language_csv = pd.read_csv(sys.argv[1])
    updated_csv = pd.DataFrame(columns=FRASER_FEATURES)
    for idx in language_csv.index:
        updated_csv = updated_csv.append({
            TRANSCRIPT_ID: language_csv[TRANSCRIPT_ID][idx],
            NUMBER_OF_PAUSES: language_csv[COOKIE_NUMBER_OF_PAUSES][idx],
            MAXIMUM_PAUSE_DURATION: language_csv[COOKIE_MAXIMUM_PAUSE_DURATION][idx],
            PAUSE_RATE: language_csv[COOKIE_PAUSE_RATE][idx],
            DURATION: language_csv[COOKIE_DURATION][idx],
            PHONATION_TIME: language_csv[COOKIE_PHONATION_TIME][idx],
            PROPORTION_OF_TIME_SPENT_SPEAKING: language_csv[COOKIE_PROPORTION_OF_TIME_SPENT_SPEAKING][idx],
            MEAN_PAUSE_LENGTH: language_csv[COOKIE_MEAN_PAUSE_LENGTH][idx],
            STD_PAUSE_LENGTH: language_csv[COOKIE_STD_PAUSE_LENGTH][idx],
            SPEECH_RATE: language_csv[COOKIE_SPEECH_RATE][idx],
            AVERAGE_SYLLABLE_DURATION: language_csv[COOKIE_AVERAGE_SYLLABLE_DURATION][idx],
            PAUSE_PER_SYLLABLE: language_csv[COOKIE_PAUSE_PER_SYLLABLE][idx],
            SYLLABLE_COUNT: language_csv[COOKIE_SYLLABLE_COUNT][idx],
            TASK: COOKIE_THEFT_TASK
        }, ignore_index=True)

        updated_csv = updated_csv.append({
            TRANSCRIPT_ID: language_csv[TRANSCRIPT_ID][idx],
            NUMBER_OF_PAUSES: language_csv[READING_NUMBER_OF_PAUSES][idx],
            MAXIMUM_PAUSE_DURATION: language_csv[READING_MAXIMUM_PAUSE_DURATION][idx],
            PAUSE_RATE: language_csv[READING_PAUSE_RATE][idx],
            DURATION: language_csv[READING_DURATION][idx],
            PHONATION_TIME: language_csv[READING_PHONATION_TIME][idx],
            PROPORTION_OF_TIME_SPENT_SPEAKING: language_csv[READING_PROPORTION_OF_TIME_SPENT_SPEAKING][idx],
            MEAN_PAUSE_LENGTH: language_csv[READING_MEAN_PAUSE_LENGTH][idx],
            STD_PAUSE_LENGTH: language_csv[READING_STD_PAUSE_LENGTH][idx],
            SPEECH_RATE: language_csv[READING_SPEECH_RATE][idx],
            AVERAGE_SYLLABLE_DURATION: language_csv[READING_AVERAGE_SYLLABLE_DURATION][idx],
            PAUSE_PER_SYLLABLE: language_csv[READING_PAUSE_PER_SYLLABLE][idx],
            SYLLABLE_COUNT: language_csv[READING_SYLLABLE_COUNT][idx],
            TASK: READING_TASK
        }, ignore_index=True)

        updated_csv = updated_csv.append({
            TRANSCRIPT_ID: language_csv[TRANSCRIPT_ID][idx],
            NUMBER_OF_PAUSES: language_csv[MEMORY_NUMBER_OF_PAUSES][idx],
            MAXIMUM_PAUSE_DURATION: language_csv[MEMORY_MAXIMUM_PAUSE_DURATION][idx],
            PAUSE_RATE: language_csv[MEMORY_PAUSE_RATE][idx],
            DURATION: language_csv[MEMORY_DURATION][idx],
            PHONATION_TIME: language_csv[MEMORY_PHONATION_TIME][idx],
            PROPORTION_OF_TIME_SPENT_SPEAKING: language_csv[MEMORY_PROPORTION_OF_TIME_SPENT_SPEAKING][idx],
            MEAN_PAUSE_LENGTH: language_csv[MEMORY_MEAN_PAUSE_LENGTH][idx],
            STD_PAUSE_LENGTH: language_csv[MEMORY_STD_PAUSE_LENGTH][idx],
            SPEECH_RATE: language_csv[MEMORY_SPEECH_RATE][idx],
            AVERAGE_SYLLABLE_DURATION: language_csv[MEMORY_AVERAGE_SYLLABLE_DURATION][idx],
            PAUSE_PER_SYLLABLE: language_csv[MEMORY_PAUSE_PER_SYLLABLE][idx],
            SYLLABLE_COUNT: language_csv[MEMORY_SYLLABLE_COUNT][idx],
            TASK: MEMORY_TASK
        }, ignore_index=True)

    updated_csv.to_csv('jan27_language_features.csv', header=True)





def main():
    # csv_to_txt()
    # extract_syllable_features_from_txt()
    # combine_syllable_csv()
    # extract_pause_features()
    # combine_pause_syllable()
    create_logistic_regression(True)
    #update_language_csv()


if __name__ == "__main__":
    main()
