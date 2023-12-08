# ACA-Harmonizer-Project

# Chordinator
Chord Prediction Project
Fall 2023 - Audio Content Analysis - Music Technology, Georgia Tech

[Chordinator Web App](https://github.com/venkatKrishnan86/ACA-Harmonizer-Project)(TBA)

See [documentation](https://github.com/venkatKrishnan86/ACA-Harmonizer-Project/tree/main/documentation) folder for paper write up. 
![alt text](https://github.com/venkatKrishnan86/ACA-Harmonizer-Project/blob/main/documentation/diagram.jpg?raw=true)

## Project Goal

Train a machine learning model to predict an accompaniment chord progression given an input melody. <br>

**Input:** a melody (raw audio, any instrument) [Q file type requirement or fs requirement?] <br>
**Output:** suggested chords (a 2D array of chords and their associated time stamps) <br>

**Note:** 2D array output consists of predicted chord and the associated time stamp <br>

For example, in the following 2D array, the song begins with C major, changes to F major at second 5, to G major at second 15, and back to C major at second 20.  
[CM FM GM  CM ]<br>
[0  5  15  20]<br>

## Set Up
Models can be found in [models folder](https://github.com/venkatKrishnan86/ACA-Harmonizer-Project/tree/main/models) for the chord detection and chord prediction (Chordinator) models.

### Templates (Chord Prediction)

1. Major as *M* 0
2. minor as *m* 1
3. diminished as *dim* or 2
4. augmented as *aug* or 3

## Chord Detection - Model Details
(used for generating data to augment the [MusDB](https://github.com/sigsep/sigsep-mus-db) dataset in order to train Chord Prediction Model) <br>

### Evaluation (Chord Detection)

#### Accuracy Metric

98.4% Reported Accuracy 

### Training Details (Chord Detection)

**During training:** <br>
**Input:** hand made audio clips of chords in 15 instruments in all 12 keys for 4 chord qualities (see ./data folder) made using [Logic](https://www.apple.com/logic-pro/) a total of 15x12x4 = 720 clips <br>
**Ground Truth:** labeled chord <br>

**After training:** <br>
**Input:** raw audio from [MusDB](https://github.com/sigsep/sigsep-mus-db) Dataset <br>
**Output:** detected chords and associated time stamps

**Note:** The detected chords supplement the [MusDB](https://github.com/sigsep/sigsep-mus-db) dataset with labeled chords in order to train the chord prediction model (chordinator).

## Chordinator (Chord Prediction) - Model Details

To predict chord accompaniment for the input melody, we use a cossine similarity metric between the input chromagram and our defined chord templates.[Q]

### Evaluation (Chord Prediction)

#### Accuracy Metric

66.04% Reported Accuracy 

#### Human Evaluation

See paper write up for more details. [documentation folder](https://github.com/venkatKrishnan86/ACA-Harmonizer-Project/tree/main/documentation)

### Training Details (Chord Prediction)

**During training:** <br>
**Input:** melody (vocals) stem from [MusDB](https://github.com/sigsep/sigsep-mus-db) dataset <br>
**Ground Truth:** labeled chords (found using chord prediction model, see above) <br>

**After training:** <br>
**Input:** a melody (raw audio, any instrument) <br>
**Output:** suggested (predicted) chords and associated time stamps <br>

## Cleaning Algorithm
The output from our models predicts a chord every 6 frames or about every ~.25 seconds. Occasionally there are small deviations in long sequences of the same chord, which we attribute to error. In order to combat this noisy output, we devised a cleaning algorithm to produce clean output. See paper write up for more details. [documentation folder](https://github.com/venkatKrishnan86/ACA-Harmonizer-Project/tree/main/documentation)
