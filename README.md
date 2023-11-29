# ACA-Harmonizer-Project

# Chordinator
Chord Prediction Project
Fall 2023 - Audio Content Analysis - Music Technology, Georgia Tech

## Project Goal

Use machine learning to create a tool that predicts a chord progression given an input melody. <br>

**Input:** a melody (raw audio, any instrument) [Q file type requirement or fs requirement?] <br>
**Output:** suggested chords (a 2D array of chords and their associated time stamps) [Q what format will our output be?] <br>

**Note:** 2D array output consists of predicted chord and the associated time stamp <br>

For example, in the following 2D array, the song begins with C major, changes to F major at second 5, to G major at second 15, and back to C major at second 20.  
[CM FM GM  CM ]<br>
[0  5  15  20]<br>

## Set Up

## Chord Detection - Model Details
(used for generating data to augment the [MusDB](https://github.com/sigsep/sigsep-mus-db) dataset in order to train Chord Prediction Model) <br>

### Training Details (Chord Detection)

**During training:** <br>
**Input:** hand made audio clips of chords in [Q] 12 instruments in all keys (see this file [Q]) made using Logic <br>
**Ground Truth:** labeled chord <br>

**After training:** <br>
**Input:** raw audio from [MusDB](https://github.com/sigsep/sigsep-mus-db) Dataset <br>
**Output:** detected chords and associated time stamps

**Note:** The detected chords supplement the [MusDB](https://github.com/sigsep/sigsep-mus-db) dataset with labeled chords in order to train the chord prediction model (chordinator).

## Chordinator (Chord Prediction) - Model Details

### Templates

1. Major as M 0
2. minor as m 1
3. diminished as dim or 2
4. augmented as aug or 3

### Training Details (Chord Prediction)

**During training:** <br>
**Input:** melody (vocals) stem from [MusDB](https://github.com/sigsep/sigsep-mus-db) dataset <br>
**Ground Truth:** labeled chords (found using chord prediction model, see above) <br>

**After training:** <br>
**Input:** a melody (raw audio, any instrument) <br>
**Output:** suggested (predicted) chords and associated time stamps <br>
