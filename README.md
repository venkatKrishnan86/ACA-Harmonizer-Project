# ACA-Harmonizer-Project

# Chordinator
Chord Prediction Project
Fall 2023 - Audio Content Analysis - Music Technology, Georgia Tech

## Project Goal

Use machine learning to create a tool that predicts a chord progression given an input melody. <br>

**Input:** a melody (raw audio, any instrument) [Q file type requirement or fs requirement?] <br>
**Output:** suggested chords (a 2D array of chords and their associated time stamps) [Q what format will our output be?] <br>

**Note:** 2D array output consists of predicted chord and the associated time stamp <br>

For example, in the following 2D array, the song begins with C, changes to F at second, to G at second 15, and back to C at second 20.  
[C F G  C ]<br>
[0 5 15 20]<br>

## Set Up

## Chord Detection Model
(used for generating training data for Chord Prediction Model) <br>

**During training:** <br>
**Input:** hand made audio clips of chords in [Q] 12 instruments in all keys (see this file [Q]) made using Logic <br>
**Ground Truth:** labeled chord <br>

**After training:** <br>
**Input:** raw audio from MusDB Dataset [Q add link]
**Output:** detected chords and associated time stamps

**Note:** The detected chords supplement the MusDB dataset with labeled chords in order to train the chord prediction model (chordinator).

## Chordinator (Chord Prediction Model)

**During training:** <br>
**Input:** melody (vocals) stem from MusDB dataset <br>
**Ground Truth:** labeled chords (found using chord prediction model, see above) <br>

**After training:** <br>
**Input:** a melody (raw audio, any instrument) <br>
**Output:** suggested (predicted) chords and associated time stamps <br>
