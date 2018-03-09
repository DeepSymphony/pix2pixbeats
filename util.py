'''
Includes Basic utilities for processing and handling midi data
'''

import numpy as np
import pretty_midi
import librosa
import os
import fnmatch
import midi2pianoroll
import matplotlib.pyplot as plt

from numpy import array

def write_piano_roll_to_midi(piano_roll, filename, program_num=0, is_drum=False, velocity=100, tempo=120.0, beat_resolution=24):

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Create an Instrument object
    instrument = pretty_midi.Instrument(program=program_num, is_drum=is_drum)
    # Set the piano roll to the Instrument object
    set_piano_roll_to_instrument(piano_roll, instrument, velocity, tempo, beat_resolution)
    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)
    # Write out the MIDI data
    midi.write(filename)


def set_piano_roll_to_instrument(piano_roll, instrument, velocity=100, tempo=120.0, beat_resolution=24):
    # Calculate time per pixel
    tpp = 60.0/tempo/float(beat_resolution)
    
    # Create piano_roll_search that captures note onsets and offsets
    piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
    
    piano_roll_diff = np.concatenate((np.zeros((1,10),dtype=int), piano_roll, np.zeros((1,10),dtype=int)))  
    piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)

    # Iterate through all possible(128) pitches
    for note_num in range(10):
        # Search for notes
        start_idx = (piano_roll_search[:,note_num] > 0).nonzero()
        start_time = tpp*(start_idx[0].astype(float))
        end_idx = (piano_roll_search[:,note_num] < 0).nonzero()
        end_time = tpp*(end_idx[0].astype(float))
        # Iterate through all the searched notes
        for idx in range(len(start_time)):
            # Create an Note object with corresponding note number, start time and end time
            note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx], end=end_time[idx])
            # Add the note to the Instrument object
            instrument.notes.append(note)
    # Sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)


#helper function for future for help with loading in midi files and converting to numpy arrays
def load_midi(filepath):
    midi_dict = {}
    for root, dirnames, filenames in os.walk(filepath):
        for filename in fnmatch.filter(filenames, '*.mid'):
            try:
                midi_dict[root].append(root+'/'+filename)
            except:
                midi_dict[root] = root+'/'+filename
    return midi_dict



def process_midis(filepath):
    '''
    Function used to extract midi data from drum dataset
    '''
    numpy_directory = './numpy_samples'
    if not os.path.exists(numpy_directory):
        os.makedirs(numpy_directory)
    #establish list to store samples and labels
    beat_list = []
    label_count = 0
    labels = []

    #walk through directories with midis and pull midi info from every bar
    for root, dirs, files in os.walk(filepath):
        for directory in dirs:
            for root2, dirs2, files2 in os.walk(root+"/"+directory):
                for file in files2:
                    #in case extraneous files exists
                    if(file.endswith("mid")):
                        beat = np.array(midi2pianoroll.midi_to_pianorolls(root2+"/"+file)[0][0])
                        
                        # with current sample rate 96 ticks equals one bar
                        if(beat.shape[0] % 96 != 0):
                            cutoff = beat.shape[0] % 96
                            beat = beat[:-cutoff]
                        
                        #bars happen every 96 divisions
                        for bar in range(beat.shape[0]/96):
                            current_loc = 96*bar
                            beat_list.append(beat[current_loc:current_loc+96])
                            labels.append(label_count)
                label_count += 1
    
    #combine accumalated beats
    all_beats = np.array(beat_list)

    #save compressed form
    # np.save(numpy_directory+'/train_x',all_beats)
    
    labels_np = array(labels)
    np.savez_compressed(numpy_directory+'/training_data',all_beats = all_beats, labels = labels_np)
    # np.save(numpy_directory+'/'+'labels',labels_np)

    #track the distribution of note frequency
    note_data = np.zeros((128,), dtype=int)
    #track min and max values
    maximum = 0
    minimum = 1000

    #iterate through all samples
    for i in range(all_beats.shape[0]):
        all_beats[i] = all_beats[i][:96]
        #iterate through all timesteps
        for j in range(all_beats[i].shape[0]):
            #iterate through all note pitches
            for k in range(all_beats[i][j].shape[0]):
                #if note is activated
                if(all_beats[i][j][k] > 0):
                    note_data[k] += 1
                    if(k > maximum):
                        maximum = k
                    if(k < minimum):
                        minimum = k
    #x axis for graph
    indexes = np.arange(128)

    #plot grap
    plt.bar(indexes, note_data, align = 'center')
    plt.ylabel('# of occurences')
    plt.savefig('./drum_histo.png')




                            
