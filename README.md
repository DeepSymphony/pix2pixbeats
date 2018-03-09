# pix2pixbeats
To accumalate numpy training samples, run process_midis in util.py with './mididrumfiles.com_.samples' as path
</br>
All data is stored in compressed numpy files. https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez_compressed.html </br>
To load the accumalated training samples, use key 'all_beats' to get numpy array with all of the beats in it. </br>
To load associated labels, use key 'labels'.