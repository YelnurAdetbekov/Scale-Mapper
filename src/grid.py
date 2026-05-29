import numpy as np

SCREEN_WIDTH = 1280
MAX_FRET = 22

distances = {
	-1: 0,
	0: 0,
	1: 53,
	2: 95,
	3: 134,
	4: 167,
	5: 198, ####
	6: 227.5,####
	7: 253,##
	8: 276, ###
	9: 299,
	10: 320,
	11: 338,
	12: 355,
	13: 372,
	14: 386,
	15: 400,
	16: 412,
	17: 425,
	18: 435.5,##
	19: 446,
	20: 455.5,
	21: 464.5, ## 
	22: 474.794
}

OPEN_POSITIONS = {
	1:4,
	2:9,
	3:2,
	4:7,
	5:11,
	6:4
}

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

MAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23, 24])
MINOR_SCALE = np.array([0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24])

SCALES = {
	"major": MAJOR_SCALE,
	"minor": MINOR_SCALE
}
chosen_scale = ['E', "major"]

def choose_scale(chosen_scale):
	return (NOTES.index(chosen_scale[0]), chosen_scale[1])

c = choose_scale(chosen_scale)

def compute_frets(c):
	scale = SCALES[c[1]]
	tonic_1 = (c[0]+12-OPEN_POSITIONS[1])%12
	tonic_2 = (c[0]+12-OPEN_POSITIONS[2])%12
	tonic_3 = (c[0]+12-OPEN_POSITIONS[3])%12
	tonic_4 = (c[0]+12-OPEN_POSITIONS[4])%12
	tonic_5 = (c[0]+12-OPEN_POSITIONS[5])%12
	tonic_6 = (c[0]+12-OPEN_POSITIONS[6])%12
	frets_1 = (tonic_1+scale)%24
	frets_2 = (tonic_2+scale)%24
	frets_3 = (tonic_3+scale)%24
	frets_4 = (tonic_4+scale)%24
	frets_5 = (tonic_5+scale)%24
	frets_6 = (tonic_6+scale)%24
	frets_1 = np.unique(frets_1[frets_1 < MAX_FRET])
	frets_2 = np.unique(frets_2[frets_2 < MAX_FRET])
	frets_3 = np.unique(frets_3[frets_3 < MAX_FRET])
	frets_4 = np.unique(frets_4[frets_4 < MAX_FRET])
	frets_5 = np.unique(frets_5[frets_5 < MAX_FRET])
	frets_6 = np.unique(frets_6[frets_6 < MAX_FRET])
	return np.array([frets_1, frets_2,frets_3,frets_4,frets_5,frets_6])

def compute_stretched_fretboard():
	factor = SCREEN_WIDTH/(list(distances.values())[-1])
	divided_distances = factor * np.array(list(distances.values()))
	frets = np.array(list(distances.keys()))
	stretched = {k:v for k,v in zip(frets, divided_distances)}

	return stretched



