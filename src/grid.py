import numpy as np

SCREEN_WIDTH = 1280

distances = {
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

def compute_stretched_fretboard():
	factor = SCREEN_WIDTH/(list(distances.values())[-1])
	divided_distances = factor * np.array(list(distances.values()))
	frets = np.array(list(distances.keys()))
	stretched = {k:v for k,v in zip(frets, divided_distances)}

	return stretched

