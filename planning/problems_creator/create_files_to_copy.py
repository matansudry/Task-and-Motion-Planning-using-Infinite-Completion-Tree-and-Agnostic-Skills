import sys
sys.path.append(".")

from utils.general_utils import load_pickle, load_yaml, save_yaml
import os

if __name__ == "__main__":
    options = load_pickle(path="all_options.pickle")
    folder = "options"
    os.makedirs(folder, exist_ok=True)
    for index, option in enumerate(options):
        file_path = os.path.join(folder, f'{index}.txt')
        f = open(file_path, 'w')
        for key in option:    
            f.write(f'\t\t(on {key} {option[key]})\n')
        f.close()

    """
    		(on cyan_box blue_box)
		(on yellow_box cyan_box)
		(on red_box yellow_box)
		(on blue_box rack)
		(on rack table)
    """