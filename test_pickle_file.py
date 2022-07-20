# Sarah Chacko

import matplotlib.pyplot as plt
import pickle

# run with
    # python3 test_pickle_file.py

# Shows the first four images of the pickled dictionary with its corresponding camera position
if __name__ == "__main__":
    unpickled_dict = {}

    with open("training_data/dict1_large.pkl", 'rb') as f:
        unpickled_dict = pickle.load(f)
    print(len(unpickled_dict))
    # show first few images

    for x in range(1, min(5, len(unpickled_dict))):
        plt.imshow(unpickled_dict[x][1])
        plt.text(0, 0, str(unpickled_dict[x][0]))
        plt.show()

    # cut to 50 images for training
    



