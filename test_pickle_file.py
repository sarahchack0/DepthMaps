# Sarah Chacko

import matplotlib.pyplot as plt
import pickle

# run with
    # python3 test_pickle_file.py

# Shows the first four images of the pickled dictionary with its corresponding camera position
if __name__ == "__main__":
    unpickled_dict = {}

    with open("training_data/test_dict_normal.pkl", 'rb') as f:
        unpickled_dict = pickle.load(f)
    print(len(unpickled_dict))
    # show first few images

    for x in range(0, min(10, len(unpickled_dict))):
        print("lat, ud: ", unpickled_dict[x][2], unpickled_dict[x][3])
        plt.imshow(unpickled_dict[x][1])
        plt.text(0, 0, str(unpickled_dict[x][0]))
        plt.show()



    # cut to 50 images for training
    



