import multiprocessing as mp
import os
import time
import pickle as pickle

import numpy as np

import ipcv.filters as f
import ipcv.textons as t
from ipcv.image import Image


# ------------------------------ Definitions ------------------------------

np.random.seed(42)

# Paths
# To test, download the CURET database from
# http://www1.cs.columbia.edu/CAVE/software/curet/index.php
DATAPATH = "../data/curet/data/"
OUTPUTPATH = "ipcv/output/"


# Constants
N_TEXTURES_FOR_BUILDING = 20  # No. of textures for building
N_TEXTURES = 20               # No. of textures for training and testing
N_BUILD = 13                  # No. of images per texture for builing
N_TRAINING = 46               # No. of images per texture for training
N_TESTING = 46                # No. of images per texture for testing


# Switches
GENERATE_RESPONSES = False
BUILD_DICTIONARY = True
TRAIN_DICTIONARY = True
MAKE_PREDICTIONS = True


# File names:
RESPONSE_NAME = OUTPUTPATH + "responses/{}.pkl"
BUILT_DICT_NAME = OUTPUTPATH + 'texton_dict_{:0>2}-{:0>2}.pkl'.format(0, N_TEXTURES_FOR_BUILDING)
TRAINED_DICT_NAME = OUTPUTPATH + 'texton_dict_trained_{:0>2}-{:0>2}.pkl'.format(0,N_TEXTURES)


# Define which texture classes should be used to build the dictionary:
BUILD_TEXTURES = np.array([1, 4, 6, 10, 12, 14, 16, 18, 20, 22, 25, 27, 30, 33, 35, 41, 45, 48, 50, 59])

# Define which texture classes should be used to train and test the dictionary:
if N_TEXTURES == 20:
    TRAIN_TEXTURES = np.array([2, 3, 5,  7,  8,  9, 15, 17, 19, 21, 24, 36, 37, 39, 43, 44, 47, 52, 54, 58])
elif N_TEXTURES == 40:
    TRAIN_TEXTURES = np.array([1, 4, 6, 10, 12, 14, 16, 18, 20, 22, 25, 27, 30, 33, 35, 41, 45, 48, 50, 59,
                               3, 5, 8, 11, 13, 15, 17, 19, 21, 24, 26, 28, 32, 34, 37, 43, 47, 49, 52, 60])



# ------------------------------ Functions ------------------------------

def get_list_of_images(directory, remove_bad_images=True):

    path, d = os.path.split(directory)

    # Get description of each image to find they view angles:
    try:
        desc = np.loadtxt("{}/{}.meas".format(directory, d))
    except FileNotFoundError:
        desc = np.loadtxt("{}/{}.avg".format(directory, d))

    # Only select images with view angles less than 60 degrees.
    idx = np.where(desc[:,2] < 60 * np.pi / 180)[0]

    names = []
    # "Bad images" as mentioned in Varma and Zisserman (2005). Note, however,
    # that they never specify exactly which images, they think are bad.
    bad_images = [[11,24,28,30,34,37,41,43,44,45],
                  [14,24,28,34,37,50], # 50/62
                  [14,24,28,37,54],
                  [14,28,39,49],
                  [26,35],
                  []]

    # Store only the good images:
    for i in idx:
        if desc[i,1] in bad_images[int(desc[i,0] - 1)] and remove_bad_images:
            pass
        else:
            names.append("{}/{}-{:0>2d}-{:0>2d}.bmp".format(
                         directory, d[-2:], int(desc[i,0]), int(desc[i,1])
                         ))
    return names



if __name__ == '__main__':
    print("=== Testing Varma-Zisserman method ===")

    # Create filter bank:
    mr8 = f.FilterBank()

    if GENERATE_RESPONSES:
        TEXTURES = np.append(BUILD_TEXTURES, TRAIN_TEXTURES)

        for texture_class in TEXTURES:
            print("== Generating responses for class {}.".format(texture_class))

            imnames = get_list_of_images("{}sample{:0>2}".format(DATAPATH, texture_class),
                                         remove_bad_images=True)

            for i in range(len(imnames)):
                impath, imname = imnames[i].rsplit('/', maxsplit=1)

                try:
                    image = Image.fromfile("{}/{}".format(impath, imname),
                                           label=texture_class)
                except OSError:
                    print("{} can't be loaded. Bad image!".format(imnames[i]))
                else:
                    # Preprocess image:
                    image.crop()
                    image.normalise()

                    a = time.time()

                    # Filter image:
                    mr8.apply(image)

                    b = time.time()
                    print("  --Generating responses for {} ({}/{}):"
                          " {:2.5f} s.".format(imname, i+1, len(imnames), b-a))

                    # Save the filtered image, which now contains the responses:
                    with open('{}responses/{}.pkl'.format(OUTPUTPATH, imname[:-4]), 'wb') as fobj:
                        pickle.dump(image, fobj, pickle.HIGHEST_PROTOCOL)


    # ----------------------------------------------------------------------

    if BUILD_DICTIONARY:
        print("Building dictionary...")

        # Instantiate texton dictionary:
        texton_dict = t.TextonDictionary()

        # Store images:
        images = []

        for texture_class in BUILD_TEXTURES:
            imnames = get_list_of_images("{}sample{:0>2}".format(DATAPATH, texture_class))
            np.random.shuffle(imnames)

            class_images = []
            for i in range(N_BUILD):
                impath, imname = os.path.split(imnames[i])
                image = np.load(RESPONSE_NAME.format(imname[:-4]))
                class_images.append(image)
            images.append(class_images)

        # Build the dictionary from the texture images:
        texton_dict.build(images)

        print("Done building texture dictionary.")

        # Save the newly built texton dictionary:
        with open(BUILT_DICT_NAME, 'wb') as fobj:
            pickle.dump(texton_dict, fobj, pickle.HIGHEST_PROTOCOL)
    else:
        # Load the built texton dictionary:
        with open(BUILT_DICT_NAME, 'rb') as fobj:
            texton_dict = pickle.load(fobj)



    # ----------------------------------------------------------------------

    # Intermezzo: fetch image file names now and store them so we can avoid
    # training and testing on the same image.
    image_names = {}
    for t in TRAIN_TEXTURES:
        filenames = get_list_of_images("{}sample{:0>2}".format(DATAPATH,t))
        np.random.shuffle(filenames)

        image_names[t] = [filenames[:N_TRAINING],
                          filenames[N_TRAINING:N_TRAINING + N_TESTING]]


    # ----------------------------------------------------------------------

    if TRAIN_DICTIONARY:
        print("Training dictionary...")

        for t,texture_class in enumerate(TRAIN_TEXTURES):
            start_time = time.time()
            images = []

            imnames = image_names[texture_class][0]

            for i in range(N_TRAINING):
                impath, imname = os.path.split(imnames[i])
                image = np.load(RESPONSE_NAME.format(imname[:-4]))
                images.append(image)

            # Train the dictionary:
            texton_dict.train(images)

            end_time = time.time()

            print("Finished training texture class {} ({}/{}):"
                  "\t{:5.3f} s".format(int(texture_class), t+1,
                                       len(TRAIN_TEXTURES), end_time-start_time))

        # Save the trained dictionary:
        with open(TRAINED_DICT_NAME, 'wb') as fobj:
            pickle.dump(texton_dict, fobj, pickle.HIGHEST_PROTOCOL)
    else:
        # Load the trained dictionary:
        with open(TRAINED_DICT_NAME, 'rb') as fobj:
            texton_dict = pickle.load(fobj)



    # ----------------------------------------------------------------------

    if MAKE_PREDICTIONS:
        print("Testing dictionary...")
        acc = np.empty(N_TEXTURES)
        for t,texture_class in enumerate(TRAIN_TEXTURES):
            images = []

            imnames = image_names[texture_class][1]

            err = 0
            predictions = {}
            plabels = []

            # Do the predictions in parallel:
            pool = mp.Pool()

            for i in range(N_TESTING):
                impath, imname = os.path.split(imnames[i])
                image = np.load(RESPONSE_NAME.format(imname[:-4]))

                plabels.append(pool.apply_async(texton_dict.predict, [image]))

            pool.close()
            pool.join()

            for p in plabels:
                # Get result from the pool:
                plabel = p.get()

                # Count our mistakes:
                if plabel - (texture_class):
                    err += 1

                # Also count how often each class was predicted:
                try:
                    predictions[plabel] += 1
                except KeyError:
                    predictions[plabel] = 1


            #
            acc[t] = 1 - err/N_TESTING
            print("\nAccuracy for class {} ({}/{}):"
                  "\t{:%}\n\t".format(int(texture_class), t+1, len(TRAIN_TEXTURES),
                                      1-err/N_TESTING), end='')

            for p, n in predictions.items():
                print("{}: {:%}".format(p, n/N_TESTING), end=', ')
            print('')


        print("\n=== Accuracy for {} texture classes:"
              " {:%} +/- {:%}".format(N_TEXTURES, np.mean(acc), np.std(acc)))
