import multiprocessing as mp

import numpy as np
import scipy.stats as st

from .response import ResponseSpace


class Model:
    """
    Class that generates a model by mapping its responses to textons.
    """
    def __init__(self, histogram, label):
        self.histogram = histogram
        self.label = label

    @classmethod
    def fromimage(cls, image, textons):
        """
        Build the model directly from an image without instantiating an object
        first. Input: Image instance.
        """

        responses = image.pixel_responses()

        # For each pixel response, find the nearest texton:
        dists = np.zeros((len(responses), len(textons)))
        for i,response in enumerate(responses):
            dists[i] = np.linalg.norm(textons - response, axis=1)

        nearest = np.argmin(dists, axis=1)

        # Create a histogram of the frequencies of the nearest textons. This
        # will be the model.
        hist = np.bincount(nearest, minlength=np.shape(textons)[0])

        return cls(hist, image.label)

    def distance(self, model):
        """
        Calculate the chi square distance between this model and another.
        """
        dist, _ = st.chisquare(self.histogram + 1, model.histogram + 1)
        return dist


class TextonDictionary:
    """
    An implementation of a texton dictionary. The dictionary can be built
    from and trained on images.
    """
    def __init__(self):
        self.textons = np.array([])
        self.models = []


    def add_textons(self, textons):
        """
        Add textons. Probably only useful within the class.
        """
        if type(textons) in (list,np.ndarray):
            if len(self.textons) == 0:
                self.textons = textons
            else:
                self.textons = np.append(self.textons, textons, axis=0)
        else:
            raise TypeError("Input must be a list or numpy.ndarray.")


    def build(self, list_of_images, nclusters=10):
        """
        Builds the texton dictionary by clustering responses of the same
        texture. Takes a list of lists of Image instances. Each sublist
        should contain images of the same texture, and each image is assumed
        to have already been filtered.
        """

        if type(list_of_images) not in (list, np.ndarray):
            raise TypeError("Input must be a list of lists of Image instances.")

        for el in list_of_images:
            if type(el) not in (list, np.ndarray):
                raise TypeError("Input must be a list of lists of Image instances.")
            for subel in el:
                if type(subel).__name__ != 'Image':
                    raise TypeError("Input must be a list of lists of Image instances.")

        self.nclusters = nclusters

        # Set up the multiprocessing pool:
        pool = mp.Pool()

        textons = []
        for i,texture_class in enumerate(list_of_images):
            # Do asynchronous parallel clustering.
            textons.append(pool.apply_async(self._cluster_responses,
                                            [texture_class]))

        pool.close()
        pool.join()

        for t in textons:
            self.add_textons(t.get())


    def _cluster_responses(self, images):
        """
        Convenience method to allow parallel clustering, since a bug in
        scikit-learn prevents the built-in parallel support being used.
        """
        rs = ResponseSpace()
        rs.add_responses(images)
        rs.cluster(nclusters=self.nclusters)
        return rs.clusters



    def train(self, list_of_images):
        """
        Train the texton dictionary by mapping textures to textons.
        Input: list of Image instances.
        """

        pool = mp.Pool()

        models = []
        for image in list_of_images:
            # Create models of the textures by mapping each image to a
            # texton distribution:
            models.append(pool.apply_async(Model.fromimage,
                                           [image, self.textons]))

        pool.close()
        pool.join()

        for model in models:
            self.models.append(model.get())


    def predict(self, image):
        """
        Predict the class of an image by finding the nearest model in the
        texton distribution space. Input: Image instance.
        """

        # Generate the model of the new image:
        model = Model.fromimage(image, self.textons)

        # Find the nearest learned model:
        dists = [model.distance(m) for m in self.models]
        nearest = np.argmin(dists)

        # Return the label of the nearest model:
        label = self.models[nearest].label

        return label
