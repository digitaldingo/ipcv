import numpy as np
import scipy.ndimage as nd
import sklearn as skl


class Image:
    """
    Class to store an image and information about it.
    """
    def __init__(self, image, label=0, path=""):
        self.image = image
        self.responses = np.array([])
        self.label = label
        self.path = path


    @classmethod
    def fromfile(cls, path, flatten=True, **kwargs):
        """
        Load an image without instantiating an object first.
        """
        image = nd.imread(path, flatten=flatten)
        return cls(image, path=path, **kwargs)


    def add_response(self, response):
        """
        Add responses. Input must a Response instance.
        """
        if type(response).__name__ != "Response":
            raise TypeError("Input must be a Response instance.")
        self.responses = np.append(self.responses, response)


    def pixel_responses(self):
        """
        Return the N dimensional pixel responses for the image. For an image
        consisting of M pixels, the output will be of shape (N,M).
        """
        return np.transpose([r.response for r in self.responses])


    def crop(self, size=(200,200)):
        """
        Crop to region around center.
        """
        x = size[0] // 2
        y = size[1] // 2

        cx,cy = np.asarray(np.shape(self.image)) // 2

        self.image = self.image[cx-x:cx+x, cy-y:cy+y]


    def normalise(self):
        """
        Scale the image to zero mean, unit variance.
        """
        self.image = skl.preprocessing.scale(self.image)


    def filter(self, filter_bank):
        # TODO: Would this be a convenient function to have?
        pass
