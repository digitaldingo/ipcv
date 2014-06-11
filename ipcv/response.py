import numpy as np
from sklearn import cluster as cl


class Response:
    """
    Class for storing the response of a filtered image.
    """

    def __init__(self, filtered_image, filter=None):
        self.org_size = np.shape(filtered_image)
        self.response = np.reshape(filtered_image,
                                   self.org_size[0] * self.org_size[1])

        # Note: response needs to be float64 due to a bug in scikit-learn's
        # K Means clustering.
        self.response = self.response.astype(np.float64)

        # The type of filter that created the response:
        self.filter = filter

    def __repr__(self):
        return "Response(filtered_image, filter={})".format(repr(self.filter))

    def as_image(self):
        "Return the response in the same shape as the original image."
        return np.reshape(self.response, self.org_size)



class ResponseSpace:
    """
    Class for storing the Response instances of Images and do clustering.
    """

    def __init__(self):
        self.responses = np.array([])

    def add_responses(self, image):
        """
        Add responses to the response space. Accepts an Image instance or a
        list of Image instances.
        """

        if type(image).__name__ == "Image":
            # Input is a single image.

            if len(image.pixel_responses()) == 0:
                raise ValueError("Image.responses must be calculated"
                                 " before adding to ResponseSpace.")

            # Fetch pixel responses:
            responses = image.pixel_responses()


        elif type(image) in (list,np.ndarray):
            # Input is a list that has to consist of Image instances.
            for im in image:
                if type(im).__name__ != "Image":
                    raise TypeError("Input list must consist of Image"
                                    " instances, not {}.".format(type(im).__name__))
                if len(im.pixel_responses()) == 0:
                    raise ValueError("Image.responses must be calculated"
                                     " before adding to ResponseSpace.")

            # Fetch pixel responses for all images:
            responses = image[0].pixel_responses()
            for im in image[1:]:
                responses = np.append(responses, im.pixel_responses(), axis=0)

        else:
            raise TypeError("Input must be an Image instance or"
                            " a list of Image instances.")

        # The responses need to be N dimensional points in a list (making
        # the list 2D) for the clustering to work.
        assert responses.ndim == 2, \
               "List of responses is not 2D, but {}D.".format(responses.ndim)

        # Store the new responses:
        if len(self.responses) == 0:
            self.responses = responses
        else:
            self.responses = np.append(self.responses, responses, axis=0)



    def cluster(self, nclusters=10):
        """
        Cluster the response space using K Means clustering.
        """
        clusters = cl.k_means(self.responses, n_clusters=nclusters)
        self.clusters = clusters[0]

        return self.clusters
