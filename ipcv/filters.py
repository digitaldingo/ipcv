import abc

import scipy.ndimage as nd
import numpy as np

def gauss(x, y, sigx, sigy):
    """
    Calculate the function value of Gaussian distribution.
    """
    r = np.exp(-((x)**2/(2*sigx**2) + (y)**2/(2*sigy**2)))
    r /= np.sum(r)
    return r


def gauss_x(x, y, sigx, sigy):
    """
    Calculate the function value of the first derivative of a Gaussian
    distribution.
    """
    r =(-x/sigx**2) * gauss(x, y, sigx, sigy)
    return r


def gauss_xx(x, y, sigx, sigy):
    """
    Calculate the function value of the second derivative of a Gaussian
    distribution.
    """
    r = ((x**2 - sigx**2) / sigx**4) * gauss(x, y, sigx, sigy)
    return r


class Filter():
    """
    Generic filter class.
    """

    def __init__(self, sigma, angle=0, accuracy=2.):

        if type(sigma) not in (tuple, list, np.ndarray):
            self.sigma = [sigma, sigma]
        else:
            self.sigma = sigma

        self.angle = angle*np.pi/180

        # Construct filter response for the convolution:
        fsize = np.ceil(np.max(self.sigma) * accuracy)
        if fsize % 2 == 0:
            fsize += 1

        values = np.arange(-fsize,fsize+1)
        x, y = np.meshgrid(values, values)

        u = x*np.cos(self.angle) + y*np.sin(self.angle)
        v = -x*np.sin(self.angle) + y*np.cos(self.angle)

        self.filter = self._filter(v, u, self.sigma[0], self.sigma[1])


    @abc.abstractmethod
    def _filter(self):
        """
        Method implementing the filter. This must be implemented by every
        inheriting class.
        """


    def normalise(self):
        """
        Subtracts the mean and L1 normalises the filter, as done in
        Varma & Zisserman (2005).
        """
        # According to the filter bank script found at
        # http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html
        # they seem to also subtract the mean of the filter. That is, however,
        # not done in the paper.

        self.filter -= np.mean(self.filter)

        self.filter /= np.sum(np.abs(self.filter))


    def apply(self, image, **kwargs):
        """
        Apply the the filter to an image.
        """

        response = nd.convolve(image, self.filter)

        return response


class GaussianFilter(Filter):
    """
    A Gaussian filter.
    """
    def __init__(self, sigma, **kwargs):
        Filter.__init__(self, sigma=sigma, **kwargs)

    def __repr__(self):
        return "GaussianFilter(sigma = {})".format(self.sigma)

    def _filter(self, x, y, sigx, sigy):
        #return nd.convolve(image, self.filter)
        return gauss(x, y, sigx, sigy)

    def normalise(self):
        """
        L1 normalises the filter, as done in Varma & Zisserman (2005).
        """
        # According to the filter bank script found at
        # http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html
        # they seem to also subtract the mean of the filter. That is, however,
        # not done in the paper.

        self.filter -= np.mean(self.filter)

        self.filter /= np.sum(np.abs(self.filter))


class LOGFilter(Filter):
    """
    A Laplacian of Gaussian (LOG) filter.
    """
    def __init__(self, sigma, **kwargs):
        Filter.__init__(self, sigma=sigma, **kwargs)

    def __repr__(self):
        return "LOGFilter(sigma = {})".format(self.sigma)

    def _filter(self, x, y, sigx, sigy):
        return gauss_xx(x, y, sigx, sigy) + gauss_xx(y, x, sigx, sigy)


class EdgeFilter(Filter):
    """
    An edge filter based on an isotropic Gaussian filter.
    """
    def __init__(self, sigma, **kwargs):
        Filter.__init__(self, sigma=sigma, **kwargs)

    def __repr__(self):
        return "EdgeFilter(sigma = {}, angle = {})".format(self.sigma, self.angle)

    def _filter(self, x, y, sigx, sigy):
        return gauss_x(x, y, sigx, sigy)


class BarFilter(Filter):
    """
    A bar filter based on an isotropic Gaussian filter.
    """
    def __init__(self, sigma, **kwargs):
        Filter.__init__(self, sigma=sigma, **kwargs)

    def __repr__(self):
        return "BarFilter(sigma = {}, angle = {})".format(self.sigma, self.angle)

    def _filter(self, x, y, sigx, sigy):
        return gauss_xx(x, y, sigx, sigy)


class StackedFilters:
    """
    Container for storing multiple filters and applying them all to images.
    """
    def __init__(self):
        self.filters = np.array([])
        self.responses = None

    def __repr__(self):
        return ", ".join([repr(f) for f in self.filters])

    def add_filter(self, filter):
        self.filters = np.append(self.filters, filter)

    def apply(self, image, n_jobs=1, **kwargs):
        """
        Apply the filters to an image.
        """

        self.responses = [filter.apply(image) for filter in
                          self.filters]

        return self

    def _get_responses(self):
        """
        Convenience method to recursively get the responses of nested
        StackedFilters.
        """
        for r,res in enumerate(self.responses):
            if type(res).__name__ == "StackedFilters":
                res._get_responses()
            elif type(res).__name__ == "ApplyResult":
                self.responses[r] = res.get()

    @property
    def maximum_response(self):
        """
        Get the maximum response of the contained responses.
        """
        if self.responses is None:
            raise ValueError("No responses detected. "
                             "Remember to call apply() on an image first.")

        return np.max(self.responses, axis=0)


    def normalise(self):
        """
        Normalise all contained filters.
        """
        for filter in self.filters:
            filter.normalise()
