import abc
import multiprocessing as mp
import multiprocessing.dummy as mpd
import os

from joblib import Parallel, delayed

import scipy.ndimage as nd
import numpy as np

from .response import Response

from IPython import embed

class Filter():
    """
    Generic filter class.
    """

    @abc.abstractmethod
    def _filter(self, image, **kwargs):
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

        #self.filter -= np.mean(self.filter)

        self.filter /= np.sum(np.abs(self.filter))


    def apply(self, image, **kwargs):
        """
        Apply the the filter to an image. Bonus if the image is an Image instance.
        """

        if type(image).__name__ == 'Image':
            res = self._filter(image.image, **kwargs)
            response = Response(res, self.__repr__)
            image.add_response(response)
        else:
            response = self._filter(image, **kwargs)

        return response


class GaussianFilter(Filter):
    """
    A Gaussian filter.
    """
    def __init__(self, sigma, order=0, mode="constant", cval=0):
        self.sigma = sigma
        self.order = order
        self.mode = mode
        self.cval = cval


        fsize = int(np.max(self.sigma) * 5)
        if fsize % 2 == 0:
            fsize += 1
        impulse = np.zeros([fsize,fsize])
        impulse[fsize // 2,fsize // 2] = 1
        self.filter = nd.gaussian_filter(impulse, sigma=self.sigma,
                                         order=self.order, mode=self.mode,
                                         cval=self.cval)


    def __repr__(self):
        return "GaussianFilter(sigma = {})".format(self.sigma)

    def _filter(self, image, **kwargs):
        return nd.convolve(image, self.filter)



class LOGFilter(GaussianFilter):
    """
    A Laplacian of Gaussian (LOG) filter.
    """
    def __init__(self, sigma, mode="constant", cval=0):
        GaussianFilter.__init__(self, sigma=sigma, order=2, mode=mode,
                                cval=cval)


    def __repr__(self):
        return "LOGFilter(sigma = {})".format(self.sigma)



class AnisotropicGaussianFilter(Filter):
    """
    An isotropic Gaussian filter.
    """
    def __init__(self, sigma, order, angle=0, mode="mirror", cval=0, factor=4.8):
        self.sigma = sigma
        self.order = order
        self.angle = angle
        self.mode = mode
        self.cval = cval

        fsize = int(np.max(self.sigma) * 5)
        if fsize % 2 == 0:
            fsize += 1
        impulse = np.zeros([fsize,fsize])
        impulse[fsize // 2,fsize // 2] = 1
        f = nd.gaussian_filter(impulse, sigma=self.sigma, order=self.order,
                               mode=self.mode, cval=self.cval)
        self.filter = nd.interpolation.rotate(f, self.angle, reshape=False)

    def __repr__(self):
        return "AnisotropicGaussianFilter(sigma = {}, order = {}, angle = {})".format(self.sigma)

    def _filter(self, image, fft_image=False):
        return nd.convolve(image, self.filter)



class EdgeFilter(AnisotropicGaussianFilter):
    """
    An edge filter based on an isotropic Gaussian filter.
    """
    def __init__(self, sigma, order=(1,0), angle=0, **kwargs):
        AnisotropicGaussianFilter.__init__(self, sigma=sigma, order=order,
                                           angle=angle, **kwargs)

    def __repr__(self):
        return "EdgeFilter(sigma = {}, angle = {})".format(self.sigma, self.angle)



class BarFilter(AnisotropicGaussianFilter):
    """
    A bar filter based on an isotropic Gaussian filter.
    """
    def __init__(self, sigma, order=(2,0), angle=0, **kwargs):
        AnisotropicGaussianFilter.__init__(self, sigma=sigma, order=order,
                                           angle=angle, **kwargs)

    def __repr__(self):
        return "BarFilter(sigma = {}, angle = {})".format(self.sigma, self.angle)

def bla(f, x):
    return f.apply(x)


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
        #self.responses = []

        #verbosity = int(os.environ['VERBOSITY'])
        # Seems to work! Perhaps try to do a plot...
        #self.responses = Parallel(n_jobs=n_jobs)(delayed(bla)(filter, image) for
        #                                         filter in self.filters)

        self.responses = [bla(filter, image) for filter in self.filters]

        #embed()
        #stop

        #pool = mp.Pool(None)
        #dpool = mpd.Pool(None)

        #for f,filter in enumerate(self.filters):
        #    embed()
        #    # FIXME: Only reasonable thing to do: use joblib NOW!
        #    if type(filter).__name__ == "StackedFilters":
        #        # Apply the StackedFilters in a separate thread to avoid
        #        # blocking the current one.
        #        self.responses.append(dpool.apply_async(filter.apply, [image]))
        #    else:
        #        self.responses.append(pool.apply_async(filter.apply, [image]))

        ## Close at join multiprocessing pool:
        #pool.close()
        #pool.join()

        ## Close at join threading pool:
        #dpool.close()
        #dpool.join()

        ## Get the results:
        #self._get_responses()

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
                embed()
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





class FilterBank:
    """
    Container class for filters.
    """

    def __init__(self, bank_type='MR8', **kwargs):
        self.bank = StackedFilters()
        self.bank_type = bank_type

        for key in kwargs.keys():
            setattr(key, kwargs[key])

        if bank_type == 'MR8':
            # Define the filter bank according to Varma & Zisserman (2005).

            if 'sigma' not in kwargs.keys():
                self.sigma = 10
            if 'scales' not in kwargs.keys():
                self.scales = ((1,3),(2,6),(4,12))
            if 'angles' not in kwargs.keys():
                self.angles = np.linspace(0, 180, 6, endpoint=False)


            self.add(GaussianFilter(self.sigma))
            self.add(LOGFilter(self.sigma))

            for scale in self.scales:
                sf = StackedFilters()
                for angle in self.angles:
                    sf.add_filter(EdgeFilter(scale, angle=angle))
                self.add(sf)

            for scale in self.scales:
                sf = StackedFilters()
                for angle in self.angles:
                    sf.add_filter(BarFilter(scale, angle=angle))
                self.add(sf)

            # Normalise all filters:
            self.bank.normalise()


    def __repr__(self):
        return "\n".join([repr(f) for f in self.bank])


    def add(self, filter):
        """
        Add filter to the filter bank.
        """
        self.bank.add_filter(filter)


    def apply(self, im):
        """
        Apply the filter bank to an image.
        """
        if type(im).__name__ == "Image":
            image = im.image
        else:
            image = im

        responses = self.bank.apply(image).responses


        if self.bank_type == "MR8":
            for r,res in enumerate(responses):
                if type(res).__name__ == "StackedFilters":
                    responses[r] = res.maximum_response

            # Convert to numpy array:
            responses = np.asarray(responses)

            # Magic contrast normalisation:
            L2 = np.sqrt(np.sum(responses**2, axis=0))
            responses *= np.log(1 + L2/0.03)/L2


        if type(im).__name__ == "Image":
            for r,res in enumerate(responses):
                response = Response(res, repr(self.bank.filters[r]))
                im.add_response(response)

        return responses

