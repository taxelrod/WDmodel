import warnings
import numpy as np
from . import io
import scipy.interpolate as spinterp
from astropy import units as u
from specutils.extinction import reddening
from scipy.ndimage.filters import gaussian_filter1d


class WDmodel(object):
    """
    Base class defines the routines to generate and work with DA White Dwarf
    model spectra. Requires the grid file - TlustyGrids.hdf5, or a custom
    user-specified grid. Look at the package level help for description on the
    grid file. There are various convenience methods that begin with an
    underscore (_) that will  not be imported by default These are intended for
    internal use, and do not have the sanity checking of the public methods.
    """

    def __init__(self, grid_file=None, grid_name=None):
        """
        constructs a white dwarf model atmosphere object
        Virtually none of the attributes should be used directly
        since it is trivially possible to break the model by redefining them
        Access to them is best through the functions connected to the models
        """
        lno     = [   1    ,   2     ,    3     ,    4    ,   5      ,  6      ]
        lines   = ['alpha' , 'beta'  , 'gamma'  , 'delta' , 'zeta'   , 'eta'   ]
        H       = [6562.857, 4861.346, 4340.478 ,4101.745 , 3970.081 , 3889.056]
        D       = [ 130.0  ,  170.0  ,  125.0   ,  75.0   ,   50.0   ,   27.0  ]
        eps     = [  10.0  ,   10.0  ,   10.0   ,   8.0   ,    5.0   ,    3.0  ]
        self._lines = dict(zip(lno, zip(lines, H, D, eps)))
        # we're passing grid_file so we know which model to init
        self._fwhm_to_sigma = np.sqrt(8.*np.log(2.))
        self.__init__tlusty(grid_file=grid_file)


    def __init__tlusty(self, grid_file=None, grid_name=None):
        """
        Initialize the Tlusty Model <grid_name> from the grid file <grid_file>
        """
        self._fluxnorm = 1. #LEGACY CRUFT

        ingrid = io.read_model_grid(grid_file, grid_name)
        self._grid_file, self._grid_name, self._wave, self._ggrid, self._tgrid, self._flux = ingrid

        # pre-init the interpolation and do it in log-space
        # note that we do the interpolation in log-log
        # this is because the profiles are linear, redward of the Balmer break in log-log
        # and the regular grid interpolator is just doing linear interpolation under the hood
        self._model = spinterp.RegularGridInterpolator((self._tgrid, self._ggrid),\
                np.log10(self._flux.T))


    def _get_model(self, teff, logg, wave, log=False):
        """
        Returns the model flux given temperature and logg at wavelengths wave
        """
        xi = (teff, logg)
        mod = self._model(xi)
        out = np.interp(wave, self._wave, mod)
        if log:
            return out
        return (10.**out)


    def _get_red_model(self, teff, logg, av, wave, rv=3.1, log=False, rvmodel='od94'):
        """
        Returns the reddened model flux given teff, logg, av, rv, rvmodel, and
        wavelengths
        """
        mod = self._get_model(teff, logg, wave, log=log)
        bluening = reddening(wave*u.Angstrom, av, r_v=rv, model=rvmodel)
        if log:
            mod = 10.**mod
        mod/=bluening
        if log:
            mod = np.log10(mod)
        return mod


    def _get_obs_model(self, teff, logg, av, fwhm, wave, rv=3.1, log=False, rvmodel='od94', pixel_scale=1.):
        """
        Returns the observed model flux given teff, logg, av, rv, rvmodel, fwhm
        (for Gaussian instrumental broadening) and wavelengths
        """
        mod = self._get_model(teff, logg, wave, log=log)
        bluening = reddening(wave*u.Angstrom, av, r_v=rv, model=rvmodel)
        if log:
            mod = 10.**mod
        mod/=bluening
        gsig = fwhm/self._fwhm_to_sigma * pixel_scale
        mod = gaussian_filter1d(mod, gsig, order=0, mode='nearest')
        if log:
            mod = np.log10(mod)
        return mod


    def _get_full_obs_model(self, teff, logg, av, fwhm, wave, rv=3.1, log=False, rvmodel='od94', pixel_scale=1.):
        """
        Convenience function that does the same thing as _get_obs_model, but
        also returns the full SED without any instrumental broadening applied
        """
        xi = (teff, logg)
        mod = self._model(xi)
        mod = 10.**mod
        bluening = reddening(self._wave*u.Angstrom, av, r_v=rv, model=rvmodel)
        mod/=bluening
        omod = np.interp(wave, self._wave, np.log10(mod))
        omod = 10.**omod
        gsig = fwhm/self._fwhm_to_sigma * pixel_scale
        omod = gaussian_filter1d(omod, gsig, order=0, mode='nearest')
        if log:
            omod = np.log10(omod)
            mod  = np.log10(mod)
        mod = np.rec.fromarrays((self._wave, mod), names='wave,flux')
        return omod, mod


    @classmethod
    def _wave_test(cls, wave):
        """
        Checks if the wavelengths passed are sane
        """
        if len(wave) == 0:
            message = 'Wavelengths not specified.'
            raise ValueError(message)

        if not np.all(wave > 0):
            message = 'Wavelengths are not all positive'
            raise ValueError(message)

        if len(wave) == 1:
            return

        dwave = np.diff(wave)
        if not(np.all(dwave > 0) or np.all(dwave < 0)):
            message = 'Wavelength array is not monotonic'
            raise ValueError(message)


    def get_model(self, teff, logg, wave=None, log=False, strict=True):
        """
        Returns the model (wavelength and flux) for some teff, logg at wavelengths wave
        If not specified, wavelengths are from 3000-9000A

        Checks inputs for consistency and calls _get_model() If you
        need the model repeatedly for slightly different parameters, use those
        functions directly
        """
        if wave is None:
            wave = self._wave

        wave = np.atleast_1d(wave)
        self._wave_test(wave)

        teff = float(teff)
        logg = float(logg)

        if not ((teff >= self._tgrid.min()) and (teff <= self._tgrid.max())):
            message = 'Temperature out of model range'
            if strict:
                raise ValueError(message)
            else:
                warnings.warn(message,RuntimeWarning)
                teff = min([self._tgrid.min(), self._tgrid.max()], key=lambda x:abs(x-teff))

        if not ((logg >= self._ggrid.min()) and (logg <= self._ggrid.max())):
            message = 'Surface gravity out of model range'
            if strict:
                raise ValueError(message)
            else:
                warnings.warn(message,RuntimeWarning)
                logg = min([self._ggrid.min(), self._ggrid.max()], key=lambda x:abs(x-logg))

        outwave = wave[((wave >= self._wave.min()) & (wave <= self._wave.max()))]

        if len(outwave) > 0:
            outflux = self._get_model(teff, logg, wave, log=log)
            return outwave, outflux
        else:
            message = 'No valid wavelengths'
            raise ValueError(message)


    def get_red_model(self, teff, logg, av, rv=3.1, rvmodel='od94', wave=None, log=False, strict=True):
        """
        Returns the model (wavelength and flux) for some teff, logg av, rv with
        reddening law "rvmodel" at wavelengths wave If not specified,
        wavelengths are from 3000-9000A Applies reddening that is specified to
        the spectrum (the model has no reddening by default)

        Checks inputs for consistency and calls get_model() If you
        need the model repeatedly for slightly different parameters, use
        _get_red_model directly.
        """
        modwave, modflux = self.get_model(teff, logg, wave=wave, log=log, strict=strict)
        av = float(av)
        rv = float(rv)
        bluening = reddening(modwave*u.Angstrom, av, r_v=rv, model=rvmodel)
        if log:
            modflux = 10.**modflux
        modflux/=bluening
        if log:
            modflux = np.log10(modflux)
        return modwave, modflux


    def get_obs_model(self, teff, logg, av, fwhm, rv=3.1, rvmodel='od94', wave=None,\
            log=False, strict=True, pixel_scale=1.):
        """
        Returns the model (wavelength and flux) for some teff, logg av, rv with
        reddening law "rvmodel" at wavelengths wave If not specified,
        wavelengths are from 3000-9000A Applies reddening that is specified to
        the spectrum (the model has no reddening by default)

        Checks inputs for consistency and calls get_red_model() If you
        need the model repeatedly for slightly different parameters, use
        _get_obs_model directly
        """
        modwave, modflux = self.get_red_model(teff, logg, av, rv=rv, rvmodel=rvmodel,\
                wave=wave, log=log, strict=strict)
        if log:
            modflux = 10.**modflux
        gsig = fwhm/self._fwhm_to_sigma * pixel_scale
        modflux = gaussian_filter1d(modflux, gsig, order=0, mode='nearest')
        if log:
            modflux = np.log10(modflux)
        return modwave, modflux


    def _normalize_model(self, spec, log=False):
        """
        Imprecise normalization for visualization only
        If you want to normalize model and data, get the model, and the data,
        compute the integrals and use the ratio to properly normalize
        """
        wave = spec.wave
        flux = spec.flux
        ind = np.where((self._wave >= wave.min()) & (self._wave <= wave.max()))[0]
        modelmedian = np.median(self._model.values[:,:, ind])
        if not log:
            modelmedian = 10.**modelmedian
        datamedian  = np.median(flux)
        self._fluxnorm = modelmedian/datamedian
        return self._fluxnorm


    @classmethod
    def _get_indices_in_range(cls, w, WA, WB, W0=None):
        """
        Accepts a wavelength array, and blue and redlimits, and returns the
        indices in the array that are between the limits
        """
        if W0 is None:
            W0 = WA + (WB-WA)/2.
        ZE  = np.where((w >= WA) & (w <= WB))
        return W0, ZE


    def _get_line_indices(self, w, line):
        """
        Returns the central wavelength, and _indices_ of the line profile
        The widths of the profile are predefined in the _lines attribute
        """
        _, W0, WID, DW = self._lines[line]
        WA  = W0 - WID - DW
        WB  = W0 + WID + DW
        return self._get_indices_in_range(w, WA, WB, W0=W0)


    def _extract_from_indices(self, w, f, ZE, df=None):
        """
        Returns the wavelength and flux of the line using the indices ZE which can be determined by _get_line_indices
        Optionally accepts the noise vector to extract as well
        """
        if df is None:
            return w[ZE], f[ZE]
        else:
            return w[ZE], f[ZE], df[ZE]


    def extract_spectral_line(self, w, f, line, df=None):
        """
        extracts a section of a line, fits a straight line to the flux outside of the line core
        to model the continuum, and then divides it out
        accepts the wavelength and the flux, and the line name (alpha, beta, gamma, delta, zeta, eta)
        returns the wavelength and normalized flux of the line
        """
        try:
            _, W0, WID, DW = self._lines[line]
        except KeyError:
            message = 'Line name %s is not valid. Must be one of (%s)' %(str(line), ','.join(self._lines.keys()))
            raise ValueError(message)

        w  = np.atleast_1d(w)
        self._wave_test(w)

        f  = np.atleast_1d(f)
        if w.shape != f.shape:
            message = 'Shape mismatch between wavelength and flux arrays'
            raise ValueError(message)

        if df is not None:
            df = np.atleast_1d(df)
            if w.shape != df.shape:
                message = 'Shape mismatch between wavelength and fluxerr arrays'
                raise ValueError(message)

        return self._extract_spectral_line(w, f, line, df=df)


    def _extract_spectral_line(self, w, f, line, df=None):
        """
        Same as extract_spectral_line() except no testing
        Used internally to extract the spectral line for the model
        """
        W0, ZE = self._get_line_indices(w,  line)
        return self._extract_from_indices(w, f, ZE, df=df)


    # these are implemented for compatibility with python's pickle
    # which in turn is required to make the code work with multiprocessing
    def __getstate__(self):
        return self.__dict__


    def __setstate__(self, d):
        self.__dict__.update(d)


    __call__ = get_model
