import sys
import os
from emcee.utils import MPIPool
import argparse
import warnings
from copy import deepcopy
import numpy as np
import pkg_resources
from collections import OrderedDict
from matplotlib.mlab import rec2txt
import json
import h5py

# Declare this tuple to init the likelihood model, and to preserve order of parameters
_PARAMETER_NAMES = ("teff", "logg", "av", "rv", "dl", "fwhm", "fsig", "tau", "fw", "mu")


def get_options(args, comm):
    """
    Get command line options for the WDmodel fitter
    """

    # create a config parser that will take a single option - param file
    conf_parser = argparse.ArgumentParser(add_help=False)

    # config options - this lets you specify a parameter configuration file,
    # set the default parameters values from it, and override them later as needed
    # if not supplied, it'll use the default parameter file included in the package
    conf_parser.add_argument("--param_file", required=False, default=None,\
            help="Specify parameter config JSON file")

    args, remaining_argv = conf_parser.parse_known_args(args)
    params = read_params(param_file=args.param_file)

    # now that we've gotten the param_file and the params (either custom, or default), create the parse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                    parents=[conf_parser],\
                    description=__doc__,\
                    epilog="If running fit_WDmodel.py with MPI using mpirun, -np must be at least 2.")

    # create a couple of custom types to use with the parser
    # this type exists to make a quasi bool type instead of store_false/store_true
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    # this type exists for parameters where we can't or don't really want the
    # user to guess a default value - better to make a guess internally than
    # have a bad  starting point
    def NoneOrFloat(v):
        if v.lower() in ("none", "null", "nan"):
            return None
        else:
            return float(v)

    parser.register('type','bool',str2bool)
    parser.register('type','NoneOrFloat',NoneOrFloat)

    # multiprocessing options
    parallel = parser.add_argument_group('parallel', 'Parallel processing options')
    mproc = parallel.add_mutually_exclusive_group()
    mproc.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    mproc.add_argument("--mpil", dest="mpil", default=False,
                       action="store_true", help="Run with MPI and enable loadbalancing.")

    # spectrum options
    spectrum = parser.add_argument_group('spectrum', 'Spectrum options')

    spectrum.add_argument('--specfile', required=True, \
            help="Specify spectrum to fit")
    spectrum.add_argument('--spectable', required=False,  default="data/spectroscopy/spectable_resolution.dat",\
            help="Specify file containing a fwhm lookup table for specfile")
    spectrum.add_argument('--lamshift', required=False, type=float, default=0.,\
            help="Specify a flat wavelength shift in Angstrom to fix  slit centering errors")
    spectrum.add_argument('--vel', required=False, type=float, default=0.,\
            help="Specify a velocity shift in kmps to apply to the spectrum")
    spectrum.add_argument('--trimspec', required=False, nargs=2, default=(None,None),
                type='NoneOrFloat', metavar=("BLUELIM", "REDLIM"), help="Trim spectrum to wavelength range")
    spectrum.add_argument('--rebin',  required=False, type=int, default=1,\
            help="Rebin the spectrum by an integer factor. Output wavelengths remain uncorrelated.")
    spectrum.add_argument('--rescale',  required=False, action="store_true", default=False,\
            help="Rescale the spectrum to make the noise ~1. Changes the value, bounds, scale on dl also")
    spectrum.add_argument('--blotch', required=False, action='store_true',\
            default=False, help="Blotch the spectrum to remove gaps/cosmic rays before fitting?")

    # photometry options
    reddeninglaws = ('od94', 'ccm89', 'f99')
    phot = parser.add_argument_group('photometry', 'Photometry options')
    phot.add_argument('--photfile', required=False,  default="data/photometry/WDphot_C22.dat",\
            help="Specify file containing photometry lookup table for objects")
    phot.add_argument('--reddeningmodel', required=False, choices=reddeninglaws, default='od94',\
            help="Specify functional form of reddening law" )
    phot.add_argument('--phot_dispersion', required=False, type=float, default=0.003,\
            help="Specify a flat photometric dispersion error in mag to add in quadrature to the measurement errors")
    phot.add_argument('--excludepb', nargs='+',\
            help="Specify passbands to exclude" )
    phot.add_argument('--ignorephot',  required=False, action="store_true", default=False,\
            help="Ignores missing photometry and does the fit with just the spectrum")

    # fitting options
    model = parser.add_argument_group('model',\
            'Model options. Modify using --param_file or CL. CL overrides. Caveat emptor.')
    for param in params:
        # we can't reasonably expect a user supplied guess or a static value to
        # work for some parameters. Allow None for these, and we'll determine a
        # good starting guess from the data. Note that we can actually just get
        # a starting guess for FWHM, but it's easier to use a lookup table.
        if param in ('fwhm','dl','mu'):
            dtype = 'NoneOrFloat'
        else:
            dtype = float

        model.add_argument('--{}'.format(param), required=False, type=dtype, default=params[param]['value'],\
                help="Specify param {} value".format(param))
        model.add_argument('--{}_fix'.format(param), required=False, default=params[param]['fixed'], type="bool",\
                help="Specify if param {} is fixed".format(param))
        model.add_argument('--{}_scale'.format(param), required=False, type=float, default=params[param]['scale'],\
                help="Specify param {} scale/step size".format(param))
        model.add_argument('--{}_bounds'.format(param), required=False, nargs=2, default=params[param]["bounds"],
                type=float, metavar=("LOWERLIM", "UPPERLIM"), help="Specify param {} bounds".format(param))

    # covariance model options
    covmodel = parser.add_argument_group('covariance model', 'Covariance model options')
    covmodel.add_argument('--covtype', required=False, choices=('White','ExpSquared','Matern32','Matern52','Exp'),\
                default='ExpSquared', help='Specify parametric form of the covariance function to model the spectrum')
    covmodel.add_argument('--usehodlr',  required=False, type="bool", default="True",\
            help="Use the HODLR solver over the Basic Solver - faster, but approximate")
    covmodel.add_argument('--hodlr_tol', required=False, type=float, default=1e-12,\
            help="Specify tolerance for HODLR solver")
    covmodel.add_argument('--hodlr_nleaf',  required=False, type=int, default=200,\
            help="Specify size of smallest matrix blocks before HODLR solves system directly")

    # MCMC config options
    mcmc = parser.add_argument_group('mcmc', 'MCMC options')
    mcmc.add_argument('--skipminuit',  required=False, action="store_true", default=False,\
            help="Skip Minuit fit - make sure to specify dl guess")
    mcmc.add_argument('--samptype', required=False, default='ensemble', choices=('ensemble', 'gibbs', 'pt'),\
            help='Specify what kind of sampler you want to use')
    mcmc.add_argument('--skipmcmc',  required=False, action="store_true", default=False,\
            help="Skip MCMC - if you skip both minuit and MCMC, simply prepares files")
    mcmc.add_argument('--ascale', required=False, type=float, default=2.0,\
            help="Specify proposal scale for MCMC")
    mcmc.add_argument('--nwalkers',  required=False, type=int, default=300,\
            help="Specify number of walkers to use (0 disables MCMC)")
    mcmc.add_argument('--ntemps', required=False, type=int, default=1,\
            help="Specify number of temperatures in ladder for parallel tempering - only available with PTSampler")
    mcmc.add_argument('--nburnin',  required=False, type=int, default=200,\
            help="Specify number of steps for burn-in")
    mcmc.add_argument('--nprod',  required=False, type=int, default=2000,\
            help="Specify number of steps for production")
    mcmc.add_argument('--everyn',  required=False, type=int, default=1,\
            help="Use only every nth point in data for computing likelihood - useful for testing.")
    mcmc.add_argument('--thin', required=False, type=int, default=1,\
            help="Save only every nth point in the chain - only works with PTSampler and Gibbs")
    mcmc.add_argument('--discard',  required=False, type=float, default=5,\
            help="Specify percentage of steps to be discarded")
    clobber = mcmc.add_mutually_exclusive_group()
    clobber.add_argument('--resume',  required=False, action="store_true", default=False,\
            help="Resume the MCMC from the last stored location")
    clobber.add_argument('--redo',  required=False, action="store_true", default=False,\
            help="Clobber existing fits")

    # visualization options
    viz = parser.add_argument_group('viz', 'Visualization options')
    viz.add_argument('-b', '--balmerlines', nargs='+', type=int, default=range(1,7,1),\
            help="Specify Balmer lines to visualize [1:7]")
    viz.add_argument('--ndraws', required=False, type=int, default=21,\
            help="Specify number of draws from posterior to overplot for model")
    viz.add_argument('--savefig',  required=False, action="store_true", default=False,\
            help="Save individual plots")

    # output options
    output = parser.add_argument_group('output', 'Output options')
    output.add_argument('--outroot', required=False,
            help="Specify a custom output root directory. Directories go under outroot/objname/subdir.")
    output.add_argument('-o', '--outdir', required=False,\
            help="Specify a custom output directory. Overrides outroot.")

    args = None
    try:
        if comm.Get_rank() == 0:
            args = parser.parse_args(args=remaining_argv)
    finally:
        args = comm.bcast(args, root=0)

    if args is None:
        sys.exit(0)

    # some sanity checking for option values
    balmer = args.balmerlines
    try:
        balmer = np.atleast_1d(balmer).astype('int')
        if np.any((balmer < 1) | (balmer > 6)):
            raise ValueError
    except (TypeError, ValueError):
        message = 'Invalid balmer line value - must be in range [1,6]'
        raise ValueError(message)

    if args.rebin < 1:
        message = 'Rebin must be integer GE 1. Note that 1 does nothing. ({:g})'.format(args.rebin)
        raise ValueError(message)

    if args.phot_dispersion < 0.:
        message = 'Photometric dispersion must be GE 0. ({:g})'.format(args.phot_dispersion)
        raise ValueError(message)

    if args.hodlr_tol <= 0:
        message = 'HODLR Solver tolerance must be greater than 0. ({:g})'.format(args.hodlr_tol)
        raise ValueError(message)

    if args.hodlr_nleaf <= 0:
        message = 'HODLR Solver nleaf must be greater than zero ({})'.format(args.hodlr_nleaf)
        raise ValueError(message)

    if args.nwalkers <= 0:
        message = 'Number of walkers must be greater than zero for MCMC ({})'.format(args.nwalkers)
        raise ValueError(message)

    if args.nwalkers%2 != 0:
        message = 'Number of walkers must be even ({})'.format(args.nwalkers)
        raise ValueError(message)

    if args.ntemps <= 0:
        message = 'Number of temperatures must be greater than zero ({})'.format(args.ntemps)
        raise ValueError(message)

    if (args.ntemps > 1) and (args.samptype == 'ensemble'):
        message = 'Multiple temperatures only available with PTSampler or Gibbs Sampler: ({})'.format(args.ntemps)
        raise ValueError(message)

    if args.nburnin <= 0:
        message = 'Number of burnin steps must be greater than zero ({})'.format(args.nburnin)
        raise ValueError(message)

    if args.nprod <= 0:
        message = 'Number of production steps must be greater than zero ({})'.format(args.nprod)
        raise ValueError(message)

    if not (0 <= args.discard < 100):
        message = 'Discard must be a percentage (0-100) ({})'.format(args.discard)
        raise ValueError(message)

    if args.everyn < 1:
        message = 'EveryN must be integer GE 1. Note that 1 does nothing. ({:g})'.format(args.everyn)
        raise ValueError(message)

    if args.thin < 1:
        message = 'Thin must be integer GE 1. Note that 1 does nothing. ({:g})'.format(args.thin)
        raise ValueError(message)

    if (args.thin > 1) and (args.samptype == 'ensemble'):
        message = 'Chain thinning only available with PTSampler or Gibbs Sampler: ({})'.format(args.thin)
        raise ValueError(message)

    # Wait for instructions from the master process if we are running MPI
    pool = None
    if args.mpi or args.mpil:
        pool = MPIPool(loadbalance=args.mpil, debug=False)
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    return args, pool


def copy_params(params):
    """
    Returns a deep copy of a dictionary
    """
    return deepcopy(params)


def write_params(params, outfile):
    """
    Dumps the parameter dictionary params to a JSON file

    params is a dict the parameter names, as defined with _PARAMETER_NAMES as keys
    Each key must have a dictionary with keys
        "value"   : value
        "fixed"   : a bool specifying if the parameter is fixed (true) or allowed to vary (false)
        "scale"   : a scale parameter used to set the step size in this dimension
        "bounds"  : An upper and lower limit on parameter values
    Any extra keys are simply written as-is

    Note that JSON doesn't preserve ordering necessarily - this is enforced by read_params
    """
    for param in params:
        if not all (key in params[param] for key in ("value","fixed","scale", "bounds")):
            message = "Parameter {} does not have value|fixed|bounds specified in params dict".format(param)
            raise KeyError(message)

    with open(outfile, 'w') as f:
        json.dump(params, f, indent=4)


def read_params(param_file=None):
    """
    Read a JSON file that configures the default guesses and bounds for the
    parameters, as well as if they should be fixed.  The JSON keys are the
    parameter names, as defined in _PARAMETER_NAMES.
    Each key must have a dictionary with keys
        "value"   : value
        "fixed"   : a bool specifying if the parameter is fixed (true) or allowed to vary (false)
        "scale"   : a scale parameter used to set the step size in this dimension
        "bounds"  : An upper and lower limit on parameter values.
    And extra keys are simply loaded as-is

    Note that the default bounds are set by the grids available for the DA
    White Dwarf atmospheres, and by reasonable plausible ranges for the other
    parameters. Don't muck with them unless you really have good reason to.

    Note that this routine does not do any checking of types, values or bounds
    This is done by get_params_from_argparse before the fit If you setup the
    fit using an external code, you should check these values

    Returns the dictionary with the parameter defaults.
    """
    if param_file is None:
        param_file = 'WDmodel_param_defaults.json'
        param_file = get_pkgfile(param_file)

    with open(param_file, 'r') as f:
        params = json.load(f)

    # JSON doesn't preserve ordering at all, but I'd like to keep it consistent
    out = OrderedDict()
    for param in _PARAMETER_NAMES:
        # note that we're only checking if we have the right keys here, not if the values are reasonable
        if not param in params:
            message = "Parameter {} not found in JSON param file {}".format(param, param_file)
            raise KeyError(message)
        if not all (key in params[param] for key in ("value","fixed","scale", "bounds")):
            message = "Parameter {} does not have value|fixed|bounds specified in param file {}".format(param, param_file)
            raise KeyError(message)
        out[param] = params[param]

    return out


def get_params_from_argparse(args):
    """
    Converts the argparse args Namespace back into an ordered parameter
    dictionary. Assumes that the argument parser options were names
        param_value  = Value of the parameter (float or None)
        param_fix    = Bool specifying if the parameter
        param_bounds = tuple with lower limit and upper limit

    Accepts Namespace from argparse
    Returns OrderedDict of parameter keywords
    """
    kwargs = vars(args)
    out = OrderedDict()
    for param in _PARAMETER_NAMES:
        keys = (param, "{}_fix".format(param),"{}_scale".format(param), "{}_bounds".format(param))
        if not all (key in kwargs for key in keys):
            message = "Parameter {} does not have value|fixed|scale|bounds specified in argparse args".format(param)
            raise KeyError(message)
        out[param]={}
        value = kwargs[param]
        out[param]['value']  = value
        out[param]['fixed']  = kwargs['{}_fix'.format(param)]

        # check if are fixed to None, which would cause likelihood to always fail
        if (out[param]['fixed'] is True) and (value is None):
            message = "Parameter {} fixed but value is None - must be specified".format(param)
            raise RuntimeError(message)
        bounds = kwargs['{}_bounds'.format(param)]

        # check if value is out of bounds, which will cause fit to fail
        if value is not None:
            if value < bounds[0] or value > bounds[1]:
                message = "Parameter {} value ({}) is out of bounds ({},{})".format(param, value, bounds[0], bounds[1])
                raise RuntimeError(message)
        out[param]['scale']  = kwargs['{}_scale'.format(param)]
        out[param]['bounds'] = bounds
    return out


def read_model_grid(grid_file=None, grid_name=None):
    """
    Read the Tlusty/Hubeny grid file (via Jay Holberg)
    NLTE grid is from an older version of Tlusty (200 vs 202 current)
    J. Holberg is working on updating the models
    """
    if grid_file is None:
        grid_file = 'TlustyGrids.hdf5'
    grid_file = get_pkgfile(grid_file)

    if grid_name is None:
        grid_name = "default"

    with h5py.File(grid_file, 'r') as grids:
        # the original IDL SAV file Tlusty grids were annoyingly broken up by wavelength
        # this was because the authors had different wavelength spacings
        # since they didn't feel that the continuum needed much spacing anyway
        # and "wanted to save disk space"
        # and then their old IDL interpolation routine couldn't handle the variable spacing
        # so they broke up the grids
        # So really, the original IDL SAV files were annoyingly broken up by wavelength because reasons...
        # We have concatenated these "large" arrays because we don't care about disk space
        # This grid is called "default", but the originals also exist
        # and you can pass grid_name to use them if you choose to
        try:
            grid = grids[grid_name]
        except KeyError,e:
            message = '{}\nGrid {} not found in grid_file {}. Accepted values are ({})'.format(e, grid_name,\
                    grid_file, ','.join(grids.keys()))
            raise ValueError(message)

        wave  = grid['wave'].value.astype('float64')
        ggrid = grid['ggrid'].value.astype('float64')
        tgrid = grid['tgrid'].value.astype('float64')
        flux  = grid['flux'].value.astype('float64')

    return grid_file, grid_name, wave, ggrid, tgrid, flux


def _read_ascii(filename, **kwargs):
    """
    Read space separated ascii file, with column names provided on first line (# optional)
    kwargs are passed along to genfromtxt
    """
    return np.recfromtxt(filename, names=True, **kwargs)


# create some aliases
# these exist so we can flesh out full functions later
# with different formats if necessary for different sorts of data
read_phot      = _read_ascii
read_spectable = _read_ascii
read_pbmap     = _read_ascii


def get_spectrum_resolution(specfile, spectable, fwhm=None):
    """
    Accepts a spectrum filename, and reads a lookup table to get the resolution
    of the spectrum spectable must contain at least two column names, specname
    and fwhm

    Note that there there's some hackish internal name fixing since T.
    Matheson's table specnames didn't match the spectrum filenames
    """
    _default_resolution = 5.0
    if fwhm is None:
        try:
            spectable = read_spectable(spectable)
        except (OSError, IOError), e:
            message = '{}\nCould not get resolution from spectable {}'.format(e, spectable)
            warnings.warn(message, RuntimeWarning)
            spectable = None
        shortfile = os.path.basename(specfile).replace('-total','')
        if shortfile.startswith('test'):
            message = 'Spectrum filename indicates this is a test - using default resolution'
            warnings.warn(message, RuntimeWarning)
            fwhm = _default_resolution
        elif spectable is not None:
            mask = (spectable.specname == shortfile)
            if len(spectable[mask]) != 1:
                message = 'Could not find an entry for this spectrum in the spectable file - using default resolution'
                warnings.warn(message, RuntimeWarning)
                fwhm = _default_resolution
            else:
                fwhm = spectable[mask].fwhm[0]
        else:
            fwhm = _default_resolution
    else:
        message = 'Smoothing factor specified on command line - overriding spectable file'
        warnings.warn(message, RuntimeWarning)
    print('Using smoothing instrumental FWHM {}'.format(fwhm))
    return fwhm


def read_spec(filename, **kwargs):
    """
    Read a space separated spectrum filei, filename, with column names
    wave flux flux_err
    column names are expected on the first line
    lines beginning with # are ignored
    Removes any NaN entries (any column)
    """
    spec = _read_ascii(filename, **kwargs)
    if np.any(~np.isfinite(spec.wave)) or np.any(~np.isfinite(spec.flux)) or np.any(~np.isfinite(spec.flux_err)):
        message = "Spectroscopy values and uncertainties must be finite."
        raise ValueError(message)

    if np.any(spec.flux_err <= 0.) or np.any(spec.flux <= 0.):
        message = "Spectroscopy values uncertainties must all be positive."
        raise ValueError(message)

    return spec


def get_phot_for_obj(objname, filename):
    """
    Accepts an object name, objname, and a lookup table filename

    The file is expected to be ascii with the first line defining column names.
    Columns names must be names of passbands (parseable by synphot) for
    magnitudes. Column names for errors in magnitudes in passband must be
    'd'+passband_name. The first column is expected to be obj for objname.
    There must be only one line per objname
    Lines beginning  with # are ignored

    Returns the photometry for objname, obj formatted as a recarray with pb, mag, mag_err

    """
    phot = read_phot(filename)
    mask = (phot.obj == objname)

    nmatch = len(phot[mask])
    if nmatch == 0:
        message = 'Got no matches for object {} in file {}. Did you want --ignorephot?'.format(objname, filename)
        raise RuntimeError(message)
    elif nmatch > 1:
        message = 'Got multiple matches for object {} in file {}'.format(objname, filename)
        raise RuntimeError(message)
    else:
        pass

    this_phot =  phot[mask][0]
    colnames  = this_phot.dtype.names
    pbnames   = [pb for pb in colnames[1:] if not pb.startswith('d')]

    mags = [this_phot[pb] for pb in pbnames]
    errs = [this_phot['d'+pb] for pb in pbnames]

    pbnames = np.array(pbnames)
    mags    = np.array(mags)
    errs    = np.array(errs)

    if np.any(~np.isfinite(mags)) or np.any(~np.isfinite(errs)):
        message = "Photometry values and uncertainties must be finite."
        raise ValueError(message)

    if np.any(errs <= 0.):
        message = "Photometry uncertainties must all be positive."
        raise ValueError(message)

    out_phot = np.rec.fromarrays([pbnames, mags, errs],names='pb,mag,mag_err')
    return out_phot


def make_outdirs(dirname, redo=False, resume=False):
    """
    Checks if output directory exists, and if it does, and we are resuming or
    redoing, simply return else creates it.
    """
    if os.path.isdir(dirname):
        if resume or redo:
            return
        else:
            message = "Output directory {} already exists. Specify --redo to clobber.".format(dirname)
            raise IOError(message)

    try:
        os.makedirs(dirname)
    except OSError as  e:
        message = '{}\nCould not create outdir {} for writing.'.format(e,dirname)
        raise OSError(message)


def set_objname_outdir_for_specfile(specfile, outdir=None, outroot=None, redo=False, resume=False):
    """
    Accepts a spectrum filename (and optionally a preset output directory) or
    an output root directory, and determines the objname.

    If output directory isn't provided, creates an output directory based on
    object name, else just uses outdir

    Returns objname and output dirname, if directories were successfully
    created/exist.
    """
    if outroot is None:
        outroot = os.path.join(os.getcwd(), "out")
    basespec = os.path.basename(specfile).replace('.flm','')
    objname = basespec.split('-')[0]
    if outdir is None:
        dirname = os.path.join(outroot, objname, basespec)
    else:
        dirname = outdir
    make_outdirs(dirname, redo=redo, resume=resume)
    return objname, dirname


def get_outfile(outdir, specfile, ext, check=False, redo=False, resume=False):
    """
    Returns the full path to a file given outdir, specfile
    Replaces .flm at the end of specfile with extension ext (i.e. you need to include the period)
    We set the output file based on the spectrum name, since we can have multiple spectra per object
    If outdir is configured by get_objname_outdir_for_specfile, it'll take care of the objname
    """
    outfile = os.path.join(outdir, os.path.basename(specfile.replace('.flm', ext)))
    if check:
        if os.path.exists(outfile) and not (resume or redo):
            message = "Output file {} already exists. Specify --redo to clobber.".format(outfile)
            raise IOError(message)
    return outfile


def get_pkgfile(infile):
    """
    Returns the full path to file inside the WDmodel package
    """
    pkgfile = pkg_resources.resource_filename('WDmodel',infile)

    if not os.path.exists(pkgfile):
        message = 'Could not find package file {}'.format(pkgfile)
        raise IOError(message)
    return pkgfile


def write_fit_inputs(spec, phot, cont_model, linedata, continuumdata,\
        rvmodel, covtype, usehodlr, nleaf, tol, phot_dispersion, scale_factor, outfile):
    """
    Save the spectrum, photometry (raw fit inputs) as well as a
    pseudo-continuum model and line data (visualization only inputs) to a file.

    Also saves the covtype, solver preferences, photometric dispersion, rvmodel and
    scale_factor

    This is intended to be called after WDmodel.fit.pre_process_spectrum() and
    WDmodel.io.get_phot_for_obj()

    This file is enough to redo the fit with the same input and
    different settings or redo the output for without redoing the fit.

    Alternatively, this file together with the HDF5 file written by
    WDmodel.fit.fit_model() with the sample chain is enough to regenerate the
    plots and output without redoing the fit.

    Accepts
        spec: recarray spectrum (wave, flux, flux_err)
        phot: recarray photometry (pb, mag, mag_err)
        cont_model: recarray continuum model (wave, flux)
        linedata: recarray linedata (wave, flux, flux_err, line_mask)
        continuumdata: data used to generate the continuum model (wave, flux, flux_err)
        rvmodel: string specifying which RV law was used to redden the spectrum
        covtype: string specifying which kernel was used to model the spectrum covariance
        usehodlr: bool specifying if the user requested that we use the HODLR solver
        nleaf: minimum matrix size before HODLR attempts to directly solve system with Eigen Cholesky
        tol: HODLR tolerance
        phot_dispersion: amount of photometric dispersion to add in quadrature with the reported uncertainties
        scale_factor: how the spectrum flux and flux_err was scaled
        outfile: output filename
    """

    outf = h5py.File(outfile, 'w')
    dset_spec = outf.create_group("spec")
    dset_spec.create_dataset("wave",data=spec.wave)
    dset_spec.create_dataset("flux",data=spec.flux)
    dset_spec.create_dataset("flux_err",data=spec.flux_err)
    dset_spec.attrs["scale_factor"] = scale_factor

    dset_cont_model = outf.create_group("cont_model")
    dset_cont_model.create_dataset("wave",data=cont_model.wave)
    dset_cont_model.create_dataset("flux",data=cont_model.flux)

    dset_linedata = outf.create_group("linedata")
    dset_linedata.create_dataset("wave",data=linedata.wave)
    dset_linedata.create_dataset("flux",data=linedata.flux)
    dset_linedata.create_dataset("flux_err",data=linedata.flux_err)
    dset_linedata.create_dataset("line_mask",data=linedata.line_mask)

    dset_continuumdata = outf.create_group("continuumdata")
    dset_continuumdata.create_dataset("wave",data=continuumdata.wave)
    dset_continuumdata.create_dataset("flux",data=continuumdata.flux)
    dset_continuumdata.create_dataset("flux_err",data=continuumdata.flux_err)

    dset_fit_config = outf.create_group("fit_config")
    dset_fit_config.attrs["covtype"]=np.string_(covtype)
    dset_fit_config.attrs["usehodlr"]=usehodlr
    dset_fit_config.attrs["nleaf"]=nleaf
    dset_fit_config.attrs["tol"]=tol
    dset_fit_config.attrs["rvmodel"]=np.string_(rvmodel)

    if phot is not None:
        dset_phot = outf.create_group("phot")
        dt = phot.pb.dtype.str.lstrip('|')
        dset_phot.create_dataset("pb", data=phot.pb, dtype=dt)
        dset_phot.create_dataset("mag",data=phot.mag)
        dset_phot.create_dataset("mag_err",data=phot.mag_err)
        dset_phot.attrs["phot_dispersion"] =phot_dispersion

    outf.close()
    print "Wrote inputs file {}".format(outfile)


def read_fit_inputs(input_file):
    """
    Read the saved HDF5 input_file and return recarrays of the contents
    input files are expected to contain at least 4 groups, with the 5th optional
    The groups, and the datasets they must have are
        spec
            wave, flux, flux_err
            and a HDF5 attribute, scale_factor specifying how the spectrum was scaled
        cont_model
            wave, flux
        linedata
            wave, flux, flux_err, line_mask
        continuumdata
            wave, flux, flux_err
        [phot]
            pb, mag, mag_err
        the fit_config group must have the following HDF5 attributes
            covtype, usehodlr, nleaf, tol, rvmodel

    Returns a tuple of recarrays and dictionary
        spec, cont_model, linedata, continuumdata, phot[=None if absent], fit_config
    """
    d = h5py.File(input_file, mode='r')

    try:
        spec_wave = d['spec']['wave'].value
        spec_flux = d['spec']['flux'].value
        spec_ferr = d['spec']['flux_err'].value
        scale_factor = d['spec'].attrs['scale_factor']
        spec = np.rec.fromarrays([spec_wave, spec_flux, spec_ferr], names='wave,flux,flux_err')

        cmod_wave = d['cont_model']['wave'].value
        cmod_flux = d['cont_model']['flux'].value
        cont_model = np.rec.fromarrays([cmod_wave, cmod_flux], names='wave,flux')

        line_wave = d['linedata']['wave'].value
        line_flux = d['linedata']['flux'].value
        line_ferr = d['linedata']['flux_err'].value
        line_mask = d['linedata']['line_mask'].value
        linedata = np.rec.fromarrays([line_wave, line_flux, line_ferr,line_mask], names='wave,flux,flux_err,line_mask')

        cont_wave = d['continuumdata']['wave'].value
        cont_flux = d['continuumdata']['flux'].value
        cont_ferr = d['continuumdata']['flux_err'].value
        continuumdata = np.rec.fromarrays([cont_wave, cont_flux, cont_ferr], names='wave,flux,flux_err')

        fit_config = {}
        fit_config['covtype'] = d['fit_config'].attrs['covtype']
        fit_config['nleaf'] = d['fit_config'].attrs['nleaf']
        fit_config['tol'] = d['fit_config'].attrs['tol']
        fit_config['rvmodel'] = d['fit_config'].attrs['rvmodel']
        fit_config['scale_factor'] = scale_factor

    except Exception as e:
        message = '{}\nCould not load all arrays from input file {}'.format(e, input_file)
        raise IOError(message)
    try:
        fit_config['usehodlr'] = d['fit_config'].attrs['usehodlr']
    except Exception as e:
        try:
            fit_config['usehodlr'] = ~d['fit_config'].attrs['usebasic']
        except Exception as e:
            message = '{}\nCould not load all arrays from input file {}'.format(e, input_file)
            raise IOError(message)


    phot = None
    if 'phot' in d.keys():
        try:
            pb  = d['phot']['pb'].value
            mag = d['phot']['mag'].value
            mag_err = d['phot']['mag_err'].value
            phot = np.rec.fromarrays([pb, mag, mag_err], names='pb,mag,mag_err')
            phot_dispersion = d['phot'].attrs['phot_dispersion']
            fit_config['phot_dispersion'] = phot_dispersion
        except KeyError as e:
            message = '{}\nFailed to restore photometry from input file {} though group exists'.format(e, input_file)
            warnings.warn(message, RuntimeWarning)
            phot = None
    return spec, cont_model, linedata, continuumdata, phot, fit_config


def read_mcmc(input_file):
    """
    Read the saved HDF5 chain_file and return samples, sample probabilities and param names

    Returns a tuple of arrays
        param_names, samples, samples_lnprob
    """
    d = h5py.File(input_file, mode='r')

    try:
        chain_params   = {}
        samples        = d['chain']['position'].value
        samples_lnprob = d['chain']['lnprob'].value
        chain_params['param_names'] = d['chain']['names'].value
        chain_params['samptype']    = d['chain'].attrs['samptype']
        chain_params['ntemps']      = d['chain'].attrs['ntemps']
        chain_params['nwalkers']    = d['chain'].attrs['nwalkers']
        chain_params['nprod']       = d['chain'].attrs['nprod']
        chain_params['ndim']        = d['chain'].attrs['nparam']
        chain_params['everyn']      = d['chain'].attrs['everyn']
        chain_params['ascale']      = d['chain'].attrs['ascale']


    except KeyError as e:
        message = '{}\nCould not load all arrays from input file {}'.format(e, input_file)
        raise IOError(message)

    return samples, samples_lnprob, chain_params


def write_spectrum_model(spec, model_spec, outfile):
    """
    Write the spectrum and the model spectrum and residuals to outfile
    Accepts
        spec: recarray spectrum (wave, flux, flux_err)
        model_spec: recarray spectrum (wave, flux, norm_flux)
        outfile: output filename
    """
    out = (spec.wave, spec.flux, spec.flux_err,\
            model_spec.norm_flux, model_spec.flux, model_spec.flux_err,\
            spec.flux-model_spec.flux)
    out = np.rec.fromarrays(out, names='wave,flux,flux_err,norm_flux,model_flux,model_flux_err,res_flux')
    with open(outfile, 'w') as f:
        f.write(rec2txt(out, precision=8)+'\n')
    print "Wrote spec model file {}".format(outfile)


def write_phot_model(phot, model_mags, outfile):
    """
    Write the photometry, model photometry and residuals to outfile
    Accepts
        phot: recarray photometry (pb, mag, mag_err)
        model_mags: recarray model photometry (pb, mag)
        outfile: output filename
    """
    out = (phot.pb, phot.mag, phot.mag_err, model_mags.mag, phot.mag-model_mags.mag)
    out = np.rec.fromarrays(out, names='pb,mag,mag_err,model_mag,res_mag')
    with open(outfile, 'w') as f:
        f.write(rec2txt(out, precision=6)+'\n')
    print "Wrote phot model file {}".format(outfile)


def write_full_model(full_model, outfile):
    """
    Write the full SED model to outfile
    Accepts
        full_model: recarray SED model (wave, flux)
        outfile: output filename
    """
    outf = h5py.File(outfile, 'w')
    dset_model = outf.create_group("model")
    dset_model.create_dataset("wave",data=full_model.wave)
    dset_model.create_dataset("flux",data=full_model.flux)
    dset_model.create_dataset("flux_err",data=full_model.flux_err)
    outf.close()
    print "Wrote full model file {}".format(outfile)
