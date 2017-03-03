"""
Routines to visualize the DA White Dwarf model atmosphere fit
"""
import os
import numpy as np
import george
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties as FM
from astropy.visualization import hist
from . import io
import corner
#from matplotlib import rc
#rc('text', usetex=True)
#rc('font', family='serif')
#rc('ps', usedistiller='xpdf')
#rc('text.latex', preamble = ','.join('''\usepackage{amsmath}'''.split()))

def plot_minuit_spectrum_fit(spec, objname, outdir, specfile, model, result, rvmodel='od94', save=True):
    """
    Quick plot to show the output from the limited Minuit fit of the spectrum.
    This fit doesn't try to account for the covariance in the data, and is not
    expected to be great - just fast, and capable of setting a reasonable
    initial guess. If this fit is very far off, refine the intial guess.

    Accepts:
        spec: the recarray spectrum
        objname: object name - cosmetic only
        outdir: controls where the plot is written out if save=True
        specfile: Used in the title, and to set the name of the outfile if save=True
        model: WDmodel.WDmodel instance 
        result: dict of parameters with keywords value, fixed, scale, bounds for each
        rvmodel: keyword allows a different model for the reddening law (default O'Donnell '94)
        save: if True, save the file

    Returns a matplotlib figure instance
    """
    
    font_s  = FM(size='small')
    font_m  = FM(size='medium')
    font_l  = FM(size='large')

    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    ax_spec  = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1])

    ax_spec.fill_between(spec.wave, spec.flux+spec.flux_err, spec.flux-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_spec.plot(spec.wave, spec.flux, color='black', linestyle='-', marker='None', label=specfile)

    teff = result['teff']['value']
    logg = result['logg']['value']
    av   = result['av']['value']
    dl   = result['dl']['value']
    rv   = result['rv']['value']
    fwhm = result['fwhm']['value']

    mod = model._get_obs_model(teff, logg, av, fwhm, spec.wave, rv=rv, rvmodel=rvmodel)
    smoothedmod = mod* (1./(4.*np.pi*(dl)**2.))
    outlabel = 'Model\nTeff = {:.1f} K\nlog(g) = {:.2f}\nAv = {:.2f} mag\ndl = {:.2f}'.format(teff, logg, av, dl)

    ax_spec.plot(spec.wave, smoothedmod, color='red', linestyle='-',marker='None', label=outlabel)

    ax_resid.fill_between(spec.wave, spec.flux-smoothedmod+spec.flux_err, spec.flux-smoothedmod-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_resid.plot(spec.wave, spec.flux-smoothedmod,  linestyle='-', marker=None,  color='black')

    ax_resid.set_xlabel('Wavelength~(\AA)',fontproperties=font_m, ha='center')
    ax_spec.set_ylabel('Normalized Flux', fontproperties=font_m)
    ax_resid.set_ylabel('Fit Residual Flux', fontproperties=font_m)
    ax_spec.legend(frameon=False, prop=font_s)
    fig.suptitle('Quick Fit: %s (%s)'%(objname, specfile), fontproperties=font_l)
    
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    if save:
        outfile = io.get_outfile(outdir, specfile, '_minuit.pdf')
        fig.savefig(outfile)
    return fig 


def plot_mcmc_spectrum_fit(spec, objname, specfile, model, result, param_names, samples,\
        rvmodel='od94', ndraws=21):
    """
    Plot the full spectrum of the DA White Dwarf 
    """

    font_s  = FM(size='small')
    font_m  = FM(size='medium')
    font_l  = FM(size='large')

    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    ax_spec  = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1])

    ax_spec.fill_between(spec.wave, spec.flux+spec.flux_err, spec.flux-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_spec.plot(spec.wave, spec.flux, color='black', linestyle='-', marker='None', label=specfile)

    this_draw = io.copy_params(result)
    draws  = samples[np.random.randint(0, len(samples), ndraws),:]

    # plot one draw of the sample, bundled into a dict
    def plot_one(this_draw, color='red', alpha=1., label=None):
        teff = this_draw['teff']['value']
        logg = this_draw['logg']['value']
        av   = this_draw['av']['value']
        rv   = this_draw['rv']['value']
        dl   = this_draw['dl']['value']
        fwhm = this_draw['fwhm']['value']
        sigf = this_draw['sigf']['value']
        tau  = this_draw['tau']['value']

        mod = model._get_obs_model(teff, logg, av, fwhm, spec.wave, rv=rv, rvmodel=rvmodel)
        smoothedmod = mod* (1./(4.*np.pi*(dl)**2.))

        res = spec.flux - smoothedmod
        kernel = (sigf**2.)*george.kernels.ExpSquaredKernel(tau)
        gp = george.GP(kernel, mean=0.)
        gp.compute(spec.wave, spec.flux_err)
        wres, cov = gp.predict(res, spec.wave)
        ax_spec.plot(spec.wave, smoothedmod+wres, color=color, linestyle='-',marker='None', alpha=alpha, label=label)
        return smoothedmod, wres, cov
        
    # for each draw, update the dict, and plot it
    out = []
    for i in range(ndraws):
        for j, param in enumerate(param_names):
            this_draw[param]['value'] = draws[i,j]
        smoothedmod, wres, cov = plot_one(this_draw, color='orange', alpha=0.3)
        out.append((smoothedmod, wres))

    outlabel = 'Model\n'
    for param in result:
        val = result[param]['value']
        errp, errm = result[param]['errors_pm']
        if param == 'sigf':
            outlabel += '{} = {:.4f} +{:.4f}/-{:.4f}\n'.format(param, val, errp, errm)
        else:
            outlabel += '{} = {:.2f} +{:.2f}/-{:.2f}\n'.format(param, val, errp, errm)

    # finally, overplot the best result draw as solid
    smoothedmod, wres, cov = plot_one(result, color='red', alpha=1., label=outlabel)
    out.append((smoothedmod, wres))
        
    # plot the residuals     
    ax_resid.fill_between(spec.wave, spec.flux-smoothedmod-wres+spec.flux_err, spec.flux-smoothedmod-wres-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_resid.plot(spec.wave, spec.flux-smoothedmod-wres,  linestyle='-', marker=None,  color='black')
    ax_resid.axhline(0., color='red', linestyle='--')
    
    # label the axes
    ax_resid.set_xlabel('Wavelength~(\AA)',fontproperties=font_m, ha='center')
    ax_spec.set_ylabel('Normalized Flux', fontproperties=font_m)
    ax_resid.set_ylabel('Fit Residual Flux', fontproperties=font_m)
    ax_spec.legend(frameon=False, prop=font_s)
    fig.suptitle('MCMC Fit: %s (%s)'%(objname, specfile), fontproperties=font_l)
    
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    return fig, out


def plot_mcmc_spectrum_nogp_fit(spec, objname, specfile, cont_model, draws):
    """
    Plot the full spectrum of the DA White Dwarf 
    """

    font_s  = FM(size='small')
    font_m  = FM(size='medium')
    font_l  = FM(size='large')

    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    ax_spec  = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1])

    ax_spec.fill_between(spec.wave, spec.flux+spec.flux_err, spec.flux-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_spec.plot(spec.wave, spec.flux, color='black', linestyle='-', marker='None', label=specfile)
    ax_spec.plot(cont_model.wave, cont_model.flux, color='blue', linestyle='--', marker='None', label='Continuum')

    smoothedmod, wres = draws[-1]
    ax_resid.fill_between(spec.wave, spec.flux-smoothedmod+spec.flux_err, spec.flux-smoothedmod-spec.flux_err,\
        facecolor='grey', alpha=0.5, interpolate=True)
    ax_resid.plot(spec.wave, spec.flux - smoothedmod, color='black', linestyle='-', marker='None')

    def plot_draw(draw, color='red', alpha=1.0, label=None):
        smoothed_mod, wres = draw
        ax_resid.plot(spec.wave, wres,  linestyle='-', marker=None,  color=color, alpha=alpha)
        ax_spec.plot(spec.wave, smoothedmod, color=color, linestyle='-', marker='None', alpha=alpha, label=label)

    for draw in draws[:-1]:
        plot_draw(draw, color='orange', alpha=0.3)
    plot_draw(draws[-1], color='red', alpha=1.0, label='Model - no Covariance')

    # label the axes
    ax_resid.set_xlabel('Wavelength~(\AA)',fontproperties=font_m, ha='center')
    ax_spec.set_ylabel('Normalized Flux', fontproperties=font_m)
    ax_resid.set_ylabel('Fit Residual Flux', fontproperties=font_m)
    ax_spec.legend(frameon=False, prop=font_s)
    fig.suptitle('MCMC Fit - No Covariance: %s (%s)'%(objname, specfile), fontproperties=font_l)
    
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

    return fig



def plot_mcmc_line_fit(spec, linedata, model, cont_model, draws, balmer=None):
    """
    Plot the full spectrum of the DA White Dwarf 
    """

    font_xs = FM(size='x-small')
    font_s  = FM(size='small')
    font_m  = FM(size='medium')
    font_l  = FM(size='large')

    # create a figure for the line profiles
    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(1, 1)
    ax_lines  = fig.add_subplot(gs[0])

    if balmer is None:
        balmer = model._lines.keys()

    # create another figure with separate axes for each of the lines
    uselines = set(np.unique(linedata.line_mask)) & set(balmer)
    nlines = len(uselines)
    Tot = nlines + 1
    Cols = 3
    Rows = Tot // Cols 
    Rows += Tot % Cols
    fig2 = plt.figure(figsize=(10,8))
    gs2 = gridspec.GridSpec(Rows, Cols )

    # get the default color cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = cycle(colors)
    
    # plot the distribution of residuals for the entire spectrum
    ax_resid  = fig2.add_subplot(gs2[0])
    smoothedmod, wres = draws[-1]
    res = spec.flux - smoothedmod - wres
    hist(res, bins='knuth', normed=True, histtype='stepfilled', color='grey', alpha=0.5, label='Residuals',ax=ax_resid)
    ax_resid.axvline(0., color='red', linestyle='--')

    # label the axes, rotate the tick labels, and get the xlim
    ax_resid.set_xlabel('Fit Residual Flux', fontproperties=font_m)
    ax_resid.set_ylabel('Norm', fontproperties=font_m)
    ax_resid.legend(loc='upper left', frameon=False, prop=font_s)
    plt.setp(ax_resid.get_xticklabels(), rotation=30, horizontalalignment='right')
    (res_xmin, res_xmax) = ax_resid.get_xlim()
    k = 1

    for i, line in enumerate(np.unique(linedata.line_mask)):

        if not line in balmer:
            continue

        # select this line
        mask = (linedata.line_mask == line)
        wave = linedata.wave[mask]

        # restore the line properties
        linename, W0, D, eps = model._lines[line] 

        # find the matching indices in the spectrum/continuum model that match the line
        ind  = np.searchsorted(cont_model.wave, wave)
        this_line_cont = cont_model.flux[ind]

        # shift the wavelength so the centroids are 0 
        shifted_wave = wave - W0
        shifted_flux = linedata.flux[mask]/this_line_cont
        shifted_ferr  = linedata.flux_err[mask]/this_line_cont

        # plot the lines, adding a small vertical offset between each
        voff = 0.2*i
        ax_lines.fill_between(shifted_wave, shifted_flux + voff + shifted_ferr, shifted_flux + voff - shifted_ferr,\
                facecolor='grey', alpha=0.5, interpolate=True)
        ax_lines.plot(shifted_wave, shifted_flux + voff, linestyle='-', marker='None', color='black')

        # add a text label for each line
        label = '{} ({:.2f})'.format(linename, W0)
        ax_lines.text(shifted_wave[-1]+10 , shifted_flux[-1] + voff, label, fontproperties=font_xs,\
                color='blue', va='top', ha='center', rotation=90)

        # plot one of the draws
        def plot_draw(draw, color='red', alpha=1.0):
            smoothedmod, wres = draw
            line_model = (smoothedmod + wres)[ind]
            line_model /= this_line_cont
            line_model += voff 
            ax_lines.plot(shifted_wave, line_model, linestyle='-', marker='None', color=color, alpha=alpha)

        # overplot the model
        for draw in draws[:-1]:
            plot_draw(draw, color='orange', alpha=0.3)
        plot_draw(draws[-1], color='red', alpha=1.0)

        # plot the residuals of this line
        ax_resid  = fig2.add_subplot(gs2[k])
        hist(linedata.flux[mask] - (smoothedmod + wres)[ind] , bins='knuth', normed=True,ax=ax_resid,\
                histtype='stepfilled', label=label, alpha=0.3, color=next(colors))
        ax_resid.axvline(0., color='red', linestyle='--')

        # label the axis and match the limits for the overall residuals
        ax_resid.set_xlabel('Fit Residual Flux', fontproperties=font_m)
        ax_resid.set_ylabel('Norm', fontproperties=font_m)
        ax_resid.set_xlim((res_xmin, res_xmax))
        ax_resid.legend(frameon=False, prop=font_s)
        plt.setp(ax_resid.get_xticklabels(), rotation=30, horizontalalignment='right')
        k+=1

    # label the axes
    ax_lines.set_xlabel('Delta Wavelength~(\AA)',fontproperties=font_m, ha='center')
    ax_lines.set_ylabel('Normalized Flux', fontproperties=font_m)

    fig.suptitle('Line Profiles', fontproperties=font_l)
    fig2.suptitle('Residual Distributions', fontproperties=font_l)

    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    gs2.tight_layout(fig2, rect=[0, 0.03, 1, 0.95])

    return fig, fig2




def plot_mcmc_model(spec, phot, linedata,\
        objname, outdir, specfile,\
        model, cont_model,\
        params, param_names, samples, samples_lnprob,\
        rvmodel='od94', balmer=None, save=True, ndraws=21, savefig=False):
    """
    Plot the full fit of the DA White Dwarf 
    """

    outfilename = os.path.join(outdir, os.path.basename(specfile.replace('.flm','_mcmc.pdf')))
    with PdfPages(outfilename) as pdf:
        # plot spectrum and model
        fig, draws  =  plot_mcmc_spectrum_fit(spec, objname, specfile, model, params, param_names, samples,\
                rvmodel=rvmodel, ndraws=ndraws)
        if savefig:
            outfile = io.get_outfile(outdir, specfile, '_mcmc_spectrum.pdf')
            fig.savefig(outfile)
        pdf.savefig(fig)

        # plot continuum, model and draws without gp
        fig = plot_mcmc_spectrum_nogp_fit(spec, objname, specfile, cont_model, draws)
        if savefig:
            outfile = io.get_outfile(outdir, specfile, '_mcmc_nogp.pdf')
            fig.savefig(outfile)
        pdf.savefig(fig)

        # plot lines
        fig, fig2 = plot_mcmc_line_fit(spec, linedata, model, cont_model, draws, balmer=balmer)
        if savefig:
            outfile = io.get_outfile(outdir, specfile, '_mcmc_lines.pdf')
            fig.savefig(outfile)
            outfile = io.get_outfile(outdir, specfile, '_mcmc_resids.pdf')
            fig2.savefig(outfile)
        pdf.savefig(fig)
        pdf.savefig(fig2)
        
        # add phot plot

        # plot corner plot
        fig = corner.corner(samples, bins=51, labels=param_names, show_titles=True,quantiles=(0.16,0.84), smooth=1.)
        if savefig:
            outfile = io.get_outfile(outdir, specfile, '_mcmc_corner.pdf')
            fig.savefig(outfile)
        pdf.savefig(fig)
        #endwith
    
