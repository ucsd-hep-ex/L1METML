import awkward as ak
import numpy as np

def read_input(inputfiles):
    import h5py
    for i, fname in enumerate(inputfiles):
        print("read file", fname)
        with h5py.File( fname, 'r') as h5f:
            if i == 0:
                X = h5f['X'][:]
                Y = h5f['Y'][:]
            else:
                X = np.concatenate((X, h5f['X']), axis=0)
                Y = np.concatenate((Y, h5f['Y']), axis=0)    
    print("finish reading files")
    return X, Y

def convertXY2PtPhi(arrayXY):
    # convert from array with [:,0] as X and [:,1] as Y to [:,0] as pt and [:,1] as phi
    nevents = arrayXY.shape[0]
    arrayPtPhi = np.zeros((nevents, 2))
    arrayPtPhi[:,0] = np.sqrt((arrayXY[:,0]**2 + arrayXY[:,1]**2))
    arrayPtPhi[:,1] = np.sign(arrayXY[:,1])*np.arccos(arrayXY[:,0]/arrayPtPhi[:,0])
    return arrayPtPhi

def preProcessing(A, normFac, EVT=None):
    """ pre-processing input """

    norm = normFac

    #pt = A[:,:,0:1] / norm
    px = A[:,:,0:1] / norm
    py = A[:,:,1:2] / norm
    eta = A[:,:,2:3]
    #phi = A[:,:,4:5]
    puppi = A[:,:,3:4]

    # remove outliers
    #pt[ np.where(np.abs(pt>500)) ] = 0.
    px[ np.where(np.abs(px>500)) ] = 0.
    py[ np.where(np.abs(py>500)) ] = 0.

    #inputs = np.concatenate((pt, eta, phi, puppi, px, py), axis=2)
    inputs = np.concatenate((eta, puppi, px, py), axis=2)

    inputs_cat0 = A[:,:,4:5] # encoded PF pdgId
    inputs_cat1 = A[:,:,5:6] # encoded PF charge

    return inputs, inputs_cat0, inputs_cat1

def MakePlots(trueXY, mlXY, puppiXY, path_out):
    # make the 1d distribution, response, resolution,
    # and response-corrected resolution plots
    # input has [:,0] as X and [:,1] as Y
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    
    true_ptPhi = convertXY2PtPhi(trueXY)
    ml_ptPhi = convertXY2PtPhi(mlXY)
    puppi_ptPhi = convertXY2PtPhi(puppiXY)
    # [:,0] is pt; [:,1] is phi
    
    Make1DHists(trueXY[:,0], mlXY[:,0], puppiXY[:,0], -400, 400, 40, False, 'MET X [GeV]', 'A.U.', f'{path_out}MET_x.png')
    Make1DHists(trueXY[:,1], mlXY[:,1], puppiXY[:,1], -400, 400, 40, False, 'MET Y [GeV]', 'A.U.', f'{path_out}MET_y.png')
    Make1DHists(true_ptPhi[:,0], ml_ptPhi[:,0], puppi_ptPhi[:,0], 0, 400, 40, False, 'MET Pt [GeV]', 'A.U.', f'{path_out}MET_pt.png')
    
    # do statistics
    from scipy.stats import binned_statistic
    binnings = np.linspace(0, 400, num=21)
    truth_means, bin_edges, binnumber = binned_statistic(true_ptPhi[:,0], true_ptPhi[:,0], statistic='mean', bins=binnings, range=(0,400))
    ml_means,  _, _ = binned_statistic(true_ptPhi[:,0], ml_ptPhi[:,0],  statistic='mean', bins=binnings, range=(0,400))
    puppi_means, _, _ = binned_statistic(true_ptPhi[:,0], puppi_ptPhi[:,0], statistic='mean', bins=binnings, range=(0,400))
    
    # plot response
    plt.figure()
    plt.hlines(truth_means/truth_means, bin_edges[:-1], bin_edges[1:], colors='k', lw=5,
           label='Truth', linestyles='solid')
    plt.hlines(ml_means/truth_means, bin_edges[:-1], bin_edges[1:], colors='r', lw=5,
           label='Predict', linestyles='solid')
    plt.hlines(puppi_means/truth_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
           label='PUPPI', linestyles='solid')
    plt.xlim(0,400.0)
    plt.ylim(0,1.1)
    plt.xlabel('Truth MET [GeV]')
    plt.legend(loc='lower right')
    plt.ylabel('<MET Estimation>/<MET Truth>')
    plt.savefig(f"{path_out}MET_response.png")
    plt.close()
    
    # response correction factors
    sfs_ML = np.take(ml_means/truth_means,  np.digitize(true_ptPhi[:,0], binnings)-1, mode='clip')
    sfs_PUPPI = np.take(puppi_means/truth_means, np.digitize(true_ptPhi[:,0], binnings)-1, mode='clip')
    
    # define the resolution statistic and potential bounds
    def resolqt(y):
        return(np.percentile(y,84)-np.percentile(y,16))/2.0
    def resolqt_upper(y):
        return(np.percentile(y,90)-np.percentile(y,10))/2.0
    def resolqt_lower(y):
        return(np.percentile(y,76)-np.percentile(y,22))/2.0

    # compute resolutions inside each bin
    bin_resolX_ML, bin_edges, binnumber = binned_statistic(true_ptPhi[:,0], trueXY[:,0] - mlXY[:,0] * sfs_ML, statistic=resolqt, bins=binnings, range=(0,400))
    bin_resolY_ML, _, _ = binned_statistic(true_ptPhi[:,0], trueXY[:,1] - mlXY[:,1] * sfs_ML, statistic=resolqt, bins=binnings, range=(0,400))
    bin_resolX_PUPPI, _, _ = binned_statistic(true_ptPhi[:,0], trueXY[:,0] - puppiXY[:,0] * sfs_PUPPI, statistic=resolqt, bins=binnings, range=(0,400))
    bin_resolY_PUPPI, _, _ = binned_statistic(true_ptPhi[:,0], trueXY[:,1] - puppiXY[:,1] * sfs_PUPPI, statistic=resolqt, bins=binnings, range=(0,400))

    # and the upper and lower bounds
    bin_resolX_ML_upper, bin_edges, binnumber = binned_statistic(true_ptPhi[:,0], trueXY[:,0] - mlXY[:,0] * sfs_ML, statistic=resolqt_upper, bins=binnings, range=(0,400))
    bin_resolY_ML_upper, _, _ = binned_statistic(true_ptPhi[:,0], trueXY[:,1] - mlXY[:,1] * sfs_ML, statistic=resolqt_upper, bins=binnings, range=(0,400))
    bin_resolX_PUPPI_upper, _, _ = binned_statistic(true_ptPhi[:,0], trueXY[:,0] - puppiXY[:,0] * sfs_PUPPI, statistic=resolqt_upper, bins=binnings, range=(0,400))
    bin_resolY_PUPPI_upper, _, _ = binned_statistic(true_ptPhi[:,0], trueXY[:,1] - puppiXY[:,1] * sfs_PUPPI, statistic=resolqt_upper, bins=binnings, range=(0,400))

    bin_resolX_ML_lower, bin_edges, binnumber = binned_statistic(true_ptPhi[:,0], trueXY[:,0] - mlXY[:,0] * sfs_ML, statistic=resolqt_lower, bins=binnings, range=(0,400))
    bin_resolY_ML_lower, _, _ = binned_statistic(true_ptPhi[:,0], trueXY[:,1] - mlXY[:,1] * sfs_ML, statistic=resolqt_lower, bins=binnings, range=(0,400))
    bin_resolX_PUPPI_lower, _, _ = binned_statistic(true_ptPhi[:,0], trueXY[:,0] - puppiXY[:,0] * sfs_PUPPI, statistic=resolqt_lower, bins=binnings, range=(0,400))
    bin_resolY_PUPPI_lower, _, _ = binned_statistic(true_ptPhi[:,0], trueXY[:,1] - puppiXY[:,1] * sfs_PUPPI, statistic=resolqt_lower, bins=binnings, range=(0,400))

    # calculate ML and PUPPI resolutions weighted difference
    weights = []
    nEvents = len(binnumber)
    for bin in np.arange(len(binnings)-1):
        weights.append(len(binnumber[binnumber==bin+1])/nEvents) #number of events in bin / total number of events
    xRes_avgDif = np.average(bin_resolX_PUPPI-bin_resolX_ML, weights=weights)
    yRes_avgDif = np.average(bin_resolY_PUPPI-bin_resolY_ML, weights=weights)

    #make the plots
    plt.hlines(bin_resolX_ML, bin_edges[:-1], bin_edges[1:], colors='r', lw=5,
               label='ML', linestyles='solid')
    plt.hlines(bin_resolX_PUPPI, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
               label='PUPPI', linestyles='solid')
    xRes_ML_error = abs([bin_resolX_ML_lower, bin_resolX_ML_upper]-bin_resolX_ML)
    plt.errorbar(bin_edges[:-1]+15, bin_resolX_ML, yerr= xRes_ML_error, fmt='none', color='r')
    xRes_PUPPI_error = abs([bin_resolX_PUPPI_lower, bin_resolX_PUPPI_upper]-bin_resolX_PUPPI)
    plt.errorbar(bin_edges[:-1]+5, bin_resolX_PUPPI, yerr= xRes_PUPPI_error, fmt='none', color='g')
    plt.legend(loc='lower right')
    plt.xlim(0,400.0)
    plt.ylim(0,200)
    plt.xlabel('Truth MET [GeV]')
    plt.ylabel('RespCorr $\sigma$(METX) [GeV]')
    plt.title(f'Average $\sigma$(METX) Difference = {round(xRes_avgDif,3)}', fontsize = 22)
    plt.savefig(f"{path_out}resolution_metx.png")

    plt.hlines(bin_resolY_ML, bin_edges[:-1], bin_edges[1:], colors='r', lw=5,
               label='Predict', linestyles='solid')
    plt.hlines(bin_resolY_PUPPI, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
               label='PUPPI', linestyles='solid')
    yRes_ML_error = abs([bin_resolY_ML_lower, bin_resolY_ML_upper]-bin_resolY_ML)
    plt.errorbar(bin_edges[:-1]+12, bin_resolY_ML, yerr= yRes_ML_error, fmt='none', color='r')
    yRes_PUPPI_error = abs([bin_resolY_PUPPI_lower, bin_resolY_PUPPI_upper]-bin_resolY_PUPPI)
    plt.errorbar(bin_edges[:-1]+8, bin_resolX_PUPPI, yerr= xRes_PUPPI_error, fmt='none', color='g')
    plt.legend(loc='lower right')
    plt.xlim(0,400.0)
    plt.ylim(0,200.0)
    plt.xlabel('Truth MET [GeV]')
    plt.ylabel('RespCorr $\sigma$(METY) [GeV]')
    plt.title(f'Average $\sigma$(METY) Difference = {round(yRes_avgDif,3)}', fontsize = 22)
    plt.savefig(f"{path_out}resolution_mety.png")

    plt.hlines(resolY_dif, bin_edges[:-1], bin_edges[1:], lw=5, linestyles='solid')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.xlim(0,400.0)
    plt.ylim(-20,20)
    plt.xlabel('Truth MET [GeV]')
    plt.ylabel('PUPPI-ML $\sigma$(METY) [GeV]')
    plt.title(f'Average $\sigma$(METY) Difference = {round(yRes_avgDif,3)}', fontsize = 22)
    plt.savefig(f"{path_out}resolutionDif_metY.png")
    
    plt.hlines(resolY_dif, bin_edges[:-1], bin_edges[1:], lw=5, linestyles='solid')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.xlim(0,400.0)
    plt.ylim(-20,20)
    plt.xlabel('Truth MET [GeV]')
    plt.ylabel('PUPPI-ML $\sigma$(METY) [GeV]')
    plt.title(f'Average $\sigma$(METX) Difference = {round(xRes_avgDif,3)}', fontsize = 22)
    plt.savefig(f"{path_out}resolutionDif_metX.png")

def Make1DHists(truth, predict, PUPPI, xmin=0, xmax=400, nbins=100, density=False, xname="pt [GeV]", yname = "A.U.", outputname="1ddistribution.png"):
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(10,8))
    plt.hist(truth,    bins=nbins, range=(xmin, xmax), density=density, histtype='step', facecolor='k', label='Truth')
    plt.hist(predict,  bins=nbins, range=(xmin, xmax), density=density, histtype='step', facecolor='r', label='Predict')
    plt.hist(PUPPI, bins=nbins, range=(xmin, xmax), density=density, histtype='step', facecolor='g', label='PUPPI')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(outputname)
    plt.close()

def to_np_array(ak_array, maxN=100, pad=0):
    return ak.fill_none(ak.pad_none(ak_array,maxN,clip=True,axis=-1),pad).to_numpy()
