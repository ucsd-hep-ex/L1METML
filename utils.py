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

    pt = A[:,:,0:1] / norm
    px = A[:,:,1:2] / norm
    py = A[:,:,2:3] / norm
    eta = A[:,:,3:4]
    phi = A[:,:,4:5]
    puppi = A[:,:,5:6]

    # remove outliers
    pt[ np.where(np.abs(pt>500)) ] = 0.
    px[ np.where(np.abs(px>500)) ] = 0.
    py[ np.where(np.abs(py>500)) ] = 0.

    inputs = np.concatenate((pt, eta, phi, puppi), axis=2)
    pxpy = np.concatenate((px, py), axis=2)

    inputs_cat0 = A[:,:,6:7] # encoded PF pdgId
    inputs_cat1 = A[:,:,7:8] # encoded PF charge

    return inputs, pxpy, inputs_cat0, inputs_cat1

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
    
    nbins = 20
    binnings = np.linspace(0, 400, num=nbins+1) # create 20 bins for pt from 0 to 400 GeV
    phiBinnings = np.linspace(-3.15,3.15, num =nbins+1)
    truth_means, bin_edges, binnumber = binned_statistic(true_ptPhi[:,0], true_ptPhi[:,0], statistic='mean', bins=binnings, range=(0,400))
    ml_means,  _, _ = binned_statistic(true_ptPhi[:,0], ml_ptPhi[:,0],
                        statistic='mean', bins=binnings, range=(0,400))
    puppi_means, _, _ = binned_statistic(true_ptPhi[:,0], puppi_ptPhi[:,0],
                        statistic='mean', bins=binnings, range=(0,400))
    
    # plot response
    plt.figure()
    plt.hlines(truth_means/truth_means, bin_edges[:-1], bin_edges[1:], colors='k', lw=5,
           label='Truth', linestyles='solid')
    plt.hlines(ml_means/truth_means, bin_edges[:-1], bin_edges[1:], colors='r', lw=5,
           label='ML', linestyles='solid')
    plt.hlines(puppi_means/truth_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
           label='PUPPI', linestyles='solid')
    plt.xlim(0,400.0)
    plt.ylim(0,1.1)
    plt.xlabel('Truth MET [GeV]')
    plt.legend(loc='lower right')
    plt.ylabel('<MET Estimation>/<MET Truth>')
    plt.savefig(f"{path_out}MET_response.png")
    plt.close()
    
    #width of a distribution at 1 standard deviation
    def resolqt(y):
        return(np.percentile(y,84)-np.percentile(y,16))/2.0
    
    # response correction factors
    responseCorrection_ml = np.take(ml_means/truth_means,  np.digitize(true_ptPhi[:,0], binnings)-1, mode='clip')
    responseCorrection_puppi = np.take(puppi_means/truth_means, np.digitize(true_ptPhi[:,0], binnings)-1, mode='clip')

    # compute resolutions inside all 20 bins
    bin_resolX_ml, bin_edges, binnumber = binned_statistic(true_ptPhi[:,0], trueXY[:,0] - mlXY[:,0] * responseCorrection_ml,
                                            statistic=resolqt, bins=binnings, range=(0,400))
    bin_resolY_ml, _, _ = binned_statistic(true_ptPhi[:,0], trueXY[:,1] - mlXY[:,1] * responseCorrection_ml,
                            statistic=resolqt, bins=binnings, range=(0,400))
    bin_resolPt_ml,_,_ = binned_statistic(true_ptPhi[:,0], true_ptPhi[:,0] - ml_ptPhi[:,0] * responseCorrection_ml,
                            statistic=resolqt, bins=binnings, range=(0,400))
    bin_resolPhi_ml,bin_edgesPhi, binnumberPhi = binned_statistic(true_ptPhi[:,1], true_ptPhi[:,1] - ml_ptPhi[:,1],
                            statistic=resolqt, bins=phiBinnings, range=(-3.15,3.15))
    
    bin_resolX_puppi, _, _ = binned_statistic(true_ptPhi[:,0], trueXY[:,0] - puppiXY[:,0] *responseCorrection_puppi,
                                statistic=resolqt, bins=binnings, range=(0,400))
    bin_resolY_puppi, _, _ = binned_statistic(true_ptPhi[:,0], trueXY[:,1] - puppiXY[:,1] *responseCorrection_puppi,
                                statistic=resolqt, bins=binnings, range=(0,400))
    bin_resolPt_puppi,_,_ = binned_statistic(true_ptPhi[:,0], true_ptPhi[:,0] - puppi_ptPhi[:,0] * responseCorrection_puppi,
                            statistic=resolqt, bins=binnings, range=(0,400))
    bin_resolPhi_puppi,_,_ = binned_statistic(true_ptPhi[:,1], true_ptPhi[:,1] - puppi_ptPhi[:,1],
                            statistic=resolqt, bins=phiBinnings, range=(-3.15,3.15))

    # calclate the resolution "magnitude" inside all 20 bins
    bin_resolXYmagnitude_ml = (bin_resolX_ml**2+bin_resolY_ml**2)**.5
    bin_resolXYmagnitude_puppi = (bin_resolX_puppi**2+bin_resolY_puppi**2)**.5
    bin_resolXYmagnitude_difference = bin_resolXYmagnitude_puppi - bin_resolXYmagnitude_ml
    
    # transverse MET resolution difference
    bin_resolPt_difference = bin_resolPt_puppi - bin_resolPt_ml
    
    # compute the resolution over the entire dataset (1 bin)
    average_xRes_ml = resolqt(trueXY[:,0] - mlXY[:,0] * responseCorrection_ml)
    average_yRes_ml = resolqt(trueXY[:,1] - mlXY[:,1] * responseCorrection_ml)
    average_ptRes_ml = resolqt(true_ptPhi[:,0] - ml_ptPhi[:,0] * responseCorrection_ml)
    
    average_xRes_puppi = resolqt(trueXY[:,0] - puppiXY[:,0] * responseCorrection_puppi)
    average_yRes_puppi = resolqt(trueXY[:,1] - puppiXY[:,1] *responseCorrection_puppi)
    average_ptRes_puppi = resolqt(true_ptPhi[:,0] - puppi_ptPhi[:,0] * responseCorrection_puppi)

    # and the resolution "magnitudes" and the corresponding difference between the puppi and ml predictions
    averageXYmag_Res_puppi = (average_xRes_puppi**2+average_yRes_puppi**2)**(.5)
    averageXYmag_Res_ml = (average_xRes_ml**2+average_yRes_ml**2)**(.5)
    
    averageXYmag_Res_difference = averageXYmag_Res_puppi-averageXYmag_Res_ml
    averagePt_Res_difference = average_ptRes_puppi - average_ptRes_ml
    # these two similar metrics can be used to compare the performance across trainings
    # for now, i will compute both

    # the square root of the number of events in each bin
    rootN=[]
    for bin in range(nbins):
        nEvents_inBin = len(binnumber[binnumber==bin+1])
        rootN.append((nEvents_inBin)**(.5))
    # is used to calculate the error bars for each bin = res/rootN
    
    #locations of error bars
    binWidth = binnings[1] # =20
    # +8 and +12 put the error bars slightly off the center of the horizontal lines
    binCenter = binWidth/2
    leftOfBinCenter = .4*binWidth # =8
    rightOfBinCenter = .6*binWidth # =12
    
    # plot x resolution 20 bins
    plt.figure(figsize=(10,8))
    plt.hlines(bin_resolX_ml, bin_edges[:-1], bin_edges[1:], colors='r', lw=3,
               label='ML', linestyles='solid')
    plt.hlines(bin_resolX_puppi, bin_edges[:-1], bin_edges[1:], colors='g', lw=3,
               label='PUPPI', linestyles='solid')
    plt.errorbar(bin_edges[:-1]+rightOfBinCenter, bin_resolX_ml,
                 yerr= bin_resolX_ml/rootN, fmt='none', color='r')
    plt.errorbar(bin_edges[:-1]+leftOfBinCenter, bin_resolX_puppi,
                 yerr= bin_resolX_puppi/rootN, fmt='none', color='g')
    plt.legend(loc='lower right')
    plt.xlim(0,400.0)
    plt.ylim(0,200)
    plt.xlabel('Truth MET [GeV]')
    plt.ylabel('RespCorr $\sigma$(METX) [GeV]')
    plt.title('METx Resolution', fontsize = 22)
    plt.savefig(f"{path_out}resolution_metx.png")

    # plot y resolutions 20 bins
    plt.figure(figsize=(10,8))
    plt.hlines(bin_resolY_ml, bin_edges[:-1], bin_edges[1:], colors='r', lw=3,
               label='ML', linestyles='solid')
    plt.hlines(bin_resolY_puppi, bin_edges[:-1], bin_edges[1:], colors='g', lw=3,
               label='PUPPI', linestyles='solid')
    plt.errorbar(bin_edges[:-1]+rightOfBinCenter, bin_resolY_ml,
                 yerr= bin_resolY_ml/rootN, fmt='none', color='r')
    plt.errorbar(bin_edges[:-1]+leftOfBinCenter, bin_resolY_puppi,
                 yerr= bin_resolY_puppi/rootN, fmt='none', color='g')
    plt.legend(loc='lower right')
    plt.xlim(0,400.0)
    plt.ylim(0,200.0)
    plt.xlabel('Truth MET [GeV]')
    plt.ylabel('RespCorr $\sigma$(METY) [GeV]')
    plt.title('METy Resolution', fontsize = 22)
    plt.savefig(f"{path_out}resolution_mety.png")
    
    # plot pt resolutions 20 bins
    plt.figure(figsize=(10,8))
    plt.hlines(bin_resolPt_ml, bin_edges[:-1], bin_edges[1:], colors='r', lw=3,
               label='ML', linestyles='solid')
    plt.hlines(bin_resolPt_puppi, bin_edges[:-1], bin_edges[1:], colors='g', lw=3,
               label='PUPPI', linestyles='solid')
    plt.errorbar(bin_edges[:-1]+rightOfBinCenter, bin_resolPt_ml,
                 yerr= bin_resolPt_ml/rootN, fmt='none', color='r')
    plt.errorbar(bin_edges[:-1]+leftOfBinCenter, bin_resolPt_puppi,
                 yerr= bin_resolPt_puppi/rootN, fmt='none', color='g')
    plt.legend(loc='lower right')
    plt.xlim(0,400.0)
    plt.ylim(0,200.0)
    plt.xlabel('Truth MET [GeV]')
    plt.ylabel('RespCorr $\sigma$(MET) [GeV]')
    plt.title('MET (Pt) Resolution', fontsize = 22)
    plt.savefig(f"{path_out}resolution_met_pt.png")
    
    # plot phi resolutions 20 bins
    plt.figure(figsize=(10,8))
    plt.hlines(bin_resolPhi_ml, bin_edgesPhi[:-1], bin_edgesPhi[1:], colors='r', lw=3,
               label='ML', linestyles='solid')
    plt.hlines(bin_resolPhi_puppi, bin_edgesPhi[:-1], bin_edgesPhi[1:], colors='g', lw=3,
               label='PUPPI', linestyles='solid')
    plt.errorbar(bin_edgesPhi[:-1]+.13, bin_resolPhi_ml,
                 yerr= bin_resolPhi_ml/rootN, fmt='none', color='r')
    plt.errorbar(bin_edgesPhi[:-1]+.17, bin_resolPhi_puppi,
                 yerr= bin_resolPhi_puppi/rootN, fmt='none', color='g')
    plt.legend(loc='lower right')
    plt.xlim(-3.3,3.3)
    plt.ylim(0,5)
    plt.xlabel('Truth MET [GeV]')
    plt.ylabel('RespCorr $\sigma$(MET) Phi')
    plt.title('MET (Phi) Resolution', fontsize = 22)
    plt.savefig(f"{path_out}resolution_met_phi.png")
    
    # plot resolution (both XY magnitude and pt) differences
    plt.figure(figsize=(10,8))
    plt.hlines(bin_resolXYmagnitude_difference, bin_edges[:-1], bin_edges[1:], lw=5, linestyles='solid', label='XYmag res difference' )
    plt.hlines(bin_resolPt_difference, bin_edges[:-1], bin_edges[1:], lw=5, linestyles='solid', color='r', label='pt res difference')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.ylim(-20,20)
    plt.xlabel('Truth MET [GeV]')
    plt.ylabel('PUPPI - ML $\sigma$(MET) [GeV]')
    plt.legend(loc='lower left')
    plt.text(0, -10, f'average XY magnitude resolution difference ={round(averageXYmag_Res_difference,3)}', fontsize=13)
    plt.text(0, -12, f'average Pt resolution difference ={round(averagePt_Res_difference,3)}', fontsize=13)
    plt.title(f'Resolution Differences (PUPPI - ML)', fontsize = 16)
    plt.savefig(f"{path_out}resolution_metDif.png")

def to_np_array(ak_array, maxN=100, pad=0):
    return ak.fill_none(ak.pad_none(ak_array,maxN,clip=True,axis=-1),pad).to_numpy()
