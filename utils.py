import math
import awkward as ak
import numpy as np


def read_input(inputfiles):
    import h5py
    for i, fname in enumerate(inputfiles):
        print("read file", fname)
        with h5py.File(fname, 'r') as h5f:
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
    arrayPtPhi[:, 0] = np.sqrt((arrayXY[:, 0]**2 + arrayXY[:, 1]**2))
    arrayPtPhi[:, 1] = np.arctan2(arrayXY[:, 1], arrayXY[:, 0])
    return arrayPtPhi


def preProcessing(A, normFac, EVT=None):
    """ pre-processing input """

    norm = normFac

    pt = A[:, :, 0:1] / norm
    px = A[:, :, 1:2] / norm
    py = A[:, :, 2:3] / norm
    eta = A[:, :, 3:4]
    phi = A[:, :, 4:5]
    puppi = A[:, :, 5:6]

    # remove outliers
    pt[np.where(np.abs(pt > 500/norm))] = 0.
    px[np.where(np.abs(px > 500/norm))] = 0.
    py[np.where(np.abs(py > 500/norm))] = 0.

    inputs = np.concatenate((pt, eta, phi, puppi), axis=2)
    pxpy = np.concatenate((px, py), axis=2)

    inputs_cat0 = A[:, :, 6]  # encoded PF pdgId
    inputs_cat1 = A[:, :, 7]  # encoded PF charge

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

    Make1DHists(trueXY[:, 0], mlXY[:, 0], puppiXY[:, 0], -400, 400, 40, False, 'MET X [GeV]', 'A.U.', f'{path_out}MET_x.png')
    Make1DHists(trueXY[:, 1], mlXY[:, 1], puppiXY[:, 1], -400, 400, 40, False, 'MET Y [GeV]', 'A.U.', f'{path_out}MET_y.png')
    Make1DHists(true_ptPhi[:, 0], ml_ptPhi[:, 0], puppi_ptPhi[:, 0], 0, 400, 40, False, 'MET Pt [GeV]', 'A.U.', f'{path_out}MET_pt.png')

    # do statistics
    from scipy.stats import binned_statistic

    nbins = 20
    binnings = np.linspace(0, 400, num=nbins+1)  # create 20 bins for pt from 0 to 400 GeV
    phiBinnings = np.linspace(-3.15, 3.15, num=nbins+1)
    truth_means, bin_edges, binnumber = binned_statistic(true_ptPhi[:, 0], true_ptPhi[:, 0], statistic='mean', bins=binnings, range=(0, 400))
    ml_means,  _, _ = binned_statistic(true_ptPhi[:, 0], ml_ptPhi[:, 0],
                                       statistic='mean', bins=binnings, range=(0, 400))
    puppi_means, _, _ = binned_statistic(true_ptPhi[:, 0], puppi_ptPhi[:, 0],
                                         statistic='mean', bins=binnings, range=(0, 400))

    # plot response
    plt.figure()
    plt.hlines(truth_means/truth_means, bin_edges[:-1], bin_edges[1:], colors='k', lw=5,
               label='Truth', linestyles='solid')
    plt.hlines(ml_means/truth_means, bin_edges[:-1], bin_edges[1:], colors='r', lw=5,
               label='ML', linestyles='solid')
    plt.hlines(puppi_means/truth_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
               label='PUPPI', linestyles='solid')
    plt.xlim(0, 400.0)
    plt.ylim(0, 1.1)
    plt.xlabel('Truth MET [GeV]')
    plt.legend(loc='lower right')
    plt.ylabel('<MET Estimation>/<MET Truth>')
    plt.savefig(f"{path_out}MET_response.png")
    plt.close()

    # width of a distribution at 1 standard deviation
    def resolqt(y):
        return (np.percentile(y, 84)-np.percentile(y, 16))/2.0

    # response correction factors
    # the events are split into 20 bins based on true_pt and get assigned the corresponding `truth_means/ml_means` of all events in that bin
    # when ml/puppi_pt is scaled by this response correction, the mean of this distributiobn will coincide with the mean of true_pt
    responseCorrection_ml = np.take(truth_means/ml_means,  np.digitize(true_ptPhi[:, 0], binnings)-1, mode='clip')
    responseCorrection_puppi = np.take(truth_means/puppi_means, np.digitize(true_ptPhi[:, 0], binnings)-1, mode='clip')

    # Phi calculation
    Phi_diff_ml = true_ptPhi[:, 1] - ml_ptPhi[:, 1]
    Phi_diff_ml = np.where(Phi_diff_ml < -math.pi, Phi_diff_ml + 2*math.pi, Phi_diff_ml)
    Phi_diff_ml = np.where(Phi_diff_ml > math.pi, Phi_diff_ml - 2*math.pi, Phi_diff_ml)

    Phi_diff_puppi = true_ptPhi[:, 1] - puppi_ptPhi[:, 1]
    Phi_diff_puppi = np.where(Phi_diff_puppi < -math.pi, Phi_diff_puppi + 2*math.pi, Phi_diff_puppi)
    Phi_diff_puppi = np.where(Phi_diff_puppi > math.pi, Phi_diff_puppi - 2*math.pi, Phi_diff_puppi)

    # compute resolutions inside all 20 bins
    # the dirstribution true_pt - respCor*ml/puppi_pt should be centered at 0
    bin_resolX_ml, bin_edges, binnumber = binned_statistic(true_ptPhi[:, 0], trueXY[:, 0] - mlXY[:, 0] * responseCorrection_ml,
                                                           statistic=resolqt, bins=binnings, range=(0, 400))
    bin_resolY_ml, _, _ = binned_statistic(true_ptPhi[:, 0], trueXY[:, 1] - mlXY[:, 1] * responseCorrection_ml,
                                           statistic=resolqt, bins=binnings, range=(0, 400))
    bin_resolPt_ml, _, _ = binned_statistic(true_ptPhi[:, 0], true_ptPhi[:, 0] - ml_ptPhi[:, 0] * responseCorrection_ml,
                                            statistic=resolqt, bins=binnings, range=(0, 400))
    bin_resolPhi_ml, bin_edgesPhi, binnumberPhi = binned_statistic(true_ptPhi[:, 1], Phi_diff_ml,
                                                                   statistic=resolqt, bins=phiBinnings, range=(-3.15, 3.15))

    bin_resolX_puppi, _, _ = binned_statistic(true_ptPhi[:, 0], trueXY[:, 0] - puppiXY[:, 0] * responseCorrection_puppi,
                                              statistic=resolqt, bins=binnings, range=(0, 400))
    bin_resolY_puppi, _, _ = binned_statistic(true_ptPhi[:, 0], trueXY[:, 1] - puppiXY[:, 1] * responseCorrection_puppi,
                                              statistic=resolqt, bins=binnings, range=(0, 400))
    bin_resolPt_puppi, _, _ = binned_statistic(true_ptPhi[:, 0], true_ptPhi[:, 0] - puppi_ptPhi[:, 0] * responseCorrection_puppi,
                                               statistic=resolqt, bins=binnings, range=(0, 400))
    bin_resolPhi_puppi, _, _ = binned_statistic(true_ptPhi[:, 1], Phi_diff_puppi,
                                                statistic=resolqt, bins=phiBinnings, range=(-3.15, 3.15))

    # calclate the resolution "magnitude" inside all 20 bins
    bin_resolXYmagnitude_ml = (bin_resolX_ml**2+bin_resolY_ml**2)**.5
    bin_resolXYmagnitude_puppi = (bin_resolX_puppi**2+bin_resolY_puppi**2)**.5
    bin_resolXYmagnitude_difference = bin_resolXYmagnitude_puppi - bin_resolXYmagnitude_ml

    # transverse MET resolution difference
    bin_resolPt_difference = bin_resolPt_puppi - bin_resolPt_ml

    # compute the resolution over the entire dataset (1 bin)
    average_xRes_ml = resolqt(trueXY[:, 0] - mlXY[:, 0] * responseCorrection_ml)
    average_yRes_ml = resolqt(trueXY[:, 1] - mlXY[:, 1] * responseCorrection_ml)
    average_ptRes_ml = resolqt(true_ptPhi[:, 0] - ml_ptPhi[:, 0] * responseCorrection_ml)

    average_xRes_puppi = resolqt(trueXY[:, 0] - puppiXY[:, 0] * responseCorrection_puppi)
    average_yRes_puppi = resolqt(trueXY[:, 1] - puppiXY[:, 1] * responseCorrection_puppi)
    average_ptRes_puppi = resolqt(true_ptPhi[:, 0] - puppi_ptPhi[:, 0] * responseCorrection_puppi)

    # and the resolution "magnitudes" and the corresponding difference between the puppi and ml predictions
    averageXYmag_Res_puppi = (average_xRes_puppi**2+average_yRes_puppi**2)**(.5)
    averageXYmag_Res_ml = (average_xRes_ml**2+average_yRes_ml**2)**(.5)

    averageXYmag_Res_difference = averageXYmag_Res_puppi-averageXYmag_Res_ml
    averagePt_Res_difference = average_ptRes_puppi - average_ptRes_ml
    # these two similar metrics can be used to compare the performance across trainings
    # for now, i will compute both

    # the square root of the number of events in each bin
    nEvents_inBin, _ = np.histogram(binnumber, bins=nbins, range=(1, nbins))
    rootN = np.sqrt(nEvents_inBin)
    nEvents_inBin_phi, _ = np.histogram(binnumberPhi, bins=nbins, range=(1, nbins))
    rootN_phi = np.sqrt(nEvents_inBin_phi)
    # is used to calculate the error bars for each bin = res/rootN

    # locations of error bars
    binWidth = binnings[1]  # =20
    # +8 and +12 put the error bars slightly off the center of the horizontal lines
    binCenter = binWidth/2
    leftOfBinCenter = .4*binWidth  # =8
    rightOfBinCenter = .6*binWidth  # =12

    fig1 = plt.figure(figsize=(15, 12), tight_layout=True)
    fig2 = plt.figure(figsize=(15, 12), tight_layout=True)
    plt.subplots_adjust(wspace=.2,
                        hspace=0)

    # plot x resolution 20 bins
    ax11 = fig1.add_subplot(2, 2, 1)
    ax11.hlines(bin_resolX_ml, bin_edges[:-1], bin_edges[1:], colors='r', lw=3,
                label='ML', linestyles='solid')
    ax11.hlines(bin_resolX_puppi, bin_edges[:-1], bin_edges[1:], colors='g', lw=3,
                label='PUPPI', linestyles='solid')
    ax11.errorbar(bin_edges[:-1]+rightOfBinCenter, bin_resolX_ml,
                  yerr=bin_resolX_ml/rootN, fmt='none', color='r')
    ax11.errorbar(bin_edges[:-1]+leftOfBinCenter, bin_resolX_puppi,
                  yerr=bin_resolX_puppi/rootN, fmt='none', color='g')
    ax11.grid()
    ax11.set_ylabel(r'$\sigma(MET)$ [GeV]', fontsize=19)
    ax11.set_title('MET-x Resolution', fontsize=22)

    # plot y resolutions 20 bins
    ax12 = fig1.add_subplot(2, 2, 2, sharey=ax11)
    ax12.hlines(bin_resolY_ml, bin_edges[:-1], bin_edges[1:], colors='r', lw=3,
                label='$\\sigma_{ML}$', linestyles='solid')
    ax12.hlines(bin_resolY_puppi, bin_edges[:-1], bin_edges[1:], colors='g', lw=3,
                label='$\\sigma_{PUPPI}$', linestyles='solid')
    ax12.errorbar(bin_edges[:-1]+rightOfBinCenter, bin_resolY_ml,
                  yerr=bin_resolY_ml/rootN, fmt='none', color='r')
    ax12.errorbar(bin_edges[:-1]+leftOfBinCenter, bin_resolY_puppi,
                  yerr=bin_resolY_puppi/rootN, fmt='none', color='g')
    ax12.legend(loc='upper center', prop={'size': 19})
    ax12.grid()
    ax12.set_title('MET-y Resolution', fontsize=22)

    # plot resolution XY magnitude absolute differences
    ax13 = fig1.add_subplot(2, 2, 3, sharex=ax11)
    ax13.hlines(bin_resolXYmagnitude_difference, bin_edges[:-1], bin_edges[1:], lw=5, linestyles='solid')
    ax13.axhline(y=0, color='black', linestyle='-')
    ax13.set_xlabel('Truth MET [GeV]', fontsize=19)
    ax13.set_ylabel(r'$\sigma_{PUPPI} - \sigma_{ML}$ [GeV]', fontsize=19)
    ax13.grid()
    ax13.set_title('Absolute Resolution Differences (PUPPI - ML)', fontsize=22)

    # relative differences
    ax14 = fig1.add_subplot(2, 2, 4, sharex=ax12)
    ax14.hlines(bin_resolXYmagnitude_difference/truth_means, bin_edges[:-1], bin_edges[1:], lw=5, linestyles='solid')
    ax14.axhline(y=0, color='black', linestyle='-')
    ax14.set_ylim(min(bin_resolXYmagnitude_difference/truth_means)-.1, max(bin_resolXYmagnitude_difference/truth_means)+.1)
    ax14.set_xlabel('Truth MET [GeV]', fontsize=19)
    ax14.set_ylabel(r'$(\sigma_{PUPPI} - \sigma_{ML})$ / $\mu_{bin}$', fontsize=19)
    ax14.grid()
    ax14.set_title('Relative Resolution Differences', fontsize=22)

    trainingName = path_out.split('/')[-2]
    fig1.text(0, 1.06, f'training: {trainingName}', fontsize=19)
    fig1.text(0, 1.03, r'$\sigma_{PUPPI} - \sigma_{ML}=\sigma_{xyDIF}$;  $Mean(\sigma_{xyDIF})=$'+f'{round(averageXYmag_Res_difference,3)}', fontsize=19)

    # plot pt resolutions 20 bins
    ax21 = fig2.add_subplot(2, 2, 1)
    ax21.hlines(bin_resolPt_ml, bin_edges[:-1], bin_edges[1:], colors='r', lw=3,
                label='ML', linestyles='solid')
    ax21.hlines(bin_resolPt_puppi, bin_edges[:-1], bin_edges[1:], colors='g', lw=3,
                label='PUPPI', linestyles='solid')
    ax21.errorbar(bin_edges[:-1]+rightOfBinCenter, bin_resolPt_ml,
                  yerr=bin_resolPt_ml/rootN, fmt='none', color='r')
    ax21.errorbar(bin_edges[:-1]+leftOfBinCenter, bin_resolPt_puppi,
                  yerr=bin_resolPt_puppi/rootN, fmt='none', color='g')
    ax21.set_xlabel('truth met [gev]', fontsize=19)
    ax21.set_ylabel(r'$\sigma(MET)$ [GeV]', fontsize=20)
    ax21.grid()
    ax21.set_title('MET-pt Resolution', fontsize=22)

    # plot phi resolutions 20 bins
    ax22 = fig2.add_subplot(2, 2, 2)
    ax22.hlines(bin_resolPhi_ml, bin_edgesPhi[:-1], bin_edgesPhi[1:], colors='r', lw=3,
                label='$\\sigma_{ML}$', linestyles='solid')
    ax22.hlines(bin_resolPhi_puppi, bin_edgesPhi[:-1], bin_edgesPhi[1:], colors='g', lw=3,
                label='$\\sigma_{PUPPI}$', linestyles='solid')
    ax22.errorbar(bin_edgesPhi[:-1]+.13, bin_resolPhi_ml,
                  yerr=bin_resolPhi_ml/rootN_phi, fmt='none', color='r')
    ax22.errorbar(bin_edgesPhi[:-1]+.17, bin_resolPhi_puppi,
                  yerr=bin_resolPhi_puppi/rootN_phi, fmt='none', color='g')
    ax22.set_ylabel('radian', fontsize=20)
    ax22.set_ylim(0., 1.)
    ax22.grid()
    ax22.set_xlabel(r'$\phi$ angle', fontsize=19)
    ax22.legend(loc='upper center', prop={'size': 19})
    ax22.set_title(r'MET-$\Phi$ Resolution', fontsize=20)

    # plot resolution pt absolute differences
    ax23 = fig2.add_subplot(2, 2, 3, sharex=ax11)
    ax23.hlines(bin_resolPt_difference, bin_edges[:-1], bin_edges[1:], lw=5, linestyles='solid')
    ax23.axhline(y=0, color='black', linestyle='-')
    ax23.set_xlabel('Truth MET [GeV]', fontsize=19)
    ax23.set_ylabel(r'$\sigma_{PUPPI} - \sigma_{ML}$ [GeV]', fontsize=20)
    ax23.grid()
    ax23.set_title('Absolute Resolution Differences', fontsize=22)

    # relative differences
    ax24 = fig2.add_subplot(2, 2, 4, sharex=ax12)
    ax24.hlines(bin_resolPt_difference/truth_means, bin_edges[:-1], bin_edges[1:], lw=5, linestyles='solid')
    ax24.axhline(y=0, color='black', linestyle='-')
    ax24.set_ylim(min(bin_resolPt_difference/truth_means)-.1, max(bin_resolPt_difference/truth_means)+.1)
    ax24.set_xlabel('Truth MET [GeV]', fontsize=19)
    ax24.set_ylabel(r'$(\sigma_{PUPPI} - \sigma_{ML})$ / $\mu_{bin}$', fontsize=19)
    ax24.grid()
    ax24.set_title(f'Relative Resolution Differences', fontsize=22)

    fig2.text(0, 1.06, f'training: {trainingName}', fontsize=19)
    fig2.text(0, 1.03, r'$\sigma_{PUPPI} - \sigma_{ML}=\sigma_{DIF}$;  $Mean(\sigma_{DIF})=$'+f'{round(averagePt_Res_difference,3)}', fontsize=19)

    fig1.savefig(f"{path_out}XY_resolution_plots.png", bbox_inches="tight")
    fig2.savefig(f"{path_out}pt_resolution_plots.png", bbox_inches="tight")


def Make1DHists(truth, ML, PUPPI, xmin=0, xmax=400, nbins=100, density=False, xname="pt [GeV]", yname="A.U.", outputname="1ddistribution.png"):
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(10, 8))
    plt.hist(truth,    bins=nbins, range=(xmin, xmax), density=density, histtype='step', facecolor='k', label='Truth')
    plt.hist(ML,  bins=nbins, range=(xmin, xmax), density=density, histtype='step', facecolor='r', label='ML')
    plt.hist(PUPPI, bins=nbins, range=(xmin, xmax), density=density, histtype='step', facecolor='g', label='PUPPI')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(outputname)
    plt.close()


def to_np_array(ak_array, maxN=100, pad=0):
    return ak.fill_none(ak.pad_none(ak_array, maxN, clip=True, axis=-1), pad).to_numpy()
