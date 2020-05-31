from ROOT import *
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def Write_MET_binned_histogram(Predict_array, Gen_array, bin_number, bin_minimum, bin_median, bin_maximum, name='title'):
    
    # book histograms
    def book(h,name,n,a,b,title=""):
        h[name]=TH1F(name,title,n,a,b)
        h[name].Sumw2()
    def book2(h,name,nx,ax,bx,ny,ay,by,title=""):
        h[name]=TH2F(name,title,nx,ax,bx,ny,ay,by)
        h[name].Sumw2()
    def SgnPara(para,z):
        return para.Mod()*(1. if abs(para.DeltaPhi(z))>TMath.Pi()/2 else -1.)
    def SgnPerp(perp,z):
        return perp.Mod()*(1. if perp.DeltaPhi(z)>0 else -1.)


    # Separate MET-bin
    bin_ = bin_number
    bin_mini = bin_minimum
    bin_medi = bin_median
    bin_maxi = bin_maximum

    binning_le = (bin_medi - bin_mini)/(bin_/2.)
    binning_gr = (bin_maxi - bin_medi)/(bin_/2.)
    hists=OrderedDict()

    for i in range(int(bin_/2.)):
        book(hists, "predict_perp_"+str(int(i*binning_le))+"-"+str(int((i+1)*binning_le))+"", 100, -bin_maxi, bin_maxi)
        book(hists, "predict_para_"+str(int(i*binning_le))+"-"+str(int((i+1)*binning_le))+"", 100, -bin_maxi, bin_maxi)
        book(hists, "v_gen_"+str(int(i*binning_le))+"-"+str(int((i+1)*binning_le))+"", 1000, i*binning_le, (i+1)*binning_le)

    for i in range(int(bin_/2.)):
        book(hists, "predict_perp_"+str(int(i*binning_gr+bin_medi))+"-"+str(int((i+1)*binning_gr+bin_medi))+"", 100, -bin_maxi, bin_maxi)
        book(hists, "predict_para_"+str(int(i*binning_gr+bin_medi))+"-"+str(int((i+1)*binning_gr+bin_medi))+"", 100, -bin_maxi, bin_maxi)
        book(hists, "v_gen_"+str(int(i*binning_gr+bin_medi))+"-"+str(int((i+1)*binning_gr+bin_medi))+"", 1000, i*binning_gr+bin_medi, (i+1)*binning_gr+bin_medi)

    v_gen =	TVector2()
    v_para_PUPPI = TVector2()
    v_perp_PUPPI = TVector2()
    v_predict = TVector2()
    v_para_predict = TVector2()
    v_perp_predict = TVector2()
    
    test_events = Predict_array.shape[0]
    
    for i in range(test_events):
        v_predict.SetMagPhi(Predict_array[i,0], Predict_array[i,1])
        v_gen.SetMagPhi(Gen_array[i,0], Gen_array[i,1])
        v_para_predict = v_predict.Proj(v_gen)
        v_perp_predict = v_predict.Norm(v_gen)
        for j in range(int(bin_/2.)):
            if (j*binning_le < v_gen.Mod()) and (v_gen.Mod() <= (j+1)*binning_le):
                hists["v_gen_"+str(int(j*binning_le))+"-"+str(int((j+1)*binning_le))+""].Fill(v_gen.Mod())
                hists["predict_para_"+str(int(j*binning_le))+"-"+str(int((j+1)*binning_le))+""].Fill(SgnPara(v_para_predict, v_gen))
                hists["predict_perp_"+str(int(j*binning_le))+"-"+str(int((j+1)*binning_le))+""].Fill(SgnPerp(v_perp_predict, v_gen))
        for j in range(int(bin_/2.)):
            if (j*binning_gr+bin_medi < v_gen.Mod()) and (v_gen.Mod() <= (j+1)*binning_gr+bin_medi):
                hists["v_gen_"+str(int(j*binning_gr+bin_medi))+"-"+str(int((j+1)*binning_gr+bin_medi))+""].Fill(v_gen.Mod())
                hists["predict_para_"+str(int(j*binning_gr+bin_medi))+"-"+str(int((j+1)*binning_gr+bin_medi))+""].Fill(SgnPara(v_para_predict, v_gen))
                hists["predict_perp_"+str(int(j*binning_gr+bin_medi))+"-"+str(int((j+1)*binning_gr+bin_medi))+""].Fill(SgnPerp(v_perp_predict, v_gen))

    fout = TFile(name, "recreate")
    for n in hists:
        hists[n].Write()



def MET_rel_error_bad(predict_met, gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)/gen_met

    mask = (rel_err[:] < 3)
    rel_err = rel_err[~mask]


    mean = np.mean(rel_err)
    std = np.std(rel_err)

    entry = rel_err.shape[0]
    #for i in range(rel_err.shape[0]):
    #    std += (mean - rel_err[i]) **2

    #std = std/rel_err.shape[0]
    #std = math.sqrt(std)

    mean = mean * 1000
    mean = int(mean)
    mean = float(mean) / 1000
    std = std * 1000
    std = int(std)
    std = float(std) / 1000

    plt.figure()
    plt.hist(rel_err, bins=np.linspace(3., 50., 50+1), label='mean : '+str(mean)+'\nstandard deviation : '+str(std)+'\nentry : '+str(entry)+'')
    plt.xlabel("relative error (predict - true)/true", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.legend()
    plt.savefig(name)
    plt.show()




def MET_rel_error(predict_met, gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)/gen_met

    mask = (rel_err[:] > 3)
    rel_err = rel_err[~mask]


    mean = np.mean(rel_err)
    std = np.std(rel_err)

    entry = rel_err.shape[0]

    mean = mean * 1000
    mean = int(mean)
    print(mean)
    mean = float(mean) / 1000
    std = std * 1000
    std = int(std)
    std = float(std) / 1000

    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-3., 3., 50+1), label='mean : '+str(mean)+'\nstandard deviation : '+str(std)+'\nentry : '+str(entry)+'')
    plt.xlabel("relative error (predict - true)/true", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.legend()
    plt.savefig(name)
    plt.show()




def MET_abs_error(predict_met, gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)

    mask = gen_met[:] > 100
    rel_err = rel_err[~mask]

    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-150., 150., 50+1))
    plt.xlabel("abs error (predict - true)")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)
    plt.show()



def Phi_abs_error(predict_met, gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-3.5, 3.5, 50+1))
    plt.xlabel("abs error (predict - true)")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)
    plt.show()

def dist(predict_met, name='dist.pdf'):
    rel_err = predict_met
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(0, 500, 50+1))
    plt.xlabel("MET [GeV]")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)
    plt.show()

def dist_xy(predict_met, name='dist.pdf'):
    rel_err = predict_met
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-500, 500, 50+1))
    plt.xlabel("MET [GeV]")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)
    plt.show()

def MET_binned_predict_mean(predict_met, gen_met, binning, mini, maxi, genMET_cut, corr_check, name='predict_mean.pdf'):
    bin_ = (maxi - mini)/binning
    X_genMET = np.zeros(bin_)
    X_error = np.zeros(bin_)
    y_predict = np.zeros(bin_)
    y_error = np.zeros(bin_)
    entry = np.zeros(bin_)

    for i in range(predict_met.shape[0]):
        for j in range(bin_):
            if ((j * binning) <= gen_met[i] < ((j + 1) * binning)):
                X_genMET[j] += gen_met[i]
                y_predict[j] += predict_met[i]
                entry[j] += 1
                break

    X_genMET = X_genMET/entry
    y_predict = y_predict/entry

    for i in range(predict_met.shape[0]):
        for j in range(bin_):
            if ((j * binning) <= gen_met[i] < ((j + 1) * binning)):
                X_error[j] += (gen_met[i] - X_genMET[j]) ** 2
                y_error[j] += (predict_met[i] - y_predict[j]) ** 2
                break

    X_error = np.sqrt(X_error/entry)
    y_error = np.sqrt(y_error/entry)

    plt.errorbar(X_genMET, y_predict, xerr = X_error, yerr = y_error, label='cut = '+str(genMET_cut)+', '+str(corr_check)+'.')

    ## x = y plot
    X = np.arange(mini, maxi, binning)
    plt.plot(X, X, 'r-')
    ##

    plt.xlim(mini, maxi)
    plt.ylim(mini, 700)
    plt.xlabel('Gen MET mean [GeV]', fontsize = 16)
    #plt.ylabel('PUPPI MET mean [GeV]', fontsize = 16)
    plt.ylabel('predicted MET mean [GeV]', fontsize = 16)
    plt.legend()
    plt.savefig(name)
    plt.show()


def MET_binned_predict_ratio(predict_met, gen_met, binning, mini, maxi, genMET_cut, comment, name='predict_mean.pdf'):
    bin_ = (maxi - mini)/binning
    X_genMET = np.zeros(bin_)
    X_error = np.zeros(bin_)
    y_predict = np.zeros(bin_)
    y_error = np.zeros(bin_)
    entry = np.zeros(bin_)

    for i in range(predict_met.shape[0]):
        for j in range(bin_):
            if ((j * binning) <= gen_met[i] < ((j + 1) * binning)):
                X_genMET[j] += gen_met[i]
                y_predict[j] += predict_met[i]/gen_met[i]
                entry[j] += 1
                break

    X_genMET = X_genMET/entry
    y_predict = y_predict/entry

    for i in range(predict_met.shape[0]):
        for j in range(bin_):
            if ((j * binning) <= gen_met[i] < ((j + 1) * binning)):
                X_error[j] += (gen_met[i] - X_genMET[j]) ** 2
                y_error[j] += (predict_met[i]/gen_met[i] - y_predict[j]) ** 2
                break

    X_error = np.sqrt(X_error/entry)
    y_error = np.sqrt(y_error/entry)

    plt.errorbar(X_genMET, y_predict, xerr = X_error, yerr = y_error, label='cut = '+str(genMET_cut)+', '+str(comment)+'.')

    ## y = 1 plot
    X = np.arange(mini, maxi, binning)
    y = np.zeros(bin_)
    y[:] = 1
    plt.plot(X, y, 'r-')
    ##

    plt.xlim(mini, maxi)
    plt.ylim(mini, 3)
    plt.xlabel('Gen MET mean [GeV]', fontsize = 16)
    plt.ylabel('(predicted MET/Gen MET) mean [GeV]', fontsize = 16)
    plt.legend()
    plt.savefig(name)
    plt.show()


def extract_result(feat_array, targ_array, path, genMET_cut, max_genMET_cut):
    feat = open(''+path+'feature_array_MET_'+str(genMET_cut)+'-'+str(max_genMET_cut)+'.txt', 'w')
    for i in range(feat_array.shape[0]):
        data = '%f' %feat_array[i,0]
        feat.write(data)
        feat.write('\n')
    feat_phi = open(''+path+'feature_array_phi_'+str(genMET_cut)+'-'+str(max_genMET_cut)+'.txt', 'w')
    for i in range(feat_array.shape[0]):
        data = '%f' %feat_array[i,1]
        feat_phi.write(data)
        feat_phi.write('\n')
    targ = open(''+path+'target_array_MET_'+str(genMET_cut)+'-'+str(max_genMET_cut)+'.txt', 'w')
    for i in range(feat_array.shape[0]):
        data = '%f' %targ_array[i,0]
        targ.write(data)
        targ.write('\n')
    targ_phi = open(''+path+'target_array_phi_'+str(genMET_cut)+'-'+str(max_genMET_cut)+'.txt', 'w')
    for i in range(feat_array.shape[0]):
        data = '%f' %targ_array[i,1]
        targ_phi.write(data)
        targ_phi.write('\n')
    feat.close()
    targ.close()
    feat_phi.close()
    targ_phi.close()


def histo_2D(predict_pT, gen_pT, name = '2D_histo.png'):
    X_hist = np.arange(0,500, 20)
    Y_hist = 2.*X_hist#1.25*X_hist
    #Y_hist_1 = 0.75*X_hist
    plt.plot(X_hist, Y_hist, '-r')
    #plt.plot(X_hist, Y_hist_1, '-r')
    x_bins = np.linspace(0, 500, 50)
    y_bins = np.linspace(0, 500, 50)
    plt.hist2d(gen_pT, predict_pT,  bins=[x_bins, y_bins], cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('gen MET [GeV]')
    plt.ylabel('predicted MET [GeV]')
    plt.savefig(name)
    plt.show()
