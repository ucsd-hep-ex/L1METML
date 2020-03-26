from ROOT import *
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
                hists["predict_para_"+str(int(j*binning_le))+"-"+str(int((j+1)*binning_le))+""].Fill(v_para_predict.Mod())
                hists["predict_perp_"+str(int(j*binning_le))+"-"+str(int((j+1)*binning_le))+""].Fill(v_perp_predict.Mod())
        for j in range(int(bin_/2.)):
            if (j*binning_gr+bin_medi < v_gen.Mod()) and (v_gen.Mod() <= (j+1)*binning_gr+bin_medi):
                hists["v_gen_"+str(int(j*binning_gr+bin_medi))+"-"+str(int((j+1)*binning_gr+bin_medi))+""].Fill(v_gen.Mod())
                hists["predict_para_"+str(int(j*binning_gr+bin_medi))+"-"+str(int((j+1)*binning_gr+bin_medi))+""].Fill(v_para_predict.Mod())
                hists["predict_perp_"+str(int(j*binning_gr+bin_medi))+"-"+str(int((j+1)*binning_gr+bin_medi))+""].Fill(v_perp_predict.Mod())

    fout = TFile(name, "recreate")
    for n in hists:
        hists[n].Write()



def MET_rel_error(predict_met, gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)/np.clip(gen_met,0.1, 1000)
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-3., 3., 50+1))
    plt.xlabel("rel error (predict - true)/true")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)



def MET_abs_error(predict_met,gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-500., 500., 50+1))
    plt.xlabel("abs error (predict - true)")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)



def Phi_abs_error(predict_met,gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-3.5, 3.5, 50+1))
    plt.xlabel("abs error (predict - true)")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)

def dist(predict_met, name='dist.pdf'):
    rel_err = predict_met
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(0, 500, 50+1))
    plt.xlabel("MET [GeV]")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)
    #plt.show()

def dist_xy(predict_met, name='dist.pdf'):
    rel_err = predict_met
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-500, 500, 50+1))
    plt.xlabel("MET [GeV]")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)
    #plt.show()
