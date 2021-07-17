import math
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def Write_MET_binned_histogram(Predict_array, Gen_array, bin_number, bin_minimum, bin_median, bin_maximum, name='title'):
    import ROOT as rt

    # book histograms
    def book(h,name,n,a,b,title=""):
        h[name]=rt.TH1F(name,title,n,a,b)
        h[name].Sumw2()
    def book2(h,name,nx,ax,bx,ny,ay,by,title=""):
        h[name]=rt.TH2F(name,title,nx,ax,bx,ny,ay,by)
        h[name].Sumw2()
    def SgnPara(para,z):
        return para.Mod()*(1. if abs(para.DeltaPhi(z))>rt.TMath.Pi()/2 else -1.)
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

    v_gen =    rt.TVector2()
    v_para_PUPPI = rt.TVector2()
    v_perp_PUPPI = rt.TVector2()
    v_predict = rt.TVector2()
    v_para_predict = rt.TVector2()
    v_perp_predict = rt.TVector2()
    
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

    fout = rt.TFile(name, "recreate")
    for n in hists:
        hists[n].Write()

def response_ab(predict_met, gen_met, bin_number, bin_minimum, bin_median, bin_maximum, path_, name='response.pdf'):


    # Separate MET-bin
    bin_ = bin_number
    bin_mini = bin_minimum
    bin_medi = bin_median
    bin_maxi = bin_maximum

    binning_le = (bin_medi - bin_mini)/(bin_/2.)
    binning_gr = (bin_maxi - bin_medi)/(bin_/2.)

    predict_array = np.zeros(bin_)
    predict_RMS = np.zeros(bin_)
    gen_array = np.zeros(bin_)
    gen_RMS = np.zeros(bin_)
    gen_array_number = np.zeros(bin_)

    for i in range(int(bin_/2.)):
        print(i)
        for j in range(gen_met.shape[0]):
            if ((i * binning_le) < gen_met[j, 0] <= ((i + 1) * binning_le)):
                predict_array[i] += predict_met[j, 0]
                gen_array[i] += gen_met[j, 0]
                gen_array_number[i] += 1

    for i in range(int(bin_/2.)):
        print(i)
        for j in range(gen_met.shape[0]):
            if ((i * binning_gr) + 100 < gen_met[j, 0] <= ((i + 1) * binning_gr) + 100):
                predict_array[i + 10] += predict_met[j, 0]
                gen_array[i + 10] += gen_met[j, 0]
                gen_array_number[i + 10] += 1

    predict_mean = predict_array / gen_array_number
    gen_mean = gen_array / gen_array_number


    for i in range(int(bin_/2.)):
        print(i)
        for j in range(gen_met.shape[0]):
            if ((i * binning_le) < gen_met[j, 0] <= ((i + 1) * binning_le)):
                predict_RMS[i] += (predict_mean[i] - predict_met[j, 0]) ** 2
                gen_RMS[i] += (gen_mean[i] - gen_met[j, 0]) ** 2

    for i in range(int(bin_/2.)):
        print(i)
        for j in range(gen_met.shape[0]):
            if ((i * binning_gr) + 100 < gen_met[j, 0] <= ((i + 1) * binning_gr) + 100):
                predict_RMS[i + 10] += (predict_mean[i + 10] - predict_met[j, 0]) ** 2
                gen_RMS[i + 10] += (gen_mean[i + 10] - gen_met[j, 0]) ** 2

    predict_RMS = np.sqrt(predict_RMS / gen_array_number)
    gen_RMS = np.sqrt(gen_RMS / gen_array_number)

    response = predict_mean / gen_mean
    response_error = response * np.sqrt((predict_RMS/predict_mean) ** 2 + (gen_RMS/gen_mean) ** 2)

    plt.figure()
    plt.errorbar(x = gen_mean, y = response, yerr = response_error, xerr = gen_RMS)
    plt.xlabel("Gen MET [GeV]", fontsize=16)
    plt.ylabel("response", fontsize=16)
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")


    feat = open(''+path_+'response_dat.txt', 'w')
    for i in range(bin_):
        data = '%f, %f, %f, %f' % (gen_mean[i], gen_RMS[i], response[i], response_error[i])
        feat.write(data)
        feat.write('\n')

def response_parallel(predict_met, gen_met, bin_number, bin_minimum, bin_median, bin_maximum, path_,  name='response.pdf'):

    # phi calc
    def phidiff(phi1, phi2):
        phi_r = phi1 - phi2
        if phi_r > math.pi : phi_r = phi_r - 2 * math.pi
        if phi_r < -math.pi : phi_r = phi_r + 2 * math.pi
        return phi_r

    # Separate MET-bin
    bin_ = bin_number
    bin_mini = bin_minimum
    bin_medi = bin_median
    bin_maxi = bin_maximum

    binning_le = (bin_medi - bin_mini)/(bin_/2.)
    binning_gr = (bin_maxi - bin_medi)/(bin_/2.)

    predict_array = np.zeros(bin_)
    predict_RMS = np.zeros(bin_)
    gen_array = np.zeros(bin_)
    gen_RMS = np.zeros(bin_)
    gen_array_number = np.zeros(bin_)

    for i in range(int(bin_/2.)):
        for j in range(gen_met.shape[0]):
            if ((i * binning_le) < gen_met[j, 0] <= ((i + 1) * binning_le)):
                predict_array[i] += predict_met[j, 0] * abs(math.cos(phidiff(predict_met[j,1], gen_met[j,1])))
                gen_array[i] += gen_met[j, 0]
                gen_array_number[i] += 1

    for i in range(int(bin_/2.)):
        for j in range(gen_met.shape[0]):
            if ((i * binning_gr) + 100 < gen_met[j, 0] <= ((i + 1) * binning_gr) + 100):
                predict_array[i + 10] += predict_met[j, 0] * abs(math.cos(phidiff(predict_met[j,1], gen_met[j,1])))
                gen_array[i + 10] += gen_met[j, 0]
                gen_array_number[i + 10] += 1

    predict_mean = predict_array / gen_array_number
    gen_mean = gen_array / gen_array_number


    for i in range(int(bin_/2.)):
        for j in range(gen_met.shape[0]):
            if ((i * binning_le) < gen_met[j, 0] <= ((i + 1) * binning_le)):
                predict_RMS[i] += (predict_mean[i] - predict_met[j, 0] * abs(math.cos(phidiff(predict_met[j,1], gen_met[j,1])))) ** 2
                gen_RMS[i] += (gen_mean[i] - gen_met[j, 0]) ** 2

    for i in range(int(bin_/2.)):
        for j in range(gen_met.shape[0]):
            if ((i * binning_gr) + 100 < gen_met[j, 0] <= ((i + 1) * binning_gr) + 100):
                predict_RMS[i + 10] += (predict_mean[i + 10] - predict_met[j, 0] * abs(math.cos(phidiff(predict_met[j,1], gen_met[j,1])))) ** 2
                gen_RMS[i + 10] += (gen_mean[i + 10] - gen_met[j, 0]) ** 2

    predict_RMS = np.sqrt(predict_RMS / gen_array_number)
    gen_RMS = np.sqrt(gen_RMS / gen_array_number)

    response = predict_mean / gen_mean
    response_error = response * np.sqrt((predict_RMS/predict_mean) ** 2 + (gen_RMS/gen_mean) ** 2)

    plt.figure()
    plt.errorbar(x = gen_mean, y =  response, yerr = response_error, xerr = gen_RMS, color='r')
    plt.xlabel("Gen MET [GeV]", fontsize=16)
    plt.ylabel("response", fontsize=16)
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.grid(True)
    plt.savefig(''+name+'.png')
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")

    feat = open(''+name+'.txt', 'w')
    for i in range(bin_):
        data = '%f, %f, %f, %f' % (gen_mean[i], gen_RMS[i], response[i], response_error[i])
        feat.write(data)
        feat.write('\n')


def response_parallel_opaque(predict_met, puppi_met, gen_met, bin_number, bin_minimum, bin_median, bin_maximum, path_,  name='response.pdf'):

    # phi calc
    def phidiff(phi1, phi2):
        phi_r = phi1 - phi2
        if phi_r > math.pi : phi_r = phi_r - 2 * math.pi
        if phi_r < -math.pi : phi_r = phi_r + 2 * math.pi
        return phi_r

    # Separate MET-bin
    bin_ = bin_number
    bin_mini = bin_minimum
    bin_medi = bin_median
    bin_maxi = bin_maximum

    binning_le = (bin_medi - bin_mini)/(bin_/2.)
    binning_gr = (bin_maxi - bin_medi)/(bin_/2.)

    predict_array = np.zeros(bin_)
    predict_RMS = np.zeros(bin_)
    puppi_array = np.zeros(bin_)
    puppi_RMS = np.zeros(bin_)
    gen_array = np.zeros(bin_)
    gen_RMS = np.zeros(bin_)
    gen_array_number = np.zeros(bin_)

    for i in range(int(bin_/2.)):
        for j in range(gen_met.shape[0]):
            if ((i * binning_le) < gen_met[j, 0] <= ((i + 1) * binning_le)):
                predict_array[i] += predict_met[j, 0] * abs(math.cos(phidiff(predict_met[j,1], gen_met[j,1])))
                puppi_array[i] += puppi_met[j, 0] * abs(math.cos(phidiff(puppi_met[j,1], gen_met[j,1])))
                gen_array[i] += gen_met[j, 0]
                gen_array_number[i] += 1

    for i in range(int(bin_/2.)):
        for j in range(gen_met.shape[0]):
            if ((i * binning_gr) + 100 < gen_met[j, 0] <= ((i + 1) * binning_gr) + 100):
                predict_array[i + 10] += predict_met[j, 0] * abs(math.cos(phidiff(predict_met[j,1], gen_met[j,1])))
                puppi_array[i + 10] += puppi_met[j, 0] * abs(math.cos(phidiff(puppi_met[j,1], gen_met[j,1])))
                gen_array[i + 10] += gen_met[j, 0]
                gen_array_number[i + 10] += 1

    predict_mean = predict_array / gen_array_number
    puppi_mean = puppi_array / gen_array_number
    gen_mean = gen_array / gen_array_number


    for i in range(int(bin_/2.)):
        for j in range(gen_met.shape[0]):
            if ((i * binning_le) < gen_met[j, 0] <= ((i + 1) * binning_le)):
                predict_RMS[i] += (predict_mean[i] - predict_met[j, 0] * abs(math.cos(phidiff(predict_met[j,1], gen_met[j,1])))) ** 2
                puppi_RMS[i] += (puppi_mean[i] - puppi_met[j, 0] * abs(math.cos(phidiff(puppi_met[j,1], gen_met[j,1])))) ** 2
                gen_RMS[i] += (gen_mean[i] - gen_met[j, 0]) ** 2

    for i in range(int(bin_/2.)):
        for j in range(gen_met.shape[0]):
            if ((i * binning_gr) + 100 < gen_met[j, 0] <= ((i + 1) * binning_gr) + 100):
                predict_RMS[i + 10] += (predict_mean[i + 10] - predict_met[j, 0] * abs(math.cos(phidiff(predict_met[j,1], gen_met[j,1])))) ** 2
                puppi_RMS[i + 10] += (puppi_mean[i + 10] - puppi_met[j, 0] * abs(math.cos(phidiff(puppi_met[j,1], gen_met[j,1])))) ** 2
                gen_RMS[i + 10] += (gen_mean[i + 10] - gen_met[j, 0]) ** 2

    predict_RMS = np.sqrt(predict_RMS / gen_array_number)
    puppi_RMS = np.sqrt(puppi_RMS / gen_array_number)
    gen_RMS = np.sqrt(gen_RMS / gen_array_number)

    response = predict_mean / gen_mean
    response_puppi = puppi_mean / gen_mean
    response_error = response * np.sqrt((predict_RMS/predict_mean) ** 2 + (gen_RMS/gen_mean) ** 2)
    response_puppi_error = response_puppi * np.sqrt((puppi_RMS/puppi_mean) ** 2 + (gen_RMS/gen_mean) ** 2)

    plt.figure()
    plt.errorbar(x = gen_mean, y =  response, yerr = response_error, xerr = gen_RMS, color='g', label='Predicted MET')
    plt.errorbar(x = gen_mean, y =  response_puppi, yerr = response_puppi_error, xerr = gen_RMS, color='r', label='PUPPI MET')
    plt.xlabel("Gen MET [GeV]", fontsize=16)
    plt.ylabel("response", fontsize=16)
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.ylim(0,4)
    plt.grid(True)
    plt.legend()
    plt.savefig(''+name+'.png')
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")

    feat = open(''+name+'.txt', 'w')
    for i in range(bin_):
        data = '%f, %f, %f, %f' % (gen_mean[i], gen_RMS[i], response[i], response_error[i])
        feat.write(data)
        feat.write('\n')

def MET_rel_error_bad(predict_met, gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)/gen_met

    mask = (rel_err < 3)
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
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")




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
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")



def MET_rel_error_opaque(predict_met, predict_met2, gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)/gen_met

    mask = (rel_err[:] > 3)
    rel_err = rel_err[~mask]


    mean = np.mean(rel_err)
    std = np.std(rel_err)

    entry = rel_err.shape[0]

    rel_err2 = (predict_met2 - gen_met)/gen_met

    mask2 = (rel_err2[:] > 3)
    rel_err2 = rel_err2[~mask2]


    mean = np.mean(rel_err)
    std = np.std(rel_err)

    mean = mean * 1000
    mean = int(mean)
    print(mean)
    mean = float(mean) / 1000
    std = std * 1000
    std = int(std)
    std = float(std) / 1000

    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-3., 3., 50+1), label='puppi', alpha=0.5, color='red')
    plt.hist(rel_err2, bins=np.linspace(-3., 3., 50+1), label='ML', alpha=0.5, color='green')
    plt.xlabel("relative error (predict - true)/true", fontsize=16)
    plt.ylabel("Events", fontsize=16)
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.title('Relative Pt error', size=18, fontweight='bold', loc='right')
    plt.legend()
    plt.savefig(name)
    plt.show(block=False)
    ##plt.pause(5)
    plt.close("all")



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
    target_array_xy1M


def Phi_abs_error(predict_met, gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)
    for i in range(rel_err.shape[0]):
        if (rel_err[i] > math.pi):
            rel_err[i] - 2*math.pi
        if (rel_err[i] < math.pi):
            rel_err[i] + 2*math.pi
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-3.5, 3.5, 50+1))
    plt.xlabel("abs error (predict - true)")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")

def Pt_abs_error_opaque(predict_met, predict_met2, gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)
    rel_err2 = (predict_met2 - gen_met)
    #minErr = min(np.array([rel_err, rel_err2]).flatten())
    #maxErr = max(np.array([rel_err, rel_err2]).flatten())
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-250, 250, 50+1), alpha=0.5, label='puppi')
    plt.hist(rel_err2, bins=np.linspace(-250, 250, 50+1), alpha=0.5, label='ML')
    plt.xlabel("abs error (predict - true)")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.legend()
    plt.title('Abs Pt error', size=18, fontweight='bold', loc='right')
    plt.savefig(name)
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")
    
def Phi_abs_error_opaque(predict_met, predict_met2,gen_met, name='Met_res.pdf'):
    rel_err = (predict_met - gen_met)
    rel_err2 = (predict_met2 - gen_met)
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-3.5, 3.5, 50+1), alpha=0.5, label='puppi')
    plt.hist(rel_err2, bins=np.linspace(-3.5, 3.5, 50+1), alpha=0.5, label='ML')
    plt.xlabel("abs error (predict - true)")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.legend()
    plt.title('Abs Phi error', size=18, fontweight='bold', loc='right')
    plt.savefig(name)
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")


def dist(predict_met, min_, max_, bin_, name='dist.pdf'):
    rel_err = predict_met
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(min_, max_, bin_))
    plt.xlabel("MET [GeV]")
    plt.ylabel("Events")
    plt.yscale("log")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")

def dist_opaque(predict_met1, predict_met2, name='dist.pdf'):
    rel_err1 = predict_met1
    rel_err2 = predict_met2
    plt.figure()
    plt.hist(rel_err1, bins=np.linspace(0, 500, 50+1), alpha=0.5)
    plt.hist(rel_err2, bins=np.linspace(0, 500, 50+1), alpha=0.5)
    plt.xlabel("MET [GeV]")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)
    plt.legend()
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")

def dist_xy(predict_met, name='dist.pdf'):
    rel_err = predict_met
    plt.figure()
    plt.hist(rel_err, bins=np.linspace(-500, 500, 50+1))
    plt.xlabel("MET [GeV]")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(name)
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")

def MET_binned_predict_mean(predict_met, gen_met, binning, mini, maxi, genMET_cut, corr_check, name='predict_mean.pdf'):
    bin_ = int((maxi - mini)/binning)
    X_genMET = np.zeros(bin_)
    X_error = np.zeros(bin_)
    y_predict = np.zeros(bin_)
    y_error = np.zeros(bin_)

    for j in range(bin_):
        mask = (gen_met > (j * binning)) & (gen_met < ((j + 1) * binning))
        X_genMET[j] = np.mean(gen_met[mask])
        y_predict[j] = np.mean(predict_met[mask])
        X_error[j] = np.std(gen_met[mask])
        y_error[j] = np.std(predict_met[mask])

    plt.errorbar(X_genMET, y_predict, xerr = X_error, yerr = y_error,
                 label='cut = '+str(genMET_cut)+', '+str(corr_check)+'.')

    ## x = y plot
    X = np.arange(mini, maxi, binning)
    plt.plot(X, X, 'r-')
    ###

    plt.xlim(mini, maxi)
    plt.ylim(mini, 700)
    plt.xlabel('Gen MET mean [GeV]', fontsize = 16)
    #plt.ylabel('PUPPI MET mean [GeV]', fontsize = 16)
    plt.ylabel('predicted MET mean [GeV]', fontsize = 16)
    plt.legend()
    plt.savefig(name)
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")


def MET_binned_predict_mean_opaque(predict_met, predict_met2, gen_met, binning, mini, maxi, genMET_cut, corr_check, name='predict_mean.pdf'):
    bin_ = int((maxi - mini)/binning)
    X_genMET = np.zeros(bin_)
    X_error = np.zeros(bin_)
    y_predict = np.zeros(bin_)
    y_error = np.zeros(bin_)

    for j in range(bin_):
        mask = (gen_met > (j * binning)) & (gen_met < ((j + 1) * binning))
        X_genMET[j] = np.mean(gen_met[mask])
        y_predict[j] = np.mean(predict_met[mask])
        X_error[j] = np.std(gen_met[mask])
        y_error[j] = np.std(predict_met[mask])

    X_genMET2 = np.zeros(bin_)
    X_error2 = np.zeros(bin_)
    y_predict2 = np.zeros(bin_)
    y_error2 = np.zeros(bin_)

    for j in range(bin_):
        mask2 = (gen_met > (j * binning)) & (gen_met < ((j + 1) * binning))
        X_genMET2[j] = np.mean(gen_met[mask2])
        y_predict2[j] = np.mean(predict_met2[mask2])
        X_error2[j] = np.std(gen_met[mask2])
        y_error2[j] = np.std(predict_met2[mask2])


    plt.errorbar(X_genMET2, y_predict2, xerr = X_error2, yerr = y_error2,
                 label='PUPPI MET', color='green', uplims=y_error2, lolims=y_error2)
     
    plt.errorbar(X_genMET, y_predict, xerr = X_error, yerr = y_error,
                 label='Predicted MET', color='red', uplims=y_error, lolims=y_error)


    ## x = y plot
    X = np.arange(mini, maxi, binning)
    plt.plot(X, X, 'r-')
    ##

    plt.xlim(mini, maxi)
    plt.ylim(mini, maxi)
    plt.xlabel('Gen MET mean [GeV]', fontsize = 16)
    #plt.ylabel('PUPPI MET mean [GeV]', fontsize = 16)
    plt.ylabel('predicted MET mean [GeV]', fontsize = 16)
    plt.legend()
    plt.savefig(name)
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")


def MET_binned_predict_ratio(predict_met, gen_met, binning, mini, maxi, genMET_cut, comment, name='predict_mean.pdf'):
    bin_ = (maxi - mini)/binning
    X_genMET = np.zeros(bin_)
    X_error = np.zeros(bin_)
    y_predict = np.zeros(bin_)
    y_error = np.zeros(bin_)

    for j in range(bin_):
        mask = (gen_met > (j * binning)) & (gen_met < ((j + 1) * binning))
        X_genMET[j] = np.mean(gen_met[mask])
        y_predict[j] = np.mean(predict_met[mask]/gen_met[mask])
        X_error[j] = np.std(gen_met[mask])
        y_error[j] = np.std(predict_met[mask]/gen_met[mask])

    plt.errorbar(X_genMET, y_predict, xerr = X_error, yerr = y_error,
                 label='cut = '+str(genMET_cut)+', '+str(comment)+'.')

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
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")


def extract_result(feat_array, targ_array, path, name, mode):
    np.save(''+path+''+name+'_feature_array_'+mode+'MET', feat_array)
    np.save(''+path+''+name+'_target_array_'+mode+'MET', targ_array)


def histo_2D(predict_pT, gen_pT,min_, max_, name = '2D_histo.png'):
    X_hist = np.arange(0,500, 20)
    Y_hist = X_hist#1.25*X_hist
    #Y_hist_1 = 0.75*X_hist
    plt.plot(X_hist, Y_hist, '-r')
    #plt.plot(X_hist, Y_hist_1, '-r')
    x_bins = np.linspace(min_, max_, 50)
    y_bins = np.linspace(min_, max_, 50)
    plt.hist2d(gen_pT, predict_pT,  bins=[x_bins, y_bins], cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('gen MET [GeV]')
    plt.ylabel('predicted MET [GeV]')
    plt.savefig(name)
    plt.show(block=False)
    #plt.pause(5)
    plt.close("all")
