void resolution(){

	TFile *file = new TFile("histogram_PUPPIonly_100cut.root", "read");

	const int bin = 20;
	float mini = 0;
	float medi = 100;
	float maxi = 400;
	float binning_le = (medi - mini)/(bin/2.);
	float binning_gr = (maxi - medi)/(bin/2.);

	char predict_para[bin];
	char predict_perp[bin];
	char v_gen[bin];

	float hPr_pa_Mean[bin];
	float hPr_pe_Mean[bin];
	float hGe_Mean[bin];

	float hPr_pa_RMS[bin];
	float hPr_pe_RMS[bin];
	float hGe_RMS[bin];

	float hPr_pa_RMSError[bin];
	float hPr_pe_RMSError[bin];
	float hGe_RMSError[bin];

	for (int i = 0 ; i < bin/2 ; i++){

		sprintf(predict_para, "predict_para_%d-%d", i*10, (i+1)*10);
		sprintf(predict_perp, "predict_perp_%d-%d", i*10, (i+1)*10);
		sprintf(v_gen, "v_gen_%d-%d", i*10, (i+1)*10);
		
		TH1F * hPr_pa = (TH1F*) file ->Get(predict_para);
		TH1F * hPr_pe = (TH1F*) file ->Get(predict_perp);
		TH1F * hGe = (TH1F*) file ->Get(v_gen);

		hGe_Mean[i] = hGe->GetMean();

		hPr_pa_RMS[i] = hPr_pa->GetStdDev();
		hPr_pe_RMS[i] = hPr_pe->GetStdDev();
		hGe_RMS[i] = hGe->GetStdDev();

		hPr_pa_RMSError[i] = hPr_pa->GetRMSError();
		hPr_pe_RMSError[i] = hPr_pe->GetRMSError();
		hGe_RMSError[i] = hGe->GetRMSError();
	}

	for (int i = bin/2 ; i < bin ; i++){

		sprintf(predict_para, "predict_para_%d-%d", 100 + (i - bin/2)*30, 100 + ((i - bin/2)+1)*30);
		sprintf(predict_perp, "predict_perp_%d-%d", 100 + (i - bin/2)*30, 100 + ((i - bin/2)+1)*30);
		sprintf(v_gen, "v_gen_%d-%d", 100 + (i - bin/2)*30, 100 + ((i - bin/2)+1)*30);
		
		TH1F * hPr_pa = (TH1F*) file ->Get(predict_para);
		TH1F * hPr_pe = (TH1F*) file ->Get(predict_perp);
		TH1F * hGe = (TH1F*) file ->Get(v_gen);

		hGe_Mean[i] = hGe->GetMean();

		hPr_pa_RMS[i] = hPr_pa->GetStdDev();
		hPr_pe_RMS[i] = hPr_pe->GetStdDev();
		hGe_RMS[i] = hGe->GetStdDev();

		hPr_pa_RMSError[i] = hPr_pa->GetRMSError();
		hPr_pe_RMSError[i] = hPr_pe->GetRMSError();
		hGe_RMSError[i] = hGe->GetRMSError();
	}

	TCanvas * c5 = new TCanvas("Predict_para", "Predict Parallel Resolution", 600, 600);
	TCanvas * c6 = new TCanvas("Predict_perp", "Predict Perpendicular Resolution", 600, 600);
	
	TGraphErrors * gPr_pa = new TGraphErrors(bin, hGe_Mean, hPr_pa_RMS, 0, hPr_pa_RMSError);

	c5->cd();
	c5->SetMargin(0.1345382, 0.06425703, 0.1345382, 0.06425703);

	gPr_pa->SetMarkerStyle(20);
	gPr_pa->SetLineWidth(3);
	gPr_pa->SetLineColor(kGreen);
	gPr_pa->GetXaxis()->SetTitle("Gen MET [GeV]");
	gPr_pa->GetXaxis()->SetTitleSize(0.05);
	gPr_pa->GetYaxis()->SetRangeUser(0,140);
	gPr_pa->GetYaxis()->SetTitle("#sigma(MET_{#parallel})");
	gPr_pa->GetYaxis()->SetTitleSize(0.05);
	gPr_pa->SetTitle("Predict Para Res");
	gPr_pa->Draw("APL");

	TGraphErrors * gPr_pe = new TGraphErrors(bin, hGe_Mean, hPr_pe_RMS, 0, hPr_pe_RMSError);

	c6->cd();
	c6->SetMargin(0.1345382, 0.06425703, 0.1345382, 0.06425703);

	gPr_pe->SetMarkerStyle(20);
	gPr_pe->SetLineWidth(3);
	gPr_pe->SetLineColor(kGreen);
	gPr_pe->GetXaxis()->SetTitle("Gen MET [GeV]");
	gPr_pe->GetXaxis()->SetTitleSize(0.05);
	gPr_pe->GetYaxis()->SetRangeUser(0,140);
	gPr_pe->GetYaxis()->SetTitle("#sigma(MET_{#perp})");
	gPr_pe->GetYaxis()->SetTitleSize(0.05);
	gPr_pe->SetTitle("Predict Perp Res");
	gPr_pe->Draw("APL");

	TFile *ff = new TFile("resolution_VBF_PUPPIonly_features_100_cut.root", "RECREATE");
	
	c5->Write();
	c6->Write();
}
