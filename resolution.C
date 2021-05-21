void resolution(){

	TFile *file = new TFile("./result/result_2021-02-16/noFlat_response/histogram_predicted_0.root", "read");
	const int bin = 20;
	float mini = 0;
	float medi = 100;
	float maxi = 300;
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

		sprintf(predict_para, "predict_para_%d-%d", 100 + (i - bin/2)*20, 100 + ((i - bin/2)+1)*20);
		sprintf(predict_perp, "predict_perp_%d-%d", 100 + (i - bin/2)*20, 100 + ((i - bin/2)+1)*20);
		sprintf(v_gen, "v_gen_%d-%d", 100 + (i - bin/2)*20, 100 + ((i - bin/2)+1)*20);
		
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



	TFile *file1 = new TFile("./result/result_2021-02-16/noFlat_response/histogram_puppi_0.root", "read");

	float hPu_pa_Mean[bin];
	float hPu_pe_Mean[bin];

	float hPu_pa_RMS[bin];
	float hPu_pe_RMS[bin];

	float hPu_pa_RMSError[bin];
	float hPu_pe_RMSError[bin];

	for (int i = 0 ; i < bin/2 ; i++){

		sprintf(predict_para, "predict_para_%d-%d", i*10, (i+1)*10);
		sprintf(predict_perp, "predict_perp_%d-%d", i*10, (i+1)*10);
		
		TH1F * hPu_pa = (TH1F*) file1 ->Get(predict_para);
		TH1F * hPu_pe = (TH1F*) file1 ->Get(predict_perp);

		hPu_pa_RMS[i] = hPu_pa->GetStdDev();
		hPu_pe_RMS[i] = hPu_pe->GetStdDev();

		hPu_pa_RMSError[i] = hPu_pa->GetRMSError();
		hPu_pe_RMSError[i] = hPu_pe->GetRMSError();
	}

	for (int i = bin/2 ; i < bin ; i++){

		sprintf(predict_para, "predict_para_%d-%d", 100 + (i - bin/2)*20, 100 + ((i - bin/2)+1)*20);
		sprintf(predict_perp, "predict_perp_%d-%d", 100 + (i - bin/2)*20, 100 + ((i - bin/2)+1)*20);
		
		TH1F * hPu_pa = (TH1F*) file1 ->Get(predict_para);
		TH1F * hPu_pe = (TH1F*) file1 ->Get(predict_perp);

		hPu_pa_RMS[i] = hPu_pa->GetStdDev();
		hPu_pe_RMS[i] = hPu_pe->GetStdDev();

		hPu_pa_RMSError[i] = hPu_pa->GetRMSError();
		hPu_pe_RMSError[i] = hPu_pe->GetRMSError();
	}





	TCanvas * c5 = new TCanvas("Predict_para", "Predict Parallel Resolution", 600, 600);
	TCanvas * c6 = new TCanvas("Predict_perp", "Predict Perpendicular Resolution", 600, 600);
	TLegend * L1 = new TLegend(.1, .7, .3, .9);
	TLegend * L2 = new TLegend(.1, .7, .3, .9);
	
	TGraphErrors * gPr_pa = new TGraphErrors(bin, hGe_Mean, hPr_pa_RMS, 0, hPr_pa_RMSError);

	c5->cd();
	c5->SetMargin(0.1345382, 0.06425703, 0.1345382, 0.06425703);
	c5->SetGrid();

	L1->AddEntry(gPr_pa, "Predicted MET resolution");
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

	TGraphErrors * gPu_pa = new TGraphErrors(bin, hGe_Mean, hPu_pa_RMS, 0, hPu_pa_RMSError);

	L1->AddEntry(gPu_pa, "PUPPI MET resolution");
	gPu_pa->SetMarkerStyle(20);
	gPu_pa->SetLineWidth(3);
	gPu_pa->SetLineColor(kRed);
	gPu_pa->Draw("PL");

	L1->Draw();



	TGraphErrors * gPr_pe = new TGraphErrors(bin, hGe_Mean, hPr_pe_RMS, 0, hPr_pe_RMSError);

	c6->cd();
	c6->SetMargin(0.1345382, 0.06425703, 0.1345382, 0.06425703);
	c6->SetGrid();

	L2->AddEntry(gPr_pe, "Predicted MET resolution");
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

	TGraphErrors * gPu_pe = new TGraphErrors(bin, hGe_Mean, hPu_pe_RMS, 0, hPu_pe_RMSError);

	L2->AddEntry(gPu_pe, "PUPPI MET resoluiton");
	gPu_pe->SetMarkerStyle(20);
	gPu_pe->SetLineWidth(3);
	gPu_pe->SetLineColor(kRed);
	gPu_pe->Draw("PL");

	L2->Draw();

	TFile *ff = new TFile("resolution_VBF_PUPPIonly_features_100_cut.root", "RECREATE");
	
	c5->Write();
	c6->Write();
}
