#include "PCANet_util.h"

// Parameters
const char* v1 = "mnist_pred/v1.csv";
const char* v2 = "mnist_pred/v2.csv";

// PCANet param
int NumStages = 2;
int PatchSize = 7;
int NumFilters = 8;
int HistBlockSize = 7; 
double BlkOverLapRatio = 0.5;

void writeMatToFile(cv::Mat& m, const char* filename)
{
	ofstream fout(filename);

	if(!fout)
	{
		cout<<"File Not Opened"<<endl;  return;
	}

	for(int i=0; i<m.rows; i++)
	{
		for(int j=0; j<m.cols; j++)
		{
			fout<<m.at<double>(i,j)<<"\t";
			
		}
		fout<<endl;
	}

	fout.close();
}


Mat im2colstep(Mat img, int PatchSize, int step, bool hist){
	int width = img.cols;
	int height = img.rows;
	Rect roi;
	Mat image_roi,vec;
	Mat im;
	vector<Mat> imv;
	for (int i = 0; i <= height-PatchSize; i += step){
		for (int j = 0; j <= width-PatchSize; j += step){
			roi = Rect(i, j, PatchSize, PatchSize);
			image_roi = img(roi).clone();
			
			if(hist){

				int histSize = 256;
				float range[] = { 0, 255 } ;
				const float* histRange = { range };
				bool uniform = true; bool accumulate = false;
				calcHist(&image_roi,1, 0, Mat(), vec, 1, &histSize, &histRange, uniform, accumulate );
			}
			else{	
				image_roi = image_roi.t();
				vec = image_roi.reshape(0,1); 
			}
			
			imv.push_back(vec.t());
		}
	}
	hconcat( imv, im );
	return im;
}


vector<Mat> featureExtraction(const char* ImName){
	int ImgSize = 28;
	Mat image = imread(ImName,1);
	cvtColor(image,image,CV_RGB2GRAY);
	image.convertTo(image,CV_64FC1);	
		 
	vector<Mat> VV;
	CvMLData dataFile;
	if (dataFile.read_csv(v1) != 0){
		fprintf(stderr, "Can't read csv file %s\n", ImName);
	}
	else 
		cout<< "data read success!" << endl;
	Mat vv1(dataFile.get_values());
	VV.push_back(vv1);

	CvMLData dataFile2;
	if (dataFile2.read_csv(v2) != 0){
		fprintf(stderr, "Can't read csv file %s\n", ImName);
	}
	else 
		cout<< "data read success!" << endl;
	Mat vv2(dataFile2.get_values());
	VV.push_back(vv2);
	
	vector<Mat> OutImg,InImg;
	OutImg.push_back(image);
	
	int ImgIdx = 1;
	int j = 0;
	for(int stage = 0; stage< NumStages; stage++){
		InImg.clear();
		for(j = 0; j < OutImg.size();j++)
			InImg.push_back(OutImg[j]);
		Mat V;		
		size_t ImgZ = OutImg.size();
		OutImg.clear();
		int mag = (PatchSize-1)/2;

		for (size_t i = 0; i < ImgZ; i++){
			Mat img;
			img.create(ImgSize+PatchSize-1,ImgSize+PatchSize-1,img.type());
			copyMakeBorder(InImg[i],img,mag,mag,mag,mag,BORDER_CONSTANT,Scalar(0));

			Mat im = im2colstep(img, PatchSize,1, false);
			Mat im1(im.rows, im.cols, CV_64FC1);
			im1.data = im.data;

			for (j = 0; j < NumFilters; j++){
				Mat t = VV[stage].col(j).t();
				
				t.convertTo(t, CV_64FC1);
				// cout << im1.type() << " " << t.type() << endl;
				Mat tmp = t * im1;
				//gemm(t,im1,1,NULL,0,tmp);
				
				OutImg.push_back(tmp.reshape(0, 28).t());				
				
			}
			
		}

	}
	
	return OutImg;
}


Mat Heaviside(Mat m){
	Mat x;
	m.convertTo(m,CV_32FC1);
	threshold( m, x, 0, 1 ,THRESH_BINARY );
	return x;
}


Mat HashingHist(vector<Mat> OutImg){
	Mat feature;
	int map_weights[8] = { 128, 64, 32, 16, 8, 4, 2, 1};
	// Mat map_weights = Mat(1, 8, CV_64FC1, data);
	vector<int> Idx_span;
	int i;
	for (i = 1; i <= 64; i++)
		Idx_span.push_back(i);
	//Mat Idx_span = Mat(1,64, CV_64FC1, dataV);
	// cout << map_weights.at<int>(0,0) << endl;
	int NumImginO = 8;
	vector<Mat> Bhish;
	for (i = 0; i < NumImginO; i++){
		Mat T;
		for (int j = 1; j <= NumFilters; j++){
			int index = Idx_span[NumFilters*i+j-1];
			//cout << OutImg[index-1].type() << endl;
			Mat X = Heaviside(OutImg[index-1]);
			T = T + map_weights[j-1]*X; 
			/*if(j == 3)
			for(int m = 0; m < 28; m++)
			for(int n = 0; n < 28; n++)
			cout << T.at<float>(n,m) << endl;*/

		}
		Bhish.push_back( im2colstep(T, PatchSize,4,true) );
	}
	
	
	hconcat( Bhish, feature );
	return feature;
}



