#include<opencv2/opencv.hpp>
#include <time.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

// Parameters
const char* Tfeature = "mnist_pred/ff.csv";
const char* Tlabel = "mnist_pred/ll.csv";
const char* testFeature = "mnist_pred/3.csv";
const char* modelFile =  "mnist_pred/model.xml";
const char* ImName =  "mnist_pred/image8.jpg";
const char* v1 = "mnist_pred/v1.csv";
const char* v2 = "mnist_pred/v2.csv";

// PCANet param
int NumStages = 2;
int PatchSize = 7;
int NumFilters = 8;
int HistBlockSize = 7; 
double BlkOverLapRatio = 0.5;


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
				/*vector<Mat> bgr_planes;
				split( image_roi, bgr_planes );*/
				int histSize = 256;
				float range[] = { 0, 255 } ;
				const float* histRange = { range };
				bool uniform = true; bool accumulate = false;
				calcHist(&image_roi,1, 0, Mat(), vec, 1, &histSize, &histRange, uniform, accumulate );
			}
			else{
				vec = image_roi.reshape(0,1);
			}
			
			transpose(vec,vec);
			imv.push_back(vec);
		}
	}
	hconcat( imv, im );
	return im;
}


vector<Mat> featureExtraction(const char* ImName){
	int ImgSize = 28;
	Mat image = imread(ImName,1);
	cvtColor(image,image,CV_RGB2GRAY);
	vector<Mat> VV;
	CvMLData dataFile;
	if (dataFile.read_csv(v1) != 0){
		fprintf(stderr, "Can't read csv file %s\n", Tfeature);
	}
	else 
		cout<< "data read success!" << endl;
	Mat vv1(dataFile.get_values());
	VV.push_back(vv1);

	if (dataFile.read_csv(v2) != 0){
		fprintf(stderr, "Can't read csv file %s\n", Tfeature);
	}
	else 
		cout<< "data read success!" << endl;
	Mat vv2(dataFile.get_values());
	VV.push_back(vv2);

	vector<Mat> OutImg,InImg;
	OutImg.push_back(image);
	
	int ImgIdx = 1;
	int j = 0;
	for(int stage = 0; stage< NumStages; stage++){
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
				OutImg.push_back(tmp.reshape(0, 28));
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
		for (int j = 1; j <=NumFilters; j++){
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
			fout<<m.at<float>(i,j)<<"\t";
		}
		fout<<endl;
	}

	fout.close();
}
int main(){
	/*float a[2][2] = {{2,2},{-2,-2}};
	Mat m(2,2,CV_32FC1,a);
	cout << m.at<float>(0,0) << endl;
	cout << m.at<float>(1,1) << endl;
	threshold( m, m, 0, 1 ,THRESH_BINARY );
	cout << m.at<float>(0,0) << endl;
	cout << m.at<float>(1,1) << endl;*/

	vector<Mat> OutImg = featureExtraction(ImName);

	Mat feature = HashingHist(OutImg);
	//writeMatToFile(feature,"ffff.txt");

	bool training = false;

	CvSVM svm =CvSVM();

	CvSVMParams param;

	CvTermCriteria criteria;

	if(training){

			CvMLData dataFile;

			if (dataFile.read_csv(Tfeature) != 0){
				fprintf(stderr, "Can't read csv file %s\n", Tfeature);
				getchar();
				return -1;
			}
			else 
				cout<< "data read success!" << endl;

			Mat dataMat(dataFile.get_values()); // Default data type is float

			Mat TrnData;
			transpose(dataMat, TrnData);

			CvMLData LabelFile;


			if (LabelFile.read_csv(Tlabel) != 0){
				fprintf(stderr, "Can't read csv file %s\n", Tlabel);
				getchar();
				return -1;
			}
			else 
				cout<< "label read success!" << endl;

			Mat LabelMat(LabelFile.get_values()); 

			Mat TrnLabel;


			cout << TrnData.rows << endl;
			cout << TrnData.cols << endl;


			criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000,FLT_EPSILON);

			param    = CvSVMParams(CvSVM::C_SVC,CvSVM::LINEAR, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1,NULL, criteria);


			CvMat train_data = TrnData;
			CvMat train_label = LabelMat;

			svm.train(&train_data, &train_label,NULL,NULL,param);
			svm.save(modelFile);
	}

	else
	{		// ================ predict ======================
		
			clock_t nTimeStart;      
			svm.load(modelFile);
		
			clock_t nTimeStop;       
			nTimeStart = clock();

			CvMLData testData;

			if (testData.read_csv(testFeature) != 0){
				fprintf(stderr, "Can't read csv file %s\n", testFeature);
				getchar();
				return -1;
			}
			else 
				cout<< "data read success!" << endl;

			Mat testMat(testData.get_values()); 

			//Mat TestData;
			//transpose(testMat, TestData);


			float label = svm.predict(feature);
			nTimeStop = clock();
			cout <<"the average running time:"<<(double)(nTimeStop - nTimeStart)/CLOCKS_PER_SEC << "seconds"<< endl;
			cout << label << endl;
	}

	system("pause");

	return 0;

}
