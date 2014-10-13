#include "PCANet_util.h"


// Parameters
bool training = false;
const char* Tfeature = "mnist_pred/ff.csv";
const char* Tlabel = "mnist_pred/ll.csv";
const char* ImName =  "mnist_pred/image6.jpg";
const char* testFeature = "mnist_pred/3.csv";
const char* modelFile =  "mnist_pred/model.xml";

bool model_file_exist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
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

	CvSVM svm ;

	CvSVMParams param;

	CvTermCriteria criteria;

	if(!model_file_exist(modelFile))
		training = true;

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
