#include "svm.h"

#include <iostream>
#include "fstream"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"


using namespace cv;
using namespace std;

const char *CSV_FILE = "heart_scale.csv";
const char *MODEL_FILE = "heart_scale.model";
 float **m;

double inputArr[10][13] = 
{
	1,0.708333,1,1,-0.320755,-0.105023,-1,1,-0.419847,-1,-0.225806,0,1, 

	-1,0.583333,-1,0.333333,-0.603774,1,-1,1,0.358779,-1,-0.483871,0,-1,

	1,0.166667,1,-0.333333,-0.433962,-0.383562,-1,-1,0.0687023,-1,-0.903226,-1,-1,

	-1,0.458333,1,1,-0.358491,-0.374429,-1,-1,-0.480916,1,-0.935484,0,-0.333333,

	-1,0.875,-1,-0.333333,-0.509434,-0.347032,-1,1,-0.236641,1,-0.935484,-1,-0.333333,

	-1,0.5,1,1,-0.509434,-0.767123,-1,-1,0.0534351,-1,-0.870968,-1,-1,

	1,0.125,1,0.333333,-0.320755,-0.406393,1,1,0.0839695,1,-0.806452,0,-0.333333,

	1,0.25,1,1,-0.698113,-0.484018,-1,1,0.0839695,1,-0.612903,0,-0.333333,

	1,0.291667,1,1,-0.132075,-0.237443,-1,1,0.51145,-1,-0.612903,0,0.333333,

	1,0.416667,-1,1,0.0566038,0.283105,-1,1,0.267176,-1,0.290323,0,1
};



double testArr[]={
	/*0.25,1,1,-0.226415,-0.506849,-1,-1,0.374046,-1,-0.83871,0,-1*/
	/*67,1,4,160,286,0,2,108,1,1.5,2,3,3*/
	-1,	57	,1	,4	,140	,192	,0	,0	,148,	0	,0.4,	2	,0

};

 
void DefaultSvmParam(struct svm_parameter *param)
{

	param->svm_type = C_SVC;

	param->kernel_type = RBF;

	param->degree = 3;

	param->gamma = 0; // 1/num_features

	param->coef0 = 0;

	param->nu = 0.5;

	param->cache_size = 100;

	param->C = 1;

	param->eps = 1e-3;

	param->p = 0.1;

	param->shrinking = 1;

	param->probability = 0;

	param->nr_weight = 0;

	param->weight_label = NULL;

	param->weight = NULL;

}

 

void SwitchForSvmParma(struct svm_parameter *param, char ch, char *strNum, int nr_fold, int cross_validation){

	switch(ch)

		{

		case 's':

		{

		param->svm_type = atoi(strNum);

		break;

		}

		case 't':

		{

		param->kernel_type = atoi(strNum);

		break;

		}

		case 'd':

		{

		param->degree = atoi(strNum);

		break;

		}

		case 'g':

		{

		param->gamma = atof(strNum);

		break;

		}

		case 'r':

		{

		param->coef0 = atof(strNum);

		break;

		}

		case 'n':

		{

		param->nu = atof(strNum);

		break;

		}

		case 'm':

		{

		param->cache_size = atof(strNum);

		break;

		}

		case 'c':

		{

		param->C = atof(strNum);

		break;

		}

		case 'e':

		{

		param->eps = atof(strNum);

		break;

		}

		case 'p':

		{

		param->p = atof(strNum);

		break;

		}

		case 'h':

		{

		param->shrinking = atoi(strNum);

		break;

		}

		case 'b':

		{

		param->probability = atoi(strNum);

		break;

		}

		case 'q':

		{

		break;

		}

		case 'v':

		{

		cross_validation = 1;

		nr_fold = atoi(strNum);

		if (nr_fold < 2)		{
			cout<<"nr_fold should > 2!!! file: "<<__FILE__<<" function: ";
			cout<<__FUNCTION__<<" line: "<<__LINE__<<endl;
		}

		break;

		}

		case 'w':
		{
			++param->nr_weight;
			param->weight_label = (int *)realloc(param->weight_label,sizeof(int)*param->nr_weight);
			param->weight = (double *)realloc(param->weight,sizeof(double)*param->nr_weight);
			param->weight_label[param->nr_weight-1] = atoi(strNum);			 
			param->weight[param->nr_weight-1] = atof(strNum);

		break;

		}

		default:

		{

		break;

		}

	}

}

 
void SetSvmParam(struct svm_parameter *param, char *str, int cross_validation, int nr_fold)

{

	DefaultSvmParam(param);

	cross_validation = 0;

	 

	char ch = ' ';

	int strSize = strlen(str);

	for (int i=0; i<strSize; i++){

		if (str[i] == '-'){

			ch = str[i+1];

			int length = 0;

			for (int j=i+3; j<strSize; j++)	{
				if (isdigit(str[j]))
					length++;
				else
					break;
			}

		char *strNum = new char[length+1];

		int index = 0;

		for (int j=i+3; j<i+3+length; j++){
			strNum[index] = str[j];
			index++;
		}

		strNum[length] = '/0';

		SwitchForSvmParma(param, ch, strNum, nr_fold, cross_validation);

		delete strNum;

		}

	}

}

 
void SvmTraining(char *option){

	struct svm_parameter param;

	struct svm_problem prob;

	struct svm_model *model;

	struct svm_node *x_space;

	int cross_validation = 0;

	int nr_fold = 0;

 	int sampleCount = 200;

	int featureDim = 14;

	prob.l =  sampleCount;

	prob.y = new double[sampleCount];

	prob.x = new struct svm_node*[sampleCount];

	x_space = new struct svm_node[(featureDim+1)*sampleCount];

 
	SetSvmParam(&param, option, cross_validation, nr_fold);

	for (int i=0; i<sampleCount; i++)
		prob.y[i] = m[i][0];


	int j = 0;

	for (int i=0; i<sampleCount; i++){
		prob.x[i] = &x_space[j];
		for (int k=1; k<=featureDim; k++){
			x_space[i*featureDim+k].index = k;
			x_space[i*featureDim+k].value = m[i][k];
		}
		x_space[(i+1)*featureDim].index = -1;
		j = (i+1)*featureDim + 1; 
	}

 
	model = svm_train(&prob, &param);

	const char* model_file_name = "model.txt";

	svm_save_model(model_file_name, model);


	svm_free_model_content(model);

	svm_destroy_param(&param);

	delete[] prob.y;

	delete[] prob.x;

	delete[] x_space; 

}


int SvmPredict(const char* modelAdd){
	struct svm_node *testX;
	struct svm_model* testModel;
	testModel = svm_load_model(modelAdd);
	int featureDim = 12;
	testX = new struct svm_node[featureDim+1];
	for (int i=0; i<featureDim; i++){
		testX[i].index = i+1;
		//testX[i].value = testArr[i];
		testX[i].value = m[269][i];
	}

	testX[featureDim].index = -1;

	double p = svm_predict(testModel, testX);

	svm_free_model_content(testModel);

	delete[] testX;


	if (p > 0.5)
		return 1;
	else
		return -1;

}

void mat2array(Mat mat){
 

 // allocate and initialize a 2d float array
 m = new float*[mat.rows];
 for (int r = 0; r < mat.rows; ++r)
 {
    m[r] = new float[mat.cols];
    for (int c = 0; c < mat.cols; ++c)
    {
       m[r][c] = mat.at<float>(r, c);
    }
 }

 // (use m for something)

 // don't forget to clean up!
 //for (int r = 0; r < mat.rows; ++r)
 //{
 //   delete[] m[r];
 //}
 //delete[] m;

 }
int main(int argc, char* argv[]){
	// Load matrix data in csv format
	CvMLData dataFile;
   
    if (dataFile.read_csv(CSV_FILE) != 0){
        fprintf(stderr, "Can't read csv file %s\n", CSV_FILE);
		getchar();
        return -1;
    }
	else 
		cout<< "read success!" << endl;
    
    Mat dataMat(dataFile.get_values()); // Default data type is float
	//Mat A = Mat(10, 13, CV_32FC1, inputArr); // save array to mat
	//Mat B = (Mat_<float>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	mat2array(dataMat);
	cout<< m[1][13]<<endl;// check test result


	/* ==== training model ==== */
	SvmTraining("-c 100 -t 0");

	int flag = SvmPredict("model.txt");

	cout<<"flag = "<< flag <<endl;


	system("pause");

	return 0;

}
