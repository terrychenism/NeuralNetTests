///////////////////////////////////////////////////////////
////// DecoupledNet.cpp
////// 2015-06-23
////// Tairui Chen
///////////////////////////////////////////////////////////

#include <cuda_runtime.h>

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>

#include <opencv/highgui.h>
#include <opencv/cv.h> //cvResize()

#include "caffe/caffe.hpp"



#define IMAGE_SIZE 224

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;



vector<vector<float>>  get_pixel(char fileneme[256]){
	std::fstream myfile(fileneme, std::ios_base::in);
	float a;
	vector<vector<float>> pTable;
	while(myfile >> a){
		vector<float> v;
		v.push_back(a);
		for(int i = 0; i < 2 && myfile >> a; i++)
			v.push_back(a);
		pTable.push_back(v);
		
	}
	return pTable;
}

void initGlog()
{
	FLAGS_log_dir=".\\log\\";
	_mkdir(FLAGS_log_dir.c_str());
	std::string LOG_INFO_FILE;
	std::string LOG_WARNING_FILE;
	std::string LOG_ERROR_FILE;
	std::string LOG_FATAL_FILE;
	std::string now_time=boost::posix_time::to_iso_extended_string(boost::posix_time::second_clock::local_time());
	now_time[13]='-';
	now_time[16]='-';
	LOG_INFO_FILE = FLAGS_log_dir + "INFO" + now_time + ".txt";
	google::SetLogDestination(google::GLOG_INFO,LOG_INFO_FILE.c_str());
	LOG_WARNING_FILE = FLAGS_log_dir + "WARNING" + now_time + ".txt";
	google::SetLogDestination(google::GLOG_WARNING,LOG_WARNING_FILE.c_str());
	LOG_ERROR_FILE = FLAGS_log_dir + "ERROR" + now_time + ".txt";
	google::SetLogDestination(google::GLOG_ERROR,LOG_ERROR_FILE.c_str());
	LOG_FATAL_FILE = FLAGS_log_dir + "FATAL" + now_time + ".txt";
	google::SetLogDestination(google::GLOG_FATAL,LOG_FATAL_FILE.c_str());
}

void GlobalInit(int* pargc, char*** pargv) {
	// Google flags.
	::gflags::ParseCommandLineFlags(pargc, pargv, true);
	// Google logging.
	::google::InitGoogleLogging(*(pargv)[0]);
	// Provide a backtrace on segfault.
	//::google::InstallFailureSignalHandler();
	initGlog();
}

void bubble_sort(float *feature, int *sorted_idx)
{
	int i=0, j=0;
	float tmp;
	int tmp_idx;

	for(i=0; i < 20; i++)
		sorted_idx[i] = i;

	for(i=0; i< 20; i++)
	{
		for(j=0; j < 19; j++)
		{
			if(feature[j] < feature[j+1])
			{
				tmp = feature[j];
				feature[j] = feature[j+1];
				feature[j+1] = tmp;

				tmp_idx = sorted_idx[j];
				sorted_idx[j] = sorted_idx[j+1];
				sorted_idx[j+1] = tmp_idx;
			}
		}
	}
}

void get_top5(float *feature, int arr[5])
{
	int i=0;
	int sorted_idx[20];

	bubble_sort(feature, sorted_idx);

	for(i=0; i<5; i++)
	{
		arr[i] = sorted_idx[i];
	}
}

int main(int argc, char** argv)
{
	bool debug_info_ = false;
	if(!debug_info_) caffe::GlobalInit(&argc, &argv);

	vector<vector<float>> pTable = get_pixel( "G:/EDU/_SOURCE_CODE/caffe/caffe-decouple/examples/seg_map/pixel.txt");
	// Test mode
	Caffe::set_phase(Caffe::TEST);

	// mode setting - CPU/GPU
	Caffe::set_mode(Caffe::GPU);

	// gpu device number
	int device_id = 0;
	Caffe::SetDevice(device_id);

	// prototxt
	//Net<float> caffe_test_net("C:/Users/Terry/Desktop/DecoupledNet/model/DecoupledNet_Full_anno/DecoupledNet_Full_anno_inference_deploy.prototxt");

	//// caffemodel
	//caffe_test_net.CopyTrainedLayersFrom("C:/Users/Terry/Desktop/DecoupledNet/model/DecoupledNet_Full_anno/DecoupledNet_Full_anno_inference.caffemodel");

	Net<float> caffe_test_net("G:/EDU/_SOURCE_CODE/caffe/caffe-decouple/examples/DeconvNet/model/DeconvNet/DeconvNet_inference_deploy.prototxt");
	// caffemodel
	//caffe_test_net.CopyTrainedLayersFrom("C:/Users/cht2pal/Desktop/caffe-old-unpool/examples/live_net/VGG_ILSVRC_19_layers.caffemodel");
	caffe_test_net.CopyTrainedLayersFrom("G:/EDU/_SOURCE_CODE/caffe/caffe-decouple/examples/DeconvNet/model/DeconvNet/DeconvNet_trainval_inference.caffemodel");


	//Debug
	//caffe_test_net.set_debug_info(true);

	clock_t begin = clock();
  
	int i=0, j=0, k=0;
	float mean_val[3] = {104.006, 116.668, 122.678}; // bgr mean

	// input
	vector<Blob<float>*> input_vec;
	Blob<float> blob(1, 3, IMAGE_SIZE, IMAGE_SIZE);
	
	IplImage *small_image = cvCreateImage(cvSize(IMAGE_SIZE, IMAGE_SIZE), 8, 3);


	caffe::Datum datum;
	const char* img_name = "G:/EDU/_SOURCE_CODE/caffe/caffe-decouple/examples/seg_map/000079.jpg"; //argv[1]; //
	IplImage *img = cvLoadImage(img_name);
	cvShowImage("raw_image", img);

	cvResize(img, small_image);

	for (k=0; k<3; k++)
	{
		for (i=0; i<IMAGE_SIZE; i++)
		{
			for (j=0; j< IMAGE_SIZE; j++)
			{
				blob.mutable_cpu_data()[blob.offset(0, k, i, j)] = (float)(unsigned char)small_image->imageData[i*small_image->widthStep+j*small_image->nChannels+k] - mean_val[k];
			}
		}
	}
	input_vec.push_back(&blob);

	float loss;

	// Do Forward
	const vector<Blob<float>*>& result = caffe_test_net.Forward(input_vec, &loss);
	LOG(INFO) << "result blobs: " << result.size() << endl;
	for(auto &out_blob : result){
		LOG(INFO) << out_blob->num() << " " << out_blob->channels() << " " 
			<< out_blob->height() << " " << out_blob->width() ;
	}
	//Blob<float> *out_blob= result[0];


	// Softmax
	vector<Blob<float>*> top;
	vector<Blob<float>*> bottom;
	top.push_back(result[0]);
	bottom.push_back(result[0]);

	top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
		bottom[0]->height(), bottom[0]->width());
	Blob<float> scale_;
	Blob<float> sum_multiplier_;
	sum_multiplier_.Reshape(1, bottom[0]->channels(), 1, 1);
	float* multiplier_data = sum_multiplier_.mutable_cpu_data();
	for (int i = 0; i < sum_multiplier_.count(); ++i) {
		multiplier_data[i] = 1.;
	}
	scale_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
	const float* bottom_data = bottom[0]->cpu_data();
	float* scale_data = scale_.mutable_cpu_data();

	float* top_data = top[0]->mutable_cpu_data();
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int dim = bottom[0]->count() / bottom[0]->num();
	int spatial_dim = bottom[0]->height() * bottom[0]->width();
	caffe_copy(bottom[0]->count(), bottom_data, top_data);

	for (int i = 0; i < num; ++i) {
		// initialize scale_data to the first plane
		caffe_copy(spatial_dim, bottom_data + i * dim, scale_data);
		for (int j = 0; j < channels; j++) {
			for (int k = 0; k < spatial_dim; k++) {
				scale_data[k] = std::max(scale_data[k],
					bottom_data[i * dim + j * spatial_dim + k]);
			}
		}
		if(debug_info_) for(int m = 0;m < 10; m++) cout << bottom_data[m] << " " << scale_data[m] <<" " << top_data[m] << endl;
		
		// subtraction
		caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
			1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data + i * dim);

		if(debug_info_) for(int m = 0;m < 10; m++) cout << bottom_data[m] << " " << scale_data[m] <<" " << top_data[m] << endl;
		
		// exponentiation
		caffe_exp<float>(dim, top_data + i * dim, top_data + i * dim);

		if(debug_info_) for(int m = 0;m < 10; m++) cout << bottom_data[m] << " " << scale_data[m] <<" " << top_data[m] << endl;

		// sum after exp
		caffe_cpu_gemv<float>(CblasTrans, channels, spatial_dim, 1.,
			top_data + i * dim, sum_multiplier_.cpu_data(), 0., scale_data);

		if(debug_info_) for(int m = 0;m < 10; m++) cout << bottom_data[m] << " " << scale_data[m] <<" " << top_data[m] << endl;

		// division
		for (int j = 0; j < channels; j++) {
			caffe_div(spatial_dim, top_data + top[0]->offset(i, j), scale_data,
				top_data + top[0]->offset(i, j));
		}
	}

	Blob<float> *out_blob= top[0];

	vector<Mat> maps;
	double sum_pix = 0;
	int idx = 0;
	int cnt_max = INT_MIN;
	vector<pair<double, cv::Mat>> mp;
	
	sum_pix = 0;
	Mat seg_map(IMAGE_SIZE,IMAGE_SIZE, CV_32FC1, Scalar(0,0,0));
	for(int i = 0; i < out_blob->height(); i++){
		for(int j=0; j < out_blob->width(); j++){
			vector<pair<float, int>> v;
			for(int c = 0; c < out_blob->channels(); c++){ // for 21 channels
				v.push_back( make_pair(out_blob->cpu_data()[idx + IMAGE_SIZE*IMAGE_SIZE*c], c) );
			}
			std::sort(v.begin(), v.end(), [](const std::pair<float, int> &left, const std::pair<float, int> &right) {
					return left.first >  right.first;} );	
			//apply pixel
			seg_map.at<float>(i,j) = v[0].second;
			//seg_map.at<float>(i,j) = top[0]->cpu_data()[IMAGE_SIZE*IMAGE_SIZE*7 + idx];
			idx++;
		}
	}
	imshow("seg_map", seg_map);



	// restore seg map to image
	Mat	res_map;
	resize(seg_map, res_map, Size(img->width, img->height));
	sum_pix = 0;
	IplImage *final_seg = cvCreateImage(cvSize(img->width, img->height), 8, 3);
	for(int h = 0; h < res_map.rows; h++){
		for(int w = 0; w < res_map.cols; w++){
			int p = res_map.at<float>(h,w);
			for (int c = 0; c < 3; c++){
				final_seg->imageData[h*img->widthStep+w*img->nChannels+c] = pTable[p][c]*255;
			}
		}
	}

	cvShowImage("Test", final_seg);

	float output[20];
	double max_prob = INT_MIN;
	int pred_label;
	for(int i=0; i<20; i++)
	{
		output[i] = result[0]->cpu_data()[i];
		cout << output[i] << " " << max_prob;
		if(output[i]  > max_prob){
			max_prob = output[i];	
			pred_label = i;
		}
	}
	cout << endl << pred_label + 1 << endl;
	clock_t end = clock();
	LOG(INFO) <<  "Process time: " << double(end - begin) / CLOCKS_PER_SEC;

	waitKey();
	return 0;
}
