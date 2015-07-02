///////////////////////////////////////////////////////////
////// FCN.cpp
////// Created by Tairui Chen on 2015-07-01
////// Copyright (c) 2015 Tairui Chen. All rights reserved.
///////////////////////////////////////////////////////////

#include <cuda_runtime.h>

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>

#include <opencv/highgui.h>
#include <opencv/cv.h> //cvResize()

#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"


#define IMAGE_SIZE 320

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;

bool debug_info_ = false;



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

vector<Blob<float>*> do_softmax(Blob<float>* result){
	vector<Blob<float>*> top;
	vector<Blob<float>*> bottom;
	top.push_back(result);
	bottom.push_back(result);

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
	return top;
}
	 
int main(int argc, char** argv)
{

	if(!debug_info_) caffe::GlobalInit(&argc, &argv);

	vector<vector<float>> pTable = get_pixel(argv[1]);
	// Test mode
	Caffe::set_phase(Caffe::TEST);

	// mode setting - CPU/GPU
	Caffe::set_mode(Caffe::GPU);

	// gpu device number
	int device_id = 0;
	Caffe::SetDevice(device_id);

	// prototxt
	Net<float> caffe_test_net(argv[2]);

	// caffemodel
	caffe_test_net.CopyTrainedLayersFrom(argv[3]);

	//Debug
	//caffe_test_net.set_debug_info(true);

	clock_t begin = clock();
  
	int i=0, j=0, k=0;
	float mean_val[3] = {104.006, 116.668, 122.678}; // bgr mean

	// input
	vector<Blob<float>*> input_vec;
	Blob<float> blob(1, 3, IMAGE_SIZE, IMAGE_SIZE);
	Blob<float> label_blob(1, 20, 1, 1);

	IplImage *small_image = cvCreateImage(cvSize(IMAGE_SIZE, IMAGE_SIZE), 8, 3);


	caffe::Datum datum;
	//const char* img_name = argv[3]; 
	IplImage *img = cvLoadImage(argv[4]);
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
	if(debug_info_){
		LOG(INFO) << "result blobs: " << result.size() << endl;
		for(auto &out_blob : result){
			LOG(INFO) << out_blob->num() << " " << out_blob->channels() << " " 
				<< out_blob->height() << " " << out_blob->width() ;
		}
	}
	//Blob<float> *out_blob= result[0];
	/*ofstream outfile("cnn_output.txt");
	for (int ind = 0;ind < result[2]->height() * result[2]->width() ; ind++)
	outfile <<  result[2]->cpu_data()[ind] << "   ";
	outfile.close();*/

	for(int p = 0; p < 20; p++){
		if(result[0]->cpu_data()[p] > 0.5)
			label_blob.mutable_cpu_data()[label_blob.offset(0,p,0,0)] = result[0]->cpu_data()[p];
		else
			label_blob.mutable_cpu_data()[label_blob.offset(0,p,0,0)] = 0;
	}
	input_vec.push_back(&label_blob);

	const vector<Blob<float>*>& result_bk = caffe_test_net.Forward(input_vec, &loss);

	Blob<float> *out_blob= do_softmax(result_bk[2])[0];
	Blob<float> *cls_score = result_bk[0];

	vector<Mat> score_map;
	double sum_pix = 0;
	int idx = 0;
	int cnt_max = INT_MIN;
	vector<pair<double, cv::Mat>> mp;
	
	sum_pix = 0;
	Mat seg_map(IMAGE_SIZE,IMAGE_SIZE, CV_32FC1, Scalar(0,0,0));
	for(int i = 0; i < out_blob->height(); i++){
		for(int j=0; j < out_blob->width(); j++){
			seg_map.at<float>(i,j) = out_blob->cpu_data()[idx];	
			idx++;
		}
	}
	score_map.push_back(seg_map);
	imshow("seg_map", seg_map);
	seg_map.release();

	for(int i = 0; i < 20; i++){
		input_vec.clear();
		Mat seg_map(IMAGE_SIZE,IMAGE_SIZE, CV_32FC1, Scalar(0,0,0));
		Blob<float> labels(1, 20, 1, 1);
		for(int p = 0; p < 20; p++)
			labels.mutable_cpu_data()[labels.offset(0,p,0,0)] = 0;

		if(cls_score->cpu_data()[i] > 0.5){
			// push data
			input_vec.push_back(&blob);
			// push label
			labels.mutable_cpu_data()[labels.offset(0,i,0,0)] = cls_score->cpu_data()[i];
			input_vec.push_back(&labels);
			float loss_;
			const vector<Blob<float>*>& cnn_output = caffe_test_net.Forward(input_vec, &loss_);
			Blob<float> *softmax_score = do_softmax(cnn_output[2])[0];

			int ind = 0;
			for(int i = 0; i < softmax_score->height(); i++){
				for(int j = 0; j < softmax_score->width(); j++){
					seg_map.at<float>(i,j) = softmax_score->cpu_data()[ind+IMAGE_SIZE*IMAGE_SIZE];	
					ind++;		
				}
			}

		}
		score_map.push_back(seg_map);
		//imshow("each_channel_map", seg_map);
		//waitKey();
		seg_map.release();
	}
	cout << score_map.size() << endl;

	Mat res_map(IMAGE_SIZE,IMAGE_SIZE, CV_32FC1, Scalar(0,0,0));
	for(int i = 0; i < IMAGE_SIZE; i++){
		for(int j = 0; j < IMAGE_SIZE; j++){
			vector<pair<float, int>> v;
			for(int c = 0; c < score_map.size(); c++){ // for 21 channels
				v.push_back( make_pair(score_map[c].at<float>(i,j), c) );
			}

			std::sort(v.begin(), v.end(), [](const std::pair<float, int> &left, const std::pair<float, int> &right) {
					return left.first >  right.first;} );	
			//apply pixel
			res_map.at<float>(i,j) = v[0].second;
		}
	}

	// restore seg map to image
	Mat	cropped_score_map;
	resize(res_map, cropped_score_map, Size(img->width, img->height));
	sum_pix = 0;
	IplImage *final_seg = cvCreateImage(cvSize(img->width, img->height), 8, 3);
	for(int h = 0; h < cropped_score_map.rows; h++){
		for(int w = 0; w < cropped_score_map.cols; w++){
			int p = cropped_score_map.at<float>(h,w);
			for (int c = 0; c < 3; c++){
				final_seg->imageData[h*img->widthStep+w*img->nChannels+c] = pTable[p][2-c]*255;
			}
		}
	}
			
	cvShowImage("Test", final_seg);

	float output[20];
	double max_prob = INT_MIN;
	int pred_label;
	for(int i = 0; i < 20; i++)
	{
		output[i] = result[0]->cpu_data()[i];
		cout << output[i] << " " << max_prob << endl;
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
