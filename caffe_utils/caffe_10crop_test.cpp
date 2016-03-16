#include "caffe/caffe.hpp"
#include "opencv2/opencv.hpp"
#include "gflags/gflags.h"
#include "caffe/layers/memory_data_layer.hpp"

using namespace caffe;
using namespace cv;
using namespace std;

DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_string(imagefile, "",
  "The image file path.");
DEFINE_string(labelfile, "",
  "The label file path.");

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}


int main(int argc, char* argv[])
{
  FLAGS_alsologtostderr = 1;
  caffe::GlobalInit(&argc, &argv);
   // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  
  
  LOG(INFO)<<"reading model from "<<FLAGS_model;
  Net<float> caffe_test_net(FLAGS_model,TEST);
  LOG(INFO)<<"reading weights from "<<FLAGS_weights;
  caffe_test_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO)<<"reading label info from "<<FLAGS_labelfile;
  NetParameter net_param;
  // For intermediate results, we will also dump the gradient values.
  caffe_test_net.ToProto(&net_param, false);
  // char iter_str_buffer[256];
  // sprintf(iter_str_buffer, 256, "thinned_net.caffemodel");
  // string iter_str_buffer = "thinned_net.caffemodel";
  // LOG(INFO) << "Snapshotting to " << iter_str_buffer;
  // WriteProtoToBinaryFile(net_param, iter_str_buffer.c_str());
  // ifstream labels_file(FLAGS_labelfile,ios_base::in);
  std::ifstream labels_file(FLAGS_labelfile.c_str());
  vector<string> labels;
  char *buf = new char[256];
  while(!labels_file.eof())
  {
    labels_file.getline(buf,256);
    labels.push_back(string(buf));
  }

  std::ifstream infile("val.txt");
    std::vector<std::pair<std::string, int> > lines;
    std::string filename;
    int label;
    while (infile >> filename >> label) {
      lines.push_back(std::make_pair(filename, label));
    }
    LOG(INFO) << "A total of " << lines.size() << " images.";


  //cv::Mat image = cv::imread(FLAGS_imagefile, 1);
  int W = 256, H = 256;
  //resize(image,image,cv::Size(W,H));
  ////image = image.t();
  //cv::Mat imgFloat;
  //image.convertTo(imgFloat,CV_32F);//,0.00390625
  /*ReadImageToDatum(FLAGS_imagefile,1,W,H,1,&datum);*/
  int line_num = 0;
  float top1_acc = 0;
  float top5_acc = 0;
  Datum datum;
  for(int line_id  = 0; line_id  < lines.size(); line_id++){
    line_num++;
    cv::Mat sample_resized; // = ReadImageToCVMat(FLAGS_imagefile, W, H, 1);
    string im_name = "/home/tairuic/Downloads/caffe_pkg/caffe/data/ilsvrc12/ILSVRC2012_img_val/" + lines[line_id].first;
    bool oversample = true;
    vector<Datum> datum_vector;
    
      cv::Mat cv_img_origin = cv::imread(im_name, 1);
      label = lines[line_id].second;
      int resize_width, resize_height;
      if (cv_img_origin.cols < cv_img_origin.rows){
        resize_width = std::max<int>(0, W);
        resize_height = resize_width * cv_img_origin.rows / cv_img_origin.cols;
      }
      else{
        resize_height = std::max<int>(0, H);
        resize_width = resize_height * cv_img_origin.cols / cv_img_origin.rows;
      }
      cv::resize(cv_img_origin, sample_resized, cv::Size(resize_width, resize_height));
      // cv::Scalar channel_mean(104.007, 116.669, 122.679);
      // cv::Mat mean_ = cv::Mat(cv::Size(resize_width, resize_height), cv_img_origin.type(), channel_mean);
      
      // cv::imshow("image", sample_resized);
      // cv::waitKey();

      // cv::Mat sample_normalized;
      // cv::subtract(sample_resized, mean_, sample_normalized);

      // vector<Datum> datum_vector;
      int crop_dims = 224;
      //LOG(INFO) << "RESIZE: " << resize_width << " " << resize_height;
      int im_center_w = resize_width / 2;
      int im_center_h = resize_height / 2;
      int myints[] = { 0, resize_width - crop_dims};
      std::vector<int> w_indices(myints, myints + sizeof(myints) / sizeof(int));

      int myints2[] = { 0, resize_height - crop_dims};
      std::vector<int> h_indices(myints2, myints2 + sizeof(myints2) / sizeof(int));
      vector<cv::Rect> myROIs;
      for (int i = 0; i < w_indices.size(); i++ ){
        for (int j = 0; j < h_indices.size(); j++ ){
          // LOG(INFO) << w_indices[i] << " " <<  h_indices[j] << " " 
          //    << w_indices[i] +crop_dims <<  " " << h_indices[j] +crop_dims;
          myROIs.push_back(cv::Rect(w_indices[i],h_indices[j], crop_dims, crop_dims));
        }
      }
      myROIs.push_back(cv::Rect(im_center_w - crop_dims / 2, im_center_h - crop_dims / 2, crop_dims, crop_dims));
      
    if(oversample){
      for (int i = 0; i < myROIs.size();i++){
        //LOG(INFO) << "crop data start";
        cv::Rect myROI = myROIs[i];
        
        //LOG(INFO) << myROI.x << myROI.y << myROI.width << myROI.height;
        cv::Mat img = sample_resized(myROI);
        // cv::imshow("croppedImage", croppedImage);
        // cv::waitKey();
        cv::Mat padded;
        int padding = 16;
        padded.create(img.rows + 2*padding, img.cols + 2*padding, img.type());
        padded.setTo(cv::Scalar::all(0));

        img.copyTo(padded(Rect(padding, padding, img.cols, img.rows)));
        // cv::imshow("croppedImage", padded);
        // cv::waitKey();


        CVMatToDatum(padded, &datum);
        datum.set_label(label);
        datum_vector.push_back(datum);

        cv::Mat flip_image;
        cv::flip(img, flip_image, 1);

        padded.create(flip_image.rows + 2*padding, flip_image.cols + 2*padding, flip_image.type());
        padded.setTo(cv::Scalar::all(0));

        flip_image.copyTo(padded(Rect(padding, padding, flip_image.cols, flip_image.rows)));
        // cv::imshow("croppedImage", padded);
        // cv::waitKey();


        CVMatToDatum(padded, &datum);
        datum.set_label(label);
        datum_vector.push_back(datum);

      }
    }
    else{
        cv::Rect myROI = myROIs[4];
        
        //LOG(INFO) << myROI.x << myROI.y << myROI.width << myROI.height;
        cv::Mat img = sample_resized(myROI);
        // cv::imshow("croppedImage", croppedImage);
        // cv::waitKey();

        cv::Mat padded;
        int padding = 16;
        padded.create(img.rows + 2*padding, img.cols + 2*padding, img.type());
        padded.setTo(cv::Scalar::all(0));

        img.copyTo(padded(Rect(padding, padding, img.cols, img.rows)));


        CVMatToDatum(padded, &datum);
        datum.set_label(label);
        datum_vector.push_back(datum);
    }
    // LOG(INFO) << "crop data end";
    MemoryDataLayer<float>* data_layer_ptr = (MemoryDataLayer<float>*)&(*caffe_test_net.layers()[0]);
    data_layer_ptr->AddDatumVector(datum_vector);
    const vector<Blob<float>*>& result = caffe_test_net.Forward();
    // LOG(INFO) << "RESULT SIZE: " << result.size();
    /*for(int i=0;i<5;i++)
    {
      LOG(ERROR)<<"top "<<i+1<<":"<< labels[ result[0]->cpu_data()[i] ]<<" "<<result[0]->cpu_data()[i+5];
    }*/
    // LOG(ERROR) << "top "  << result[0]->shape_string(); //label
    // LOG(ERROR) << "top "  << result[1]->shape_string(); //prob
    vector<float> mean_prob;
    // // float check_sum = 0;
    for(int i = 0; i < 1000; i++){
      float mean_sum = 0;
      for(int j = 0; j < 10; j++){
        mean_sum  += result[1]->cpu_data()[1000 * j + i];
      }
      // check_sum += mean_sum/10;
      mean_prob.push_back(mean_sum/10);
    }

    // for(int i = 0; i < 1000; i++){
    //   mean_prob.push_back(result[1]->cpu_data()[i]);
    // }

    // LOG(INFO) << check_sum;
    // LOG(INFO) << "prob size: " << mean_prob.size();

    std::vector<std::pair<float, int> > pairs;
      for (size_t i = 0; i < mean_prob.size(); ++i)
        pairs.push_back(std::make_pair(mean_prob[i], i));
      std::partial_sort(pairs.begin(), pairs.begin() + 1000, pairs.end(), PairCompare);

      // std::vector<int> top5;
      
      if(pairs[0].second == label) top1_acc++;
      for (int i = 0; i < 5; ++i){
        // top5.push_back(pairs[i].second);
        LOG(INFO) << pairs[i].second;
        if(pairs[i].second == label){
          top5_acc++;
          // break;
        }
      }

      LOG(INFO) << "image: "<< line_num << " top1 acc: " <<  top1_acc/line_num<< " top5 acc: " << top5_acc/line_num;


  }
    // LOG(INFO) << pairs[0].second;



  // for (int i = 0; i < 20; i++)
  // {
  //  //LOG(ERROR)<<"top "<<i+1<<":"<< labels[ result[0]->cpu_data()[i] ]<<" "<<result[0]->cpu_data()[i+5];
  //  LOG(ERROR) << "crop " << i + 1 << ":" << labels[result[0]->cpu_data()[i]] << " " << result[0]->cpu_data()[i + 1];
  // }
  //test_accuracy+=result[0]->cpu_data()[0];

  return 0;
}

