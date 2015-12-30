#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <sstream>


#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;



struct IdxCompare
{
    std::vector<float>& target;

    IdxCompare(std::vector<float>& target): target(target) {}

    bool operator()(float a, float b) const { return target[a] < target[b]; }
};

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<float> Classify(const cv::Mat& img, std::ofstream &lfw_feature, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  // Blob<float>* output_layer = net_->output_blobs()[0];
  // CHECK_EQ(labels_.size(), output_layer->channels())
  //   << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<float> Classifier::Classify(const cv::Mat& img, std::ofstream &lfw_feature, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  //std::cout << output.size() << std::endl;
  for (int i = 0; i < output.size()-1; ++i) {
    // int idx = maxN[i];
    // predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    lfw_feature << output[i] << ",";
  }
  lfw_feature << output[output.size()-1] << "\n";

  return output;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  //net_->ForwardPrefilled();
  net_->ForwardPrefilled();
  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}



float L2Distance(std::vector<float> f1, std::vector<float> f2) {
  float ret = 0.0;
  for(int i = 0; i < f1.size(); i++) {
    float dist = f1[i] - f2[i];
    ret += dist * dist;
  }
  return ret > 0.0 ? sqrt(ret) : 0.0;
}

int main(int argc, char** argv) {
  // if (argc != 6) {
  //   std::cerr << "Usage: " << argv[0]
  //             << " deploy.prototxt network.caffemodel"
  //             << " mean.binaryproto labels.txt img.jpg" << std::endl;
  //   return 1;
  // }

  // ::google::InitGoogleLogging(argv[0]);

  // string model_file   = argv[1];
  // string trained_file = argv[2];
  // string mean_file    = argv[3];
  // string label_file   = argv[4];
  const int index = atoi(argv[1]);
  string model_file   = "models/VGG16_M/vgg_m_16_layers.prototxt";
  string trained_file = "models/VGG16_M/vgg_m_16_layers.caffemodel";
  string mean_file    = "data/ilsvrc12/imagenet_mean.binaryproto";
  string label_file   = "data/ilsvrc12/synset_words.txt";

  Classifier classifier(model_file, trained_file, mean_file, label_file);

  std::ifstream image_list;
  image_list.open ("data/lfw_list.txt", ios::out | ios::app | ios::binary); 

//#define write_feat
#ifdef write_feat
  std::ofstream lfw_feature;
  lfw_feature.open ("data/lfw_feature.bin");
  int cnt = 0;
#endif
  // //lfw_feature << "Writing this to a file.\n";
  std::vector<string> v_image_list;
  if (image_list.is_open()){
    string file; //;= "examples/images/cat.jpg";
    
    while(!image_list.eof())
    {
        image_list >> file;       
        //std::cout << cnt << "---------- Prediction for "
              //<< file << " ----------" << std::endl;      
        string img_name = "/home/tairuic/Downloads/" + file;
        v_image_list.push_back(img_name);

#ifdef write_feat
        cnt++;
        std::cout << cnt << std::endl;
        cv::Mat img = cv::imread(img_name, -1);
        CHECK(!img.empty()) << "Unable to decode image " << file;
        std::vector<float> predictions = classifier.Classify(img, lfw_feature);
#endif
    }
  }
#ifdef write_feat
  lfw_feature.close();
#endif


  std::ifstream infile;
  infile.open ("data/lfw_feature.bin", ios::out | ios::app | ios::binary); 
  std::vector< std::vector<float> > v_feature;
  std::vector<float> v;
  // char cNum[20] ;
  // int feat_num = 0;
  if (infile.is_open())
  {
      // while (infile.good())
      // {
      //         infile.getline(cNum, 256, ',');
      //         v.push_back(atof(cNum)) ;
      //         //std::cout << atof(cNum) << " ";
      //         feat_num++;
      // }
      // infile.close();

      // string line;
      // while(!infile.eof())
      // {
      //     infile >> line;
      //     //std::cout << line;
      //     feat_num++;
      // }

      // string line;
      while(!infile.eof()){
        string s;
        infile >> s;
        std::stringstream ss(s);
        vector <float> record;

        while (ss)
        {
          string s;
          if (!getline( ss, s, ',' )) break;
          record.push_back( atof(s.c_str()) );
        }

        v_feature.push_back( record );

      }

  }
  else
  {
          std::cout << "Error opening file";
  }
 

  std::ofstream lfw_feature_one;
  lfw_feature_one.open ("data/lfw_feature_one.bin");
  cv::Mat img = cv::imread(v_image_list[index], -1);
  std::vector<float> v1 = classifier.Classify(img, lfw_feature_one);
  std::cout << "number of features: " << v1.size() << std::endl;
  lfw_feature_one.close();
  // std::vector<float> v2 = v_feature[100];
  std::cout << v_feature.size() << std::endl;
  clock_t begin = clock();
  std::vector<float> scores;
  for(int i = 0; i < v_feature.size() -1;i++ ){
    std::vector<float> v2 = v_feature[i];
    scores.push_back(L2Distance(v1,v2));
  }
  //std::cout << "scores: "<< scores.size() << std::endl;
  std::vector<int> y;
  // initialize indexes
  for(size_t i = 0; i < scores.size(); ++i)
      y.push_back(i);

  std::sort(y.begin(), y.end(), IdxCompare(scores));
  std::cout << "top 5: " <<  std::endl;
  clock_t end = clock();
  imshow("raw", img);
  for(size_t i = 0; i < 5; ++i){
        std::cout << y[i] << '\n';
        cv::Mat ref_img = cv::imread(v_image_list[y[i]], -1);
        if(i != 0){
          imshow("ref", ref_img);
          cv::waitKey();
        }
  }
  //std::cout << "L2 Distance: " << L2Distance(v1,v2) << std::endl;
  
  cv::waitKey();
  
  std::cout << "Process time: " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
