#include "light_glue.h"
#include "super_glue.h"
#include <cfloat>
#include <utility>
#include <unordered_map>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;

LightGlue::LightGlue(const LightGlueConfig &lightglue_config) : lightglue_config_(lightglue_config), engine_(nullptr) {
    setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}
bool LightGlue::deserialize_engine() {
    std::ifstream file(lightglue_config_.engine_file, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        char *model_stream = new char[size];
        file.read(model_stream, size);
        file.close();
        IRuntime *runtime = createInferRuntime(gLogger);
        if (runtime == nullptr) return false;
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
        if (engine_ == nullptr) return false;
        return true;
    }
    return false;
}
void LightGlue::save_engine() {
    if (lightglue_config_.engine_file.empty()) return;
    if (engine_ != nullptr) {
        nvinfer1::IHostMemory *data = engine_->serialize();
        std::ofstream file(lightglue_config_.engine_file, std::ios::binary);;
        if (!file) return;
        file.write(reinterpret_cast<const char *>(data->data()), data->size());
    }
}

bool LightGlue::build() {
    if(deserialize_engine()){
        return true;
    }

    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }

    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) {
        return false;
    }

    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }

    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        return false;
    }
    // keypoint
    profile->setDimensions(lightglue_config_.input_tensor_names[0].c_str(), OptProfileSelector::kMIN, Dims3(1, 1, 2));
    profile->setDimensions(lightglue_config_.input_tensor_names[0].c_str(), OptProfileSelector::kOPT, Dims3(1, 512, 2));
    profile->setDimensions(lightglue_config_.input_tensor_names[0].c_str(), OptProfileSelector::kMAX,
                           Dims3(1, 1024, 2));

    profile->setDimensions(lightglue_config_.input_tensor_names[1].c_str(), OptProfileSelector::kMIN, Dims3(1, 1, 2));
    profile->setDimensions(lightglue_config_.input_tensor_names[1].c_str(), OptProfileSelector::kOPT, Dims3(1, 512, 2));
    profile->setDimensions(lightglue_config_.input_tensor_names[1].c_str(), OptProfileSelector::kMAX,
                           Dims3(1, 1024, 2));
 
    profile->setDimensions(lightglue_config_.input_tensor_names[2].c_str(), OptProfileSelector::kMIN, Dims3(1, 1, 256));
    profile->setDimensions(lightglue_config_.input_tensor_names[2].c_str(), OptProfileSelector::kOPT,
                           Dims3(1, 512, 256));
    profile->setDimensions(lightglue_config_.input_tensor_names[2].c_str(), OptProfileSelector::kMAX,
                           Dims3(1, 1024, 256));

    profile->setDimensions(lightglue_config_.input_tensor_names[3].c_str(), OptProfileSelector::kMIN, Dims3(1, 1, 256));
    profile->setDimensions(lightglue_config_.input_tensor_names[3].c_str(), OptProfileSelector::kOPT,
                           Dims3(1, 512, 256));
    profile->setDimensions(lightglue_config_.input_tensor_names[3].c_str(), OptProfileSelector::kMAX,
                           Dims3(1, 1024, 256));
    config->addOptimizationProfile(profile);

    auto constructed = construct_network(builder, network, config, parser);
    if (!constructed) {
        return false;
    }

    auto profile_stream = makeCudaStream();
    if (!profile_stream) {
        return false;
    }
    config->setProfileStream(*profile_stream);

    TensorRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    TensorRTUniquePtr<IRuntime> runtime{createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime) {
        return false;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
        return false;
    }

    save_engine();

    ASSERT(network->getNbInputs() == 4);
    keypoints_0_dims_ = network->getInput(0)->getDimensions();
    keypoints_1_dims_ = network->getInput(1)->getDimensions();
    descriptors_0_dims_ = network->getInput(2)->getDimensions();
    descriptors_1_dims_ = network->getInput(3)->getDimensions();
    assert(keypoints_0_dims_.d[1] == -1);
    assert(descriptors_0_dims_.d[1] == -1);
    assert(keypoints_1_dims_.d[1] == -1);
    assert(descriptors_1_dims_.d[1] == -1);
    return true;
}

bool LightGlue::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                                  TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                  TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                  TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
    auto parsed = parser->parseFromFile(lightglue_config_.onnx_file.c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }
    config->setMaxWorkspaceSize(512_MiB);
    config->setFlag(BuilderFlag::kFP16);
    enableDLA(builder.get(), config.get(), lightglue_config_.dla_core);
    return true;
}

Eigen::Matrix<double, 259, Eigen::Dynamic> LightGlue::normalize_keypoints(const Eigen::Matrix<double, 259,
                                                                          Eigen::Dynamic> &features,
                                                                          int width, int height) {
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features;
  norm_features.resize(259, features.cols());
  norm_features = features;
  for (int col = 0; col < features.cols(); ++col) {
    norm_features(1, col) =
        (features(1, col) - width / 2) / (std::max(width, height) * 0.7);
    norm_features(2, col) =
        (features(2, col) - height / 2) / (std::max(width, height) * 0.7);
  }
  return norm_features;
}


int LightGlue::matching_points(Eigen::Matrix<double, 259, Eigen::Dynamic>& features0,
                                  Eigen::Matrix<double, 259, Eigen::Dynamic>& features1,
                                  std::vector<cv::DMatch>& matches, bool outlier_rejection){
  matches.clear();
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features0 = normalize_keypoints(features0,
                                                              lightglue_config_.image_width, lightglue_config_.image_height);
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features1 = normalize_keypoints(features1,
                                                              lightglue_config_.image_width, lightglue_config_.image_height);
  Eigen::VectorXi indices0, indices1;
  Eigen::VectorXd mscores0, mscores1;
  infer(norm_features0, norm_features1, indices0, indices1, mscores0, mscores1);

  int num_match = 0;
  std::vector<cv::Point2f> points0, points1;
  std::vector<int> point_indexes;
  for(size_t i = 0; i < indices0.size(); i++){
    if(indices0(i) < indices1.size() && indices0(i) >= 0 && indices1(indices0(i)) == i){
      double d = 1.0 - (mscores0[i] + mscores1[indices0[i]]) / 2.0;
      matches.emplace_back(i, indices0[i], d);
      points0.emplace_back(features0(1, i), features0(2, i));
      points1.emplace_back(features1(1, indices0(i)), features1(2, indices0(i)));
      num_match++;
    }
  }

  if(outlier_rejection){
    std::vector<uchar> inliers;
    cv::findFundamentalMat(points0, points1, cv::FM_RANSAC, 3, 0.99, inliers);
    int j = 0;
    for(int i = 0; i < matches.size(); i++){
      if(inliers[i]){
        matches[j++] = matches[i];
      }
    }
    matches.resize(j);
  }

  return matches.size();
}


bool LightGlue::infer(const Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                      const Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
                      Eigen::VectorXi &indices0,
                      Eigen::VectorXi &indices1,
                      Eigen::VectorXd &mscores0,
                      Eigen::VectorXd &mscores1) {
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            return false;
    }
    }

    assert(engine_->getNbBindings() == 5);

    const int keypoints_0_index = engine_->getBindingIndex(lightglue_config_.input_tensor_names[0].c_str());
    const int keypoints_1_index = engine_->getBindingIndex(lightglue_config_.input_tensor_names[1].c_str());
    const int descriptors_0_index = engine_->getBindingIndex(lightglue_config_.input_tensor_names[2].c_str());
    const int descriptors_1_index = engine_->getBindingIndex(lightglue_config_.input_tensor_names[3].c_str());

    const int output_score_index = engine_->getBindingIndex(lightglue_config_.output_tensor_names[0].c_str());


    context_->setBindingDimensions(keypoints_0_index, Dims3(1, features0.cols(), 2));
    context_->setBindingDimensions(keypoints_1_index, Dims3(1, features1.cols(), 2));
    context_->setBindingDimensions(descriptors_0_index, Dims3(1, features0.cols(), 256));
    context_->setBindingDimensions(descriptors_1_index, Dims3(1, features1.cols(), 256));

    keypoints_0_dims_ = context_->getBindingDimensions(keypoints_0_index);
    descriptors_0_dims_ = context_->getBindingDimensions(descriptors_0_index);
    keypoints_1_dims_ = context_->getBindingDimensions(keypoints_1_index);
    descriptors_1_dims_ = context_->getBindingDimensions(descriptors_1_index);

    output_scores_dims_ = context_->getBindingDimensions(output_score_index);

    BufferManager buffers(engine_, 0, context_.get());

    ASSERT(lightglue_config_.input_tensor_names.size() == 4);
    if (!process_input(buffers, features0, features1)) {
        return false;
    }

    buffers.copyInputToDevice();

    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        return false;
    }
    buffers.copyOutputToHost();

    if (!process_output(buffers, indices0, indices1, mscores0, mscores1)) {
        return false;
    }

    return true;
}

bool LightGlue::process_input(const BufferManager &buffers,
                              const Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                              const Eigen::Matrix<double, 259, Eigen::Dynamic> &features1) {
    auto *keypoints_0_buffer = static_cast<float *>(buffers.getHostBuffer(lightglue_config_.input_tensor_names[0]));
    auto *keypoints_1_buffer = static_cast<float *>(buffers.getHostBuffer(lightglue_config_.input_tensor_names[1]));
    auto *descriptors_0_buffer = static_cast<float *>(buffers.getHostBuffer(lightglue_config_.input_tensor_names[2]));
    auto *descriptors_1_buffer = static_cast<float *>(buffers.getHostBuffer(lightglue_config_.input_tensor_names[3]));


    for (int colk0 = 0; colk0 < features0.cols(); ++colk0) {
        for (int rowk0 = 1; rowk0 < 3; ++rowk0) {
            keypoints_0_buffer[colk0 * 2 + (rowk0 - 1)] = features0(rowk0, colk0);
        }
    }

    for (int rowd0 = 3; rowd0 < features0.rows(); ++rowd0) {
        for (int cold0 = 0; cold0 < features0.cols(); ++cold0) {
            descriptors_0_buffer[(rowd0 - 3) * features0.cols() + cold0] = features0(rowd0, cold0);
        }
    }

    for (int colk1 = 0; colk1 < features1.cols(); ++colk1) {
        for (int rowk1 = 1; rowk1 < 3; ++rowk1) {
            keypoints_1_buffer[colk1 * 2 + (rowk1 - 1)] = features1(rowk1, colk1);
        }
    }

    for (int rowd1 = 3; rowd1 < features1.rows(); ++rowd1) {
        for (int cold1 = 0; cold1 < features1.cols(); ++cold1) {
            descriptors_1_buffer[(rowd1 - 3) * features1.cols() + cold1] = features1(rowd1, cold1);
        }
    }

    return true;
}


void LightGlue::decode(float *scores, int h, int w, std::vector<int> &indices0, std::vector<int> &indices1,
            std::vector<double> &mscores0, std::vector<double> &mscores1) {
    auto *max_indices0 = new int[h - 1];
    auto *max_indices1 = new int[w - 1];
    auto *max_values0 = new float[h - 1];
    auto *max_values1 = new float[w - 1];
    SuperGlue::max_matrix(scores, max_indices0, max_values0, h, w, 2);
    SuperGlue::max_matrix(scores, max_indices1, max_values1, h, w, 1);
    auto *mutual0 = new int[h - 1];
    auto *mutual1 = new int[w - 1];
    SuperGlue::equal_gather(max_indices1, max_indices0, mutual0, h - 1);
    SuperGlue::equal_gather(max_indices0, max_indices1, mutual1, w - 1);
    SuperGlue::where_exp(mutual0, max_values0, mscores0, h - 1);
    SuperGlue::where_gather(mutual1, max_indices1, mscores0, mscores1, w - 1);
    auto *valid0 = new int[h - 1];
    auto *valid1 = new int[w - 1];
    SuperGlue::and_threshold(mutual0, valid0, mscores0, 0.01);
    SuperGlue::and_gather(mutual1, valid0, max_indices1, valid1, w - 1);
    SuperGlue::where_negative_one(valid0, max_indices0, h - 1, indices0);
    SuperGlue::where_negative_one(valid1, max_indices1, w - 1, indices1);
    delete[] max_indices0;
    delete[] max_indices1;
    delete[] max_values0;
    delete[] max_values1;
    delete[] mutual0;
    delete[] mutual1;
    delete[] valid0;
    delete[] valid1;
}


bool LightGlue::process_output(const BufferManager &buffers,
                               Eigen::VectorXi &indices0,
                               Eigen::VectorXi &indices1,
                               Eigen::VectorXd &mscores0,
                               Eigen::VectorXd &mscores1) {
    indices0_.clear();
    indices1_.clear();
    mscores0_.clear();
    mscores1_.clear();
    auto *output_score = static_cast<float *>(buffers.getHostBuffer(lightglue_config_.output_tensor_names[0]));

    int scores_map_h = output_scores_dims_.d[1];
    int scores_map_w = output_scores_dims_.d[2];
    auto *scores = new float[(scores_map_h + 1) * (scores_map_w + 1)];
    // log_optimal_transport(output_score, scores, scores_map_h, scores_map_w);
    // scores_map_h = scores_map_h + 1;
    // scores_map_w = scores_map_w + 1;
    decode(output_score, scores_map_h, scores_map_w, indices0_, indices1_, mscores0_, mscores1_);
    indices0.resize(indices0_.size());
    indices1.resize(indices1_.size());
    mscores0.resize(mscores0_.size());
    mscores1.resize(mscores1_.size());
    for (int i0 = 0; i0 < indices0_.size(); ++i0) {
        indices0(i0) = indices0_[i0];
    }
    for (int i1 = 0; i1 < indices1_.size(); ++i1) {
        indices1(i1) = indices1_[i1];
    }
    for (int j0 = 0; j0 < mscores0_.size(); ++j0) {
        mscores0(j0) = mscores0_[j0];
    }
    for (int j1 = 0; j1 < mscores1_.size(); ++j1) {
        mscores1(j1) = mscores1_[j1];
    }
    return true;
}


