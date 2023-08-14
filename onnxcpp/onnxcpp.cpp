#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <onnxruntime_cxx_api.h>
using namespace std;
std::vector<std::string> loadLabels(const std::string& filename);
int main()
{
    const std::string labelFile = "C:/Users/chesa/source/repos/onnxcpp/onnxcpp/assets/lb.txt";
    auto modelPath = L"C:/Users/chesa/source/repos/onnxcpp/onnxcpp/assets/colormodel.onnx";
    // Load the model and create InferenceSession
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(env, modelPath, Ort::SessionOptions{ nullptr });

    std::vector<std::string> labels = loadLabels(labelFile);
    if (labels.empty()) {
        std::cout << "Failed to load labels: " << labelFile << std::endl;
        return 1;
    }
    constexpr int64_t numClasses = 11;
   


    const std::array<int64_t, 2> inputShape = { 1, 3 };
    const std::array<int64_t, 2> outputShape = { 1, numClasses };

    std::array<float, 3> input = {255, 190, 3};
    std::array<float, numClasses> results;
    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    // define names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char*, 1> inputNames = { inputName.get() };
    const std::array<const char*, 1> outputNames = { outputName.get() };
    inputName.release();
    outputName.release();


    // run inference
    try {
        session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    // sort results
    std::vector<std::pair<size_t, float>> indexValuePairs;
    for (size_t i = 0; i < results.size(); ++i) {
        indexValuePairs.emplace_back(i, results[i]);
    }
    //std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
    // show Top5
    for (size_t i = 0; i < 11; ++i) {
        const auto& result = indexValuePairs[i];
        std::cout << i + 1 << ": " << labels[result.first] << " " << result.second << std::endl;
    }
}

std::vector<std::string> loadLabels(const std::string& filename)
{
    std::vector<std::string> output;

    std::ifstream file(filename);
    if (file) {
        std::string s;
        while (getline(file, s)) {
            output.emplace_back(s);
        }
        file.close();
    }

    return output;
}