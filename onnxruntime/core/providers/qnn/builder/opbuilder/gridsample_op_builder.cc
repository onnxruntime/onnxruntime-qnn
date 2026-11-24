// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/qnn_graph_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

// GridSample can arrive in several domains. They share the QNN lowering but differ in attribute encoding and layout:
//  - kOnnx: ONNX GridSample (or its NHWC-internal variant). String mode/padding_mode. ORT's layout transformer
//    converts the data layout, so the builder emits no transposes.
//  - kMsContrib: com.microsoft:GridSample. String attributes like ONNX, but the node stays NCHW (4D only), so the
//    builder inserts the transposes itself.
//  - kOrtContrib: ai.onnx.contrib:GridSample. INT attributes using the QNN enum values, NC(D)HW input and a
//    channel-last grid. The builder inserts the transposes itself.
enum class GridSampleDomain { kOnnx,
                              kMsContrib,
                              kOrtContrib,
                              kUnsupported };

GridSampleDomain GetGridSampleDomain(const OrtNodeUnit& node_unit) {
  const std::string domain = node_unit.Domain();
  if (domain == kOnnxDomain || domain == kMSInternalNHWCDomain) {
    return GridSampleDomain::kOnnx;
  }
  if (domain == kMSDomain) {
    return GridSampleDomain::kMsContrib;
  }
  if (domain == kOrtContribDomain) {
    return GridSampleDomain::kOrtContrib;
  }
  return GridSampleDomain::kUnsupported;
}

struct GridSampleParams {
  uint32_t mode = QNN_OP_GRID_SAMPLE_MODE_BILINEAR;
  uint32_t padding_mode = QNN_OP_GRID_SAMPLE_PADDING_MODE_ZEROS;
  bool align_corners = false;
};

// Maps an attribute C++ type to its ONNX attribute type tag and a printable name.
template <typename T>
struct AttrTraits;
template <>
struct AttrTraits<int64_t> {
  static constexpr OrtOpAttrType kType = ORT_OP_ATTR_INT;
  static constexpr const char* kName = "INT";
};
template <>
struct AttrTraits<std::string> {
  static constexpr OrtOpAttrType kType = ORT_OP_ATTR_STRING;
  static constexpr const char* kName = "STRING";
};

// Reads an attribute of type T. An absent attribute yields def_val; an attribute of any other type is an error
// (OrtNodeAttrHelper::Get silently returns the default on a type mismatch).
template <typename T>
Ort::Status GetStrictAttr(const OrtNodeUnit& node_unit, const std::string& key, const T& def_val, T& out) {
  Ort::ConstOpAttr attr;
  Ort::Status status = Ort::ConstNode(&node_unit.GetNode()).GetAttributeByName(key, attr);
  if (!status.IsOK() || attr == nullptr) {
    out = def_val;
    return Ort::Status();
  }
  RETURN_IF_NOT(attr.GetType() == AttrTraits<T>::kType,
                ("GridSample attribute '" + key + "' must be of type " + AttrTraits<T>::kName + ".").c_str());
  return attr.GetValue<T>(out);
}

// One accepted attribute value (K is int64_t for ai.onnx.contrib, const char* for ONNX / com.microsoft) and the QNN
// enum it lowers to.
template <typename K>
struct EnumEntry {
  K key;
  uint32_t qnn_value;
};

std::string DescribeKey(int64_t key) { return std::to_string(key); }
std::string DescribeKey(const std::string& key) { return key; }
std::string DescribeKey(const char* key) { return key; }

// Reads attribute `key` (type T) and maps it through `table` to a QNN enum value. Absent -> `def_val`.
// An unlisted value is an error that names the accepted values, so the table is the single source of truth.
template <typename T, typename K, size_t N>
Ort::Status ParseEnumAttr(const OrtNodeUnit& node_unit, const std::string& key, const T& def_val,
                          const EnumEntry<K> (&table)[N], uint32_t& out) {
  T value;
  RETURN_IF_ERROR(GetStrictAttr<T>(node_unit, key, def_val, value));
  for (const auto& entry : table) {
    if (value == entry.key) {
      out = entry.qnn_value;
      return Ort::Status();
    }
  }
  std::string supported;
  for (const auto& entry : table) {
    supported += (supported.empty() ? "" : ", ") + DescribeKey(entry.key);
  }
  return MAKE_EP_FAIL(("GridSample attribute '" + key + "' has unsupported value '" + DescribeKey(value) +
                       "'. Supported values: " + supported + ".")
                          .c_str());
}

// ai.onnx.contrib uses INT enums with the QNN/PyTorch numbering. Only values with a QNN equivalent are listed
// (mode 2 = bicubic has none).
constexpr EnumEntry<int64_t> kContribModes[] = {{0, QNN_OP_GRID_SAMPLE_MODE_BILINEAR},
                                                {1, QNN_OP_GRID_SAMPLE_MODE_NEAREST}};
constexpr EnumEntry<int64_t> kContribPaddingModes[] = {{0, QNN_OP_GRID_SAMPLE_PADDING_MODE_ZEROS},
                                                       {1, QNN_OP_GRID_SAMPLE_PADDING_MODE_BORDER},
                                                       {2, QNN_OP_GRID_SAMPLE_PADDING_MODE_REFLECTION}};

// ONNX and com.microsoft use string attributes. cubic/bicubic have no QNN equivalent and are not listed.
constexpr EnumEntry<const char*> kStringModes[] = {{"linear", QNN_OP_GRID_SAMPLE_MODE_BILINEAR},
                                                   {"bilinear", QNN_OP_GRID_SAMPLE_MODE_BILINEAR},
                                                   {"nearest", QNN_OP_GRID_SAMPLE_MODE_NEAREST}};
constexpr EnumEntry<const char*> kStringPaddingModes[] = {{"zeros", QNN_OP_GRID_SAMPLE_PADDING_MODE_ZEROS},
                                                          {"border", QNN_OP_GRID_SAMPLE_PADDING_MODE_BORDER},
                                                          {"reflection", QNN_OP_GRID_SAMPLE_PADDING_MODE_REFLECTION}};

// Parses the node's attributes once; the result feeds both validation and lowering.
Ort::Status ParseGridSampleAttrs(const OrtNodeUnit& node_unit, GridSampleDomain domain, GridSampleParams& params) {
  RETURN_IF(domain == GridSampleDomain::kUnsupported,
            ("QNN EP does not support GridSample in domain '" + node_unit.Domain() + "'.").c_str());

  int64_t align_corners = 0;
  RETURN_IF_ERROR(GetStrictAttr<int64_t>(node_unit, "align_corners", 0, align_corners));
  RETURN_IF(align_corners != 0 && align_corners != 1, "GridSample align_corners must be 0 or 1.");
  params.align_corners = align_corners != 0;

  if (domain == GridSampleDomain::kOrtContrib) {
    RETURN_IF_ERROR(ParseEnumAttr<int64_t>(node_unit, "mode", 0, kContribModes, params.mode));
    return ParseEnumAttr<int64_t>(node_unit, "padding_mode", 0, kContribPaddingModes, params.padding_mode);
  }

  RETURN_IF_ERROR(ParseEnumAttr<std::string>(node_unit, "mode", "linear", kStringModes, params.mode));
  return ParseEnumAttr<std::string>(node_unit, "padding_mode", "zeros", kStringPaddingModes, params.padding_mode);
}

Ort::Status AddGridSampleParams(QnnModelWrapper& qnn_model_wrapper,
                                const OrtNodeUnit& node_unit,
                                const GridSampleParams& params,
                                std::vector<std::string>& param_tensor_names) {
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), params.align_corners,
                                     QNN_OP_GRID_SAMPLE_PARAM_ALIGN_CORNERS, param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), params.mode,
                                         QNN_OP_GRID_SAMPLE_PARAM_MODE, param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         params.padding_mode, QNN_OP_GRID_SAMPLE_PARAM_PADDING_MODE,
                                         param_tensor_names));
  return Ort::Status();
}

// GridSample for com.microsoft / ai.onnx.contrib: the node reaches the builder in NC(D)HW layout (ORT's layout
// transformer does not convert these domains), so emit X -> Transpose(NHWC) -> QNN GridSample -> Transpose(NCHW).
// The grid is already channel-last and is not transposed.
Ort::Status AddGridSampleWithLayoutTransposes(QnnModelWrapper& qnn_model_wrapper,
                                              const OrtNodeUnit& node_unit,
                                              const std::vector<std::string>& input_names,
                                              std::vector<std::string>&& param_tensor_names,
                                              bool do_op_validation) {
  RETURN_IF_NOT(input_names.size() == 2, "GridSample requires exactly 2 inputs.");

  TensorInfo x_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], x_info));
  TensorInfo y_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], y_info));

  const size_t rank = x_info.shape.size();
  RETURN_IF_NOT(rank == 4 || rank == 5, "GridSample input must be rank 4 or 5.");
  RETURN_IF_NOT(y_info.shape.size() == rank, "GridSample output rank must equal input rank.");

  const std::vector<uint32_t> cf_to_cl = utils::ChannelFirstToLastPerm(rank);
  const std::vector<uint32_t> cl_to_cf = utils::ChannelLastToFirstPerm(rank);
  const std::vector<uint32_t> x_cl_shape = utils::ApplyPermToShape(x_info.shape, cf_to_cl);
  const std::vector<uint32_t> y_cl_shape = utils::ApplyPermToShape(y_info.shape, cf_to_cl);

  auto& names = utils::UniqueNameGenerator();
  const std::string x_cl_name = names.New(node_unit.Name() + "_x_cl");
  const std::string y_cl_name = names.New(node_unit.Name() + "_y_cl");

  RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), input_names[0], x_cl_name,
                                                     x_info.shape, cf_to_cl, x_cl_shape,
                                                     x_info.qnn_data_type, x_info.quant_param,
                                                     do_op_validation,
                                                     /*is_for_input=*/false, /*is_for_output=*/false));

  QnnTensorWrapper y_cl_tensor(y_cl_name, QNN_TENSOR_TYPE_NATIVE, y_info.qnn_data_type,
                               y_info.quant_param.Copy(), std::vector<uint32_t>(y_cl_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(y_cl_tensor)),
                "Failed to add GridSample channel-last output tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(names.New(node_unit), QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_GRID_SAMPLE, {x_cl_name, input_names[1]}, {y_cl_name},
                                                std::move(param_tensor_names), do_op_validation),
                "Failed to add GridSample node.");

  const std::string& y_name = node_unit.Outputs()[0].name;
  return qnn_model_wrapper.AddTransposeNode(node_unit.Index(), y_cl_name, y_name,
                                            y_cl_shape, cl_to_cf, y_info.shape,
                                            y_info.qnn_data_type, y_info.quant_param,
                                            do_op_validation,
                                            /*is_for_input=*/false,
                                            /*is_for_output=*/qnn_model_wrapper.IsGraphOutput(y_name));
}

}  // namespace

// GridSample for all supported domains (ai.onnx, com.microsoft, ai.onnx.contrib); see GridSampleDomain above.
class GridSampleOpBuilder : public BaseOpBuilder {
 public:
  GridSampleOpBuilder() : BaseOpBuilder("GridSampleOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GridSampleOpBuilder);

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status ExplicitOpCheck(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                              GridSampleDomain domain) const;
};

// Tensor-level checks (dtype, rank, grid shape). Attributes are validated by ParseGridSampleAttrs.
Ort::Status GridSampleOpBuilder::ExplicitOpCheck(QnnModelWrapper& qnn_model_wrapper,
                                                 const OrtNodeUnit& node_unit,
                                                 GridSampleDomain domain) const {
  // QNN HTP supports rank <= 5 for 8/16-bit fixed-point GridSample, rank <= 4 for float.
  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
  const bool is_fixed_point = utils::IsQuant8bit(input_info.qnn_data_type) ||
                              utils::IsQuant16bit(input_info.qnn_data_type);
  size_t max_rank = is_fixed_point ? 5 : 4;

  // The QNN CPU backend rejects 16-bit fixed-point GridSample and its Transposes (backendValidateOpConfig error
  // 3110). Validation is skipped for the ONNX domain, so reject here; otherwise pass 2 fails the session.
  // Verified by QnnCPUBackendTests.GridSample_U16_Probe; u8 rank 5 passes on CPU (GridSample_U8_Rank5_Probe).
  if (qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::CPU) {
    RETURN_IF(utils::IsQuant16bit(input_info.qnn_data_type),
              "QNN EP does not support 16-bit quantized GridSample on CPU backend.");
  }

  const size_t input_rank = input_info.shape.size();
  RETURN_IF(input_rank < 4, "QNN EP does not support GridSample with input rank < 4.");
  RETURN_IF(domain == GridSampleDomain::kMsContrib && input_rank != 4,
            "QNN EP supports com.microsoft:GridSample only with rank 4 input.");
  RETURN_IF(input_rank > max_rank,
            ("QNN EP does not support GridSample with input rank > " + std::to_string(max_rank) +
             (is_fixed_point ? " (fixed-point)." : " (floating-point)."))
                .c_str());

  TensorInfo grid_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[1], grid_info));
  RETURN_IF(grid_info.shape.size() != input_rank,
            "QNN EP requires GridSample grid rank to equal input rank.");

  return Ort::Status();
}

Ort::Status GridSampleOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                             const OrtNodeUnit& node_unit,
                                                             std::vector<std::string>&& input_names,
                                                             const Ort::Logger& logger,
                                                             bool do_op_validation) const {
  if (input_names.size() < 1) {
    return Ort::Status();
  }

  // Domain and attributes are resolved once and shared by validation and lowering.
  const GridSampleDomain domain = GetGridSampleDomain(node_unit);
  GridSampleParams params;
  RETURN_IF_ERROR(ParseGridSampleAttrs(node_unit, domain, params));

  if (do_op_validation) {
    RETURN_IF_ERROR(ExplicitOpCheck(qnn_model_wrapper, node_unit, domain));
    // NOTE: because QNN validation is skipped below for the ONNX domain, ExplicitOpCheck must be exhaustive. A
    // backend rejection in the second validation pass fails the session instead of falling back to ORT CPU
    // (see graph_partitioner.cc:488-497).
    // The skip applies only to the ONNX domain (layout converted by ORT afterwards). com.microsoft and
    // ai.onnx.contrib nodes are lowered with builder-inserted transposes, so they run the QNN validation.
    if (node_unit.Domain() == kOnnxDomain) {
      return Ort::Status();
    }
  }

  std::vector<std::string> param_tensor_names;
  RETURN_IF_ERROR(AddGridSampleParams(qnn_model_wrapper, node_unit, params, param_tensor_names));

  if (domain == GridSampleDomain::kMsContrib || domain == GridSampleDomain::kOrtContrib) {
    return AddGridSampleWithLayoutTransposes(qnn_model_wrapper, node_unit, input_names,
                                             std::move(param_tensor_names), do_op_validation);
  }

  return ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), std::move(param_tensor_names),
                        logger, do_op_validation, QNN_OP_GRID_SAMPLE);
}

void CreateGridSampleOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GridSampleOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
