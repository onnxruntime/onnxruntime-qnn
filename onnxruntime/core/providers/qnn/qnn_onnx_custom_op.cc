// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/qnn_onnx_custom_op.h"

namespace onnxruntime {

/*static*/
int64_t QtiAiswPlaceholderOp::ReadIntAttr(const OrtApi& api, OrtShapeInferContext* ctx,
                                          const char* name, int64_t fallback) {
  const OrtOpAttr* attr = nullptr;
  if (api.ShapeInferContext_GetAttribute(ctx, name, &attr) != nullptr || attr == nullptr) {
    return fallback;
  }
  int64_t value = 0;
  size_t out = 0;
  if (api.ReadOpAttr(attr, ORT_OP_ATTR_INT, &value, sizeof(value), &out) != nullptr) {
    return fallback;
  }
  return value;
}

/*static*/
bool QtiAiswPlaceholderOp::ReadStringAttr(const OrtApi& api, OrtShapeInferContext* ctx,
                                          const char* name, std::string& out_str) {
  const OrtOpAttr* attr = nullptr;
  if (api.ShapeInferContext_GetAttribute(ctx, name, &attr) != nullptr || attr == nullptr) {
    return false;
  }
  size_t len = 0;
  // First call with len=0 to query the size, then read. The sizing call returns a non-OK status
  // (buffer too small) but still writes the required length to `len`; discard that status.
  if (OrtStatusPtr size_st = api.ReadOpAttr(attr, ORT_OP_ATTR_STRING, nullptr, 0, &len);
      size_st != nullptr) {
    api.ReleaseStatus(size_st);
  }
  if (len == 0) {
    return false;
  }
  out_str.resize(len);
  size_t out = 0;
  if (api.ReadOpAttr(attr, ORT_OP_ATTR_STRING, out_str.data(), len, &out) != nullptr) {
    return false;
  }
  return true;
}

/*static*/
OrtStatusPtr QtiAiswPlaceholderOp::InferOutputShapeImpl(const OrtCustomOp* op,
                                                        OrtShapeInferContext* ctx) {
  const OrtApi& api = Ort::GetApi();

  size_t num_inputs = 0;
  if (OrtStatusPtr st = api.ShapeInferContext_GetInputCount(ctx, &num_inputs); st != nullptr) {
    return st;
  }
  if (num_inputs == 0) {
    return nullptr;  // nothing to anchor output 0 on
  }

  // Input 0 (activation) type + shape. `in0` is owned by the context — do not release it.
  OrtTensorTypeAndShapeInfo* in0 = nullptr;
  if (OrtStatusPtr st = api.ShapeInferContext_GetInputTypeShape(ctx, 0, &in0); st != nullptr) {
    return st;
  }
  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  if (OrtStatusPtr st = api.GetTensorElementType(in0, &elem_type); st != nullptr) {
    return st;
  }
  size_t x_rank = 0;
  if (OrtStatusPtr st = api.GetDimensionsCount(in0, &x_rank); st != nullptr) {
    return st;
  }
  std::vector<int64_t> x_dims(x_rank, -1);
  if (x_rank > 0) {
    if (OrtStatusPtr st = api.GetDimensions(in0, x_dims.data(), x_rank); st != nullptr) {
      return st;
    }
  }

  const std::string op_name = static_cast<const QtiAiswPlaceholderOp*>(op)->name_;

  // Output dims, expressed as (integer value, symbolic name). A dim is concrete when its integer
  // value is >= 0 (symbolic name empty); it is unknown when the integer value is < 0 and a
  // non-empty symbolic name is supplied (ONNX requires a name for unknown dims).
  std::vector<int64_t> int_dims;
  std::vector<std::string> sym_dims;
  auto add_known = [&](int64_t v) {
    int_dims.push_back(v);
    sym_dims.emplace_back("");
  };
  auto add_unknown = [&](const char* name) {
    int_dims.push_back(-1);
    sym_dims.emplace_back(name);
  };
  // Append input 0's dims verbatim (preserving known values; naming unknowns).
  auto add_input_dims = [&]() {
    for (size_t i = 0; i < x_dims.size(); ++i) {
      if (x_dims[i] >= 0) {
        add_known(x_dims[i]);
      } else {
        add_unknown("qti_aisw_x");
      }
    }
  };

  if (op_name == "Buffer") {
    // out[0] = input shape with dim[buffer_dim] replaced by buffer_size.
    const int64_t buffer_dim = ReadIntAttr(api, ctx, "buffer_dim", -1);
    const int64_t buffer_size = ReadIntAttr(api, ctx, "buffer_size", -1);
    for (size_t i = 0; i < x_dims.size(); ++i) {
      if (static_cast<int64_t>(i) == buffer_dim && buffer_size > 0) {
        add_known(buffer_size);
      } else if (x_dims[i] >= 0) {
        add_known(x_dims[i]);
      } else {
        add_unknown("qti_aisw_x");
      }
    }
  } else if (op_name == "StatefulLstm" || op_name == "StatefulGru") {
    // ONNX LSTM/GRU (layout=0): X = [seq_length, batch_size, input_size];
    // out[0] Y = [seq_length, num_directions, batch_size, hidden_size].
    // Emit the full rank-4 shape whenever hidden_size is known, using the input's seq/batch dims
    // when available and a symbolic (unknown) dim otherwise. Getting the rank right is what
    // matters for the downstream Squeeze/MatMul consumers common in real models.
    const int64_t hidden_size = ReadIntAttr(api, ctx, "hidden_size", -1);
    std::string direction;
    const int64_t num_dir = (ReadStringAttr(api, ctx, "direction", direction) &&
                             direction == "bidirectional")
                                ? 2
                                : 1;
    if (hidden_size > 0) {
      // seq
      if (x_dims.size() == 3 && x_dims[0] >= 0) {
        add_known(x_dims[0]);
      } else {
        add_unknown("qti_aisw_seq");
      }
      add_known(num_dir);
      // batch
      if (x_dims.size() == 3 && x_dims[1] >= 0) {
        add_known(x_dims[1]);
      } else {
        add_unknown("qti_aisw_batch");
      }
      add_known(hidden_size);
    } else {
      add_input_dims();  // cannot compute without hidden_size
    }
  } else {
    add_input_dims();  // unknown op: best-effort passthrough
  }

  // Build the output type+shape info and set output 0.
  OrtTensorTypeAndShapeInfo* out_info = nullptr;
  if (OrtStatusPtr st = api.CreateTensorTypeAndShapeInfo(&out_info); st != nullptr) {
    return st;
  }
  std::vector<const char*> sym_ptrs(sym_dims.size());
  for (size_t i = 0; i < sym_dims.size(); ++i) {
    sym_ptrs[i] = sym_dims[i].c_str();
  }
  OrtStatusPtr result = api.SetTensorElementType(out_info, elem_type);
  if (result == nullptr) {
    result = api.SetDimensions(out_info, int_dims.data(), int_dims.size());
  }
  if (result == nullptr) {
    result = api.SetSymbolicDimensions(out_info, sym_ptrs.data(), sym_ptrs.size());
  }
  if (result == nullptr) {
    result = api.ShapeInferContext_SetOutputTypeShape(ctx, 0, out_info);
  }
  api.ReleaseTensorTypeAndShapeInfo(out_info);

  // Only output 0 is set; outputs 1+ would crash via unchecked OOB in ORT's getOutputType
  // (no ShapeInferContext API to query the node's output count).
  return result;
}

}  // namespace onnxruntime
