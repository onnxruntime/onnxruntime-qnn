import numpy as np
from math     import ceil
import onnx
from onnx     import helper, TensorProto
import onnxruntime as ort
from onnx import shape_inference, numpy_helper
import numpy
import os
import torch 
from transformers import AutoModelForCausalLM
import argparse
from onnxruntime.quantization import quantize, QuantType, CalibrationDataReader
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

def tensor_types_to_np_types(tensor_type):
    if tensor_type == "tensor(float)":
        return np.float32
    elif tensor_type == "tensor(uint16)":
        return np.uint16
    elif tensor_type == "tensor(uint8)":
        return np.uint8
    else:
        raise ValueError(f"Unsupported tensor type {tensor_type}")

class RandomDataReader(CalibrationDataReader):
    def __init__(self, model_path, max_samples=1):
        # Initialize the ONNX session within each process
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # Suppress warnings
        self.model = ort.InferenceSession(model_path, sess_options=session_options)
        self.max_samples = max_samples
        self.count = 0
        self.input_shapes = []
        self.input_types = []
        self.input_names = []
        for input_meta in self.model.get_inputs():
            shape = [dim if isinstance(dim, int) else 1 for dim in input_meta.shape]
            self.input_shapes.append(shape)
            self.input_types.append(tensor_types_to_np_types(input_meta.type))
            self.input_names.append(input_meta.name)

    def get_next(self):
        if self.count >= self.max_samples:
            return None
        inputs = {}
        for i in range(len(self.input_shapes)):
            shape = self.input_shapes[i]
            dtype = self.input_types[i]
            name = self.input_names[i]
            data = np.random.rand(*shape).astype(dtype)
            inputs[name] = data
        self.count += 1
        return inputs
    
class BasicDataReader(CalibrationDataReader):
    def __init__(self, data):
        self.data = data
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None
        data_out = self.data[self.index]
        self.index += 1
        return data_out


def make_unsqueeze_node(inputs, outputs, axes_tensor_name, node_name):
    axes_tensor = numpy_helper.from_array(
        np.array([1], dtype=np.int64),
        name=axes_tensor_name
    )

    inputs.extend([axes_tensor_name])
    unsqueeze_node = helper.make_node(
        op_type = "Unsqueeze",
        inputs = inputs,
        outputs = outputs,
        name = node_name
    )
    return axes_tensor, unsqueeze_node


def make_matmul_nbits_model(path, weights_uint8: np.ndarray, scales: np.ndarray, zeros_uint8: np.ndarray, block_size=64):

    weights_uint8 = weights_uint8.reshape(-1, 3072,  4)
    weights_uint8 = weights_uint8.transpose(1, 0, 2).flatten()

    zeros_uint8 = zeros_uint8.reshape(-1, 3072,  4)
    zeros_uint8 = zeros_uint8.transpose(1, 0, 2).flatten()

    scales = scales.reshape(-1, 3072)
    scales = scales.transpose(1, 0).flatten()

    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 1, K])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 1, N])

    init_B      = helper.make_tensor('B',          TensorProto.UINT8,   weights_uint8.shape,  weights_uint8)
    init_scales = helper.make_tensor('scales',     TensorProto.FLOAT,   scales.shape,   scales)
    init_zeros  = helper.make_tensor('zero_points',TensorProto.UINT8,   zeros_uint8.shape,    zeros_uint8)

    node = helper.make_node(
        'MatMulNBits',
        ['A', 'B', 'scales', 'zero_points'],
        ['Y'],
        name='MatMulNBits',
        domain='com.microsoft',
        K=K, N=N, bits=2, block_size=block_size
    )

    graph = helper.make_graph(
        nodes=[node], name='MatMulNBitsInt32',
        inputs=[A], outputs=[Y],
        initializer=[init_B, init_scales, init_zeros]
    )
    opset_version = 17  # Example opset version
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("", opset_version)])
    onnx.save(model, path)


def compare_models(m1: str, m2: str, so_path: str):
    so = ort.SessionOptions()
    so.register_custom_ops_library(so_path)
    s1 = ort.InferenceSession(m1, so)
    s2 = ort.InferenceSession(m2, so)

    batch = 1
    K     = s1.get_inputs()[0].shape[1]
    A_np  = np.random.randn(batch, K).astype(np.float32)

    y1 = s1.run(None, {'A': A_np})[0]
    y2 = s2.run(None, {'A': A_np})[0]
    return np.max(np.abs(y1 - y2))

def make_matmul32bit_model(path, W: np.ndarray):
    N, K = W.shape
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, K])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, N])

    init_B = helper.make_tensor('B', TensorProto.FLOAT, W.shape, W.transpose().flatten().tolist())

    node = helper.make_node(
        'MatMul',
        ['A', 'B'],
        ['Y'],
        name='MatMul',
    )

    graph = helper.make_graph(
        nodes=[node], name='MatMulInt32',
        inputs=[A], outputs=[Y],
        initializer=[init_B]
    )
    opset_version = 17  # Example opset version
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("", opset_version)])
    onnx.save(model, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MatmulNBits test')
    parser.add_argument('--load_weights_from_hf', action='store_true', help='Load weights from Hugging Face model, used on HDDR2')
    parser.add_argument('--two_bit_weight_folder', type=str, default='/public_data/llm/models/Phi-3.5-mini-instruct-quantized-w2g064-repacked-transposed-orig/model.layers.0.self_attn.o_proj/weight_data', help='Path to the two bit weight folder')
    parser.add_argument("--ep", type=str, default="CPU", help="Execution provider to use")
    parser.add_argument("--make_onnx", action="store_true", help="Make the onnx model")
    args = parser.parse_args()

    

    K, N, block_size = 3072, 3072, 64

    n_bits_file = "model_nbits.onnx"
    n_bits_qdq_file = "model_nbits_qdq.onnx"

    test_weight_folder = args.two_bit_weight_folder
    scales_file = os.path.join(test_weight_folder, "scales_repacked.data")
    zeros_file  = os.path.join(test_weight_folder, "qzeros_repacked.data")
    weights_file = os.path.join(test_weight_folder, "qweight_repacked.data")

    # load the weights
    scales = np.fromfile(scales_file, dtype=np.float32)
    zeros_uint8  = np.fromfile(zeros_file, dtype=np.uint8)
    weights_uint8 = np.fromfile(weights_file, dtype=np.uint8)
    zeros_uint32 = np.fromfile(zeros_file, dtype=np.uint32)
    weights_uint32 = np.fromfile(weights_file, dtype=np.uint32)


    load_weights_from_hf = False
    if load_weights_from_hf:
        # get floating point reference directly from the f' model.
        f_prime_directory = "/public_data/llm_2bit_reconstructed/Phi-3.5-mini-instructed-quantized-w_f_prime_reconstructed/"
        f_prime_model = AutoModelForCausalLM.from_pretrained(f_prime_directory, torch_dtype=torch.float32, device_map="auto")
        weight_prime = f_prime_model.model.layers[0].self_attn.o_proj.weight.data.cpu().numpy()

        # save the weight prime to a file
        weight_prime.tofile("weight_prime.npy")
    else:
        # load the weight prime from a file
        weight_prime = np.fromfile("weight_prime.npy", dtype=np.float32)
        weight_prime = weight_prime.reshape(3072, 3072)

    if args.make_onnx:
        # make the 2 models
        make_matmul32bit_model('model_32bit.onnx', weight_prime)
        make_matmul_nbits_model(n_bits_file, weights_uint8, scales, zeros_uint8, block_size)

    # set the seed
    np.random.seed(0)
    # make a random input
    A = np.random.randn(1, 1, 3072).astype(np.float32)

    # set up the reference model
    ref_providers = ["CPUExecutionProvider"]
    ref_options = ort.SessionOptions()

    # set up the test model
    if args.ep == "CPU":
        test_providers = ["CPUExecutionProvider"]
        test_options = ort.SessionOptions()
        # options.log_severity_level = 0

        test_provider_options = [{}]
    elif args.ep == "QNN":
        test_providers = ["QNNExecutionProvider"]
        test_options = ort.SessionOptions()
        test_options.log_severity_level = 0  # full verbose logging
        test_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        test_options.add_session_config_entry("ep.context_embed_mode", "0")
        test_options.add_session_config_entry("ep.context_enable", "1")

        test_options.intra_op_num_threads = 2
        test_options.inter_op_num_threads = 1

        test_provider_options = [
            {
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "sustained_high_performance",
                "device_id": "0",
                "htp_graph_finalization_optimization_mode": "3",
                "soc_model": "60",
                "htp_arch": "73",
                "vtcm_mb": "8",
                # "profiling_level": "off" if use_ep_profiler else "detailed",
                # "profiling_file": "image_processing_stack_session.txt" if use_ep_profiler else "",
            }
        ]
    else:
        raise ValueError(f"Unknown execution provider: {args.ep}")
    


    session_cpu_nbits = ort.InferenceSession(n_bits_file, sess_options=ref_options, providers=ref_providers)
    session_ref = ort.InferenceSession("model_32bit.onnx", sess_options=ref_options, providers=ref_providers)
    # run the model
    # reshape the input to [1, 1, 1, 3072]
    A_reshape = np.reshape(A, (1, 1, 1, 3072)).astype(np.float32)
    Y_onnx = session_cpu_nbits.run(None, {"A": A_reshape})[0]

    # compare the two outputs
    Y_ref = session_ref.run(None, {"A": A})[0]
    diff_float = np.max(np.abs(Y_ref - Y_onnx))
    print("Max (Y_ref - Y_onnx): ", diff_float)

    # static quantize the model
    if args.make_onnx:
        # random_data_reader = RandomDataReader(n_bits_file, max_samples=1)
        basic_data_reader = BasicDataReader([{"A": A_reshape}])

        qnn_config = get_qnn_qdq_config(
            n_bits_file,
            basic_data_reader,
            activation_type=QuantType.QUInt16,
            weight_type=QuantType.QUInt8,
            per_channel=False,
            init_overrides=None,
            activation_symmetric=False,
            keep_removable_activations=True,
        )
        qnn_config.extra_options["ForceQuantizeNoInputCheck"] = True

        quantize(
            n_bits_file,
            n_bits_qdq_file,
            qnn_config,
        )

    # test the accuracy of the quantized model
    session_qdq = ort.InferenceSession(n_bits_qdq_file, sess_options=test_options, providers=test_providers, provider_options=test_provider_options)
    Y_qdq = session_qdq.run(None, {"A": A_reshape})[0]
    diff_qdq_ref = np.max(np.abs(Y_ref-Y_qdq))
    print("Max (Y_ref-Y_qdq): ", diff_qdq_ref)
    diff_qdq_ref_rmse = np.sqrt(np.mean((Y_ref-Y_qdq)**2))
    print("RMSE (Y_ref-Y_qdq): ", diff_qdq_ref_rmse)



