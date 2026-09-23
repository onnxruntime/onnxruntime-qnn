// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include "HTP/QnnHtpGraph.h"

#include "core/providers/qnn/builder/qnn_configs_helper.h"
#include "core/providers/qnn/builder/qnn_def.h"

#include <gsl/gsl>

namespace onnxruntime {

inline void PopulateHtpGraphConfigs(
    qnn::QnnBackendType backend_type,
    const qnn::HtpGraphConfigs_t& configs,
    qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t>& configs_builder) {
  if (backend_type != qnn::QnnBackendType::HTP) {
    return;
  }

  if (configs.htp_graph_finalization_opt_mode != qnn::HtpGraphFinalizationOptimizationMode::kDefault) {
    gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_opt_config = configs_builder.PushCustomConfig();
    htp_graph_opt_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    htp_graph_opt_config->optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
    htp_graph_opt_config->optimizationOption.floatValue = static_cast<float>(configs.htp_graph_finalization_opt_mode);

    gsl::not_null<QnnGraph_Config_t*> graph_opt_config = configs_builder.PushConfig();
    graph_opt_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_opt_config->customConfig = htp_graph_opt_config;
  }

  if (configs.vtcm_size_in_mb > 0) {
    gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_opt_config_vtcm = configs_builder.PushCustomConfig();
    htp_graph_opt_config_vtcm->option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
    htp_graph_opt_config_vtcm->vtcmSizeInMB = static_cast<uint32_t>(configs.vtcm_size_in_mb);

    gsl::not_null<QnnGraph_Config_t*> graph_opt_config_vtcm = configs_builder.PushConfig();
    graph_opt_config_vtcm->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_opt_config_vtcm->customConfig = htp_graph_opt_config_vtcm;
  }

  if (configs.htp_num_cores > 0) {
    gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_num_cores_config = configs_builder.PushCustomConfig();
    htp_num_cores_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES;
    htp_num_cores_config->numCores = configs.htp_num_cores;

    gsl::not_null<QnnGraph_Config_t*> graph_num_cores_config = configs_builder.PushConfig();
    graph_num_cores_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_num_cores_config->customConfig = htp_num_cores_config;
  }

  if (configs.enable_htp_fp16_precision) {
    gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_precision_config = configs_builder.PushCustomConfig();
    htp_graph_precision_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    htp_graph_precision_config->precision = QNN_PRECISION_FLOAT16;

    gsl::not_null<QnnGraph_Config_t*> graph_precision_config = configs_builder.PushConfig();
    graph_precision_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_precision_config->customConfig = htp_graph_precision_config;
  }

  if (configs.enable_htp_monolithic_lstm) {
    gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_monolithic_lstm_config = configs_builder.PushCustomConfig();
    htp_graph_monolithic_lstm_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_MONOLITHIC_LSTM;
    htp_graph_monolithic_lstm_config->monolithicLstm = true;

    gsl::not_null<QnnGraph_Config_t*> graph_config = configs_builder.PushConfig();
    graph_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_config->customConfig = htp_graph_monolithic_lstm_config;
  }

  if (configs.enable_htp_fp16_clamp_overflow) {
#ifdef QNN_HTP_FP16_CLAMP_OVERFLOW_AVAILABLE
    gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_fp16_clamp_config = configs_builder.PushCustomConfig();
    htp_fp16_clamp_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_FP16_CLAMP_OVERFLOW;
    htp_fp16_clamp_config->fp16ClampOverflow = true;

    gsl::not_null<QnnGraph_Config_t*> graph_config = configs_builder.PushConfig();
    graph_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_config->customConfig = htp_fp16_clamp_config;
#endif
  }

#if ORT_QNN_HTP_MATMUL_LUT_SUPPORTED
  if (configs.enable_htp_matmul_lut) {
    gsl::not_null<QnnHtpGraph_CustomConfig_t*> matmul_lut_config = configs_builder.PushCustomConfig();
    matmul_lut_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG;
    matmul_lut_config->finalizeConfig.key = "enable_matmul_lut";
    matmul_lut_config->finalizeConfig.value.dataType = QNN_DATATYPE_BOOL_8;
    matmul_lut_config->finalizeConfig.value.bool8Value = 1;

    gsl::not_null<QnnGraph_Config_t*> graph_config = configs_builder.PushConfig();
    graph_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_config->customConfig = matmul_lut_config;
  }
#endif
}

}  // namespace onnxruntime
