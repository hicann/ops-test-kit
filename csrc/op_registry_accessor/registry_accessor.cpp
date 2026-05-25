/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <memory>

#include "json.hpp"
#include "register/op_impl_kernel_registry.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/tiling_context.h"
#include "base/runtime/runtime_attrs_def.h"
#include "graph/utils/type_utils.h"

using json = nlohmann::json;

namespace registry_accessor {

static constexpr size_t kTilingInputOtherNum = 5;

// ── Per-tensor descriptor (one CompileTimeTensorDesc entry) ──
struct IrTensorDesc {
    ge::DataType dtype = ge::DT_UNDEFINED;
    ge::Format format = ge::FORMAT_ND;
    ge::Format ori_format = ge::FORMAT_ND;
};

// ── Per-anchor descriptor (one IR input/output position) ──
//   empty  = optional not provided (null in JSON)
//   1 elem = single tensor (REQUIRED_INPUT / OPTIONAL_INPUT provided)
//   N elem = TensorList (DYNAMIC_INPUT)
struct IrInputDesc {
    std::vector<IrTensorDesc> tensors;
    bool IsProvided() const { return !tensors.empty(); }
    size_t Count() const { return tensors.size(); }
};

// ── Attribute descriptor parsed from JSON ──
struct AttrDesc {
    std::string name;
    std::string dtype;
    json value;           // may be null for float special values
    std::string null_desc; // "inf", "-inf", "nan", or empty
};

// ── Extra parameters parsed from JSON ──
struct ExtraParams {
    std::string op_name;
    int32_t deterministic = 0;
};

// ── dtype string → ge::DataType ──
static ge::DataType ParseDtype(const std::string &s) {
    if (s.empty()) return ge::DT_UNDEFINED;
    std::string upper = "DT_" + s;
    for (auto &c : upper) c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
    return ge::TypeUtils::SerialStringToDataType(upper);
}

// ── format string → ge::Format ──
static ge::Format ParseFormat(const std::string &s) {
    if (s.empty()) return ge::FORMAT_ND;
    std::string upper = s;
    for (auto &c : upper) c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
    return ge::TypeUtils::SerialStringToFormat(upper);
}

// ── Parse a single tensor JSON object → IrTensorDesc ──
static IrTensorDesc ParseOneTensor(const json &obj) {
    IrTensorDesc td;
    td.dtype = ParseDtype(obj.value("dtype", ""));
    td.format = ParseFormat(obj.value("format", "ND"));
    // ori_format falls back to format when not specified
    std::string ori_str = obj.value("ori_format", "");
    td.ori_format = ori_str.empty() ? td.format : ParseFormat(ori_str);
    return td;
}

// ── Parse inputs/outputs JSON array → vector<IrInputDesc> ──
//   Three-branch: null → empty; object → 1 tensor; array → TensorList
static bool ParseIrDescs(const char *json_str, std::vector<IrInputDesc> &descs) {
    descs.clear();
    if (!json_str || json_str[0] == '\0') return true;
    try {
        auto arr = json::parse(json_str);
        if (!arr.is_array()) return false;
        for (auto &elem : arr) {
            IrInputDesc desc;
            if (elem.is_null()) {
                // optional not provided — tensors stays empty
            } else if (elem.is_object()) {
                // single tensor
                desc.tensors.push_back(ParseOneTensor(elem));
            } else if (elem.is_array()) {
                // TensorList
                for (auto &sub : elem) {
                    if (sub.is_object()) {
                        desc.tensors.push_back(ParseOneTensor(sub));
                    } else {
                        return false;
                    }
                }
            } else {
                return false;
            }
            descs.push_back(desc);
        }
    } catch (...) {
        return false;
    }
    return true;
}

// ── Count total sub-tensors across all anchors ──
static size_t CountTotalTensors(const std::vector<IrInputDesc> &descs) {
    size_t n = 0;
    for (auto &d : descs)
        n += d.Count();
    return n;
}

// ── Parse attrs JSON array ──
static bool ParseAttrs(const char *json_str, std::vector<AttrDesc> &attrs) {
    attrs.clear();
    if (!json_str) return true;
    try {
        auto arr = json::parse(json_str);
        if (!arr.is_array()) return false;
        for (auto &elem : arr) {
            AttrDesc ad;
            ad.name = elem.value("name", "");
            ad.dtype = elem.value("dtype", "");
            ad.value = elem.value("value", json());
            ad.null_desc = elem.value("value_null_desc", "");
            attrs.push_back(ad);
        }
    } catch (...) {
        return false;
    }
    return true;
}

// ── Parse extra_params JSON ──
static bool ParseExtraParams(const char *json_str, ExtraParams &ep) {
    if (!json_str) return true;
    try {
        auto obj = json::parse(json_str);
        ep.op_name = obj.value("op_name", "");
        ep.deterministic = obj.value("deterministic", 0);
    } catch (...) {
        return false;
    }
    return true;
}

// ── Forward declarations ──
static bool BuildTilingContext(
    const char *op_type,
    const std::vector<IrInputDesc> &inputs,
    const std::vector<IrInputDesc> &outputs,
    const std::vector<AttrDesc> &attrs,
    const ExtraParams &extra_params,
    gert::TilingContext *&tiling_ctx,
    std::vector<uint8_t> &memory_holder);

static size_t CalcRuntimeAttrsSize(const std::vector<AttrDesc> &attrs, size_t &out_data_offset);
static void FillRuntimeAttrs(uint8_t *base, const std::vector<AttrDesc> &attrs, size_t data_offset);

// ══════════════════════════════════════════════════════════════
//  extern "C" exports — split into FindFuncs + InvokeFuncs
// ══════════════════════════════════════════════════════════════

extern "C" {

__attribute__((visibility("default")))
int FindGenSimplifiedKeyFuncs(const char *op_type, void **handle) {
    if (!op_type || !handle) return 1;
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (registry == nullptr) return 1;
    const auto *funcs = registry->GetOpImpl(op_type);
    if (funcs == nullptr || funcs->gen_simplifiedkey == nullptr) {
        funcs = registry->GetOpImpl("DefaultImpl");
        if (funcs == nullptr || funcs->gen_simplifiedkey == nullptr) return 1;
    }
    *handle = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(funcs);
    return 0;
}

__attribute__((visibility("default")))
int InvokeGenSimplifiedKey(
    void *handle,
    const char *op_type,
    const char *inputs_json,
    const char *outputs_json,
    const char *attrs_json,
    const char *extra_params_json,
    char *result_buf)
{
    if (!handle) return 1;
    if (!op_type || !result_buf) return 2;

    auto *funcs = static_cast<const gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(handle);
    if (!funcs->gen_simplifiedkey) return 1;

    std::vector<IrInputDesc> inputs, outputs;
    if (!ParseIrDescs(inputs_json, inputs)) return 2;
    if (!ParseIrDescs(outputs_json, outputs)) return 2;

    std::vector<AttrDesc> attrs;
    if (!ParseAttrs(attrs_json, attrs)) return 2;

    ExtraParams extra_params;
    if (!ParseExtraParams(extra_params_json, extra_params)) return 2;

    gert::TilingContext *tiling_ctx = nullptr;
    std::vector<uint8_t> memory_holder;
    if (!BuildTilingContext(op_type, inputs, outputs, attrs, extra_params,
                            tiling_ctx, memory_holder)) {
        return 2;
    }

    // CANN callbacks use internal DEST_MAX (typically 30-50) for strcat_s.
    // When result_buf already contains a long prefix (e.g. "LpNormReduceV2/d=0,p=0/" = 23 bytes),
    // the callback's strcat_s may exceed DEST_MAX and zero out buf[0] on error.
    // Solution: pass a separate empty buffer to the callback, then assemble prefix + result.
    // This matches the CANN UT convention (tiling_test_help.cpp uses empty 30-byte buf).
    static constexpr size_t kCallbackBufSize = 256;
    char callback_buf[kCallbackBufSize] = {0};

    auto ret = funcs->gen_simplifiedkey(tiling_ctx, callback_buf);
    if (ret != 0) return 3;

    // Assemble: result_buf already contains the prefix from Python caller.
    // Append the callback's output after the existing prefix.
    size_t prefix_len = std::strlen(result_buf);
    size_t callback_len = std::strlen(callback_buf);
    if (prefix_len + callback_len >= kCallbackBufSize) return 3;
    std::memcpy(result_buf + prefix_len, callback_buf, callback_len + 1);

    return 0;
}

} // extern "C"

// ══════════════════════════════════════════════════════════════
//  TilingContext construction
// ══════════════════════════════════════════════════════════════

static bool BuildTilingContext(
    const char *op_type,
    const std::vector<IrInputDesc> &inputs,
    const std::vector<IrInputDesc> &outputs,
    const std::vector<AttrDesc> &attrs,
    const ExtraParams & /*extra_params*/,
    gert::TilingContext *&tiling_ctx,
    std::vector<uint8_t> &memory_holder)
{
    size_t ir_inputs = inputs.size();
    size_t ir_outputs = outputs.size();
    size_t real_inputs = CountTotalTensors(inputs);
    size_t real_outputs = CountTotalTensors(outputs);

    // Calculate RuntimeAttrs size
    size_t attr_data_offset = 0;
    size_t attr_total_size = CalcRuntimeAttrsSize(attrs, attr_data_offset);

    // Allocate ComputeNodeInfo using public API
    size_t cni_size = 0;
    if (gert::ComputeNodeInfo::CalcSize(ir_inputs, ir_outputs,
                                         real_inputs, real_outputs, cni_size) !=
        ge::GRAPH_SUCCESS) {
        return false;
    }

    size_t kei_size = sizeof(gert::KernelExtendInfo);
    size_t values_needed = real_inputs + real_outputs + kTilingInputOtherNum;
    size_t krc_alloc = sizeof(KernelRunContext) + sizeof(AsyncAnyValue *) * (values_needed > 0 ? values_needed - 1 : 0);

    size_t total_cni = cni_size + attr_total_size;

    memory_holder.resize(total_cni + kei_size + krc_alloc, 0);
    uint8_t *base = memory_holder.data();

    auto *cni = reinterpret_cast<gert::ComputeNodeInfo *>(base);
    auto *kei = reinterpret_cast<gert::KernelExtendInfo *>(base + total_cni);
    auto *krc = reinterpret_cast<KernelRunContext *>(base + total_cni + kei_size);

    // Initialize ComputeNodeInfo
    cni->Init(ir_inputs, ir_outputs, real_inputs, real_outputs,
              attr_total_size, op_type, op_type);

    // Fill AnchorInstanceInfo for inputs (IR-level, one per anchor)
    size_t compile_desc_idx = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto *info = cni->MutableInputInstanceInfo(i);
        if (!inputs[i].IsProvided()) {
            info->SetInstanceStart(compile_desc_idx);
            info->SetInstantiationNum(0);
        } else {
            info->SetInstanceStart(compile_desc_idx);
            info->SetInstantiationNum(inputs[i].Count());
            compile_desc_idx += inputs[i].Count();
        }
    }

    // Fill CompileTimeTensorDesc for inputs (flat, one per sub-tensor)
    compile_desc_idx = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (auto &t : inputs[i].tensors) {
            auto *td = cni->MutableInputTdInfo(compile_desc_idx);
            td->SetDataType(t.dtype);
            td->SetStorageFormat(t.format);
            td->SetOriginFormat(t.ori_format);
            ++compile_desc_idx;
        }
    }

    // Fill CompileTimeTensorDesc for outputs (flat, one per sub-tensor)
    size_t output_desc_idx = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        for (auto &t : outputs[i].tensors) {
            auto *td = cni->MutableOutputTdInfo(output_desc_idx);
            td->SetDataType(t.dtype);
            td->SetStorageFormat(t.format);
            td->SetOriginFormat(t.ori_format);
            ++output_desc_idx;
        }
    }

    // Fill RuntimeAttrsDef
    if (attr_total_size > 0) {
        const auto *raw_attrs = cni->GetAttrs();
        auto *attr_def = reinterpret_cast<RuntimeAttrsDef *>(
            const_cast<gert::RuntimeAttrs *>(raw_attrs));
        FillRuntimeAttrs(reinterpret_cast<uint8_t *>(attr_def), attrs, attr_data_offset);
    }

    // Fill AnchorInstanceInfo for outputs (IR-level, one per anchor)
    size_t output_anchor_idx = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto *info = cni->MutableOutputInstanceInfo(i);
        if (!outputs[i].IsProvided()) {
            info->SetInstanceStart(output_anchor_idx);
            info->SetInstantiationNum(0);
        } else {
            info->SetInstanceStart(output_anchor_idx);
            info->SetInstantiationNum(outputs[i].Count());
            output_anchor_idx += outputs[i].Count();
        }
    }

    // Initialize KernelExtendInfo
    kei->SetKernelName(op_type);
    kei->SetKernelType(op_type);

    // Initialize KernelRunContext
    krc->input_size = real_inputs + real_outputs + kTilingInputOtherNum;
    krc->output_size = 0;
    krc->compute_node_info = cni;
    krc->kernel_extend_info = kei;
    krc->output_start = nullptr;

    // Cast to TilingContext
    tiling_ctx = reinterpret_cast<gert::TilingContext *>(krc);
    return true;
}

static size_t Align8(size_t n) { return (n + 7) & ~(size_t)7; }

static size_t AttrDataSize(const AttrDesc &attr) {
    const auto &dtype = attr.dtype;
    const auto &val = attr.value;

    if (dtype == "bool" || dtype == "int" || dtype == "int32" || dtype == "int64") {
        return 8;
    }
    if (dtype == "float" || dtype == "float32") {
        return 8;
    }
    if (dtype == "float64" || dtype == "double") {
        return 8;
    }
    if (dtype == "str") {
        std::string s = val.is_string() ? val.get<std::string>() : "";
        return Align8(s.size() + 1);
    }
    if (dtype == "list_bool") {
        size_t n = val.is_array() ? val.size() : 0;
        return sizeof(gert::ContinuousVector) + n;
    }
    if (dtype == "list_int" || dtype == "list_int32" || dtype == "list_int64") {
        size_t n = val.is_array() ? val.size() : 0;
        return sizeof(gert::ContinuousVector) + n * sizeof(int64_t);
    }
    if (dtype == "list_float" || dtype == "list_float32") {
        size_t n = val.is_array() ? val.size() : 0;
        return sizeof(gert::ContinuousVector) + n * sizeof(float);
    }
    if (dtype == "list_str") {
        size_t total = sizeof(gert::ContinuousVector);
        if (val.is_array()) {
            for (auto &elem : val) {
                std::string s = elem.is_string() ? elem.get<std::string>() : "";
                total += Align8(s.size() + 1);
            }
        }
        return total;
    }
    if (dtype == "list_list_int" || dtype == "list_list_int32" || dtype == "list_list_int64") {
        size_t outer_n = val.is_array() ? val.size() : 0;
        size_t elem_type_size = sizeof(int64_t);
        size_t overhead = gert::ContinuousVectorVector::GetOverHeadLength(outer_n);
        size_t total = overhead;
        if (val.is_array()) {
            for (auto &inner : val) {
                size_t inner_n = inner.is_array() ? inner.size() : 0;
                total += sizeof(gert::ContinuousVector) + inner_n * elem_type_size;
            }
        }
        return total;
    }
    return 8;
}

static size_t CalcRuntimeAttrsSize(const std::vector<AttrDesc> &attrs, size_t &out_data_offset) {
    if (attrs.empty()) {
        out_data_offset = 0;
        return 0;
    }
    size_t attr_num = attrs.size();
    out_data_offset = sizeof(RuntimeAttrsDef) + sizeof(size_t) * attr_num;
    out_data_offset = Align8(out_data_offset);

    size_t data_total = 0;
    for (auto &attr : attrs) {
        data_total += AttrDataSize(attr);
    }
    return out_data_offset + data_total;
}

static void WriteAttrData(uint8_t *ptr, const AttrDesc &attr) {
    const auto &dtype = attr.dtype;
    const auto &val = attr.value;

    if (dtype == "bool") {
        *(reinterpret_cast<int64_t *>(ptr)) = 0;
        *(reinterpret_cast<bool *>(ptr)) = val.is_boolean() ? val.get<bool>() : false;
    } else if (dtype == "int" || dtype == "int32" || dtype == "int64") {
        *(reinterpret_cast<int64_t *>(ptr)) = val.is_number() ? val.get<int64_t>() : 0;
    } else if (dtype == "float" || dtype == "float32") {
        *(reinterpret_cast<int64_t *>(ptr)) = 0;
        float f = 0.0f;
        if (val.is_number()) {
            f = val.get<float>();
        } else if (val.is_null() && !attr.null_desc.empty()) {
            if (attr.null_desc == "inf") f = std::numeric_limits<float>::infinity();
            else if (attr.null_desc == "-inf") f = -std::numeric_limits<float>::infinity();
            else if (attr.null_desc == "nan") f = std::numeric_limits<float>::quiet_NaN();
        }
        *(reinterpret_cast<float *>(ptr)) = f;
    } else if (dtype == "float64" || dtype == "double") {
        double d = 0.0;
        if (val.is_number()) {
            d = val.get<double>();
        } else if (val.is_null() && !attr.null_desc.empty()) {
            if (attr.null_desc == "inf") d = std::numeric_limits<double>::infinity();
            else if (attr.null_desc == "-inf") d = -std::numeric_limits<double>::infinity();
            else if (attr.null_desc == "nan") d = std::numeric_limits<double>::quiet_NaN();
        }
        *(reinterpret_cast<double *>(ptr)) = d;
    } else if (dtype == "str") {
        std::string s = val.is_string() ? val.get<std::string>() : "";
        memcpy(ptr, s.c_str(), s.size() + 1);
    } else if (dtype == "list_bool") {
        auto *cv = reinterpret_cast<gert::ContinuousVector *>(ptr);
        size_t n = val.is_array() ? val.size() : 0;
        cv->Init(n);
        cv->SetSize(n);
        if (n > 0) {
            auto *data = reinterpret_cast<bool *>(cv->MutableData());
            for (size_t i = 0; i < n; ++i) {
                data[i] = val[i].is_boolean() ? val[i].get<bool>() : false;
            }
        }
    } else if (dtype == "list_int" || dtype == "list_int32" || dtype == "list_int64") {
        auto *cv = reinterpret_cast<gert::ContinuousVector *>(ptr);
        size_t n = val.is_array() ? val.size() : 0;
        cv->Init(n);
        cv->SetSize(n);
        if (n > 0) {
            auto *data = reinterpret_cast<int64_t *>(cv->MutableData());
            for (size_t i = 0; i < n; ++i) {
                data[i] = val[i].is_number() ? val[i].get<int64_t>() : 0;
            }
        }
    } else if (dtype == "list_float" || dtype == "list_float32") {
        auto *cv = reinterpret_cast<gert::ContinuousVector *>(ptr);
        size_t n = val.is_array() ? val.size() : 0;
        cv->Init(n);
        cv->SetSize(n);
        if (n > 0) {
            auto *data = reinterpret_cast<float *>(cv->MutableData());
            for (size_t i = 0; i < n; ++i) {
                data[i] = val[i].is_number() ? val[i].get<float>() : 0.0f;
            }
        }
    } else if (dtype == "list_str") {
        auto *cv = reinterpret_cast<gert::ContinuousVector *>(ptr);
        size_t n = val.is_array() ? val.size() : 0;
        cv->Init(n);
        cv->SetSize(n);
        uint8_t *str_ptr = ptr + sizeof(gert::ContinuousVector);
        if (n > 0) {
            for (size_t i = 0; i < n; ++i) {
                std::string s = val[i].is_string() ? val[i].get<std::string>() : "";
                memcpy(str_ptr, s.c_str(), s.size() + 1);
                str_ptr += Align8(s.size() + 1);
            }
        }
    } else if (dtype == "list_list_int" || dtype == "list_list_int32" || dtype == "list_list_int64") {
        size_t outer_n = val.is_array() ? val.size() : 0;
        auto *cvv = reinterpret_cast<gert::ContinuousVectorVector *>(ptr);
        cvv->Init(outer_n);
        if (val.is_array()) {
            for (size_t i = 0; i < outer_n; ++i) {
                size_t inner_n = val[i].is_array() ? val[i].size() : 0;
                auto *inner_cv = cvv->Add<int64_t>(inner_n);
                if (inner_cv && inner_n > 0) {
                    auto *data = reinterpret_cast<int64_t *>(inner_cv->MutableData());
                    for (size_t j = 0; j < inner_n; ++j) {
                        data[j] = val[i][j].is_number() ? val[i][j].get<int64_t>() : 0;
                    }
                }
            }
        }
    }
}

static void FillRuntimeAttrs(uint8_t *base, const std::vector<AttrDesc> &attrs, size_t data_offset) {
    if (attrs.empty()) return;

    auto *attr_def = reinterpret_cast<RuntimeAttrsDef *>(base);
    attr_def->attr_num = attrs.size();

    uint8_t *data_ptr = base + data_offset;
    for (size_t i = 0; i < attrs.size(); ++i) {
        attr_def->offset[i] = static_cast<size_t>(data_ptr - base);
        WriteAttrData(data_ptr, attrs[i]);
        data_ptr += AttrDataSize(attrs[i]);
    }
}

} // namespace registry_accessor
