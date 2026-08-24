#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""
FrameworkApiInstance — InstanceBase implementation for framework_api tests.
"""

import logging
import os

from ttk.core_modules.infra.instance_base import InstanceBase
from ttk.utilities.container_utils import get_global_storage

from .backends import get_backend
from .object import FrameworkApiProfileObject


class FrameworkApiInstance(InstanceBase):
    """Entry instance for framework-level API testing."""

    def __init__(self):
        super().__init__()
        switches = get_global_storage()
        if switches.backend == "npusim":
            self._inject_camodel_env(switches)
        framework = getattr(switches, "framework", "torch")
        self.backend = get_backend(switches.force_cpu, framework=framework)
        if not self.backend.has_device():
            switches.proc_no_reuse = True
        logging.info(f"Framework API mode: backend={self.backend.device_type()}, framework={framework}")

    @staticmethod
    def _inject_camodel_env(switches):
        """Prepare the camodel simulation environment for the E2E profiling workers.

        E2E profiling workers are forked (forkserver) from this process and
        inherit its ``LD_LIBRARY_PATH``, so prepending the camodel lib dir here
        makes every worker load the camodel runtime instead of the real device
        runtime. Must run before ``get_backend()`` probes
        ``torch.npu.is_available()`` so the NPU backend is selected.
        """
        if switches.force_cpu:
            raise ValueError("--cpu cannot be combined with --backend npusim")
        from ttk.core_modules.simulator.config import resolve_camodel_lib_dir

        camodel = resolve_camodel_lib_dir(switches.sim_soc_version)
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        camodel_str = str(camodel)
        # Exact entry check on the ":"-split list (mirrors _env._restore_ld_paths):
        # substring matching on the joined string would skip injection when an
        # unrelated entry merely contains the camodel path (e.g. ".../camodel_bak").
        if camodel_str not in existing.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{camodel_str}:{existing}" if existing else camodel_str
        log_dir = os.path.join(switches.sim_output_dir, "camodel_log")
        os.environ.setdefault("CAMODEL_LOG_PATH", log_dir)
        os.environ.setdefault("STARS_LOG_PATH", log_dir)
        logging.info("E2E npusim: camodel runtime injected from %s", camodel)

    def env_prepare(self):
        import multiprocessing
        self._register_mc2_apis()
        self.mp_context = multiprocessing.get_context("fork")

    @staticmethod
    def _register_mc2_apis():
        from ttk.utilities.simple_param_extractor import (
            APIParamInfo, OverloadInfo, ParamInfo, register_api_params,
        )
        mc2_apis = {
            "torch_npu.npu_mm_all_reduce_base": APIParamInfo(
                api_name="torch_npu.npu_mm_all_reduce_base",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="x1", type="Tensor"),
                    ParamInfo(name="x2", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="reduce_op", type="str", is_optional=True, default="sum", is_keyword_only=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="anticipate_dequant_time", type="bool", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="dequant_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x3", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="anticipate_return_dim", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="anticipate_comm_mode", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="anticant_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="antiquant_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="antiquant_offset", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="antiquant_group_size", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x2_dtype", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="comm_mode", type="str", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="stream_mode", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="comm_turn", type="int", is_optional=True, default=0, is_keyword_only=True),
                ], return_count=1)],
            ),
            "torch_npu.npu_all_gather_quant_mm": APIParamInfo(
                api_name="torch_npu.npu_all_gather_quant_mm",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="x1", type="Tensor"),
                    ParamInfo(name="x2", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="world_size", type="int", is_optional=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x1_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x2_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="quant_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="quant_mode", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="gather_index", type="int", is_optional=True, default=0, is_keyword_only=True),
                    ParamInfo(name="gather_output", type="bool", is_optional=True, default=True, is_keyword_only=True),
                    ParamInfo(name="comm_turn", type="int", is_optional=True, default=0, is_keyword_only=True),
                    ParamInfo(name="group_size", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="comm_mode", type="str", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="stream_mode", type="int", is_optional=True, is_keyword_only=True),
                ], return_count=3)],
            ),
            "torch_npu.npu_quant_matmul_all_to_all": APIParamInfo(
                api_name="torch_npu.npu_quant_matmul_all_to_all",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="x1", type="Tensor"),
                    ParamInfo(name="x2", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="world_size", type="int", is_optional=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x1_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x2_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="common_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x1_offset", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x2_offset", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x1_quant_mode", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x2_quant_mode", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="common_quant_mode", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="group_size", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="all2all_axes", type="int[]", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="y_dtype", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="comm_mode", type="str", is_optional=True, is_keyword_only=True),
                ], return_count=1)],
            ),
            "torch_npu.npu_quant_gmm_alltoallv": APIParamInfo(
                api_name="torch_npu.npu_quant_gmm_alltoallv",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="gmm_x", type="Tensor"),
                    ParamInfo(name="gmm_weight", type="Tensor"),
                    ParamInfo(name="gmm_x_scale", type="Tensor"),
                    ParamInfo(name="gmm_weight_scale", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="ep_world_size", type="int", is_optional=True),
                    ParamInfo(name="send_counts", type="int[]", is_optional=True),
                    ParamInfo(name="recv_counts", type="int[]", is_optional=True),
                    ParamInfo(name="send_counts_tensor", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="recv_counts_tensor", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_x", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_weight", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_x_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_weight_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(
                        name="trans_gmm_weight", type="bool", is_optional=True, default=False, is_keyword_only=True),
                    ParamInfo(
                        name="trans_mm_weight", type="bool", is_optional=True, default=False, is_keyword_only=True),
                    ParamInfo(name="gmm_y_dtype", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_y_dtype", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="group_size", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="comm_quant_mode", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="comm_quant_dtype", type="int", is_optional=True, is_keyword_only=True),
                ], return_count=2)],
            ),
            "torch_npu.npu_alltoallv_quant_gmm": APIParamInfo(
                api_name="torch_npu.npu_alltoallv_quant_gmm",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="gmm_x", type="Tensor"),
                    ParamInfo(name="gmm_weight", type="Tensor"),
                    ParamInfo(name="gmm_x_scale", type="Tensor"),
                    ParamInfo(name="gmm_weight_scale", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="ep_world_size", type="int", is_optional=True),
                    ParamInfo(name="send_counts", type="int[]", is_optional=True),
                    ParamInfo(name="recv_counts", type="int[]", is_optional=True),
                    ParamInfo(name="send_counts_tensor", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="recv_counts_tensor", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_x", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_weight", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_x_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_weight_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(
                        name="trans_gmm_weight", type="bool", is_optional=True, default=False, is_keyword_only=True),
                    ParamInfo(
                        name="trans_mm_weight", type="bool", is_optional=True, default=False, is_keyword_only=True),
                    ParamInfo(
                        name="permute_out_flag", type="bool", is_optional=True, default=False, is_keyword_only=True),
                    ParamInfo(name="gmm_y_dtype", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_y_dtype", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="group_size", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="comm_quant_mode", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="comm_quant_dtype", type="int", is_optional=True, is_keyword_only=True),
                ], return_count=3)],
            ),
            "torch_npu.npu_all_gather_base_mm": APIParamInfo(
                api_name="torch_npu.npu_all_gather_base_mm",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="input", type="Tensor"),
                    ParamInfo(name="x2", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="world_size", type="int", is_optional=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="gather_index", type="int", is_optional=True, default=0, is_keyword_only=True),
                    ParamInfo(name="gather_output", type="bool", is_optional=True, default=True, is_keyword_only=True),
                    ParamInfo(name="comm_turn", type="int", is_optional=True, default=0, is_keyword_only=True),
                ], return_count=2)],
            ),
            "torch_npu.npu_mm_reduce_scatter_base": APIParamInfo(
                api_name="torch_npu.npu_mm_reduce_scatter_base",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="input", type="Tensor"),
                    ParamInfo(name="x2", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="world_size", type="int", is_optional=True),
                    ParamInfo(name="reduce_op", type="str", is_optional=True, default="sum", is_keyword_only=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="comm_turn", type="int", is_optional=True, default=0, is_keyword_only=True),
                ], return_count=1)],
            ),
            "torch_npu.npu_matmul_all_to_all": APIParamInfo(
                api_name="torch_npu.npu_matmul_all_to_all",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="x1", type="Tensor"),
                    ParamInfo(name="x2", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="world_size", type="int", is_optional=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True),
                    ParamInfo(name="all2all_axes", type="int[]", is_optional=True),
                    ParamInfo(name="comm_mode", type="str", is_optional=True),
                ], return_count=1)],
            ),
            "torch_npu.npu_all_to_all_matmul": APIParamInfo(
                api_name="torch_npu.npu_all_to_all_matmul",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="x1", type="Tensor"),
                    ParamInfo(name="x2", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="world_size", type="int", is_optional=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True),
                    ParamInfo(name="all2all_axes", type="int[]", is_optional=True),
                    ParamInfo(name="all2all_out_flag", type="bool", is_optional=True, default=True),
                    ParamInfo(name="comm_mode", type="str", is_optional=True),
                ], return_count=2)],
            ),
            "torch_npu.npu_all_to_all_quant_matmul": APIParamInfo(
                api_name="torch_npu.npu_all_to_all_quant_matmul",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="x1", type="Tensor"),
                    ParamInfo(name="x2", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="world_size", type="int", is_optional=True),
                    ParamInfo(
                        name="all2all_out_flag", type="bool", is_optional=True, default=True, is_keyword_only=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x1_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x2_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="common_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x1_offset", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x2_offset", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x1_quant_mode", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="x2_quant_mode", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="common_quant_mode", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="group_sizes", type="int[]", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="all2all_axes", type="int[]", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="y_dtype", type="int", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="comm_mode", type="str", is_optional=True, is_keyword_only=True),
                ], return_count=2)],
            ),
            "torch_npu.npu_gmm_alltoallv": APIParamInfo(
                api_name="torch_npu.npu_gmm_alltoallv",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="gmm_x", type="Tensor"),
                    ParamInfo(name="gmm_weight", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="ep_world_size", type="int", is_optional=True),
                    ParamInfo(name="send_counts", type="int[]", is_optional=True),
                    ParamInfo(name="recv_counts", type="int[]", is_optional=True),
                    ParamInfo(name="send_counts_tensor", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="recv_counts_tensor", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_x", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_weight", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(
                        name="trans_gmm_weight", type="bool", is_optional=True, default=False, is_keyword_only=True),
                    ParamInfo(
                        name="trans_mm_weight", type="bool", is_optional=True, default=False, is_keyword_only=True),
                ], return_count=2)],
            ),
            "torch_npu.npu_alltoallv_gmm": APIParamInfo(
                api_name="torch_npu.npu_alltoallv_gmm",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="gmm_x", type="Tensor"),
                    ParamInfo(name="gmm_weight", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="ep_world_size", type="int", is_optional=True),
                    ParamInfo(name="send_counts", type="int[]", is_optional=True),
                    ParamInfo(name="recv_counts", type="int[]", is_optional=True),
                    ParamInfo(name="send_counts_tensor", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="recv_counts_tensor", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_x", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="mm_weight", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(
                        name="trans_gmm_weight", type="bool", is_optional=True, default=False, is_keyword_only=True),
                    ParamInfo(
                        name="trans_mm_weight", type="bool", is_optional=True, default=False, is_keyword_only=True),
                    ParamInfo(
                        name="permute_out_flag", type="bool", is_optional=True, default=False, is_keyword_only=True),
                ], return_count=3)],
            ),
            "mindspeed.npu_bmm_reducescatter_alltoall": APIParamInfo(
                api_name="mindspeed.npu_bmm_reducescatter_alltoall",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="x", type="Tensor"),
                    ParamInfo(name="weight", type="Tensor"),
                    ParamInfo(name="group_ep", type="str", is_optional=True),
                    ParamInfo(name="group_ep_worldsize", type="int", is_optional=True),
                    ParamInfo(name="group_tp", type="str", is_optional=True),
                    ParamInfo(name="group_tp_worldsize", type="int", is_optional=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="shard_type", type="int", is_optional=True, default=0, is_keyword_only=True),
                ], return_count=1)],
            ),
            "mindspeed.npu_alltoall_allgather_bmm": APIParamInfo(
                api_name="mindspeed.npu_alltoall_allgather_bmm",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="x", type="Tensor"),
                    ParamInfo(name="weight", type="Tensor"),
                    ParamInfo(name="group_ep", type="str", is_optional=True),
                    ParamInfo(name="group_ep_worldsize", type="int", is_optional=True),
                    ParamInfo(name="group_tp", type="str", is_optional=True),
                    ParamInfo(name="group_tp_worldsize", type="int", is_optional=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="shard_type", type="int", is_optional=True, default=0, is_keyword_only=True),
                    ParamInfo(name="act_type", type="str", is_optional=True, is_keyword_only=True),
                    ParamInfo(
                        name="need_allgather_out", type="bool", is_optional=True, default=False, is_keyword_only=True),
                    ParamInfo(
                        name="need_activation_feature", type="bool", is_optional=True, default=False,
                        is_keyword_only=True),
                ], return_count=3)],
            ),
            "mindspeed.npu_mm_all_reduce_add_rms_norm": APIParamInfo(
                api_name="mindspeed.npu_mm_all_reduce_add_rms_norm",
                source="manual-mc2",
                overloads=[OverloadInfo(params=[
                    ParamInfo(name="x1", type="Tensor"),
                    ParamInfo(name="x2", type="Tensor"),
                    ParamInfo(name="residual", type="Tensor"),
                    ParamInfo(name="gamma", type="Tensor"),
                    ParamInfo(name="hcom", type="str", is_optional=True),
                    ParamInfo(name="reduce_op", type="str", is_optional=True, default="sum", is_keyword_only=True),
                    ParamInfo(name="epsilon", type="float", is_optional=True, default=1e-6, is_keyword_only=True),
                    ParamInfo(name="bias", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="antiquant_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="antiquant_offset", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(name="dequant_scale", type="Tensor", is_optional=True, is_keyword_only=True),
                    ParamInfo(
                        name="antiquant_group_size", type="int", is_optional=True, default=0, is_keyword_only=True),
                    ParamInfo(name="comm_turn", type="int", is_optional=True, default=0, is_keyword_only=True),
                ], return_count=2)],
            ),
        }
        for api_name, info in mc2_apis.items():
            info.params = info.overloads[0].params
            register_api_params(api_name, info.params, info.source)
            from ttk.core_modules.framework_api.framework_api_info_keeper import FrameworkApiInfoKeeper
            FrameworkApiInfoKeeper()._cache[api_name] = info
        logging.info(f"Registered {len(mc2_apis)} mc2 torch_npu APIs")

    def get_device_count(self):
        switches = get_global_storage()
        if switches.device_count == -1:
            switches.device_count = self.backend.device_count()
        logging.info(f"Device count: {switches.device_count}")

    def get_device_platform(self):
        switches = get_global_storage()
        if switches.dev_plat == "AUTO":
            switches.dev_plat = self.backend.device_name()
        switches.short_soc_version = self.backend.soc_series()
        logging.info(f"Device platform: {switches.dev_plat}")

    def setup_profile_object(self):
        self.profile_object = FrameworkApiProfileObject(self.task_keeper, self.mp_context, self.backend)

    def device_info(self, dev_id: int) -> str:
        return f"{self.backend.device_type()}:{dev_id}"
