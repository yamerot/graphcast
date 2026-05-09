import dataclasses
import functools

import os
import numpy as np
import pandas as pd
import xarray as xr
import jax
import haiku as hk

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast.data_utils import add_tisr_var, add_derived_vars

# 模型权重文件路径（.npz）
CHECKPOINT_PATH = 'graphcast/params/graphcast_params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz'
DATA_PATH = "/path/to/your/input.nc"        # 输入气象数据文件路径
STATS_DIR = "graphcast/stats/"              # 标准化统计量所在目录
OUTPUT_PATH = "graphcast/output/24.zarr"    # 预测结果保存路径
PRED_STEPS = 4                         # 预测步数（每步 6 小时）
DEVICE = "gpu"                              # 计算设备：可选 "cpu", "gpu", "tpu"


def load_checkpoint(ckpt_path: str):
    """加载模型参数和配置"""
    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    print("model description:\n", ckpt.description)
    print("license:\n", ckpt.license)
    print("model configuration:\n", ckpt.model_config)
    print("task configuration:\n", ckpt.task_config)
    if hasattr(ckpt, 'state'):
        state = ckpt.state
    else:
        state = {}
    return ckpt.params, state, ckpt.model_config, ckpt.task_config

def load_data(data_paths: list, stats_dir: str):
    ds = xr.open_mfdataset(data_paths).rename({})

    diffs = xr.load_dataset(f"{stats_dir}/diffs_stddev_by_level.nc").compute()
    mean = xr.load_dataset(f"{stats_dir}/mean_by_level.nc").compute()
    stddev = xr.load_dataset(f"{stats_dir}/stddev_by_level.nc").compute()

    return ds, diffs, mean, stddev

def prepare_data_from_hourly(
    ds: xr.Dataset, input_vars: tuple, target_vars: tuple, forcing_vars: tuple, plevs: tuple, 
    input_duration: str = "12h",
    target_hours: list = [6, 12, 18, 24],
):
    # 1. 检查 datetime 坐标
    assert 'datetime' in ds.coords, 'datetime not found in dataset'
    ds = ds.sortby('datetime')

    # 2. 为原始历史数据添加 TISR 和派生变量（用于输入部分）
    data_utils.add_tisr_var(ds)
    data_utils.add_derived_vars(ds)

    # 3. 重组输入
    # 原始数据为逐小时连续时间块 [datetime, ...], 重组为 [batch, time, ...] 结构
    # 其中 time 代表模型输入时刻 (eg. [-6h, 0h]) 而非真实时间
    # 在不同真实时刻上所取的输入被划分为不同 batch, 
    # 比如 batch[0] <- [T00, T06], batch[1] <- [T01, T07] ...
    onehour = pd.Timedelta('1h')
    input_hours = list(range(- pd.Timedelta(input_duration) // onehour + 6, 6, 6))  # [-6, 0]
    inputs = []
    chunksize = len(ds.time.values)
    batchsize = chunksize + input_hours[0]
    for t in input_hours:
        shift = t - input_hours[0]
        moment = ds.isel(datetime=np.arange(shift, shift + batchsize)).drop_vars('datetime').rename(datetime='batch')
        inputs.append(moment)
    inputs = xr.concat(inputs, dim='time')
    input_times = [pd.Timedelta(hours=h) for h in input_hours]
    inputs = inputs.assign_coords(time=input_times)

    target_times = [pd.Timedelta(hours=h) for h in target_hours]

    # 提取所有可能的窗口起始索引（保证输入部分完整）
    time_idx = np.arange(len(ds_step['datetime']))
    # 最后一个可用的起始索引：len - input_steps + 1? 但 input_steps 窗口包含索引 start...start+input_steps-1
    start_indices = time_idx[: -input_steps + 1] if input_steps > 1 else time_idx

    batch_inputs, batch_targets, batch_forcings = [], [], []

    # 静态变量（无时间维度）
    static_vars = [v for v in input_vars
                   if v in ds_step.data_vars and 'datetime' not in ds_step[v].dims]
    dyn_input_vars = [v for v in input_vars if v not in static_vars]
    dyn_target_vars = [v for v in target_vars if v not in static_vars]
    dyn_forcing_vars = [v for v in forcing_vars if v not in static_vars]

    for start in start_indices:
        # ---- 构建 inputs ----
        input_data = ds_step.isel(datetime=slice(start, start + input_steps))
        ref_time = input_data['datetime'].isel(datetime=input_steps - 1)  # 参考时刻（输入最后一步）
        # 转为相对时间 (timedelta)
        input_rel_times = input_data['datetime'] - ref_time
        input_data = input_data.assign_coords(datetime=input_rel_times)
        input_data = input_data.rename({'datetime': 'time'})

        # ---- 构建未来的目标时间坐标 ----
        target_rel_times = np.array(target_timedeltas)  # 已经是相对于 ref_time 的偏移
        # 构造一个只包含时间、纬度、经度的临时 Dataset，用于计算 TISR 和派生变量
        target_ds = xr.Dataset(
            coords={
                'datetime': target_rel_times,   # 实际是 timedelta，但 forceding 函数需要 datetime 坐标
                'lat': ds_step.coords['lat'],
                'lon': ds_step.coords['lon'],
            }
        )
        # 注意：add_tisr_var 和 add_derived_vars 需要 datetime 坐标是真正的 datetime64 类型，但这里只是 timedelta？不行！
        # 我们需要将目标相对时间转换为绝对时间：ref_time + offset
        # ref_time 是 pandas Timestamp 或 numpy datetime64
        abs_target_times = ref_time.values + target_rel_times  # 得到绝对时间数组
        target_ds = target_ds.assign_coords(datetime=abs_target_times)

        # 计算目标时刻的 TISR 和派生变量
        add_tisr_var(target_ds)
        add_derived_vars(target_ds)

        # 从 target_ds 中提取 forcing 变量
        forcing_data = target_ds[list(set(dyn_forcing_vars) & set(target_ds.data_vars))]
        # 将时间坐标改为相对时间 (timedelta)，以便与 targets 对齐
        forcing_data = forcing_data.assign_coords(datetime=target_rel_times)
        forcing_data = forcing_data.rename({'datetime': 'time'})

        # ---- 构建全 NaN 的目标模板 ----
        target_data = target_ds.copy()
        # 将目标变量（target_vars）全部设为 NaN，保留维度
        for var in dyn_target_vars:
            # 目标变量应该存在于 target_ds 中（因为 add_* 只添加了 forcing 相关），
            # 但我们的 target_ds 刚开始是空的，只有坐标，需要先添加同名变量并设为 NaN
            # 所以更简单：先构建一个具有相同时间/空间坐标的 DataArray，全 NaN
            # 这里用 xr 广播机制：创建一个 NaN 模板
            pass

        # 推荐方式：直接用 xr 构造全 NaN 的目标，保持与 input 相同的空间维度
        # 先利用 target_ds 的坐标创建模板，再赋予 NaN
        target_template = xr.Dataset(
            {var: (('datetime',) + (('level',) if 'level' in input_data[var].dims else ()) + ('lat', 'lon'),
                   np.full_like(
                       input_data[var].isel(time=0).values if 'level' not in input_data[var].dims else
                       input_data[var].isel(time=0).values, np.nan, dtype=np.float32))
             for var in dyn_target_vars if var in dyn_input_vars or var in static_vars},
            coords={
                'datetime': target_rel_times,
                'lat': ds_step.coords['lat'],
                'lon': ds_step.coords['lon'],
            }
        )
        # 如果目标变量有 level 维度，需要加入 level 坐标
        if 'level' in input_data.dims:
            target_template.coords['level'] = input_data.coords['level']
        # 现在将值全部设为 NaN，并扩展时间维度（广播）
        # 上述 np.full_like 已生成 NaN，但需要每个时间步都复制？我们用 ones_like 然后乘 NaN
        # 简化：直接用 xr 的 broadcasting
        target_data = target_template
        # 其实 target_template 已经是 (datetime, lat, lon) 的多维 NaN 了，但 datetime 只有一个值，我们需要为每个目标时间复制
        # 所以需要 expand_dims 然后 concat 或 tile。不如用循环：
        target_list = []
        for t in target_rel_times:
            dt = target_template.isel(datetime=0)  # 取第一个模板
            dt = dt.expand_dims('time')
            dt['time'] = [t]
            target_list.append(dt)
        target_data = xr.concat(target_list, dim='time')
        # 修正坐标名称
        target_data = target_data.rename({'time': 'time'})  # 已经叫 time 了，但之前是 datetime，无所谓

        # 但我们还需确保目标变量中没有 NaN 之外的问题。上面生成已经都是 NaN。

        # 现在将时间坐标变为 timedelta（与输入一致）
        target_data = target_data.assign_coords(time=target_rel_times)

        # 添加静态变量到输入（扩展时间维度）
        for var in static_vars:
            if var in ds_step.data_vars:
                # 静态变量没有时间维度，直接复制到输入中
                input_data[var] = ds_step[var]

        # 添加 batch 维度
        input_data = input_data.expand_dims('batch')
        target_data = target_data.expand_dims('batch')
        forcing_data = forcing_data.expand_dims('batch')

        batch_inputs.append(input_data)
        batch_targets.append(target_data)
        batch_forcings.append(forcing_data)

    # 沿 batch 维度拼接
    inputs = xr.concat(batch_inputs, dim='batch')
    targets = xr.concat(batch_targets, dim='batch')
    forcings = xr.concat(batch_forcings, dim='batch')

    # 按气压层筛选（对 level 维进行选择）
    if 'level' in inputs.dims:
        inputs = inputs.sel(level=list(pressure_levels))
    if 'level' in targets.dims:
        targets = targets.sel(level=list(pressure_levels))
    # forcings 通常不含 level 维度，sel 会忽略

    return inputs, targets, forcings

def prepare_inputs_targets_forcings(ds: xr.Dataset, task_config):
    """根据任务配置提取输入、目标和强迫项（目标仅用于构造模板）"""

    # 计算最大可用预测步数（除去两个输入时次）
    max_steps = ds.sizes["time"] - 2
    lead_steps = min(PRED_STEPS, max_steps)
    target_lead_times = slice("6h", f"{lead_steps*6}h")

    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        ds,
        target_lead_times=target_lead_times,
        input_variables=task_config.input_variables,
        target_variables=task_config.target_variables,
        forcing_variables=task_config.forcing_variables,
        pressure_levels=task_config.pressure_levels,
        input_duration=task_config.input_duration,
    )

    print(f"input dims: {inputs.dims}")
    print(f"target dims: {targets.dims}")
    print(f"forcing dims: {forcings.dims}")

    return inputs, targets, forcings


def build_inference_fn(model_config, task_config, diffs_stddev_by_level, mean_by_level, stddev_by_level):
    """构建包装好并经过标准化处理的自回归预测函数"""
    def construct_wrapped_graphcast():
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    # 将 model_config 和 task_config 绑定到函数
    return functools.partial(run_forward, model_config=model_config, task_config=task_config)


def main():
    # 1. 加载模型
    assert DEVICE in ['cpu', 'gpu', 'tpu'], 'device should be cpu/gpu/tpu'
    jax.config.update("jax_platform_name", DEVICE)

    params, state, model_config, task_config = load_checkpoint(CHECKPOINT_PATH)
    return

    # 2. 加载数据
    print('loading data...')
    ds, diffs_stddev_by_level, mean_by_level, stddev_by_level = load_data(DATA_PATH, STATS_DIR)
    inputs, targets, forcings = prepare_inputs_targets_forcings(ds, task_config)
    

    # 3. 构建推理函数
    print("building inference functions...")
    run_forward = build_inference_fn(model_config, task_config,
                                     diffs_stddev_by_level, mean_by_level, stddev_by_level)

    # 绑定参数和状态
    bound_apply = functools.partial(run_forward.apply, params=params, state=state)

    # JIT 编译，并去掉返回值中的 state（因为模型无状态）
    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]
    run_forward_jitted = drop_state(jax.jit(bound_apply))

    # 4. 执行预测
    targets_template = targets * np.nan
    print("predicting ...")
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
    )
    print("prediction done")

    # 5. 保存结果
    predictions.to_zarr(OUTPUT_PATH)
    print(f"saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()