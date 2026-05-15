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
    ds = xr.open_mfdataset(data_paths).rename({
        't2m': '2m_temperature', 
        'msl': 'mean_sea_level_pressure', 
        'u10': '10m_v_component_of_wind', 
        'v10': '10m_u_component_of_wind', 
        't': 'temperature', 
        'z': 'geopotential', 
        'u': 'u_component_of_wind', 
        'v': 'v_component_of_wind', 
        'w': 'vertical_velocity', 
        'q': 'specific_humidity'
    })

    diffs = xr.load_dataset(f"{stats_dir}/diffs_stddev_by_level.nc").compute()
    mean = xr.load_dataset(f"{stats_dir}/mean_by_level.nc").compute()
    stddev = xr.load_dataset(f"{stats_dir}/stddev_by_level.nc").compute()

    return ds, diffs, mean, stddev

def prepare_data_from_hourly(
    ds: xr.Dataset, input_vars: tuple, target_vars: tuple, 
    input_duration: str = "12h",
    target_hours: list = [6, 12, 18, 24],
):
    def batching(ds: xr.Dataset, window_slice) -> xr.Dataset:
        """
        ds: 逐小时连续时间块 [time, ...]
        window_slice: 需要滑动选取的相对时刻窗口 (eg. [0, 6]) 
        将 ds 重组为 [batch, time, ...] 结构, 其中 time 维度被替换为相对时刻, 在不同真实时刻上所取的输入被划分为不同 batch, 比如 batch[0] <- [T00, T06], batch[1] <- [T01, T07] ...
        """
        moments = []
        chunksize = len(ds.time.values)
        batchsize = chunksize - window_slice[-1]
        for t in window_slice:
            moment = ds.isel(time=np.arange(t, t + batchsize))\
                    .rename(time='batch').assign_coords(batch=np.arange(0, batchsize))
            moments.append(moment)
        return xr.concat(moments, dim='time').assign_coords(time=np.array(window_slice))
    
    # 1. 前置处理: 检查 datetime 坐标, 纬度坐标按升序排列
    if 'datetime' not in ds.coords:
        assert 'time' in ds.coords, 'datetime not found'
        ds = ds.rename_vars(time='datetime').assign_coords(time=ds.time)
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.reindex (lat=list(reversed(ds.lat)))

    # 2. 输入数据处理: 添加 TISR 和派生变量, 时间维度重组
    data_utils.add_tisr_var(ds)
    data_utils.add_derived_vars(ds)
    onehour = pd.Timedelta('1h')
    slice_in = np.arange(0, 6, pd.Timedelta(input_duration) // onehour)  # 12h -> (0, 6)
    inputs = batching(ds[input_vars], slice_in)
    input_last_datetime = inputs.datetime.sel(batch=1)

    # 3. 从目标时间轴构建 forcing 数据, 计算 TISR 和派生变量
    target_datetime = xr.concat(
        [input_last_datetime + pd.Timedelta(hours=h) for h in target_hours], 
        dim='time')
    forcings = xr.Dataset(
            coords={
                'time': (target_datetime - target_datetime[0]) // onehour, 
                'level': ds.level, 
                'lat': ds.lat, 'lon': ds.lon, 
                'datetime': target_datetime, 
            }
        )
    data_utils.add_tisr_var(forcings)
    data_utils.add_derived_vars(forcings)
    slice_tg = np.array(target_hours) - target_hours[0]
    forcings = batching(forcings, slice_tg)

    # 4. 构建 target 数据, 用 nan 填充
    surface_shape = (len(forcings.batch), len(forcings.time), len(forcings.lat), len(forcings.lon))
    upper_shape = (len(forcings.batch), len(forcings.time), len(forcings.level), len(forcings.lat), len(forcings.lon))

    surface_nan = (('batch', 'time', 'lat', 'lon'), np.full(surface_shape, np.nan))
    upper_nan = (('batch', 'time', 'level', 'lat', 'lon'), np.full(upper_shape, np.nan))

    targets = xr.Dataset(
        data_vars={var: surface_nan if 'level' in ds[var].dims else upper_nan 
         for var in target_vars}, 
        coords=forcings.coords
    )

class Loader:
    def __init__(ds, chunksize, batchsize):
        pass

    def batching(ds, window_slice):
        pass

    def update(idx):
        pass

    def claim():
        pass

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