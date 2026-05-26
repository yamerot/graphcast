import dataclasses
import functools

import os
import sys
import logging
import numpy as np
import pandas as pd
import xarray as xr
import jax
import haiku as hk

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast.data_utils import add_tisr_var, add_derived_vars
from zarr.codecs import BloscCodec

# nohup python run_gr.py &

# 模型权重文件路径（.npz）
CHECKPOINT_PATH = 'graphcast/params/graphcast_params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz'
UPPER_PATH = '/home/chengzy/dlink/era5_zarr_v3/gtest.zarr'
SURFACE_PATH = '/home/chengzy/dlink/era5_zarr_v3/slev.zarr'
STATIC_PATH = '/home/chengzy/dlink/era5_zarr_v3/static.zarr'        # 输入气象数据文件路径
STATS_DIR = "graphcast/stats/graphcast_stats_"              # 标准化统计量路径前缀
OUTPUT_PATH = "graphcast/output/try.zarr"    # 预测结果保存路径
DEVICE = "gpu"                              # 计算设备：可选 "cpu", "gpu", "tpu"


def load_checkpoint(ckpt_path: str):
    """加载模型参数和配置"""
    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    logger.info(ckpt.description)
    # logger.info(ckpt.license)
    logger.info(ckpt.model_config)
    logger.info(ckpt.task_config)
    if hasattr(ckpt, 'state'):
        state = ckpt.state
    else:
        state = {}
    return ckpt.params, state, ckpt.model_config, ckpt.task_config

def load_data(upper_path, surface_path, static_path, stats_prefix):
    """预加载数据集, 应用一些固定的前处理"""
    def preprocess(ds, rename_dict):
        """前置处理, 使其符合 graphcast 官方提供的规范"""
        # time 维度转为整数, 新的 datetime 维度储存原来的真实日期
        if 'datetime' not in ds.coords:
            assert 'time' in ds.coords, 'datetime not found'
            ds = ds.assign_coords(datetime=ds.time)
            time_index = ((ds.time - ds.time[0]) // pd.Timedelta('1h')).astype(np.int32)
            ds = ds.assign_coords(time=time_index)
        # 纬度升序排序
        if ds.lat.values[0] > ds.lat.values[-1]:
            ds = ds.reindex(lat=list(reversed(ds.lat)))
        # 气压升序排序
        if 'lev' in list(ds.coords) and ds.lev.values[0] > ds.lev.values[-1]:
            ds = ds.reindex(lev=list(reversed(ds.lev)))
        # 将变量名替换为长名
        if rename_dict:
            ds = ds.rename(rename_dict)
        return ds

    upper_dict = {
        'lev': 'level', 
        't': 'temperature', 
        'z': 'geopotential', 
        'u': 'u_component_of_wind', 
        'v': 'v_component_of_wind', 
        'w': 'vertical_velocity', 
        'q': 'specific_humidity'
    }
    ds = preprocess(xr.open_dataset(upper_path), upper_dict)
    surface_dict = {
        't2m': '2m_temperature', 
        'msl': 'mean_sea_level_pressure', 
        'u10': '10m_v_component_of_wind', 
        'v10': '10m_u_component_of_wind', 
    }
    ds_surface = preprocess(xr.open_dataset(surface_path), surface_dict)

    ds_static = xr.load_dataset(static_path).compute()
    diffs = xr.load_dataset(f"{stats_prefix}diffs_stddev_by_level.nc").compute()
    mean = xr.load_dataset(f"{stats_prefix}mean_by_level.nc").compute()
    stddev = xr.load_dataset(f"{stats_prefix}stddev_by_level.nc").compute()

    return ds, ds_surface, ds_static, diffs, mean, stddev

class Loader:
    """
    对于大规模数据集, 采取懒加载, 逐个一定时间内的连续小时数据再划分为 batch
    """
    def __init__(self, ds, ds_surface, ds_static, 
                 input_vars: list | tuple, 
                 target_vars: list | tuple, 
                 batchsize: int = 6, 
                 input_duration: str = "12h",
                 target_hours: list = [6, 12, 18, 24]):

        
        sel_vars = [var for var in list(ds.data_vars) if var in input_vars]
        ds_sel = ds[sel_vars]

        sel_vars = [var for var in list(ds_surface.data_vars) if var in input_vars]
        ds_surface_sel = ds_surface[sel_vars]

        sel_vars = [var for var in list(ds_static.data_vars) if var in input_vars]
        self.ds_static = ds_static[sel_vars]

        # 1. 取样的相对时刻窗口, 
        # e.g. input_duration: '12h' -> input_hours: (-6, 0)
        onehour = pd.Timedelta('1h')
        self.input_hours = np.arange(- pd.Timedelta(input_duration) // onehour, 0, 6) + 6
        self.batchsize = batchsize
        self.continuous_chunksize = batchsize + 6 * (len(self.input_hours) - 1)
        self.target_hours = np.array(target_hours)

        # 2. 时间连续形式的 forcings 模板
        self.forcings_template = xr.Dataset(
            coords={'time': np.arange(0, batchsize + target_hours[-1] - target_hours[0]), 
                    'level': ds.level, 
                    'lat': ds.lat, 
                    'lon': ds.lon}
        )

        # 3. batch 形式的 targets 模板, 用 nan 填充
        surface_shape = (batchsize, len(target_hours), len(ds.lat), len(ds.lon))
        upper_shape = (batchsize, len(target_hours), len(ds.level), len(ds.lat), len(ds.lon))
        surface_nan = (('batch', 'time', 'lat', 'lon'), np.full(surface_shape, np.nan))
        upper_nan = (('batch', 'time', 'level', 'lat', 'lon'), np.full(upper_shape, np.nan))

        self.targets_template = xr.Dataset(
            data_vars={var: upper_nan if var in list(ds.data_vars) else surface_nan
            for var in target_vars}, 
            coords={'time': target_hours, 
                    'level': ds.level, 
                    'lat': ds.lat, 
                    'lon': ds.lon}
        )
        
        # 4. 数据暂存区
        self.upper_cache = ChunkCache(ds_sel)
        self.surface_cache = ChunkCache(ds_surface_sel)

    def batching(self, ds: xr.Dataset, hours) -> xr.Dataset:
        """
        ds: 逐小时连续时间块 [time, ...] * 注意: time 维度需要从 0 开始
        hours: 需要滑动选取的相对时刻窗口 (eg. [0, 6], [6, 12, 18] 等) 
        将 ds 重组为 [batch, time, ...] 结构, 其中 time 维度被替换为相对时刻, 在不同真实时刻上所取的输入被划分为不同 batch, 比如 batch[0] <- [T00, T06], batch[1] <- [T01, T07] ...
        """
        moments = []
        window_slice = hours - hours[0]
        batchsize = len(ds.time) - window_slice[-1]
        for t in window_slice:
            moment = ds.isel(time=np.arange(t, t + batchsize))\
                    .rename(time='batch').drop_vars('batch')\
                    .reset_coords('datetime')   # 将 datetime 坐标转为一维变量, 才能进行拼接
            moments.append(moment)
        minibatch = xr.concat(moments, dim='time').assign_coords(time=np.array(hours))\
               .set_coords('datetime')  # 此时 datetime 变为二维坐标
        return minibatch

    def update(self, start):
        idx_range = np.arange(start, start + self.continuous_chunksize)
        inputs = self.get_continuous(idx_range)
        add_tisr_var(inputs)
        add_derived_vars(inputs)
        inputs = self.batching(inputs, self.input_hours)
        inputs.update(self.ds_static)
        
        # forcings 和 targets 的时间信息是一致的
        # 在输入的基准时间上加上预测时效，得到目标时间 (连续形式), 写入 forcings 模板
        targets_datetime = np.concatenate(
            [inputs.datetime.sel(time=0).data + pd.Timedelta(hours=h) for h in self.target_hours]
        )
        forcings = self.forcings_template.assign_coords(datetime=('time', targets_datetime))
        add_tisr_var(forcings)
        add_derived_vars(forcings)
        forcings = self.batching(forcings, self.target_hours)

        target_batch_datetime = forcings.datetime
        inputs = inputs.drop_vars(['datetime', 'day_progress', 'year_progress'])
        forcings = forcings.drop_vars(['datetime', 'day_progress', 'year_progress'])

        logger.debug(inputs)
        logger.debug(self.targets_template)
        logger.debug(forcings)

        return target_batch_datetime, inputs, self.targets_template, forcings

    def get_continuous(self, idx) -> xr.Dataset:
        return xr.merge([self.upper_cache.claim(idx), 
                         self.surface_cache.claim(idx)], compat='no_conflicts')

class ChunkCache:
    """
    graphcast 需要两个时次输入, 因此每个时刻的数据将会被读取两次
    为了节省读取开销, 建立一个暂存区管理数据, 防止重复从硬盘载入
    暂存区一次只在内存存储两个块
    """
    def __init__(self, ds: xr.Dataset):
        assert 'time' in list(ds.coords), 'time must be in the dataset\'s coords'

        self.ds = ds
        self.chunksize = ds[list(ds.data_vars)[0]].encoding['chunks'][0]

        self.current = np.arange(0, 0 + self.chunksize)
        self.cache = [ds.isel(time=self.current).compute(), 
                      ds.isel(time=self.current + self.chunksize).compute()]

    def claim(self, idx):
        assert idx[0] >= self.current[0], 'no backward-loading'
        assert idx[0] < self.current[-1] + self.chunksize, 'no jumping over chunks'
        assert idx[-1] - idx[0] < self.chunksize, 'batchsize should <= chunksize'
        if idx[0] > self.current[-1]:
            self.cache.pop(0)
            print('moving to the next chunk')
            self.current = self.current + self.chunksize
            self.cache.append(self.ds.isel(time=self.current + self.chunksize).compute())
        if idx[-1] <= self.current[-1]:
            return self.cache[0].isel(time=idx)
        else:
            idx1 = idx[idx <= self.current[-1]]
            idx2 = idx[idx > self.current[-1]]
            return xr.concat([self.cache[0].isel(time=idx1), 
                              self.cache[1].isel(time=idx2)], dim='time')
        
def build_inference_fn(params, state, model_config, task_config, diffs_stddev_by_level, mean_by_level, stddev_by_level):
    """构建包装好并经过标准化处理的自回归预测函数"""
    def construct_wrapped_graphcast(model_config, task_config):
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
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    # 将固定参数绑定到函数
    return drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

def save_data(ds, path):
    """保存预测结果, 未完成: 将 batch 维度还原到 datetime, 按照预测时效分开保存数据"""
    if os.path.exists(path):
        ds.to_zarr(path, mode='a', append_dim='batch')
    else:
        compressor = BloscCodec(cname='zstd', clevel=5, shuffle='shuffle')
        encoding = {
            var: {
                'compressors': [compressor],
                "dtype": "float32"
            } for var in ds.data_vars}
        ds.to_zarr(path, mode='w', zarr_format=3, encoding=encoding)

def setup_logging(level=logging.INFO):
    logger = logging.getLogger('graphcast')
    logger.setLevel(level)

    # 文件处理器
    file_handler = logging.FileHandler('./run.log', mode='w')
    file_handler.setLevel(level)

    # 格式器：包含时间、级别、消息
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def main():
    # 1. 加载模型
    assert DEVICE in ['cpu', 'gpu', 'tpu'], 'device should be cpu/gpu/tpu'
    jax.config.update('jax_platform_name', DEVICE)
    jax.config.update('jax_traceback_filtering', 'off')
    # 模型缓存路径
    jax.config.update('jax_compilation_cache_dir', 'graphcast/params/tmp/jax_cache')

    params, state, model_config, task_config = load_checkpoint(CHECKPOINT_PATH)

    # 2. 加载数据
    logger.info('loading data...')
    batchsize = 1
    ds, ds_surface, ds_static, diffs_stddev_by_level, mean_by_level, stddev_by_level = load_data(UPPER_PATH, SURFACE_PATH, STATIC_PATH, STATS_DIR)
    loader = Loader(ds, ds_surface, ds_static, 
                    task_config.input_variables, task_config.target_variables, 
                    batchsize, '12h', [6, 12, 18, 24])
    
    # 3. 构建推理函数
    logger.info("building inference functions...")
    run_forward_jitted = build_inference_fn(params, state, model_config, task_config,
                                     diffs_stddev_by_level, mean_by_level, stddev_by_level)

    # 4. 执行预测
    logger.info("predicting ...")
    datelist = ds.datetime.data
    start = 0
    end = 12
    # end = len(ds.time) - loader.continuous_chunksize + 1
    for idx in np.arange(start, end, batchsize):
        logger.info(f'proceeding {datelist[idx]}')
        dt, inputs, targets, forcings = loader.update(idx)
        predictions = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets,
            forcings=forcings,
        )
        print(predictions)
        print(dt)
        save_data(predictions, OUTPUT_PATH, )
        
    logger.info("prediction done")

if __name__ == "__main__":
    os.chdir('/home/chengzy/graphcast')
    logger = setup_logging(logging.DEBUG)
    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception
    main()