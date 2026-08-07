import dataclasses
import functools

import os
import sys
import logging
from zarr.codecs import BloscCodec
import concurrent
import numpy as np
import pandas as pd
import xarray as xr
import jax
import haiku as hk
import threading
from queue import Queue

# --- 修改 1: 导入 GenCast 相关模块 ---
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning
from graphcast import checkpoint
from graphcast import normalization
from graphcast import rollout
from graphcast.data_utils import add_derived_vars

# nohup python run_ge_con.py &
# CUDA_LAUNCH_BLOCKING=1 JAX_DISABLE_CUDA_GRAPH=1 nohup python run_ge_con.py &

# 模型权重文件路径（.npz）
CHECKPOINT_PATH = 'gencast/params/gencast_params_GenCast 0p25deg _2019.npz' 
UPPER_PATH = '/home/chengzy/dlink/era5_zarr_v3/plev.zarr'
SURFACE_PATH = '/home/chengzy/dlink/era5_zarr_v3/slev.zarr'
STATIC_PATH = '/home/chengzy/dlink/era5_zarr_v3/static.zarr'        # 输入气象数据文件路径
STATS_DIR = "gencast/stats/gencast_stats_"              # 标准化统计量路径前缀
OUTPUT_DIR = "gencast/output/gencast_"    # 预测结果保存路径前缀
DEVICE = "gpu"                              # 计算设备：可选 "cpu", "gpu", "tpu"
BATCHSIZE = 1


def load_checkpoint(ckpt_path: str):
    """加载模型参数和配置"""
    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)

    state = {}
    
    # Replace attention mechanism.
    splash_spt_cfg = ckpt.denoiser_architecture_config.sparse_transformer_config
    tbd_spt_cfg = dataclasses.replace(splash_spt_cfg, attention_type="triblockdiag_mha", mask_type="full")
    denoiser_architecture_config = dataclasses.replace(ckpt.denoiser_architecture_config, sparse_transformer_config=tbd_spt_cfg)

    logger.info(ckpt.description)
    logger.info(f'{ckpt.task_config}\n')
    
    return (ckpt.params, state, ckpt.task_config, ckpt.sampler_config, 
            ckpt.noise_config, ckpt.noise_encoder_config, denoiser_architecture_config)

def load_data(upper_path, surface_path, static_path, stats_prefix):
    """预加载数据集, 应用一些固定的前处理"""
    def preprocess(ds: xr.Dataset, rename_dict):
        """前置处理, 使其符合官方提供的规范"""
        if not ds.time.to_index().is_monotonic_increasing:
            logger.warning("Detected unsorted time coordinates! Applying sortby('time'). This may impact I/O performance.")
            ds = ds.sortby('time')
        # time 维度转为整数, 新的 datetime 维度储存原来的真实日期
        if 'datetime' not in ds.coords:
            assert 'time' in ds.coords, 'datetime not found'
            ds = ds.assign_coords(datetime=ds.time)
            time_index = ((ds.time - ds.time[0]) // pd.Timedelta('1h')).astype(np.int32)
            ds = ds.assign_coords(time=time_index)
        # 纬度升序排序
        if ds.lat.values[0] > ds.lat.values[-1]:
            logger.debug('data reindexed by lat')
            ds = ds.reindex(lat=list(reversed(ds.lat)))
        # 气压升序排序
        if 'lev' in list(ds.coords):
            ds = ds.rename({'lev': 'level'})
            if ds.level.values[0] > ds.level.values[-1]:
                logger.debug('data reindexed by lev')
                ds = ds.reindex(level=list(reversed(ds.level)))
        # 将变量名替换为长名
        if rename_dict:
            ds = ds.rename(rename_dict)
        return ds

    ds = preprocess(xr.open_dataset(upper_path), upper_dict)
    ds_surface = preprocess(xr.open_dataset(surface_path), surface_dict)

    ds_static = xr.load_dataset(static_path).compute()
    if ds_static.lat.values[0] > ds_static.lat.values[-1]:
        logger.debug('static data reindexed by lat')
        ds_static = ds_static.reindex(lat=list(reversed(ds_static.lat.values)))

    diffs = xr.load_dataset(f"{stats_prefix}diffs_stddev_by_level.nc").compute()
    mean = xr.load_dataset(f"{stats_prefix}mean_by_level.nc").compute()
    stddev = xr.load_dataset(f"{stats_prefix}stddev_by_level.nc").compute()
    
    # --- 修改 3: 加载 min_by_level 用于 NaNCleaner ---
    min_vals = xr.load_dataset(f"{stats_prefix}min_by_level.nc").compute()

    logger.info('data preprocess done')

    return ds, ds_surface, ds_static, diffs, mean, stddev, min_vals

class Loader:
    def __init__(self, ds, ds_surface, ds_static, 
                 input_vars: list | tuple, 
                 target_vars: list | tuple, 
                 batchsize: int = 1, 
                 input_duration: str = "24h",
                 target_hours: list = [12, 24], 
                 start_idx: int = 0):
        
        sel_vars = [var for var in list(ds.data_vars) if var in input_vars]
        ds_sel = ds[sel_vars]

        sel_vars = [var for var in list(ds_surface.data_vars) if var in input_vars]
        ds_surface_sel = ds_surface[sel_vars]

        sel_vars = [var for var in list(ds_static.data_vars) if var in input_vars]
        self.ds_static = ds_static[sel_vars]

        onehour = pd.Timedelta('1h')
        self.input_hours = np.arange(- pd.Timedelta(input_duration) // onehour, 0, 12) + 12
        input_idx = self.input_hours - self.input_hours[0]
        self.input_batch_idx = np.concatenate([np.arange(x, x + batchsize) for x in input_idx])
        
        self.target_hours = np.array(target_hours)
        target_idx = self.target_hours - self.input_hours[0]
        self.target_batch_idx = np.concatenate([np.arange(x, x + batchsize) for x in target_idx])

        self.batchsize = batchsize

        self.forcings_template = xr.Dataset(
            coords={'time': self.target_batch_idx, 
                    'level': ds.level, 
                    'lat': ds.lat, 
                    'lon': ds.lon}
        )

        surface_shape = (batchsize, len(target_hours), len(ds.lat), len(ds.lon))
        upper_shape = (batchsize, len(target_hours), len(ds.level), len(ds.lat), len(ds.lon))
        surface_nan = (('batch', 'time', 'lat', 'lon'), np.full(surface_shape, np.nan))
        upper_nan = (('batch', 'time', 'level', 'lat', 'lon'), np.full(upper_shape, np.nan))

        self.targets_template = xr.Dataset(
            data_vars={var: upper_nan if var in list(ds.data_vars) else surface_nan
            for var in target_vars}, 
            coords={'time': self.target_hours, 
                    'level': ds.level, 
                    'lat': ds.lat, 
                    'lon': ds.lon}
        )
        
        self.upper_cache = ChunkCache(ds_sel, start_idx)
        self.surface_cache = ChunkCache(ds_surface_sel, start_idx)
        self.full_datetime = ds_sel.datetime.data

    def batching(self, ds: xr.Dataset, hours, batchsize) -> xr.Dataset:
        return ds.drop_vars('time')\
                 .assign_coords(
                    batch=('time', np.tile(np.arange(0, batchsize), len(hours))), 
                    moment=('time', np.repeat(np.arange(0, len(hours)), batchsize))
                 )\
                 .set_index(time=['batch', 'moment']).unstack('time')\
                 .rename(moment='time')\
                 .transpose('batch', 'time' ,'level', 'lat', 'lon')\
                 .assign_coords(time=hours)

    def update(self, start):
        inputs = self.get_chunk(start + self.input_batch_idx)
        add_derived_vars(inputs)
        inputs = self.batching(inputs, self.input_hours, self.batchsize)
        inputs.update(self.ds_static)
        
        targets_datetime = self.full_datetime[start] + self.target_batch_idx.astype('timedelta64[h]')
        forcings = self.forcings_template.assign_coords(datetime=('time', targets_datetime))
        add_derived_vars(forcings)
        forcings = self.batching(forcings, self.target_hours, self.batchsize)

        inputs = inputs.drop_vars(['day_progress', 'year_progress'])
        forcings = forcings.drop_vars(['day_progress', 'year_progress'])
        targets = self.targets_template.assign_coords(datetime=forcings.datetime)

        return inputs, targets, forcings

    def get_chunk(self, idx) -> xr.Dataset:
        return xr.merge([self.upper_cache.claim(idx), 
                         self.surface_cache.claim(idx)], compat='no_conflicts')

class ChunkCache:
    def __init__(self, ds: xr.Dataset, start_idx):
        assert 'time' in list(ds.coords), 'time must be in the dataset\'s coords'
        
        self.ds = ds
        self.chunksize = ds[list(ds.data_vars)[0]].encoding['chunks'][0]
        idxes = np.arange(0, ds.sizes['time'])
        self.idx_chunks = [idxes[i:i + self.chunksize] for i in range(0, len(idxes), self.chunksize)]

        self.current_chunk = start_idx // self.chunksize
        self.cache = []
        for i in range(2): self.claim_chunk(self.current_chunk + i)

    def claim_chunk(self, chunk_idx):
        if len(self.cache) >= 2:
            self.cache.pop(0)
        if chunk_idx < len(self.idx_chunks):
            self.cache.append(self.ds.isel(time=self.idx_chunks[chunk_idx]).compute())
        else:
            logger.debug('No chunks to claim further')

    def claim(self, abs_idx):
        idx = abs_idx - self.idx_chunks[self.current_chunk][0]
        assert idx[0] >= 0, 'no backward-loading'
        assert idx[0] < 2 * self.chunksize, 'no jumping over chunks'
        # assert idx[-1] - idx[0] < self.chunksize, 'the range to claim is too long'
        if idx[0] >= self.chunksize:
            logger.debug('moving to the next chunk')
            self.current_chunk += 1
            self.claim_chunk(self.current_chunk + 1)
            idx -= self.chunksize
        # 不同时间的数据存储在不同 chunk 中
        idx_dict = {}
        for x in idx: idx_dict.setdefault(x // self.chunksize, []).append(x % self.chunksize)
        idx_dict = {k: np.array(v) for k, v in idx_dict.items()}
        return xr.concat([self.cache[k].isel(time=v) for k, v in idx_dict.items()], dim='time')
        
def build_inference_fn(params, state, task_config, sampler_config, noise_config, 
                       noise_encoder_config, denoiser_architecture_config, 
                       diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level):
    """构建包装好并经过标准化处理的 GenCast 预测函数"""
    def construct_wrapped_gencast():
        predictor = gencast.GenCast(
            sampler_config=sampler_config,
            task_config=task_config,
            denoiser_architecture_config=denoiser_architecture_config,
            noise_config=noise_config,
            noise_encoder_config=noise_encoder_config,
        )

        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )

        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=min_by_level,
            var_to_clean='sea_surface_temperature',
        )

        return predictor

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_gencast()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    # 将固定参数绑定到函数，抛出 state
    return drop_state(with_params(jax.jit(run_forward.apply)))

def get_data(queue: Queue, all_idx_range, datelist, loader: Loader):
    for idx in all_idx_range:
        logger.info(f'proceeding {datelist[idx]}') if idx % 24 == 0 else logger.debug(f'proceeding {datelist[idx]}')
        data = loader.update(idx)
        queue.put(data)
    logger.info('all data claimed')
    queue.put((None, None, None))

def write_to_zarr(ds_to_write: xr.Dataset, save_path, chunks: tuple):
    if os.path.exists(save_path):
        ds_to_write.to_zarr(save_path, mode='a', append_dim='time')
    else:
        compressor = BloscCodec(cname='zstd', clevel=5, shuffle='shuffle')
        encoding = {
            **{var: {
                'compressors': [compressor],
                'dtype': 'float32', 
                'chunks': chunks
            } for var in ds_to_write.data_vars}, 
            **{'time': {
                'units': 'hours since 1970-01-01T00:00:00',
                'dtype': 'int64',
                '_FillValue': None,
            }}
        }
        ds_to_write.to_zarr(save_path, mode='w', zarr_format=3, encoding=encoding)

def save_data(queue: Queue, path_prefix, upper_chunks, surface_chunks, rename_dict):
    upper_buffers = {}   
    surface_buffers = {} 

    upper_target_len = upper_chunks[0]     
    surface_target_len = surface_chunks[0] 

    with concurrent.futures.ThreadPoolExecutor(max_workers=2 * len(save_hours)) as executor:
        futures = []
        
        while True:
            ds: xr.Dataset = queue.get()
            if ds is None: 
                break 
            
            if ds.lat.values[0] < ds.lat.values[-1]:
                ds = ds.reindex(lat=list(reversed(ds.lat)))
            if 'level' in list(ds.coords):
                ds = ds.rename({'level': 'lev'})
                if ds.lev.values[0] < ds.lev.values[-1]:
                    ds = ds.reindex(lev=list(reversed(ds.lev)))
            if rename_dict:
                ds = ds.rename(rename_dict)

            for lead in ds['time'].data:
                ds_lead = ds.sel(time=lead, drop=True)
                ds_lead = ds_lead.rename({'batch': 'time'}).assign_coords(time=ds_lead.datetime.data).drop_vars('datetime')
                
                upper_vars = [v for v in ds_lead.data_vars if 'lev' in ds_lead[v].dims]
                surface_vars = [v for v in ds_lead.data_vars if 'lev' not in ds_lead[v].dims]
                
                if lead not in upper_buffers: upper_buffers[lead] = []
                if lead not in surface_buffers: surface_buffers[lead] = []

                if upper_vars:
                    upper_buffers[lead].append(ds_lead[upper_vars])
                    if len(upper_buffers[lead]) == upper_target_len:
                        ds_write = xr.concat(upper_buffers[lead], dim='time').sortby('time')
                        save_path = f'{path_prefix}plev_{lead}.zarr'
                        
                        futures.append(executor.submit(write_to_zarr, ds_write, save_path, upper_chunks))
                        upper_buffers[lead] = [] 
                        logger.info(f'one chunk saved to {save_path}')

                if surface_vars:
                    surface_buffers[lead].append(ds_lead[surface_vars])
                    if len(surface_buffers[lead]) == surface_target_len:
                        ds_write = xr.concat(surface_buffers[lead], dim='time').sortby('time')
                        save_path = f'{path_prefix}slev_{lead}.zarr'
                        
                        futures.append(executor.submit(write_to_zarr, ds_write, save_path, surface_chunks))
                        surface_buffers[lead] = [] 
                        logger.info(f'one chunk saved to {save_path}')

        for lead, datasets in upper_buffers.items():
            if datasets:
                ds_write = xr.concat(datasets, dim='time').sortby('time')
                save_path = f'{path_prefix}plev_{lead}.zarr'
                futures.append(executor.submit(write_to_zarr, ds_write, save_path, upper_chunks))
                logger.warning(f'upper outcomes that do not fill a chunk saved to {save_path}')
        
        for lead, datasets in surface_buffers.items():
            if datasets:
                ds_write = xr.concat(datasets, dim='time').sortby('time')
                save_path = f'{path_prefix}slev_{lead}.zarr'
                futures.append(executor.submit(write_to_zarr, ds_write, save_path, surface_chunks))
                logger.warning(f'surface outcomes that do not fill a chunk saved to {save_path}')

        concurrent.futures.wait(futures)
        
    logger.info('all outcomes saved')

def setup_logging(level=logging.INFO):
    logger = logging.getLogger('graphcast')
    logger.setLevel(level)
    file_handler = logging.FileHandler('./run.log', mode='w')
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def main():
    params, state, task_config, sampler_config, noise_config, noise_encoder_config, denoiser_architecture_config = load_checkpoint(CHECKPOINT_PATH)

    logger.info('loading data...')
    upper_chunks = xr.open_dataset(UPPER_PATH).z.encoding['chunks']
    surface_chunks = xr.open_dataset(SURFACE_PATH).msl.encoding['chunks']
    batchsize = BATCHSIZE
    
    ds, ds_surface, ds_static, diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level = load_data(UPPER_PATH, SURFACE_PATH, STATIC_PATH, STATS_DIR)
    
    loader = Loader(ds, ds_surface, ds_static, 
                    task_config.input_variables, task_config.target_variables, 
                    batchsize, task_config.input_duration, save_hours, 
                    start_idx=start)
    
    logger.info("building inference functions...")
    run_forward_jitted = build_inference_fn(params, state, task_config, sampler_config, 
                                            noise_config, noise_encoder_config, denoiser_architecture_config,
                                            diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level)

    logger.info('initializing I/O threads')

    datelist = ds.datetime.data.astype('datetime64[h]')
    end = len(datelist) + loader.input_hours[0]
    end = 72
    all_idx_range = np.arange(start, end, batchsize)

    logger.info(f'prediction inputs range: {datelist[start]} ~ {datelist[end - 1]}')

    queue_in = Queue(maxsize=1)
    load_thread = threading.Thread(target=get_data, args=(queue_in, all_idx_range, datelist, loader), daemon=True)
    load_thread.start()
    
    queue_out = Queue(maxsize=1)
    save_thread = threading.Thread(target=save_data, args=(queue_out, OUTPUT_DIR, upper_chunks, surface_chunks, rev_dict))
    save_thread.start()
    
    while True:
        inputs, targets, forcings = queue_in.get()
        if inputs is None:
            break
        
        # --- 修改 6: 保持 chunked_prediction，但注意 RNG 种子 ---
        # GenCast 是基于扩散模型的，因此具备概率属性 (Stochastic)。
        # 如果你只跑 1 个 sample，固定 PRNGKey 就像跑 deterministic 模型一样。
        # 如果你想做 ensemble (集合预报)，需要在外部再套一层循环，每次传不同的 PRNGKey。
        predictions = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),  
            inputs=inputs,
            targets_template=targets,
            forcings=forcings,
        )
        queue_out.put(predictions)

    logger.info('all predictions done')
    queue_out.put(None)
    save_thread.join()

if __name__ == "__main__":
    os.chdir('/home/chengzy/graphcast')

    save_hours =  [12, 24]
    start = 0

    logger = setup_logging(logging.DEBUG)
    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    jax.config.update('jax_platform_name', DEVICE)
    jax.config.update('jax_traceback_filtering', 'off')
    jax.config.update('jax_compilation_cache_dir', 'gencast/params/tmp/jax_cache')

    upper_dict = {
        't': 'temperature', 
        'z': 'geopotential', 
        'u': 'u_component_of_wind', 
        'v': 'v_component_of_wind', 
        'w': 'vertical_velocity', 
        'q': 'specific_humidity'
    }
    surface_dict = {
        't2m': '2m_temperature', 
        'msl': 'mean_sea_level_pressure', 
        'u10': '10m_u_component_of_wind', 
        'v10': '10m_v_component_of_wind', 
        'sst': 'sea_surface_temperature'
    }
    rev_dict = {
        value: key for key, value in {**upper_dict, **surface_dict}.items()
    }

    main()