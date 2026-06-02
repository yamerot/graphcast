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

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast.data_utils import add_tisr_var, add_derived_vars

# nohup python run_gr_con.py &
# CUDA_LAUNCH_BLOCKING=1 JAX_DISABLE_CUDA_GRAPH=1 nohup python run_gr_con.py &

# 模型权重文件路径（.npz）
CHECKPOINT_PATH = 'graphcast/params/graphcast_params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz'
UPPER_PATH = '/home/chengzy/dlink/era5_zarr_v3/plev.zarr'
SURFACE_PATH = '/home/chengzy/dlink/era5_zarr_v3/slev.zarr'
STATIC_PATH = '/home/chengzy/dlink/era5_zarr_v3/static.zarr'        # 输入气象数据文件路径
STATS_DIR = "graphcast/stats/graphcast_stats_"              # 标准化统计量路径前缀
OUTPUT_DIR = "graphcast/output/hhhgraphcast_op_"    # 预测结果保存路径前缀
DEVICE = "gpu"                              # 计算设备：可选 "cpu", "gpu", "tpu"
BATCHSIZE = 1


def load_checkpoint(ckpt_path: str):
    """加载模型参数和配置"""
    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    logger.info(ckpt.description)
    # logger.info(ckpt.license)
    logger.info(ckpt.model_config)
    logger.info(f'{ckpt.task_config}\n')
    if hasattr(ckpt, 'state'):
        state = ckpt.state
    else:
        state = {}
    return ckpt.params, state, ckpt.model_config, ckpt.task_config

def load_data(upper_path, surface_path, static_path, stats_prefix):
    """预加载数据集, 应用一些固定的前处理"""
    def preprocess(ds: xr.Dataset, rename_dict):
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
        if 'lev' in list(ds.coords):
            ds = ds.rename({'lev': 'level'})
            if ds.level.values[0] > ds.level.values[-1]:
                ds = ds.reindex(lev=list(reversed(ds.level)))
        # 将变量名替换为长名
        if rename_dict:
            ds = ds.rename(rename_dict)
        return ds

    ds = preprocess(xr.open_dataset(upper_path), upper_dict)
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
                 target_hours: list = [6, 12, 18, 24], 
                 start_idx: int = 0):
        
        sel_vars = [var for var in list(ds.data_vars) if var in input_vars]
        ds_sel = ds[sel_vars]

        sel_vars = [var for var in list(ds_surface.data_vars) if var in input_vars]
        ds_surface_sel = ds_surface[sel_vars]

        sel_vars = [var for var in list(ds_static.data_vars) if var in input_vars]
        self.ds_static = ds_static[sel_vars]

        # 1. 取样的相对时刻窗口, 
        # e.g. input_duration: '12h', batchsize = 2, 计算得到以下值: 
        # input_hours: (-6, 0) -> input_idx: (0, 6) -> input_batch_idx: (0, 1, 6, 7)
        # target_hours: (6, 12) -> target_idx: (12, 18) -> target_batch_idx: (12, 13, 18, 19)
        onehour = pd.Timedelta('1h')
        self.input_hours = np.arange(- pd.Timedelta(input_duration) // onehour, 0, 6) + 6
        input_idx = self.input_hours - self.input_hours[0]
        self.input_batch_idx = np.concatenate([np.arange(x, x + batchsize) for x in input_idx])
        
        self.target_hours = np.array(target_hours)
        target_idx = self.target_hours - self.input_hours[0]
        self.target_batch_idx = np.concatenate([np.arange(x, x + batchsize) for x in target_idx])

        self.batchsize = batchsize

        # 2. 时间连续形式的 forcings 模板
        self.forcings_template = xr.Dataset(
            coords={'time': self.target_batch_idx, 
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
            coords={'time': self.target_hours, 
                    'level': ds.level, 
                    'lat': ds.lat, 
                    'lon': ds.lon}
        )
        
        # 4. 数据暂存区
        self.upper_cache = ChunkCache(ds_sel, start_idx)
        self.surface_cache = ChunkCache(ds_surface_sel, start_idx)
        self.full_datetime = ds_sel.datetime.data

    def batching(self, ds: xr.Dataset, hours, batchsize) -> xr.Dataset:
        """
        ds: 逐小时连续时间块 [time, ...]
        将 ds 重组为 [batch, time, ...] 结构, 其中 time 维度被替换为相对时刻, 在不同真实时刻上所取的输入被划分为不同 batch, 比如 batch[0] <- [T00, T06], batch[1] <- [T01, T07] ...
        """
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
        add_tisr_var(inputs)
        add_derived_vars(inputs)
        inputs = self.batching(inputs, self.input_hours, self.batchsize)
        inputs.update(self.ds_static)
        
        # forcings 和 targets 的时间信息是一致的
        targets_datetime = self.full_datetime[start] + self.target_batch_idx.astype('timedelta64[h]')
        forcings = self.forcings_template.assign_coords(datetime=('time', targets_datetime))
        add_tisr_var(forcings)
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
    """
    graphcast 需要两个时次输入, 因此每个时刻的数据将会被读取两次
    为了节省读取开销, 建立一个暂存区管理数据, 防止重复从硬盘载入
    暂存区一次只在内存存储两个块
    """
    def __init__(self, ds: xr.Dataset, start_idx):
        assert 'time' in list(ds.coords), 'time must be in the dataset\'s coords'
        
        self.ds = ds
        self.chunksize = ds[list(ds.data_vars)[0]].encoding['chunks'][0]
        idxes = np.arange(0, ds.sizes['time'])
        self.idx_chunks = [idxes[i:i + self.chunksize] for i in range(0, len(idxes), self.chunksize)]

        self.current_chunk = start_idx // self.chunksize
        self.cache = []
        self.claim_chunk(self.current_chunk) 
        self.claim_chunk(self.current_chunk + 1)

    def claim_chunk(self, chunk_idx):
        if len(self.cache) >= 2:
            self.cache.pop(0)
        if chunk_idx < len(self.idx_chunks):
            self.cache.append(self.ds.isel(time=self.idx_chunks[chunk_idx]).compute())
        else:
            logger.debug('No chunks to claim further')

    def claim(self, abs_idx):
        idx= abs_idx - self.idx_chunks[self.current_chunk][0]
        assert idx[0] >= 0, 'no backward-loading'
        assert idx[0] < 2 * self.chunksize, 'no jumping over chunks'
        assert idx[-1] - idx[0] < self.chunksize, 'the range to claim is too long'
        if idx[0] >= self.chunksize:
            logger.debug('moving to the next chunk')
            self.current_chunk += 1
            self.claim_chunk(self.current_chunk + 1)
            idx -= self.chunksize
        if idx[-1] < self.chunksize:
            return self.cache[0].isel(time=idx)
        else:
            idx1 = idx[idx < self.chunksize]
            idx2 = idx[idx >= self.chunksize] - self.chunksize
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

def get_data(queue: Queue, all_idx_range, datelist, loader: Loader):
    for idx in all_idx_range:
        logger.info(f'proceeding {datelist[idx]}') if idx % 24 == 0 else logger.debug(f'proceeding {datelist[idx]}')
        data = loader.update(idx)
        queue.put(data)
    logger.info('all data claimed')
    queue.put((None, None, None))

def write_to_zarr(ds_to_write: xr.Dataset, save_path, chunks: tuple):
    """
    独立线程 Worker: 负责将已经攒满一个完整 Chunk 的 Dataset 写入硬盘。
    """
    # 2. 线程安全地执行写入
    if os.path.exists(save_path):
        # 此时追加的数据大小正好等于 Chunk 大小（或其倍数），Zarr 会直接生成新文件块，杜绝 RMW
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
    """
    保存线程：内存攒够完整 Chunk 后，分路径、多线程并行异步写入。
    """
    # 初始化高空和地面的内存缓冲区 (Key 为各个预报时效 lead)
    upper_buffers = {}   
    surface_buffers = {} 

    # 提取期望的 Chunk 长度阈值（高空为 12，地面为 120）
    upper_target_len = upper_chunks[0]     
    surface_target_len = surface_chunks[0] 

    # 创建独立线程池，专门负责后台的磁盘 I/O 写入，不阻塞主预测循环
    with concurrent.futures.ThreadPoolExecutor(max_workers=2 * len(save_hours)) as executor:
        futures = []
        
        while True:
            ds: xr.Dataset = queue.get()
            if ds is None: 
                break # 收到结束信号，跳出循环进入 Flush 阶段
            
            # 1. 统一的公共前处理（每个 Batch 仅执行一次）
            if ds.lat.values[0] < ds.lat.values[-1]:
                ds = ds.reindex(lat=list(reversed(ds.lat)))
            if 'level' in list(ds.coords):
                ds = ds.rename({'level': 'lev'})
                if ds.lev.values[0] < ds.lev.values[-1]:
                    ds = ds.reindex(lev=list(reversed(ds.level)))
            if rename_dict:
                ds = ds.rename(rename_dict)

            # 2. 按预报时效（lead）拆分数据并分类塞入缓冲区
            for lead in ds['time'].data:
                ds_lead = ds.sel(time=lead, drop=True)
                ds_lead = ds_lead.rename({'batch': 'time'}).assign_coords(time=ds_lead.datetime.data).drop_vars('datetime')
                
                # 核心步骤：通过检查是否包含层维 'lev'，将高空变量与地面变量剥离
                upper_vars = [v for v in ds_lead.data_vars if 'lev' in ds_lead[v].dims]
                surface_vars = [v for v in ds_lead.data_vars if 'lev' not in ds_lead[v].dims]
                
                # 初始化特定时效的缓存列表
                if lead not in upper_buffers: upper_buffers[lead] = []
                if lead not in surface_buffers: surface_buffers[lead] = []

                # --- 处理高空数据 ---
                if upper_vars:
                    upper_buffers[lead].append(ds_lead[upper_vars])
                    # 检查是否攒够一个 chunk
                    if len(upper_buffers[lead]) == upper_target_len:
                        ds_write = xr.concat(upper_buffers[lead], dim='time').sortby('time')
                        save_path = f'{path_prefix}plev_{lead}.zarr'
                        
                        # 提交给线程池异步写入
                        futures.append(executor.submit(write_to_zarr, ds_write, save_path, upper_chunks))
                        upper_buffers[lead] = [] # 清空该时效的高空缓存
                        logger.info(f'one chunk saved to {save_path}')

                # --- 处理地面数据 ---
                if surface_vars:
                    surface_buffers[lead].append(ds_lead[surface_vars])
                    # 检查是否攒够一个 chunk
                    if len(surface_buffers[lead]) == surface_target_len:
                        ds_write = xr.concat(surface_buffers[lead], dim='time').sortby('time')
                        save_path = f'{path_prefix}slev_{lead}.zarr'
                        
                        # 提交给线程池异步写入
                        futures.append(executor.submit(write_to_zarr, ds_write, save_path, surface_chunks))
                        surface_buffers[lead] = [] # 清空该时效的地面缓存
                        logger.info(f'one chunk saved to {save_path}')

        # 3. Flush 阶段：当所有预测迭代结束，把缓冲区里“未凑满一整块”的残余数据强制刷入硬盘
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

        # 阻塞等待线程池中最后一批数据彻底落盘，确保进程安全退出
        concurrent.futures.wait(futures)
        
    logger.info('all outcomes saved')

def setup_logging(level=logging.INFO):
    logger = logging.getLogger('graphcast')
    logger.setLevel(level)

    # 文件处理器
    file_handler = logging.FileHandler('./run_gr_con.log', mode='w')
    file_handler.setLevel(level)

    # 格式器：包含时间、级别、消息
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def main():
    # 1. 加载模型
    params, state, model_config, task_config = load_checkpoint(CHECKPOINT_PATH)

    # 2. 加载数据
    logger.info('loading data...')
    upper_chunks = xr.open_dataset(UPPER_PATH).z.encoding['chunks']
    surface_chunks = xr.open_dataset(SURFACE_PATH).msl.encoding['chunks']
    batchsize = BATCHSIZE
    ds, ds_surface, ds_static, diffs_stddev_by_level, mean_by_level, stddev_by_level = load_data(UPPER_PATH, SURFACE_PATH, STATIC_PATH, STATS_DIR)
    loader = Loader(ds, ds_surface, ds_static, 
                    task_config.input_variables, task_config.target_variables, 
                    batchsize, '12h', save_hours, 
                    start_idx=start)
    
    # 3. 构建推理函数
    logger.info("building inference functions...")
    run_forward_jitted = build_inference_fn(params, state, model_config, task_config,
                                     diffs_stddev_by_level, mean_by_level, stddev_by_level)

    # 4. 启动 IO 线程
    logger.info('initializing I/O threads')

    datelist = ds.datetime.data.astype('datetime64[h]')
    end = len(datelist) + loader.input_hours[0]
    all_idx_range = np.arange(start, end, batchsize)

    logger.info(f'prediction inputs range: {datelist[start]} ~ {datelist[end - 1]}')

    queue_in = Queue(maxsize=1)
    load_thread = threading.Thread(target=get_data, args=(queue_in, all_idx_range, datelist, loader), daemon=True)
    load_thread.start()
    queue_out = Queue(maxsize=1)
    save_thread = threading.Thread(target=save_data, args=(queue_out, OUTPUT_DIR, upper_chunks, surface_chunks, rev_dict))
    save_thread.start()
    # 4. 执行预测
    while True:
        inputs, targets, forcings = queue_in.get()
        if inputs is None:
            break
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

    save_hours =  [6, 12, 18, 24]

    start = 34920

    # 日志设置
    logger = setup_logging(logging.DEBUG)
    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    # jax 配置
    jax.config.update('jax_platform_name', DEVICE)
    jax.config.update('jax_traceback_filtering', 'off')
    jax.config.update('jax_compilation_cache_dir', 'graphcast/params/tmp/jax_cache') # 模型缓存路径

    # 将变量 short name 调整为 long name, 保存时再重命名回去
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
        'u10': '10m_v_component_of_wind', 
        'v10': '10m_u_component_of_wind', 
    }
    rev_dict = {
        value: key for key, value in {**upper_dict, **surface_dict}.items()
    }

    main()

