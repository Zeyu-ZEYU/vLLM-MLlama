import logging
import time

import pynvml

from vllm.zeyu_utils import net as znet


class Logger(object):
    def __init__(self, job_name, file_path, log_level=logging.INFO, mode="w"):
        self.__logger = logging.getLogger(job_name)
        self.__logger.setLevel(log_level)
        self.__fh = logging.FileHandler(filename=file_path, mode=mode)
        self.__formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        self.__fh.setFormatter(self.__formatter)
        self.__logger.addHandler(self.__fh)

    @property
    def logger(self):
        return self.__logger


pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(int(0))

listener = znet.SocketMsger.tcp_listener("0.0.0.0", 55555)
logger = Logger(job_name="GPU_STAT", file_path=f"./gpu_stat.log").logger

conn, _ = listener.accept()


while True:
    # gpu_query = gpustat.new_query()
    # gpu_dev = gpu_query.gpus[gpu_id]
    # logger.info(gpu_dev.utilization)
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_mem = gpu_mem_info.used / gpu_mem_info.total
    logger.info(f"{gpu_util} {gpu_mem:.3f}")
    time.sleep(0.02)
