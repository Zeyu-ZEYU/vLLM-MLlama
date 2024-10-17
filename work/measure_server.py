import logging
import time

import psutil
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
logger = Logger(job_name="GPU_STAT", file_path=f"./logs/gpu_cpu_util_0.log").logger

conn, _ = listener.accept()


recv0 = psutil.net_io_counters(pernic=True)["eth0"].bytes_recv
time0 = time.time()
while True:
    # gpu_query = gpustat.new_query()
    # gpu_dev = gpu_query.gpus[gpu_id]
    # logger.info(gpu_dev.utilization)
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    cpu_percent = psutil.cpu_percent()
    recv1 = psutil.net_io_counters(pernic=True)["eth0"].bytes_recv
    time1 = time.time()
    time_diff = time1 - time0
    bw_in = (recv1 - recv0) / time_diff / 6553600000
    recv0 = recv1
    time0 = time1
    # gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # gpu_mem = gpu_mem_info.used / gpu_mem_info.total
    logger.info(f"{gpu_util} {cpu_percent} {bw_in}")
    time.sleep(0.02)
