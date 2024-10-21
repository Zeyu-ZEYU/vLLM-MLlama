import logging
import subprocess


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


dmon_logger = Logger("DMON", "/home/azureuser/zeyu/data/mllama/logs/sm.log").logger


def run_linux_command():
    # 使用 Popen 来启动命令并捕获输出，传递参数
    process = subprocess.Popen(
        [
            "dcgmi",
            "dmon",
            "-i",
            "0",
            "-e",
            "1002,1003",
            "-d",
            "100",
        ],  # 替换为你的命令和参数
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # 确保输出是文本格式
        bufsize=1,  # 行缓冲
        universal_newlines=True,  # 保证文本解码的正确性
    )

    # 实时读取命令的输出
    for line in process.stdout:
        if line and line[0] == "G":
            process_output(line)

    # 处理完成后，关闭子进程
    process.stdout.close()
    process.wait()


def process_output(output_line):
    # 这是你处理每一行输出的逻辑
    # 你可以根据需要进行自定义处理
    dmon_logger.info(output_line.rstrip())


if __name__ == "__main__":
    run_linux_command()
