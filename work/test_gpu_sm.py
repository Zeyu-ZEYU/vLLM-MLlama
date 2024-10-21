import os
import time

import torch
import zeyu_utils.net as znet

device = torch.device("cuda:0")

print(os.getpid())
# conn = znet.SocketMsger.tcp_connect("127.0.0.1", 38136)

# conn.send(os.getpid())


# 定义张量的大小和批处理大小
tensor_size = (4096, 4096)  # 调整大小以适应GPU内存
batch_size = 64  # 可以根据需求调整


# 创建随机张量
def generate_large_tensor(size, device):
    return torch.randn(size, device=device)


# 运行一次矩阵乘法运算
def stress_test_step(tensor1, tensor2):
    return torch.mm(tensor1, tensor2)


# 压力测试函数
def gpu_stress_test(duration_sec=60):
    start_time = time.time()
    elapsed_time = 0
    num_operations = 0

    print(f"Starting GPU stress test for {duration_sec} seconds...")

    while elapsed_time < duration_sec:
        # 生成大张量
        tensor1 = generate_large_tensor(tensor_size, device)
        tensor2 = generate_large_tensor(tensor_size, device)

        # 运行批量矩阵乘法运算
        for _ in range(batch_size):
            stress_test_step(tensor1, tensor2)
            num_operations += 1

        # 同步GPU，以确保所有计算完成
        torch.cuda.synchronize()

        # 计算时间
        elapsed_time = time.time() - start_time

    print(
        f"Completed {num_operations} matrix multiplications in {elapsed_time:.2f} seconds."
    )


# 运行压力测试（设定测试时长为60秒）
gpu_stress_test(60)
