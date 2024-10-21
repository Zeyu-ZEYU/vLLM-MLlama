import dcgm_structs as structs
import pydcgm

dcgm_handle = pydcgm.DcgmHandle(ipAddress="127.0.0.1")  # 可以指定 DCGM 运行的 IP 地址
dcgm_system = dcgm_handle.GetSystem()


gpu_ids = dcgm_system.discovery.GetAllGpuIds()
print(f"Detected GPU IDs: {gpu_ids}")


field_group_name = "dmon_fields"
field_ids = [
    1002,  # GPU 利用率
    1003,  # 功耗
]

field_group = dcgm_system.fields.CreateFieldGroup(field_group_name, field_ids)


group_name = "dmon_group"
group = dcgm_system.groups.CreateGroup(group_name)

# 添加 GPU 到监控组
for gpu_id in gpu_ids:
    group.AddGpu(gpu_id)

# 开始监控这些字段
group.WatchFields(
    field_group, updateFrequency=1000000, maxKeepAge=3600, maxKeepSamples=10
)
