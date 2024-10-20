import sys
import threading
import time

import dcgm_agent as dcgm_agent
import dcgm_fields
import dcgm_structs
import dcgmvalue
import pydcgm
import zeyu_utils.net as znet


class DcgmProf:
    def __init__(self):
        self.listener = znet.SocketMsger.tcp_listener("0.0.0.0", 38136)
        self.server_thr = threading.Thread(target=self.server_thread)

    def server_thread(self):
        while True:
            conn, _ = self.listener.accept()
            threading.Thread(target=self.process_conn, args=(conn,)).start()

    def process_conn(self, conn):
        pid = conn.recv()
        while True:
            try:
                dcgm_system.UpdateAllFields(True)
                pidInfo = gpu_group.stats.GetPidInfo(pid)
            except:
                time.sleep(1)
                continue

            ## Display some process statistics (more may be desired)
            print("Process ID      : %d" % pid)
            print("Start time      : %d" % pidInfo.summary.startTime)
            print("End time        : %d" % pidInfo.summary.endTime)
            print("Energy consumed : %d" % pidInfo.summary.energyConsumed)
            print("Max GPU Memory  : %d" % pidInfo.summary.maxGpuMemoryUsed)
            print("Avg. SM util    : %d" % pidInfo.summary.smUtilization.average)
            print("Avg. mem util   : %d" % pidInfo.summary.memoryUtilization.average)
            time.sleep(1)

    def run(self):
        self.server_thr.start()


def convert_value_to_string(value):
    v = dcgmvalue.DcgmValue(value)

    try:
        if v.IsBlank():
            return "N/A"
        else:
            return v.__str__()
    except:
        ## Exception is generally thorwn when int32 is
        ## passed as an input. Use additional methods to fix it
        sys.exc_clear()
        v = dcgmvalue.DcgmValue(0)
        v.SetFromInt32(value)

        if v.IsBlank():
            return "N/A"
        else:
            return v.__str__()


dcgm_handle = pydcgm.DcgmHandle(opMode=dcgm_structs.DCGM_OPERATION_MODE_MANUAL)
dcgm_system = dcgm_handle.GetSystem()
gpu_group = pydcgm.DcgmGroup(
    dcgm_handle, groupName="one_gpu_group", groupType=dcgm_structs.DCGM_GROUP_EMPTY
)
supportedGPUs = dcgm_system.discovery.GetAllSupportedGpuIds()
print(f"supported GPUs: {supportedGPUs}")
for i in range(8):
    gpu_group.AddGpu(supportedGPUs[i])

# field_ids = [dcgm_fields.DCGM_FI_PROF_SM_ACTIVE]
# field_group = dcgm_structs.DcgmFieldGroup("one_gpu_group", field_ids)
dcgm_system.UpdateAllFields(waitForUpdate=True)

## Get the current configuration for the group
config_values = gpu_group.config.Get(dcgm_structs.DCGM_CONFIG_CURRENT_STATE)
groupGpuIds = gpu_group.GetGpuIds()
## Display current configuration for the group
for x in range(0, len(groupGpuIds)):
    print("GPU Id      : %d" % (config_values[x].gpuId))
    print("Ecc  Mode   : %s" % (convert_value_to_string(config_values[x].mEccMode)))
    print(
        "Sync Boost  : %s"
        % (convert_value_to_string(config_values[x].mPerfState.syncBoost))
    )
    print(
        "Mem Clock   : %s"
        % (convert_value_to_string(config_values[x].mPerfState.targetClocks.memClock))
    )
    print(
        "SM  Clock   : %s"
        % (convert_value_to_string(config_values[x].mPerfState.targetClocks.smClock))
    )
    print(
        "Power Limit : %s" % (convert_value_to_string(config_values[x].mPowerLimit.val))
    )
    print("Compute Mode: %s" % (convert_value_to_string(config_values[x].mComputeMode)))
    print("\n")


gpu_group.stats.WatchPidFields(1000000, 2, 0)


DcgmProf().run()


# sampling_interval = 1

# try:
#     while True:
#         field_values = gpu_group.GetLatestValues(field_ids)

#         for gpu_id, field_value in field_values.items():
#             sm_util = field_value[dcgm_fields.DCGM_FI_PROF_SM_ACTIVE].value
#             print(f"GPU {gpu_id}: SM usage: {sm_util}%")

#         time.sleep(sampling_interval)

# finally:
#     dcgm_agent.dcgmShutdown()
