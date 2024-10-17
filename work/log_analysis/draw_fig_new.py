#! /usr/bin/env python3


from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

folder = "log"
# folder = "10_strg_large_99%"
# folder = "07_strg_99%"
# folder = "08_strg_99%"
# folder = "14_lambda_strg_iperf_full"
# folder = "11_lambda_wo_strg"
# folder = "13_lambda_strg_500mbps"


LOG_DIR = f"/Users/zeyu/Documents/Seafile/CS Learning/Paper Projects/Video-Related/llama3.2_res_profile/logs"
TIME_INTV = 0.01
START_RATIO = 0
END_RATIO = 1


# set start_time and end_time
with open(f"{LOG_DIR}/gpu_cpu_util_0.log") as f:
    line = f.readline()
    arr = line.split()
    START_TIME = datetime.timestamp(
        datetime.strptime(f"{arr[0]} {arr[1]}", "%Y-%m-%d %H:%M:%S,%f")
    )
    end_line = ""
    line = f.readline()
    while line != "":
        end_line = line
        line = f.readline()
    arr = end_line.split()
    END_TIME = datetime.timestamp(
        datetime.strptime(f"{arr[0]} {arr[1]}", "%Y-%m-%d %H:%M:%S,%f")
    )

time_delta = END_TIME - START_TIME
# START_TIME += 0.8 * time_delta
# END_TIME = START_TIME + 0.5 * time_delta
START_TIME = START_TIME + START_RATIO * time_delta
END_TIME = START_TIME + END_RATIO * time_delta
# END_TIME = START_TIME + 5


def get_time_and_gpu_cpu_usage(gpuid=0):
    times = []
    gpu_util = []
    cpu_usage = []
    bw = []
    with open(f"{LOG_DIR}/gpu_cpu_util_{gpuid}.log") as f:
        line = f.readline()
        while line != "":
            arr = line.rstrip("\n").split()
            tm = datetime.timestamp(
                datetime.strptime(f"{arr[0]} {arr[1]}", "%Y-%m-%d %H:%M:%S,%f")
            )
            if tm >= START_TIME and tm <= END_TIME:
                times.append(tm)
                gpu_util.append(float(arr[-3]))
                cpu_usage.append(float(arr[-2]) * 100 / 96)
                bw.append(float(arr[-1]) * 6553600000 / 1024 / 124)
            line = f.readline()
    return times, gpu_util, cpu_usage, bw


def get_gpu_mem(gpuid=0):
    times = []
    gpu_mem = []
    with open(f"{LOG_DIR}/gpu_mem_{gpuid}.log") as f:
        line = f.readline()
        while line != "":
            arr = line.rstrip("\n").split()
            tm = datetime.timestamp(
                datetime.strptime(f"{arr[0]} {arr[1]}", "%Y-%m-%d %H:%M:%S,%f")
            )
            if tm >= START_TIME and tm <= END_TIME:
                times.append(tm)
                gpu_mem.append(float(arr[-1]))
            line = f.readline()
    return times, gpu_mem


def get_wrk_labels(label, gid=0):
    times = []
    with open(f"{LOG_DIR}/label_time_{gid}.log") as f:
        line = f.readline()
        while line != "":
            arr = line.rstrip("\n").split()
            if label == arr[-1].split("|")[-2]:
                tm = datetime.timestamp(
                    datetime.strptime(f"{arr[0]} {arr[1]}", "%Y-%m-%d %H:%M:%S,%f")
                )
                if tm >= START_TIME and tm <= END_TIME:
                    times.append(tm)
            line = f.readline()
    return times


def get_preprocess_label():
    times = []
    with open(f"{LOG_DIR}/preprocess.log") as f:
        line = f.readline()
        while line != "":
            arr = line.rstrip("\n").split()
            tm = datetime.timestamp(
                datetime.strptime(f"{arr[0]} {arr[1]}", "%Y-%m-%d %H:%M:%S,%f")
            )
            if tm >= START_TIME and tm <= END_TIME:
                times.append(tm)
            line = f.readline()
    return times


def get_mm_data_label():
    times = []
    with open(f"{LOG_DIR}/mm_data_load.log") as f:
        line = f.readline()
        while line != "":
            arr = line.rstrip("\n").split()
            tm = datetime.timestamp(
                datetime.strptime(f"{arr[0]} {arr[1]}", "%Y-%m-%d %H:%M:%S,%f")
            )
            if tm >= START_TIME and tm <= END_TIME:
                times.append(tm)
            line = f.readline()
    return times


def draw_subfig(gid=0):
    subfig = axs

    # plt.plot(times, in_bw, "--", color="red", label=f"in")
    times, gpuutil, cpuusage, bw = get_time_and_gpu_cpu_usage()
    gmem_times, gpu_mem = get_gpu_mem()
    subfig.plot(times, gpuutil, "-", color="red", alpha=1, label=f"GPU until.")
    subfig.plot(gmem_times, gpu_mem, "-", color="blue", alpha=1, label=f"GPU mem.")
    subfig.plot(times, cpuusage, "-", color="k", alpha=1, label=f"GPU usage")
    subfig.plot(times, bw, "-", color="gold", alpha=1, label=f"BW")

    # label = "vision_model"
    # clr = "gold"
    # times = get_wrk_labels(label=label, gid=gid)
    # for tm in times:
    #     subfig.plot([tm, tm], [10, 100], "--", alpha=0.8, color=clr)

    # label = "language_model"
    # clr = "k"
    # times = get_wrk_labels(label=label, gid=gid)
    # for tm in times:
    #     subfig.plot([tm, tm], [10, 100], "--", alpha=0.3, color=clr)

    # clr = "green"
    # times = get_preprocess_label()
    # for tm in times:
    #     subfig.plot([tm, tm], [10, 100], "--", alpha=0.3, color=clr)

    clr = "k"
    times = get_mm_data_label()
    for tm in times:
        subfig.plot([tm, tm], [10, 100], "--", alpha=0.3, color=clr)


plt.rcParams.update({"xtick.labelsize": 17, "ytick.labelsize": 17})
fig, axs = plt.subplots(1, sharex=True, sharey=True, figsize=(25, 4))
fig.subplots_adjust(left=0.12, right=0.92, top=0.9, bottom=0.2)

pids = [686948, 687558, 687559, 687560, 687561, 687562, 687563, 687564]

draw_subfig(0)
# draw_subfig(1)
# draw_subfig(2)
# draw_subfig(3)

plt.xlabel("System time (s)", fontsize=20)
# plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
plt.ylabel("Usage (%)", fontsize=20)
plt.legend(loc="best", fontsize=16)
plt.show()
