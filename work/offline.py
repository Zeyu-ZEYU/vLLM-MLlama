import threading
import time

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser
from vllm.zeyu_utils import net as znet


def run_mllama():
    model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=128,
        max_num_seqs=100,
        enforce_eager=True,
        tensor_parallel_size=8,
        # max_num_batched_tokens=12800
    )

    # prompt = f"<|image|><|begin_of_text|>{question}"
    # stop_token_ids = None
    return llm


def get_multi_modal_input():
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    img_question = "What is the content of this image?"

    return {
        "data": image,
        "question": img_question,
    }


test_started = False


def main():
    global test_started

    listener = znet.SocketMsger.tcp_listener("0.0.0.0", 44478)

    def conn_thread(conn):
        data = conn.recv()
        if data:
            while True:
                data = conn.recv()
                if data is None:
                    conn.close()
                    return
                else:
                    if data == "GET":
                        conn.send(test_started)

    def listener_thread(lstner):
        while True:
            conn, _ = lstner.accept()
            if conn:
                threading.Thread(target=conn_thread, args=(conn,)).start()

    list_thread = threading.Thread(target=listener_thread, args=(listener,))
    list_thread.start()

    llm = run_mllama()
    sampling_params = SamplingParams(temperature=0.2, max_tokens=3, stop_token_ids=None)

    batch_size = 64

    time1 = time.time()
    for warmup in range(1):
        inputs = []
        for _ in range(batch_size):
            mm_input = get_multi_modal_input()
            data = mm_input["data"]
            question = mm_input["question"]
            inputs.append(
                {
                    "prompt": f"<|image|><|begin_of_text|>{question}",
                    "multi_modal_data": {"image": data},
                }
            )
        outputs = llm.generate(inputs, sampling_params=sampling_params)
    warmup_time = (time.time() - time1) * 1000
    print(f"warm-up time: {warmup_time}ms")

    print("==============================================================")

    import vllm.zeyu_utils.os as zos

    torch.multiprocessing.Process(
        target=zos.run_cmd, args=("python3 /home/zeyu/vLLM-MLlama/work/measure_sm.py",)
    ).start()
    time.sleep(1)

    inputs = []
    for _ in range(batch_size):
        mm_input = get_multi_modal_input()
        data = mm_input["data"]
        question = mm_input["question"]
        inputs.append(
            {
                "prompt": f"<|image|><|begin_of_text|>{question}",
                "multi_modal_data": {"image": data},
            }
        )
    test_started = True
    time1 = time.time()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    infer_time = (time.time() - time1) * 1000
    print(f"infer time: {infer_time}ms")

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.

    # llm.start_profile()
    # outputs = llm.generate(inputs, sampling_params=sampling_params)
    # llm.stop_profile()

    # for o in outputs:
    #     generated_text = o.outputs[0].text
    #     print(generated_text)


main()
