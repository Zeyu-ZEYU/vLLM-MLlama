from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser


def run_mllama(question: str, modality: str):
    assert modality == "image"

    model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=131072,
        max_num_seqs=100,
        enforce_eager=True,
        tensor_parallel_size=8,
    )

    prompt = f"<|image|><|begin_of_text|>{question}"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


def get_multi_modal_input():
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    img_question = "What is the content of this image?"

    return {
        "data": image,
        "question": img_question,
    }


def main():
    modality = "image"
    mm_input = get_multi_modal_input()
    data = mm_input["data"]
    question = mm_input["question"]

    llm, prompt, stop_token_ids = run_mllama(question, modality)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(
        temperature=0.2, max_tokens=64, stop_token_ids=stop_token_ids
    )

    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {modality: data},
        }
        for _ in range(2)
    ]

    # llm.start_profile()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    # llm.stop_profile()

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


main()
