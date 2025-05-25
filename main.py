# main.py
from core.config       import load_full_config, make_image_processor, make_llm
from core.pipeline     import ScamEvaluatorPipeline
from core.memory       import MemoryBuffer
from core.logger       import EvaluationLogger
from captions.blip_captioner import FastCaptioner
import time                                   
from captions.vit_gpt2_captioner import ViTGPT2Captioner

import glob, pathlib

def one_time(pipeline):
    
    image_paths = ["captures/1.png", "captures/2.png", "captures/4.png"]

    result    = pipeline.run_multi(image_paths)
    print(f"{result}")

def full_test(pipeline):

    print("SUPPOSED TO BE LEGIT:")
    for i in range(228, 500):
        if i < 10:
            i = "0" + str(i)
        t0 = time.perf_counter()              # start timer
        img_path= "data/legit/00" + str(i) + ".png"
        result = pipeline.run_batch([img_path])
        dt = time.perf_counter() - t0         # seconds elapsed
        print(f"{img_path} – {dt:.2f} s → {result}")

    print("SUPPOSED TO BE SCAM:")
    for i in range(1, 195):
        if i < 10:
            i = "0" + str(i)
        t0 = time.perf_counter()              # start timer
        img_path= "data/better_scam/00" + str(i) + ".png"
        result = pipeline.run_batch([img_path])
        dt = time.perf_counter() - t0         # seconds elapsed
        print(f"{img_path} – {dt:.2f} s → {result}")


def descriptor():
    cap = FastCaptioner(device="cpu",threads=4) 
    vit2 = ViTGPT2Captioner(device="cpu")

    image_paths = [
        "data/legit/0041.png",
        "data/legit/0000.png",
        "data/legit/0042.png",
        "data/scam/0001.png",
    ]

    for p in image_paths:
        print('Blip', pathlib.Path(p).name, "→", cap.caption(p))
        print('ViT-GPT2', pathlib.Path(p).name, "→", vit2.caption(p))
    





def main():
    cfg       = load_full_config()
    prof_key  = cfg["active_profile"]

    profile   = cfg["profiles"][prof_key]

    img_proc  = make_image_processor(profile["image_processor"])
    llm       = make_llm(profile["llm"])

    memory   = MemoryBuffer(max_size=10)
    logger   = EvaluationLogger()

    pipeline = ScamEvaluatorPipeline(
        image_processor = img_proc,
        llm             = llm,
        memory_buffer   = memory,
        logger          = logger,
    )


    # ------test---------
    #one_time(pipeline)
    full_test(pipeline)
    #descriptor()

    






if __name__ == "__main__":
    main()
