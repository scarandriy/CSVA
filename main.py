# main.py
from core.config       import load_full_config, make_image_processor, make_llm
from core.pipeline     import ScamEvaluatorPipeline
from core.memory       import MemoryBuffer
from core.logger       import EvaluationLogger

def main():
    cfg       = load_full_config()
    prof_key  = cfg["active_profile"]
    profile   = cfg["profiles"][prof_key]

    img_proc = make_image_processor(profile["image_processor"])
    llm      = make_llm(profile["llm"])
    memory   = MemoryBuffer(max_size=10)
    logger   = EvaluationLogger()

    pipeline = ScamEvaluatorPipeline(
        image_processor = img_proc,
        llm             = llm,
        memory_buffer   = memory,
        logger          = logger,
    )

    # test
    image_paths = ["data/scam/0042.png"]
    results     = pipeline.run_batch(image_paths)
    for path, res in zip(image_paths, results):
        print(f"{path} → {res}")

if __name__ == "__main__":
    main()
