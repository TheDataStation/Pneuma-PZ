import json
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pneuma import Pneuma


def main():
    # Step 1: Initialize Pneuma
    out_path = "demo"
    pneuma = Pneuma(
        out_path=out_path,
        llm_path="Qwen/Qwen2.5-7B-Instruct",
        embed_path="BAAI/bge-base-en-v1.5",
    )
    pneuma.setup()

    # Step 2: Register dataset
    data_path = "data_src/pneuma_chembl_10K"
    response = pneuma.add_tables(path=data_path, creator="pneuma_pz_demo")
    response = json.loads(response)
    print(response)

    # Step 3: Summarize dataset
    response = pneuma.summarize()
    response = json.loads(response)
    print(response)

    # Step 4: Generate index
    response = pneuma.generate_index(index_name="demo_index")
    response = json.loads(response)
    print(response)


if __name__ == "__main__":
    main()
