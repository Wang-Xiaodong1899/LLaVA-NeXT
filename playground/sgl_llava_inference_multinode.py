import argparse
import json
import time
import os
import tqdm
import sglang as sgl
from sglang.test.test_utils import select_sglang_backend
from sglang.utils import dump_state_text

@sgl.function
def image_description(s, image_file):
    prompt = "Please generate detailed descriptions of the given image."
    s += sgl.user(sgl.image(image_file) + prompt)
    s += sgl.assistant(sgl.gen("answer", max_tokens=args.max_tokens, temperature=0.0))

def load_progress(progress_file):
    print(f"Load progress from {progress_file}")
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return json.load(f)
    return None

def save_progress(progress_file, progress_data):
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2)

def find_images_in_subfolders(folder_path):
    image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp")
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def main(args):
    dist_rank = args.dist
    dist_size = args.total_dist

    progress_file = f"./progress_{dist_rank}.json"
    progress_data = load_progress(progress_file) or {"last_index": -1, "last_chunk": -1, "results": [], "annotations": []}

    image_files = find_images_in_subfolders(args.image_folder)
    image_files = image_files[: args.limit] if args.limit > 0 else image_files

    # Shard the data
    shard_size = len(image_files) // dist_size
    start_index = shard_size * dist_rank
    end_index = start_index + shard_size if dist_rank < dist_size - 1 else len(image_files)
    shard_files = image_files[start_index:end_index]

    print(f"Querying {len(shard_files)} images from index {start_index} to {end_index - 1}")

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    tic = time.time()
    batch_size = 16
    for batch_start in tqdm.tqdm(range(0, len(shard_files), batch_size)):
        batch_end = min(batch_start + batch_size, len(shard_files))
        batch_arguments = [{"image_file": image_file} for image_file in shard_files[batch_start:batch_end]]
        # Run batch
        batch_states = image_description.run_batch(
            batch_arguments,
            temperature=0,
            num_threads=args.parallel,
            progress_bar=False  # Assuming tqdm is used for the outer loop, disable inner progress bar
        )
        
        # Process and save batch results
        for i, ret in enumerate(batch_states):
            image_file = batch_arguments[i]["image_file"]
            caption = ret.text().split("ASSISTANT:")[-1].strip()
            progress_data["annotations"].append({"image_file": image_file, "caption": caption})
            progress_data["last_index"] = start_index + batch_start + i
        
        # Save progress after each batch
        print(f"Saving progress at index {start_index + batch_start + len(batch_states) - 1}")
        save_progress(progress_file, progress_data)

    latency = time.time() - tic
    print(f"Latency: {latency:.3f}")

    value = {
        "task": "image_captioning",
        "backend": args.backend,
        "num_gpus": 1,
        "latency": round(latency, 3),
        "num_requests": len(shard_files),
        "parallel": args.parallel,
        "results": progress_data["annotations"],
    }
    
    result_file = args.result_file.replace(".json", f"_shard_{dist_rank}.json")
    print(f"Write output to {result_file}")
    with open(result_file, "w") as fout:
        json.dump(value, fout, indent=2)

    save_progress(progress_file, progress_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="/mnt/bn/vl-research/data/byte_caption/labelled_images")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--backend", type=str, default="srt")
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--result_file", type=str, default="./sgl_llava_inference.json")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--dist", type=int, default=0, help="The rank of the distributed machine")
    parser.add_argument("--total_dist", type=int, default=1, help="Total number of distributed machines")
    args = parser.parse_args()
    main(args)