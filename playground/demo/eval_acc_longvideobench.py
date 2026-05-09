import json

def calculate_accuracy(jsonl_file_path):
    correct = 0
    total = 0

    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                pred = data.get("pred", "")
                gt = data.get("gt", "")
                if gt in pred:
                    correct += 1
                total += 1
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    return accuracy

jsonl_file_path = r"C:\Users\wangxiaodong\Downloads\answer-longvideobench-llava-next-video-f16.jsonl"
calculate_accuracy(jsonl_file_path)
