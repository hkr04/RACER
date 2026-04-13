import json
import argparse
from collections import defaultdict
from tabulate import tabulate

mt_bench_list = ["writing", "roleplay", "reasoning", "math" , "coding", "extraction", "stem", "humanities"]

def analyze_jsonl(file_path):
    total_tokens = 0
    total_time_ms = 0.0
    all_accept_lengths = []
    
    category_tokens = defaultdict(int)
    category_times_ms = defaultdict(float)
    category_accept_lengths = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            try:
                data = json.loads(line)
                for choice in data.get("choices", []):
                    new_tokens = choice.get("new_tokens", [])
                    wall_times = choice.get("wall_time", [])
                    accept_lengths = choice.get("accept_lengths", [])
                    category = data.get("category", "unknown")
                    
                    category_tokens[category] += sum(new_tokens)
                    category_times_ms[category] += sum(wall_times) * 1000
                    category_accept_lengths[category].extend(accept_lengths)

                    total_tokens += sum(new_tokens)
                    total_time_ms += sum(wall_times) * 1000
                    all_accept_lengths.extend(accept_lengths)
            except Exception as e:
                print(f"[ERROR] line {line_num}: {e}")

    avg_ms_per_token = total_time_ms / total_tokens if total_tokens > 0 else 0
    avg_accept_length = (
        sum(all_accept_lengths) / len(all_accept_lengths)
        if all_accept_lengths else 0
    )
    
    category_avg_ms_per_token = {
        k: v / category_tokens[k] if category_tokens[k] > 0 else 0
        for k, v in category_times_ms.items()
    }
    category_avg_accept_length = {
        k: sum(category_accept_lengths[k]) / len(category_accept_lengths[k])
        if category_accept_lengths[k] else 0
        for k in category_accept_lengths
    }
    
    mt_total_tokens = 0
    mt_total_time_ms = 0.0
    mt_accept_lengths = []
    for category in mt_bench_list:
        mt_total_tokens += category_tokens[category]
        mt_total_time_ms += category_times_ms[category]
        mt_accept_lengths.extend(category_accept_lengths[category])
        
    mt_avg_ms_per_token = (
        mt_total_time_ms / mt_total_tokens if mt_total_tokens > 0 else 0
    )
    mt_avg_accept_length = (
        sum(mt_accept_lengths) / len(mt_accept_lengths) if mt_accept_lengths else 0
    )
    
    category_avg_ms_per_token["multi_turn"] = mt_avg_ms_per_token
    category_avg_accept_length["multi_turn"] = mt_avg_accept_length

    return {
        "total_tokens": total_tokens,
        "total_time_ms": total_time_ms,
        "avg_ms_per_token": avg_ms_per_token,
        "avg_accept_length": avg_accept_length,
        "category_avg_ms_per_token": category_avg_ms_per_token,
        "category_avg_accept_length": category_avg_accept_length
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file_path",
        type=str,
        help="The file path of evaluated Speculative Decoding methods.",
    )
    parser.add_argument(
        "--base-path",
        default=None,
        type=str,
        help="The file path of evaluated baseline.",
    )
    parser.add_argument(
        "--bench-name",
        default="spec_bench",
        type=str,
        help="The name of the benchmark to analyze. Options: mt_bench, spec_bench",
    )
    
    args = parser.parse_args()
    
    result = analyze_jsonl(args.file_path)
    
    table = []
    
    if args.base_path:
        base_result = analyze_jsonl(args.base_path)
    else:
        base_result = None
    
    for category in result["category_avg_ms_per_token"].keys():
        if args.bench_name != "spec_bench" and category == "multi_turn":
            continue
        if args.bench_name == "spec_bench" and category in mt_bench_list:
            continue
        if base_result:
            table.append([
                category,
                f"{result['category_avg_ms_per_token'][category]:.2f}",
                f"{result['category_avg_accept_length'][category]:.2f}",
                f"{base_result['category_avg_ms_per_token'][category]/result['category_avg_ms_per_token'][category]:.2f}",
            ])
        else:
            table.append([
                category,
                f"{result['category_avg_ms_per_token'][category]:.2f}",
                f"{result['category_avg_accept_length'][category]:.2f}"
            ])
            
    if base_result:
        table.append([
            "Overall",
            f"{result['avg_ms_per_token']:.2f}",
            f"{result['avg_accept_length']:.2f}",
            f"{base_result['avg_ms_per_token']/result['avg_ms_per_token']:.2f}"
        ])
    else: 
        table.append([
            "Overall",
            f"{result['avg_ms_per_token']:.2f}",
            f"{result['avg_accept_length']:.2f}"
        ])
    
    if base_result:
        print(tabulate(table, headers=["Category", "Mean Speed (ms/token)", "Mean Accepted Tokens", "Speedup"], tablefmt="pretty"))
    else:
        print(tabulate(table, headers=["Category", "Mean Speed (ms/token)", "Mean Accepted Tokens"], tablefmt="pretty"))
    
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Total time (ms): {result['total_time_ms']}")
    print(f"MAT: {result['avg_accept_length']:.2f}")
    if base_result:
        print(f"Speedup: {base_result['avg_ms_per_token']/result['avg_ms_per_token']:.2f}")
    print(f"Throughput (token/s): {result['total_tokens'] * 1000 / result['total_time_ms']:.1f}")