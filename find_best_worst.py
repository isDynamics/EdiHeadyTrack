import json
import heapq
from collections import defaultdict

# Path to the predictions.jsonl file
file_path = "../../../Data/predictions.jsonl"

def find_best_and_worst_entries(file_path):
    """
    Finds the top 5 entries with the highest and lowest mean geodesic error in a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        tuple: Two lists containing the top 5 entries with the lowest and highest mean geodesic error.
    """
    lowest_entries = []
    highest_entries = []

    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            if 'geodesic_error' in entry:
                mean_error = sum(entry['geodesic_error']) / len(entry['geodesic_error'])

                # Maintain a heap of the top 5 lowest entries
                if len(lowest_entries) < 5:
                    heapq.heappush(lowest_entries, (mean_error, entry))
                else:
                    heapq.heappushpop(lowest_entries, (mean_error, entry))

                # Maintain a heap of the top 5 highest entries
                if len(highest_entries) < 5:
                    heapq.heappush(highest_entries, (-mean_error, entry))
                else:
                    heapq.heappushpop(highest_entries, (-mean_error, entry))

    # Extract entries from heaps and sort them
    lowest_entries = [entry for _, entry in sorted(lowest_entries)]
    highest_entries = [entry for _, entry in sorted(highest_entries, reverse=True)]

    return lowest_entries, highest_entries

def calculate_average_geodesic_error(file_path):
    """
    Calculates the average geodesic error for each Impact ID across all data sources.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        dict: A dictionary where keys are Impact IDs and values are the average geodesic errors.
    """
    impact_id_errors = defaultdict(list)

    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            if 'geodesic_error' in entry and 'Impact ID' in entry:
                mean_error = sum(entry['geodesic_error']) / len(entry['geodesic_error'])
                impact_id_errors[entry['Impact ID']].append(mean_error)

    # Calculate the average geodesic error for each Impact ID
    average_errors = {
        impact_id: sum(errors) / len(errors)
        for impact_id, errors in impact_id_errors.items()
    }

    return average_errors

if __name__ == "__main__":
    lowest, highest = find_best_and_worst_entries(file_path)

    print("Top 5 entries with the lowest mean geodesic error:")
    for entry in lowest:
        mean_error = sum(entry['geodesic_error']) / len(entry['geodesic_error'])
        print(f"Impact ID: {entry['Impact ID']}, Data source: {entry['Data source']}, Mean geodesic error: {mean_error:.4f}")

    print("\nTop 5 entries with the highest mean geodesic error:")
    for entry in highest:
        mean_error = sum(entry['geodesic_error']) / len(entry['geodesic_error'])
        print(f"Impact ID: {entry['Impact ID']}, Data source: {entry['Data source']}, Mean geodesic error: {mean_error:.4f}")

    average_errors = calculate_average_geodesic_error(file_path)

    print("\nAverage geodesic error for each Impact ID:")
    for impact_id, avg_error in sorted(average_errors.items(), key=lambda x: x[1]):
        print(f"Impact ID: {impact_id}, Average geodesic error: {avg_error:.4f}")