import json
import jsonlines
import os
def merge_json_lists(file1, file2, file3, output_file):
    input_files = [file1, file2, file3]
    with jsonlines.open(output_file, mode='a') as writer:
        # Iterate over the input files
        for file in input_files:
            # Open each input file
            with jsonlines.open(file) as reader:
                # Iterate over the lines in the input file
                for line in reader:
                    # Write each line to the output file
                    writer.write(line)

    print("Merging completed!")

root = "/home/pirenjie/data/refcoco/shikra_mask/anns"
merge_json_lists(os.path.join(root, "refcoco", "train_mask.jsonl"), os.path.join(root, "refcocog", "train_mask.jsonl"), os.path.join(root, "refcoco+", "train_mask.jsonl"), os.path.join(root, "train_mask_merged.jsonl"))