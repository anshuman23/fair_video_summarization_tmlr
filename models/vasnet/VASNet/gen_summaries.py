import os
import h5py
import json
import torch
import argparse
import pickle
from vasnet_model import VASNet
from vsum_tools import generate_summary, evaluate_summary

def generate_summaries(model_path, dataset_file, splits_filename, split_id, output_dir):
    # Load the trained model
    model = VASNet()
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()

    # Read the splits file
    with open(splits_filename, 'r') as splits_file:
        splits = json.load(splits_file)

    # Get the specified split
    if split_id < 0 or split_id >= len(splits):
        print(f"Invalid split ID: {split_id}. Exiting.")
        return

    split = splits[split_id]
    # split_name = split['split_name']
    test_keys = split['test_keys']

    # Create a directory for the split if it doesn't exist
    split_dir = os.path.join(output_dir, f'split{split_id}')
    os.makedirs(split_dir, exist_ok=True)

    print(test_keys)

    dataset = h5py.File(dataset_file, 'r')

    all_summaries = []
    fms = []
    # Process each video in the test split
    with torch.no_grad():
        for key in test_keys:
            # Load the video dataset
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            seq = seq.float().cuda()
            # Inference
            y, _ = model(seq, seq.shape[1])
            probs = y[0].detach().cpu().numpy()
            
            # get video info to generate summary
            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...] 

            # Generate the summary using the trained model
            machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)
            all_summaries.append(machine_summary)
            # Eval
            fm, _, _ = evaluate_summary(machine_summary, user_summary, 'avg')
            fms.append(fm)
        # Close the dataset
        dataset.close()
        print('Avg Fscore: ', sum(fms)/len(fms))
        with open(f'{split_dir}/summaries.pkl', 'wb') as file:
            pickle.dump(all_summaries, file)

    print(f"Summaries generated for {split_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summaries for a specific split using a trained model")
    parser.add_argument("--model_path", type=str, help="Path to the trained model file")
    parser.add_argument("--dataset", default="../../datasets/fvs.h5",type=str, help="Path to the h5 dataset")
    parser.add_argument("--splits_filename", type=str, default="data/splits/fvs_splits.json", help="Path to the splits file")
    parser.add_argument("--split", type=int, default=0, help="ID of the split to save summaries for")
    parser.add_argument("--out_dir", type=str, default="data/summaries",help="Directory to save the generated summaries")
    args = parser.parse_args()

    generate_summaries(args.model_path, args.dataset, args.splits_filename, args.split, args.out_dir)