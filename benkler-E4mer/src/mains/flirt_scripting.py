import os
import argparse
from src.data import load
from src.STATIC import ROOT_DIR

def load_and_process_unlabelled(batch_size):
    unlabelled_dir = os.path.join(ROOT_DIR, 'e4data/unlabelled_data/raw')
    unread_files_path = os.path.join(ROOT_DIR, 'e4data/flirt_processed/unlabelled/unread_files.txt')
    save_filepath = os.path.join(ROOT_DIR, 'e4data/flirt_processed/unlabelled/unlabelled_flirt.csv')

    unread_files = []
    with open(unread_files_path, 'r') as f:
        for file in f.readlines():
            unread_files.append(file)
    f.close()

    unreadable_files = load.batch_load_unlabelled_e4data(unlabelled_dir, subset=unread_files, 
                                                         batch_size = batch_size,
                                                         save_filepath = save_filepath)
    with open(unread_files_path, 'w') as f:
        for file in unreadable_files:
            f.write(f"{file}\n")

def load_and_process_Nurses(batch_size):
    nurse_dir = os.path.join(ROOT_DIR, '../../Summer 2024/DS-288_Capstone/Data:Code Sources/Stressed Nurses/Stress_dataset')
    unread_files_path = os.path.join(ROOT_DIR, 'e4data/flirt_processed/Nurses/unread_files.txt')
    save_filepath = os.path.join(ROOT_DIR, 'e4data/flirt_processed/Nurses/Nurses_flirt.csv')

    # unread_files = []
    # with open(unread_files_path, 'r') as f:
    #     for file in f.readlines():
    #         unread_files.append(file)
    # f.close()

    unreadable_files = load.batch_load_nurses_e4data(nurse_dir,
                                                     batch_size = batch_size,
                                                     save_filepath = save_filepath)
    with open(unread_files_path, 'w') as f:
        for file in unreadable_files:
            f.write(f"{file}\n")

def load_and_process_WESAD(batch_size):
    WESAD_dir = os.path.join(ROOT_DIR, '../../Summer 2024/DS-288_Capstone/Data:Code Sources/WESAD')
    unread_files_path = os.path.join(ROOT_DIR, 'e4data/flirt_processed/WESAD/unread_files.txt')
    save_filepath = os.path.join(ROOT_DIR, 'e4data/flirt_processed/WESAD/WESAD_flirt.csv')

    # unread_files = []
    # with open(unread_files_path, 'r') as f:
    #     for file in f.readlines():
    #         unread_files.append(file)
    # f.close()

    unreadable_files = load.batch_load_WESAD_e4data(WESAD_dir,
                                                    batch_size,
                                                     save_filepath = save_filepath)
    with open(unread_files_path, 'w') as f:
        for file in unreadable_files:
            f.write(f"{file}\n")

def main():
    parser = argparse.ArgumentParser(description="Process E4 data for different datasets.")
    parser.add_argument('dataset', choices=['unlabelled', 'Nurses', 'WESAD'], help="Specify which dataset to process.")
    parser.add_argument('--batch_size', type=int, default=20, help="Batch size for processing files incrementally.")

    args = parser.parse_args()
    
    if args.dataset == 'unlabelled':
        print("Processing Unlabelled dataset...")
        load_and_process_unlabelled(batch_size=args.batch_size)
    elif args.dataset == 'Nurses':
        print("Processing Nurses dataset...")
        load_and_process_Nurses(batch_size=args.batch_size)
    elif args.dataset == 'WESAD':
        print("Processing WESAD dataset...")
        load_and_process_WESAD(batch_size=args.batch_size)    

if __name__ == "__main__":
    main()