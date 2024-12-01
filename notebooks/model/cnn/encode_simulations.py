import argparse
import time
import random
from typing import List
from tqdm import tqdm

from chinese_checkers.simulation import S3SimulationCatalog, SimulationMetadata, GameSimulation
from chinese_checkers.archive.cnn import CnnEncoderExperience
from chinese_checkers.archive.experience import S3ExperienceCatalog


def main(simulation_name: str):
    # Initialize simulation catalog and experience catalog
    sim_catalog = S3SimulationCatalog()
    exp_catalog = S3ExperienceCatalog()

    # List all datasets and filter by simulation name
    sim_metadata: List[SimulationMetadata] = sim_catalog.list_datasets()
    simulations: List[GameSimulation] = [
        dataset
        for metadata in sim_metadata
        if metadata.name == simulation_name
            and metadata.winning_player in ["0", "3"]
        for dataset in sim_catalog.load_dataset(metadata)
    ]

    # Initialize experience encoder with the specific version
    exp_encoder = CnnEncoderExperience("v006")

    # Shuffle simulations randomly
    random.shuffle(simulations)

    total_experiences = 0
    with tqdm(simulations, desc="Generating experiences") as pbar:
        for simulation in pbar:
            # Start timing the encoding process
            encode_start_time = time.time()
            experiences = exp_encoder.encode(simulation)
            encode_end_time = time.time()
            encode_time = encode_end_time - encode_start_time

            # Start timing the upload process
            upload_start_time = time.time()
            exp_catalog.add_record_list(experiences)
            upload_end_time = time.time()
            upload_time = upload_end_time - upload_start_time

            total_experiences += len(experiences)

            # Update progress bar description with timing information
            pbar.set_description(
                f"Generating experiences (Last generated: {len(experiences)}, "
                f"Total {total_experiences}, Encode time: {encode_time:.2f}s, Upload time: {upload_time:.2f}s)"
            )

    print(f"Completed encoding and uploading experiences. Total experiences: {total_experiences}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode and upload experiences for a given simulation.")
    parser.add_argument("simulation_name", type=str, help="Name of the simulation to encode")
    args = parser.parse_args()
    main(args.simulation_name)
