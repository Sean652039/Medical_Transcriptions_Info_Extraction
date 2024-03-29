import pandas as pd

from ExtractAge import ExtractAge
from ExtractTreatment import ExtractTreatment

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program.')

    # Add arguments here using add_argument() method
    parser.add_argument('-n', '--num_topics', type=int, default=5, help='Number of topics. Default is 5.')
    parser.add_argument('-f', '--file', type=str, help='Specify a file.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # read data
    samples_add = args.file if args.file else 'mtsamples.csv'
    samples = pd.read_csv(samples_add)

    # use '' to fill the empty column
    samples.fillna({'transcription': ''}, inplace=True)

    # extraction of age
    extract_age = ExtractAge(samples)
    extract_age.train_model()

    # extraction of treatment
    extract_treatment = ExtractTreatment(samples, args.num_topics)
    extract_treatment.train_model()



