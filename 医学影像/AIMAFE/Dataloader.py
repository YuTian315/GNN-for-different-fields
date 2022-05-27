import numpy as np
import os
import csv
phenotype = r"data/Phenotypic_V1_0b_preprocessed1.csv"

def Load_Raw_Data(atlas="aal"):

    subject_IDs = np.genfromtxt(r"data/subject_IDs.txt", dtype=str)
    labels = get_subject_score(subject_IDs, score='DX_GROUP')
    # adj = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)
    num_nodes = len(subject_IDs)
    y = np.zeros([num_nodes])
    for i in range(num_nodes):
        y[i] = int(labels[subject_IDs[i]])
    Label = y - 1
    Raw_Features = np.load(rf"data/{atlas}_871.npy")
    return Raw_Features, Label

# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]
    return scores_dict
