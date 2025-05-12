import pandas as pd
import numpy as np
import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

sequences_file_path = "uniprot_sprot.fasta"
interpro_file_path = "uniprotkb_reviewed_true_2024_06_08.tsv"


def readFasta(file_path):
    names = []
    sequences = []
    sequence = ""
    with open(file_path) as file:
        for line in file.readlines():
            if line[0] == ">":
                names.append(line.split("|")[1])
                sequences.append(sequence)
                sequence = ""
            else:
                sequence += line[:-1]
        sequences.append(sequence)
        sequences.pop(0)
    return [names, sequences]


# Contient en data[0] les noms et en data[1] les séquences
data = readFasta(sequences_file_path)

# Contient dans la colone "Entry" les noms et dans la colone "InterPro" les codes Interpro
functions = pd.read_csv(filepath_or_buffer=interpro_file_path, sep="\t")
functions["InterPro"] = functions["InterPro"].str[:-1].str.split(pat=";")
functions.loc[functions["InterPro"].isnull(), "InterPro"] = functions.loc[
    functions["InterPro"].isnull(), "InterPro"
].apply(lambda x: [np.nan])


# On transforme la colone de liste en une seule longue liste et on compte les occurences
All_InterPro_codes, counts = np.unique(
    functions["InterPro"].explode().tolist(), return_counts=True
)

# On retire le compte des NaN
nan_i = np.where(All_InterPro_codes == "nan")
All_InterPro_codes = np.delete(All_InterPro_codes, nan_i)
counts = np.delete(counts, nan_i)

# On récupère les indexes des 100 plus hauts comptes et on récupère les codes InterPro associés
values, indexes = torch.topk(torch.tensor(counts), 100)
InterPro = All_InterPro_codes[indexes]


functions["Chosen"] = functions["InterPro"].apply(
    lambda code_list: any(code in InterPro for code in code_list)
)
interpro_sequences = pd.DataFrame(
    list(zip(data[0], data[1])), columns=["Names", "Sequences"]
)
interpro_sequences = interpro_sequences.sort_values("Names", ignore_index=True)
interpro_sequences = interpro_sequences[functions["Chosen"] == True]
interpro_functions = functions[functions["Chosen"] == True]
interpro_sequences["InterPro"] = interpro_functions["InterPro"]
interpro_sequences


# Générer des 3-mers à partir d'une séquence de protéine
def generate_3mers(sequence):
    n = len(sequence)
    return [sequence[i : i + 3] for i in range(n - 2)]


# Générer les 3-mers pour toutes les séquences
all_3mers = [generate_3mers(seq) for seq in interpro_sequences["Sequences"]]
print(all_3mers[0])


# Charger le modèle ProtVec (512)
protvec_model = Word2Vec.load("protvec_512.model")


# Encoder une séquence de protéine -> obtenjir des représentations vectorielles des séquences
def encode_sequence(sequence, model):
    kmers = generate_3mers(sequence)
    vectors = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


# Encoder toutes les séquences dans le DataFrame
interpro_sequences["Encoded"] = interpro_sequences["Sequences"].apply(
    lambda seq: encode_sequence(seq, protvec_model)
)

interpro_sequences


# Features X et les labels y
X = np.array(interpro_sequences["Encoded"].tolist())
y = np.array(interpro_sequences["InterPro"].apply(lambda codes: codes[0]).tolist())

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Charger le modèle SVM pré-entraîné (512)
svm_model = joblib.load("svm_model_512.joblib")
print("Modèle SVM (512) chargé avec succès.")

# Prédictions sur les données de test
y_pred = svm_model.predict(X_test)
# Enregistrer les prédictions dans un fichier .csv
predictions_df = pd.DataFrame(y_pred, columns=["Predictions"])
predictions_df.to_csv("predictions_512.csv", index=False)
print(
    "Les données de prédictions ont été enregistrées dans le fichier 'predictions_512.csv'."
)

# Évaluation des performances du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
