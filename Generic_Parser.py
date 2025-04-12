import csv
import glob
import os
import re
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import Load_MasterDictionary as LM

TARGET_FILES = r'D:/data/**/*.txt'
MASTER_DICT_PATH = r'./LoughranMcDonald_MasterDictionary_2014.csv'
HARVARD_NEG_PATH = r'./Harvard_IV_Negative_Word_List_Inf.txt'
OUTPUT_FILE = r'./final_results.csv'

OUTPUT_FIELDS = [
    'filename', 'file_size', 'num_words', 'pct_positive', 'pct_negative',
    'pct_uncertainty', 'pct_litigious', 'pct_modal_weak', 'pct_modal_moderate',
    'pct_modal_strong', 'pct_constraining', 'num_alphanum', 'num_digits',
    'num_numbers', 'avg_syllables', 'avg_word_length', 'vocabulary', 'CIK',
    'H4N-INF_pct', 'FIN-NEG_pct', 'TF-IDF', 'Term_Weight'
]


class SharedData:
    def __init__(self):
        self.lm_dict = LM.load_masterdictionary(MASTER_DICT_PATH)

        self.fin_neg_words = [word for word in self.lm_dict if self.lm_dict[word].negative]
        self.fin_neg_idx = {word: idx for idx, word in enumerate(self.fin_neg_words)}
        self.fin_neg_set = set(self.fin_neg_words)

        with open(HARVARD_NEG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            self.harvard_neg = {line.strip().upper() for line in f}

        self.token_pattern = re.compile(r'\b[A-Za-z]{2,}\b')
        self.may_pattern = re.compile(r'\b(MAY|May)\b', flags=re.IGNORECASE)
        self.number_pattern = re.compile(r'\b\d+\b')


def init_worker(shared):
    global g
    g = shared


def process_file(args):
    try:
        doc_idx, filename = args
        basename = os.path.basename(filename).rstrip('.txt')
        parts = basename.split('_')
        if len(parts) < 5:
            return None

        raw_cik = parts[4].split('-')[0]
        cik = raw_cik.zfill(10)

        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            doc = g.may_pattern.sub(' ', f.read()).upper()

        tokens = g.token_pattern.findall(doc)
        valid_words = len(tokens)
        if valid_words == 0:
            return None

        stats = {
            'positive': 0, 'negative': 0, 'uncertainty': 0,
            'litigious': 0, 'constraining': 0,
            'modal_weak': 0, 'modal_moderate': 0, 'modal_strong': 0,
            'syllables': 0, 'word_length': 0, 'vocab': set(),
            'h4n_inf': 0, 'fin_neg': 0,
            'numbers': len(g.number_pattern.findall(doc))
        }

        tf_row = np.zeros(len(g.fin_neg_words), dtype=np.float32)

        for token in tokens:
            if token in g.lm_dict:
                entry = g.lm_dict[token]
                stats['positive'] += 1 if entry.positive else 0
                stats['negative'] += 1 if entry.negative else 0
                stats['uncertainty'] += 1 if entry.uncertainty else 0
                stats['litigious'] += 1 if entry.litigious else 0
                stats['constraining'] += 1 if entry.constraining else 0
                stats['syllables'] += entry.syllables
                stats['vocab'].add(token)

                stats['modal_weak'] += 1 if entry.weak_modal else 0
                stats['modal_moderate'] += 1 if entry.moderate_modal else 0
                stats['modal_strong'] += 1 if entry.strong_modal else 0

                if token in g.fin_neg_idx:
                    tf_row[g.fin_neg_idx[token]] += 1

            stats['h4n_inf'] += 1 if token in g.harvard_neg else 0

        def safe_divide(a, b):
            return a / b if b != 0 else 0.0

        output_row = [
            os.path.basename(filename),
            os.path.getsize(filename),
            valid_words,

            safe_divide(stats['positive'], valid_words) * 100,
            safe_divide(stats['negative'], valid_words) * 100,
            safe_divide(stats['uncertainty'], valid_words) * 100,
            safe_divide(stats['litigious'], valid_words) * 100,
            safe_divide(stats['modal_weak'], valid_words) * 100,
            safe_divide(stats['modal_moderate'], valid_words) * 100,
            safe_divide(stats['modal_strong'], valid_words) * 100,
            safe_divide(stats['constraining'], valid_words) * 100,

            len(re.findall(r'[A-Za-z0-9]', doc)),  # num_alphanum
            len(re.findall(r'\d', doc)),  # num_digits
            stats['numbers'],  # num_numbers

            safe_divide(stats['syllables'], valid_words),  # avg_syllables
            safe_divide(sum(len(t) for t in tokens), valid_words),  # avg_word_length
            len(stats['vocab']),  # vocabulary

            cik,

            safe_divide(stats['h4n_inf'], valid_words) * 100,
            safe_divide(stats['negative'], valid_words) * 100,  # FIN-NEG_pct

            0.0,  # TF-IDF
            0.0  # Term_Weight
        ]

        return (doc_idx, output_row, tf_row, valid_words)

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None


def main():
    shared = SharedData()

    file_list = glob.glob(TARGET_FILES, recursive=True)
    num_docs = len(file_list)

    tf_matrix = np.zeros((num_docs, len(shared.fin_neg_words)), dtype=np.float32)
    doc_lengths = np.zeros(num_docs, dtype=np.uint32)

    with Manager() as manager:
        with Pool(
                processes=max(1, cpu_count() - 1),
                initializer=init_worker,
                initargs=(shared,)
        ) as pool:
            results = list(tqdm(
                pool.imap(process_file, enumerate(file_list)),
                total=num_docs,
                desc="Processing",
                unit="file"
            ))

        output_data = []
        valid_count = 0
        for result in results:
            if not result:
                continue
            doc_idx, row, tf_row, words = result
            output_data.append(row)
            tf_matrix[doc_idx] = tf_row
            doc_lengths[doc_idx] = words
            valid_count += 1

        print(f"Processed : {valid_count}/{num_docs}")

        # TF-IDF calculation
        df = np.sum(tf_matrix > 0, axis=0) + 1e-6
        idf = np.log(num_docs / df)
        tf = tf_matrix / doc_lengths[:, np.newaxis]
        tfidf = tf * idf
        term_weight = tfidf.sum(axis=1)

        for i, row in enumerate(output_data):
            row[-2] = np.mean(tfidf[i])
            row[-1] = term_weight[i]  # Term_Weight

        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(OUTPUT_FIELDS)
            writer.writerows(output_data)

        print(f"Output {OUTPUT_FILE}")


if __name__ == '__main__':
    main()