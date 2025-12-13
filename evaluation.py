import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from rag.rag_pipeline import build_resume_kb, answer_query
from rag.vectorstore import embed_text_list, search_faiss

# defining improved smoothing
smooth = SmoothingFunction().method4

# defining Product Manager question set
test_cases = [
    {
        "question": "How many years of experience does the candidate have",
        "gold": "Candidate has 4 years of experience in product management"
    },
    {
        "question": "What tools and skills does the candidate mention",
        "gold": "Product strategy user research agile data analysis SQL Tableau"
    },
    {
        "question": "What is the candidate's highest qualification",
        "gold": "Bachelor of Science in Information Systems"
    },
    {
        "question": "Mention one project the candidate has worked on",
        "gold": "Customer segmentation model to identify high value cohorts"
    }
]

# evaluating rag
def evaluate_rag(resume_text):
    kb = build_resume_kb(resume_text)

    retrieval_hits = 0
    hallucinations = 0

    bleu_scores = []
    rouge1_scores = []
    rougel_scores = []
    similarities = []

    qualitative_errors = []

    for case in test_cases:
        question = case["question"]
        gold = case["gold"].lower()

        # retrieving chunks
        chunks = search_faiss(question, kb["chunks"], kb["index"], k=5)
        chunk_text = " ".join(chunks).lower()

        # checking retrieval hit using gold keywords
        gold_terms = gold.split()
        hit = any(term in chunk_text for term in gold_terms)
        if hit:
            retrieval_hits += 1

        # generating model answer
        model_answer = answer_query(question, kb).lower()

        # computing semantic similarity
        v1 = embed_text_list([model_answer])
        v2 = embed_text_list([gold])
        sim = cosine_similarity(v1, v2)[0][0]
        similarities.append(sim)

        # detecting hallucination
        if not any(term in model_answer for term in gold_terms):
            hallucinations += 1

        # computing bleu
        bleu = sentence_bleu([gold.split()], model_answer.split(), smoothing_function=smooth)
        bleu_scores.append(bleu)

        # computing rouge
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(gold, model_answer)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougel_scores.append(scores["rougeL"].fmeasure)

        # collecting qualitative errors
        if not hit:
            qualitative_errors.append((question, "retrieval error"))
        elif sim < 0.55:
            qualitative_errors.append((question, "generation error"))
        elif len(model_answer.strip()) < 5:
            qualitative_errors.append((question, "incomplete answer"))

    # summarizing metrics
    retrieval_accuracy = retrieval_hits / len(test_cases)
    avg_similarity = sum(similarities) / len(similarities)
    hallucination_rate = hallucinations / len(test_cases)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rougel = sum(rougel_scores) / len(rougel_scores)

    return {
        "retrieval_accuracy": retrieval_accuracy,
        "avg_similarity": avg_similarity,
        "hallucination_rate": hallucination_rate,
        "avg_bleu": avg_bleu,
        "avg_rouge1": avg_rouge1,
        "avg_rougel": avg_rougel,
        "qualitative_errors": qualitative_errors
    }


# running evaluation
if __name__ == "__main__":
    resume_text = open("sample_resume.txt").read()

    results = evaluate_rag(resume_text)

    print("Retrieval Accuracy:", results["retrieval_accuracy"])
    print("Avg Semantic Similarity:", results["avg_similarity"])
    print("Hallucination Rate:", results["hallucination_rate"])
    print("Avg BLEU:", results["avg_bleu"])
    print("Avg ROUGE-1 F1:", results["avg_rouge1"])
    print("Avg ROUGE-L F1:", results["avg_rougel"])
    print("\nQualitative Errors:")
    for q, err in results["qualitative_errors"]:
        print(q, "=>", err)
