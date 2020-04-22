mkdir -p results
python -um pygaggle.run.evaluate_kaggle_highlighter --method bm25 > results/bm25-nq.log
python -um pygaggle.run.evaluate_kaggle_highlighter --method t5 > results/t5-nq.log
python -um python -um pygaggle.run.evaluate_kaggle_highlighter --method qa_transformer --model-name ~/models/biobert-squad1 > results/biobert-squadv1-nq.log

python -um pygaggle.run.evaluate_kaggle_highlighter --split kq --method bm25 > results/bm25-kq.log
python -um pygaggle.run.evaluate_kaggle_highlighter --split kq --method t5 > results/t5-kq.log
python -um python -um pygaggle.run.evaluate_kaggle_highlighter --method qa_transformer --model-name ~/models/biobert-squad1 --split kq > results/biobert-squadv1-kq.log

for name in results/*; do echo $name; cat $name; echo; done
