mkdir -p results
python -um pygaggle.run.evaluate_kaggle_highlighter --method bm25 > results/bm25-nq.log
python -um pygaggle.run.evaluate_kaggle_highlighter --method t5 > results/t5-nq.log
python -um pygaggle.run.evaluate_kaggle_highlighter --method transformer --model-name biobert > results/biobert-nq.log
python -um pygaggle.run.evaluate_kaggle_highlighter --method transformer --model-name allenai/scibert_scivocab_cased > results/scibert-nq.log
python -um pygaggle.run.evaluate_kaggle_highlighter --method transformer --model-name bert-base-cased > results/bert-nq.log

python -um pygaggle.run.evaluate_kaggle_highlighter --split kq --method bm25 > results/bm25-kq.log
python -um pygaggle.run.evaluate_kaggle_highlighter --split kq --method t5 > results/t5-kq.log
python -um pygaggle.run.evaluate_kaggle_highlighter --split kq --method transformer --model-name biobert > results/biobert-kq.log
python -um pygaggle.run.evaluate_kaggle_highlighter --split kq --method transformer --model-name allenai/scibert_scivocab_cased > results/scibert-kq.log
python -um pygaggle.run.evaluate_kaggle_highlighter --split kq --method transformer --model-name bert-base-cased > results/bert-kq.log

for name in results/*; do echo $name; cat $name; echo; done
