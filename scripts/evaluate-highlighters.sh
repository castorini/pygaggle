mkdir -p results
python -um pygaggle.run.evaluate_kaggle_highlighter --method bm25 > results/bm25.log
python -um pygaggle.run.evaluate_kaggle_highlighter --method t5 > results/t5.log
python -um pygaggle.run.evaluate_kaggle_highlighter --method transformer --model-name biobert > results/biobert.log
python -um pygaggle.run.evaluate_kaggle_highlighter --method transformer --model-name allenai/scibert_scivocab_cased > results/scibert.log
python -um pygaggle.run.evaluate_kaggle_highlighter --method transformer --model-name bert-base-cased > results/bert.log
for name in results/*; do echo $name; cat $name; echo; done
