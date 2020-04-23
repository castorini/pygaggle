mkdir -p results
for split in kq nq; do
  python -um pygaggle.run.evaluate_kaggle_highlighter --split $split --method random > results/random-$split.log;
  python -um pygaggle.run.evaluate_kaggle_highlighter --split $split --method bm25 > results/bm25-$split.log;
  python -um pygaggle.run.evaluate_kaggle_highlighter --split $split --method t5 > results/t5-$split.log;
  python -um pygaggle.run.evaluate_kaggle_highlighter --split $split --method transformer --model-name bert-base-cased > results/bbc-unsup-$split.log;
  python -um pygaggle.run.evaluate_kaggle_highlighter --split $split --method transformer --model-name biobert > results/biobert-unsup-$split.log;
  python -um pygaggle.run.evaluate_kaggle_highlighter --split $split --method transformer --model-name allenai/scibert_scivocab_cased > results/scibert-unsup-$split.log;
  python -um pygaggle.run.evaluate_kaggle_highlighter --split $split --method seq_class_transformer --model-name ~/models/biobert-msmarco > results/biobert-marco-$split.log;
  python -um pygaggle.run.evaluate_kaggle_highlighter --split $split --method seq_class_transformer --model-name ~/models/bbu-marco --do-lower-case > results/bert-marco-$split.log;
  python -um pygaggle.run.evaluate_kaggle_highlighter --split $split --method qa_transformer --model-name ~/models/biobert-squad1 > results/biobert-squadv1-$split.log;
done
for name in results/*; do echo $name; cat $name; echo; done
