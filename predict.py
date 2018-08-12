from utility import argparse_predict, preprocess_image, import_mapping
from model import load_model, predict

results_argparse = argparse_predict()

model = load_model(results_argparse.checkpoint, results_argparse.gpu)

image = preprocess_image(results_argparse.image_path)

if results_argparse.category_names:
    cat_to_name = import_mapping(results_argparse.category_names)
else:
    cat_to_name = None

probs, classes = predict(image, model, results_argparse.top_k, results_argparse.gpu, cat_to_name)

print("Top {} Predictions:".format(results_argparse.top_k))
for i in range(results_argparse.top_k):
    print("{}\t->\t{:6.2%}:\t{}".format(i,probs[i],classes[i]))