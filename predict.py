import argparse

parser = argparse.ArgumentParser()

parser.add_argument("image_path", help="set path to image for prediction")
parser.add_argument("checkpoint", help="set checkpoint to load the model from")
parser.add_argument("--top_k", help="return top K most likely classes", default=5)
parser.add_argument("--category_names", help="use a mapping of categories to real names", default="cat_to_name.json")
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu_training', help='set gpu-training to True')

results = parser.parse_args()
print('image_path     = {!r}'.format(results.image_path))
print('checkpoint     = {!r}'.format(results.checkpoint))
print('top_k     = {!r}'.format(results.top_k))
print('category_names     = {!r}'.format(results.category_names))
print('gpu_training     = {!r}'.format(results.gpu_training))