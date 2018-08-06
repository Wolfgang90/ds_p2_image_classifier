import argparse

parser = argparse.ArgumentParser()

parser.add_argument("data_directory", help="set directory to get the data from")
parser.add_argument("--save_dir", help="set directory to save checkpoints", default = "./")
parser.add_argument("--arch", help="choose model architecture", default="densenet121")
parser.add_argument("--learning_rate", help="choose learning rate", default=0.001)
parser.add_argument("--hidden_units", help="choose hidden units", default=512)
parser.add_argument("--epochs", help="choose epochs", default=10)
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu_training', help='set gpu-training to True')


results = parser.parse_args()
print('data_directory     = {!r}'.format(results.data_directory))
print('save_dir     = {!r}'.format(results.save_dir))
print('arch     = {!r}'.format(results.arch))
print('learning_rate     = {!r}'.format(results.learning_rate))
print('hidden_units     = {!r}'.format(results.hidden_units))
print('epochs     = {!r}'.format(results.epochs))
print('gpu_training     = {!r}'.format(results.gpu_training))