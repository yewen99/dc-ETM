import os
import argparse
from trainer_dcetm_policy import DeepCoupling_Policy_trainer
from trainer_dcetm import DeepCoupling_trainer

import utils
from mydataset import *
from clustering import _best_cluster

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

parser = argparse.ArgumentParser()
# -------------------------------------------------------------------------------------------------------------------
# device
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--gpu_idx', type=int, default=0, help='which gpu to use if any (default: 0)')

# task
parser.add_argument('--ppl', type=bool, default=False, help='whether ppl')
parser.add_argument('--classification', type=bool, default=False, help='whether cls')
parser.add_argument('--clustering', type=bool, default=True, help='whether clustering')

# mode
parser.add_argument('--train', type=bool, default=True, help="whether pretrain.")
parser.add_argument('--resume', type=bool, default=False, help='whether resume training')
parser.add_argument('--if_debug', type=bool, default=False, help="whether pretrain.")

parser.add_argument('--saw_trainer', type=bool, default=False, help='whether use rl training method.')
parser.add_argument('--partial_trainer', type=bool, default=False, help='whether use rl training method.')
parser.add_argument('--use_policy', type=bool, default=False, help='whether use rl training method.')
parser.add_argument('--use_beta', type=bool, default=True, help='whether use rl training method.')

# path
parser.add_argument('--dataset_dir', type=str, default='./dataset/20ng.pkl', help='type of dataset.')
parser.add_argument('--load_path', type=str, default='', help='load model from ...')
parser.add_argument('--save_path', type=str, default=f'./20ng_cluster_results', help='where to save results.')
parser.add_argument('--word-vector-path', type=str, default='../process_data/20ng_word_embedding.pkl', help='type of dataset.')

# model
parser.add_argument('--topic_size', type=list, default=[256, 128, 64, 32, 16], help='Number of units in hidden layer 1.')
parser.add_argument('--vocab_size', type=int, default=2000, help='Number of vocabulary')
parser.add_argument('--batch_size', type=int, default=200, help="models used.")
parser.add_argument('--hidden_size', type=int, default=256, help='Number of units in hidden layer 1.')
parser.add_argument('--embed_size', type=int, default=100, help='Number of units in hidden layer 1.')

# optimizer
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--eval_epoch_num', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--test_epoch_num', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--save_epoch_num', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--discount', type=float, default=0.98, help="the discount for the next step's reward")
parser.add_argument('--kl_weight', type=float, default=1.0, help="the discount for the next step's reward")
# -------------------------------------------------------------------------------------------------------------------

args = parser.parse_args()
args.device = torch.device("cuda:" + str(args.gpu_idx)) if torch.cuda.is_available() else torch.device("cpu")

# save path
if args.use_policy:
    args.save_path = os.path.join(args.save_path, f'rl_beta_{args.use_beta}_patial_{args.partial_trainer}')
    args.save_path = os.path.join(args.save_path, f'{len(args.topic_size)}_layers')
    args.save_path = os.path.join(args.save_path, 'seed_'+str(args.seed))
    # args.save_path = os.path.join(args.save_path, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    args.save_path = os.path.join(args.save_path, f'kl_weight_{args.kl_weight}_discount_{args.discount}')
    print(f'save_path:{args.save_path}')
else:
    args.save_path = os.path.join(args.save_path, f'no_rl_beta_{args.use_beta}_saw_{args.saw_trainer}')
    args.save_path = os.path.join(args.save_path, f'{len(args.topic_size)}_layers')
    args.save_path = os.path.join(args.save_path, 'seed_'+str(args.seed))
    # args.save_path = os.path.join(args.save_path, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    args.save_path = os.path.join(args.save_path, f'kl_weight_{args.kl_weight}_discount_{args.discount}')
    print(f'save_path:{args.save_path}')

utils.chk_mkdir(args.save_path)
utils.chk_mkdir(os.path.join(args.save_path, "model"))
utils.chk_mkdir(os.path.join(args.save_path, "theta"))

# set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.ppl:
    train_loader, vocab_size, voc = get_loader_txt_ppl(args.dataset_dir, batch_size=args.batch_size, voc_size=args.vocab_size)
    args.task = 'ppl'
elif args.clustering:
    test_loader, vocab_size, voc = get_test_loader_txt(args.dataset_dir, batch_size=args.batch_size, voc_size=args.vocab_size)
    args.task = 'clustering'

args.vocab_size = vocab_size
print(vocab_size)

if args.ppl:
    if args.use_policy:
        trainer = DeepCoupling_Policy_trainer(args, voc_path=voc)
        trainer.train(train_loader)
    else:
        trainer = DeepCoupling_trainer(args, voc_path=voc)
        trainer.train(train_loader)

elif args.clustering:
    if args.use_policy:
        trainer = DeepCoupling_Policy_trainer(args, voc_path=voc)
        _best_cluster(args, trainer, test_loader)
    else:
        trainer = DeepCoupling_trainer(args, voc_path=voc)
        _best_cluster(args, trainer, test_loader)