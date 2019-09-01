import argparse
import os

def get_train_args():
    parser = argparse.ArgumentParser("Train a DKT model")

    parser.add_argument('--eval_steps',
                        type=int,
                        default=100,
                        help='Number of steps between successive evaluations')
    parser.add_argument('--lr',
                        type=float,
                        default=0.1,
                        help='Learning rate')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=20,
                        help='Number of epochs to train')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=2.0,
                        help='Max gradient norm for gradient clipping')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='mini-batch size for dataset')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=200,
                        help='Number of features in the lstm hidden layer')
    parser.add_argument('--train_path',
                        type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'data', '0910_a_train.csv'),
                        help='Path for training data')
    parser.add_argument('--eval_path',
                        type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'data', '0910_a_test.csv'),
                        help='Path for training data')
    parser.add_argument('--input_size',
                        type=int,
                        default=124,
                        help='input features size')
    parser.add_argument('--save_path',
                        type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'data', 'model_save.pth.tar'),
                        help='Path for saved checkpoint')
    parser.add_argument('--save_fig',
                        type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'data', 'out_heatmap.png'),
                        help='Path for heatmap')
    parser.add_argument('--save_loss_plot',
                        type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'data', 'out_loss.png'),
                        help='Path for loss plot')
    
    args = parser.parse_args()

    return args
    