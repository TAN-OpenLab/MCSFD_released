import torch
from Claim_Guided.ClaimH_NET import ClaimH_NET
from Claim_Guided.CLaimH_CCFD import CliamH_CCFD_NET


import argparse
import os
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.manual_seed(6)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='4ferguson',
                        choices=['weibo', '4charliehebdo', '4ferguson','4germanwings-crash','4ottawashooting', '4sydneysiege'],
                        help='The name of dataset')
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs to run')
    parser.add_argument('--start_epoch', type=int, default= 22, help='Continue to train')
    parser.add_argument('--batch_size', type=int, default= 32, help='The size of batch')
    parser.add_argument('--num_worker', type=int, default=4)

    parser.add_argument('--text_embedding', type =tuple, default=(768,200), help=' reduce the dimension of the text vector')
    parser.add_argument('--num_posts', type=int, default=50, help='num of node in X')
    parser.add_argument('--num_heads', type=list, default=[1, 1], help='num_heads')
    parser.add_argument('--h_DUGAT', type=list, default=[200,200], help='Number of D node on hidden layer')
    parser.add_argument('--num_filter', type=int, default= 10, help='Number of D node on dense layer')
    parser.add_argument('--lr', type=tuple, default=5e-3, help='lr')
    parser.add_argument('--dropout', type=float, default=0.2, help='attention dropout in GAT')
    parser.add_argument('--weight_decay', type=float, default=10, help='weight_decay')
    parser.add_argument('--patience', type=int, default=30, help='early stopping')
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--save_dir', type=str, default=r'E:\WDK_workshop\PCL_rumor_detection\result', help='Directory name to save the model')
    parser.add_argument('--load_dir', type=str, default=r'E:\WDK_workshop\PCL_rumor_detection\result', help='Directory name to load the model')
    parser.add_argument('--model_name', type=str, default=r'TextCNN_downsample', help='Directory name to save the GAN')
    parser.add_argument('--data_dir', default=r'E:\WDK_workshop\PCL_rumor_detection\data\4charliehebdo', type=str)
    parser.add_argument('--target_domain', default=r'E:\WDK_workshop\PCL_rumor_detection\data\raw_data\charliehebdo', type=str)
    parser.add_argument('--target_domain_dataset', default=r'charliehebdo', type=str)
    parser.add_argument('--data_eval', default='', type=str)


    args = parser.parse_args()
    print(args)

    torch.set_default_dtype(torch.float32)
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

    model = ClaimH_NET(args, device)
    # model = CliamH_CCFD_NET(args, device)


    if not os.path.exists(os.path.join(args.save_dir, args.model_name)):
        os.mkdir(os.path.join(args.save_dir,  args.model_name))
    if not os.path.exists(os.path.join(args.load_dir, args.model_name, args.dataset)):
        os.mkdir(os.path.join(args.save_dir,args.model_name, args.dataset))

    if os.path.exists(os.path.join(args.load_dir, args.model_name, args.dataset, str(args.start_epoch) + '_model_states.pkl')):
        model_path = os.path.join(args.load_dir, args.model_name, args.dataset)
        start_epoch = model.load(model_path, args.start_epoch)
        start_epoch = start_epoch + 1
    else:
        start_epoch = 0
        print("start from epoch {}".format(start_epoch))

    argsDict = args.__dict__
    with open(os.path.join(args.load_dir, args.model_name, args.dataset, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


    # model.train_epoch(args.data_dir, start_epoch)
    model.test(args.target_domain)
    print(" [*] Training finished!")







