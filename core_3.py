import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import h5py
import argparse
import time
import logging
from sklearn import metrics
from utils import utilities, data_generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

try:
    import cPickle
except BaseException:
    import _pickle as cPickle


def move_data_to_gpu(x, cuda, volatile=False):
    x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    x = Variable(x, volatile=volatile)
    return x


def forward_in_batch(model, x, batch_size, cuda):
    model.eval()
    batch_num = int(np.ceil(len(x) / float(batch_size)))
    output_all = []
    cla_all = []
    norm_att_all = []
    mult_all = []
    cla2_all = []
    norm2_att_all = []
    mult2_all  = []
    b2_all  = []
    
    for i1 in range(batch_num):
        batch_x = x[i1 * batch_size: (i1 + 1) * batch_size]
        batch_x = move_data_to_gpu(batch_x, cuda, volatile=True)
        output, cla, norm_att, mult = model(batch_x)
        output_all.append(output)
        #b2_all.append(b2)
        
        cla_all.append(cla)
        norm_att_all.append(norm_att)
        mult_all.append(mult)
        
        #for multy
        #cla2_all.append(cla2)
        #norm2_att_all.append(norm_att2)
        #mult2_all.append(mult2)
        
    output_all = torch.cat(output_all, dim=0)
    single = 1
    if single == 1:
        cla_all = torch.cat(cla_all, dim=0)
        mult_all = torch.cat(mult_all, dim=0)
        norm_att_all = torch.cat(norm_att_all, dim=0)
        return output_all, cla_all, norm_att_all, mult_all
        
    #for multy
    multy = 0
    if multy == 1:
        cla_all = torch.cat(cla_all, dim=0)
        mult_all = torch.cat(mult_all, dim=0)
        norm_att_all = torch.cat(norm_att_all, dim=0)
        cla2_all = torch.cat(cla2_all, dim=0)
        mult2_all = torch.cat(mult2_all, dim=0)
        norm2_att_all = torch.cat(norm2_att_all, dim=0)
        return output_all, cla_all, norm_att_all, mult_all,  cla2_all, norm2_att_all, mult2_all
        
    avg = 0
    if avg == 1:
        b2_all = torch.cat(b2_all, dim=0)
        return output_all, b2_all
        


def evaluate(model, input, target, stats_dir, probs_dir, iteration):
    """Evaluate a model.
    Args:
      model: object
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
      stats_dir: str, directory to write out statistics.
      probs_dir: str, directory to write out output (samples_num, classes_num)
      iteration: int
    Returns:
      None
    """
    # Check if cuda
    cuda = True
    #cuda = next(model.parameters()).is_cuda
    utilities.create_folder(stats_dir)
    utilities.create_folder(probs_dir)

    # Predict presence probabilittarget
    callback_time = time.time()
    (clips_num, time_steps, freq_bins) = input.shape

    (input, target) = utilities.transform_data(input, target)

    output, cla, norm_att, mult = forward_in_batch(model, input, batch_size=350, cuda=cuda)
    
    output = output.data.cpu().numpy()  # (clips_num, classes_num)
    
    single = 1
    if single == 1:
        print("output_all cat: ",output.shape)
        print("cla_all cat: ",cla.shape)
        print("cla_all cat: ",cla)
        print("mult_all cat: ",mult)
        print("norm_att_all cat: ",norm_att)
            
        cla = cla.data.cpu().numpy()
        norm_att = norm_att.data.cpu().numpy()
        mult = mult.data.cpu().numpy()
        
    #for multy
    multy = 0
    if multy == 1:
        cla = cla.data.cpu().numpy()
        norm_att = norm_att.data.cpu().numpy()
        mult = mult.data.cpu().numpy()
        cla2 = cla2.data.cpu().numpy()
        norm_att2 = norm_att2.data.cpu().numpy()
        mult2 = mult2.data.cpu().numpy()
        print("cla_all cat: ",cla)
        print("mult_all cat: ",mult)
        print("norm_att_all cat: ",norm_att)
        print("cla_all cat: ",cla2)
        print("mult_all cat: ",mult2)
        print("norm_att_all cat: ",norm_att2)
        
        
    avg = 0  
    if avg ==1:
        print("output_all cat: ",output.shape)
        print("b2: ",b2.shape)
        b2 = b2.data.cpu().numpy()
        
        
        
    '''
    # Write out presence probabilities
    prob_path = os.path.join(probs_dir, "prob_{}_iters.p".format(iteration))
    cPickle.dump(output, open(prob_path, 'wb'))

    # Calculate statistics
    stats = utilities.calculate_stats(output, target)

    # Write out statistics
    stat_path = os.path.join(stats_dir, "stat_{}_iters.p".format(iteration))
    cPickle.dump(stats, open(stat_path, 'wb'))

    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    logging.info(
        "mAP: {:.6f}, AUC: {:.6f}, Callback time: {:.3f} s".format(
            mAP, mAUC, time.time() - callback_time))

    if False:
        logging.info("Saveing prob to {}".format(prob_path))
        logging.info("Saveing stat to {}".format(stat_path))
        
    '''
    
    #Save
    totest = 0   
    
    if totest == 0:
        #SAVE MODEL
        
        dataset = {}
        dataset['output'] = output
        
        if single == 1:
            dataset['cla'] = cla
            dataset['norm_att'] = norm_att
            dataset['mult'] = mult
            
        if multy == 1:
            dataset['cla'] = cla
            dataset['norm_att'] = norm_att
            dataset['mult'] = mult
            dataset['cla2'] = cla2
            dataset['norm_att2'] = norm_att2
            dataset['mult2'] = mult2
        
        if avg == 1:
            dataset['b2'] = b2
            
            
        path = r'C:\Users\AdexGomez\Downloads\Master\third_semester\Deep_Learning\Project\02456_project_audioset_attention\data'
        file_name = '\multyt.h5'
        file = path + file_name
        print(file)
        f = h5py.File(file,'w')
        for k in dataset.keys():
            print('\n  '+k+' type=', type(dataset[k][0]))
            if k != 'event_label' and k != 'filename': # @@@@temporal, need to be fixed!!!
                print('    ... saving '+k+'...')
                f.create_dataset(k, data=dataset[k])

def test(args):
    
    data_dir = args.data_dir
    workspace = args.workspace
    #mini_data = args.mini_data
    balance_type = args.balance_type
    #learning_rate = args.learning_rate
    filename = args.filename
    model_type = args.model_type
    model = args.model
    #batch_size = args.batch_size
    
    
    # Test data
    test_hdf5_path = os.path.join(data_dir, "eval1.h5")
    (test_x, test_y, test_id_list) = utilities.load_data(test_hdf5_path)
    
    # Output directories
    sub_dir = os.path.join(filename,
                           'balance_type={}'.format(balance_type),
                           'model_type={}'.format(model_type))

    models_dir = os.path.join(workspace, "models", sub_dir)
    utilities.create_folder(models_dir)

    stats_dir = os.path.join(workspace, "stats", sub_dir)
    utilities.create_folder(stats_dir)

    probs_dir = os.path.join(workspace, "probs", sub_dir)
    utilities.create_folder(probs_dir)
    
    iteration = 1200
    
    # Optimization method
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3,
                           betas=(0.9, 0.999),
                           eps=1e-07)
    
    
    #Loading model..

    
    PATH = os.path.join(models_dir, "md_{}_iters_300batchsize.tar".format(iteration))
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    logging.info("Training data shape: {}".format(test_x.shape))
    logging.info("Training data shape: {}".format(test_y.shape))
    
    logging.info("Test statistics:")
    evaluate(model=model,
             input=test_x,
             target=test_y,
             stats_dir=os.path.join(stats_dir, "test"),
             probs_dir=os.path.join(probs_dir, "test"),
             iteration=iteration)
    
    
    
    print('ready')



def train(args):
    """Train a model.
    """

    data_dir = args.data_dir
    workspace = args.workspace
    mini_data = args.mini_data
    balance_type = args.balance_type
    learning_rate = args.learning_rate
    filename = args.filename
    model_type = args.model_type
    model = args.model
    batch_size = args.batch_size
    #cuda = True


    # Move model to gpu
    if torch.cuda.is_available():
        model.cuda()

    # Path of hdf5 data
    bal_train_hdf5_path = os.path.join(data_dir, "bal_train.h5")
    #unbal_train_hdf5_path = os.path.join(data_dir, "unbal_train.h5")
    test_hdf5_path = os.path.join(data_dir, "eval.h5")

    # Load data
    load_time = time.time()

    if mini_data:
        # Only load balanced data
        (bal_train_x, bal_train_y, bal_train_id_list) = utilities.load_data(
            bal_train_hdf5_path)

        train_x = bal_train_x
        train_y = bal_train_y
        train_id_list = bal_train_id_list
    
      
    else:
        # Load both balanced and unbalanced data
        print("warning")
        print(71)
        (bal_train_x, bal_train_y, bal_train_id_list) = utilities.load_data(
            bal_train_hdf5_path)

        (unbal_train_x, unbal_train_y, unbal_train_id_list) = utilities.load_data(
            unbal_train_hdf5_path)

        train_x = np.concatenate((bal_train_x, unbal_train_x))
        train_y = np.concatenate((bal_train_y, unbal_train_y))
        train_id_list = bal_train_id_list + unbal_train_id_list

    # Test data
    (test_x, test_y, test_id_list) = utilities.load_data(test_hdf5_path)
    


    logging.info("Loading data time: {:.3f} s".format(time.time() - load_time))
    logging.info("Training data shape: {}".format(train_x.shape))
    logging.info("Training data shape: {}".format(train_y.shape))
    logging.info("Training data shape: {}".format(test_x.shape))
    logging.info("Training data shape: {}".format(test_y.shape))
    
    time.sleep(10)

    # Optimization method
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3,
                           betas=(0.9, 0.999),
                           eps=1e-07)

    # Output directories
    sub_dir = os.path.join(filename,
                           'balance_type={}'.format(balance_type),
                           'model_type={}'.format(model_type))

    models_dir = os.path.join(workspace, "models", sub_dir)
    utilities.create_folder(models_dir)

    stats_dir = os.path.join(workspace, "stats", sub_dir)
    utilities.create_folder(stats_dir)

    probs_dir = os.path.join(workspace, "probs", sub_dir)
    utilities.create_folder(probs_dir)

    # Data generator
    if balance_type == 'no_balance':
        DataGenerator = data_generator.VanillaDataGenerator

    elif balance_type == 'balance_in_batch':
        DataGenerator = data_generator.BalancedDataGenerator

    else:
        raise Exception("Incorrect balance_type!")

    train_gen = DataGenerator(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        shuffle=True,
        seed=1234)

    iteration = 0
    call_freq = 200
    train_time = time.time()

    for (batch_x, batch_y) in train_gen.generate():

        # Compute stats every several interations
        if iteration % call_freq == 0 and iteration > 1:

            logging.info("------------------")

            logging.info(
                "Iteration: {}, train time: {:.3f} s".format(
                    iteration, time.time() - train_time))

            logging.info("Balance train statistics:")
            evaluate(
                model=model,
                input=bal_train_x,
                target=bal_train_y,
                stats_dir=os.path.join(stats_dir, 'bal_train'),
                probs_dir=os.path.join(probs_dir, 'bal_train'),
                iteration=iteration)

            logging.info("Test statistics:")
            evaluate(
                model=model,
                input=test_x,
                target=test_y,
                stats_dir=os.path.join(stats_dir, "test"),
                probs_dir=os.path.join(probs_dir, "test"),
                iteration=iteration)

            train_time = time.time()

        (batch_x, batch_y) = utilities.transform_data(batch_x, batch_y)
        cuda = True
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        # Forward.
        model.train()
        output, cla, nomt_att, mult = model(batch_x)
        # Loss.
        
        loss = F.binary_cross_entropy(output, batch_y)

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1

        # Save model.
        if iteration % 200 == 0:
            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(), }
            save_out_path = os.path.join(
                models_dir, "md_{}_iters_300batchsize.tar".format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info("Save model to {}".format(save_out_path))

        # Stop training when maximum iteration achieves
        if iteration == 2601:
            
            break