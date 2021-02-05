import numpy as np
import os, time, subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import librosa
from helpers import *
from MIMII import *
import torch
import config
import shutil
from torchsummary.torchsummary import summary
from torchTools import *
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from models import *
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib as mpl
import matplotlib
from sklearn.svm import OneClassSVM


def train_neural_embeddings(f_train, f_val, f_test, epochs=100, center_loss=True,
                            rotate_train=False, rotate_test=False, length_s=2., channels=None, embedding_dimension=25,
                            batch_size=128, lr=1e-3, lr_center=0.01, min_delta=1e-3):
    snrs = [6, 0]
    # channels = [0]
    dataset_n = dataset(length_s=length_s, labels=['normal'], target=['type', 'id'], snrs=snrs, channels=channels)

    idx_all = dataset_n.get_all_ids()
    idx_train, idx_val, idx_test = [], [], []
    f_all = dataset_n.get_filepaths(idx_all)
    for i, f in tqdm(enumerate(f_all), desc='Identifying audio samples'):
        if f in f_train:
            idx_train.append(i)
        elif f in f_val:
            idx_val.append(i)
        elif f in f_test:
            idx_test.append(i)
        else:
            raise ValueError('ID does not belong anywhere. What a pitty, a pitty, a pitty...')

    #assert not any([i in idx_val for i in idx_train])
    #assert not any([i in idx_train for i in idx_val])
    #assert not any([i in idx_test for i in idx_train])
    #assert not any([i in idx_train for i in idx_test])

    trainloader = DataLoader(dataset_n, batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=True,
                             sampler=SubsetRandomSampler(idx_train))
    valloader = DataLoader(dataset_n, batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=False,
                           sampler=SequentialSampler(idx_val))

    if not os.path.exists('./images/'): create_folder('images')

    input_shape, output_shape = (len(dataset_n.channels), int(length_s*fs)), len(dataset_n.get_classes())
    model = RawdNet(input_shape, output_shape, embedding_dimension)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)
        summary(model, torch.rand((1, *input_shape)))
    else:
        raise ValueError('No cuda device!')

    if center_loss:
        nllloss = nn.NLLLoss().cuda()  # CrossEntropyLoss = log_softmax + NLLLoss
        centerloss = CenterLoss(len(dataset_n.get_classes()), embedding_dimension, 1.0).cuda()
        criterion = [nllloss, centerloss]

        optimizer4nn = torch.optim.Adam(model.parameters(), lr=lr)
        optimzer4center = torch.optim.SGD(centerloss.parameters(), lr=lr_center)
        optimizer = [optimizer4nn, optimzer4center]
    else:
        criterion = torch.nn.NLLLoss().cuda() # torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    early_stopping = EarlyStopping(patience=7, delta=min_delta, path=os.path.join(FOLDER, 'model_checkpoint.pt'), verbose=False)

    print('Training...')
    for epoch in range(1, epochs+1):
        t0 = time.time()
        model.train()
        # shuffle dataset and organize the indexes into batches
        running_loss, running_acc = 0., 0.
        ip1s, idxs = [], []

        for i, (x_train_batch, y_train_batch) in tqdm(enumerate(trainloader), desc="Epoch %d" % epoch):
            # rotate microphone array
            if rotate_train:
                x_train_batch = np.roll(x_train_batch, np.random.randint(x_train_batch.shape[1]), axis=1)

            x_train_batch = torch.Tensor(x_train_batch).type(torch.FloatTensor).cuda()
            y_train_batch = torch.Tensor(y_train_batch).type(torch.long).cuda()

            if center_loss:
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
            else:
                optimizer.zero_grad()

            ip1, y_hat = model(x_train_batch)
            if center_loss:
                loss = criterion[0](y_hat, torch.argmax(y_train_batch, 1)) + criterion[1](torch.argmax(y_train_batch, 1), ip1)
            else:
                loss = criterion(y_hat, torch.argmax(y_train_batch, 1))
            ip1s.append(ip1)
            idxs.append((torch.argmax(y_train_batch, 1)))
            running_acc += torch_accuracy(y_hat, y_train_batch)
            running_loss += loss.item()
            loss.backward()
            if center_loss:
                optimizer[0].step()
                optimizer[1].step()
            else:
                optimizer.step()
        train_loss = running_loss / (i + 1)
        train_acc = running_acc / (i + 1)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if embedding_dimension == 2:
            feat = torch.cat(ip1s, 0)
            labels = torch.cat(idxs, 0)
            visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), dataset_n.get_classes(), epoch)

        # validation
        with torch.no_grad():
            model.eval()
            running_loss, running_acc = 0., 0.
            for i, (x_val_batch, y_val_batch) in enumerate(valloader):

                if rotate_test:
                    x_val_batch = np.roll(x_val_batch, np.random.randint(x_val_batch.shape[1]), axis=1)

                x_val_batch = torch.Tensor(x_val_batch).type(torch.FloatTensor).cuda()
                y_val_batch = torch.Tensor(y_val_batch).type(torch.long).cuda()
                #print(i, x_val_batch.shape, y_val_batch.shape)
                ip1, y_hat = model(x_val_batch)
                if center_loss:
                    loss = criterion[0](y_hat, torch.argmax(y_val_batch, 1)) + criterion[1](torch.argmax(y_val_batch, 1), ip1)
                else:
                    loss = criterion(y_hat, torch.argmax(y_val_batch, 1))
                running_acc += torch_accuracy(y_hat, y_val_batch)
                running_loss += loss.item()
            valid_loss = running_loss / (i + 1)
            valid_acc = running_acc / (i + 1)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

        tab = ' ' if epoch < 10 else ''
        print('%ds - epoch: %s%d/%d - loss: %.4f - acc: %.3f - val_loss: %.4f - val_acc: %.3f' % (int(time.time() - t0), tab, epoch, epochs,
                                                                      train_loss, train_acc, valid_loss, valid_acc)),
 

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping!")
            break

    model = torch.load(FOLDER + '\\model_checkpoint.pt')
    torch.save(model, FOLDER + '\\model.pt')

    if True:
        from matplotlib.ticker import MaxNLocator
        ax = plt.figure().gca()
        plt.plot(range(1,len(train_losses)+1), train_losses, 'b', label='Training loss')
        plt.plot(range(1,len(valid_losses)+1), valid_losses, 'g', label='Validation loss')
        plt.axvline(early_stopping.best_epoch, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.xlim(0, len(train_losses) + 1)  # consistent scale
        plt.grid(True)
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel = str(criterion)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()
        plt.savefig(FOLDER + '/train_history.png', dpi=600)  # , bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()
        subprocess.Popen('C:\\Users\\VRYSIS\\Dropbox\\condition_monitoring\\' + FOLDER + '/train_history.png', shell=True,
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

    testloader = DataLoader(dataset_n, batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=False,
                           sampler=SequentialSampler(idx_val))

    ### evaluate
    model.eval()
    running_loss, running_acc = 0., 0.
    with torch.no_grad():
        for i, (x_val_batch, y_val_batch) in enumerate(testloader):
            x_val_batch = torch.Tensor(x_val_batch).type(torch.FloatTensor).cuda()
            y_val_batch = torch.Tensor(y_val_batch).type(torch.long).cuda()
            ip1, y_hat = model(x_val_batch)
            if center_loss:
                loss = criterion[0](y_hat, torch.argmax(y_val_batch, 1)) + criterion[1](torch.argmax(y_val_batch, 1), ip1)
            else:
                loss = criterion(y_hat, torch.argmax(y_val_batch, 1))
            running_acc += torch_accuracy(y_hat, y_val_batch)
            running_loss += loss.item()
        test_loss = running_loss / (i + 1)
        test_acc = running_acc / (i + 1)
        print('test_loss: %.3f - test_acc: %.3f ' % (test_loss, test_acc))

    return model



def predictNeuralEmbeddings(model, batch_size, f_test, length_s=1., type='fan', id=4, snr=None, channels=None, epoch=None):
    assert snr in [6, 0, -6]
    if epoch == None: epoch = ''
    # get loaders per id
    dataset_n = dataset(length_s=length_s, labels=['normal'], ids=[id], types=[type], snrs=[snr], target=['label'], channels=channels)
    dataset_ab = dataset(length_s=length_s, labels=['abnormal'], ids=[id], types=[type], snrs=[snr], target=['label'], channels=channels)

    # remove test set sample indices and clear
    idx_all = dataset_n.get_all_ids()

    idx_test, idx_train = [], []
    f_all = dataset_n.get_filepaths(idx_all)
    for i, f in enumerate(f_all):
        if f not in f_test:
            idx_train.append(i)
        else:
            idx_test.append(i)

    print('------------------------------------')

    assert not any([i in idx_test for i in idx_train])
    assert not any([i in idx_train for i in idx_test])

    # construct loaders
    trainloader = DataLoader(dataset_n, batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=False,
                             sampler=SequentialSampler(idx_train))
    testloader = DataLoader(dataset_n, batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=False,
                             sampler=SequentialSampler(idx_test))
    abtestloader = DataLoader(dataset_ab, batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=False,shuffle=False)

    model.eval()
    x_train, y_train = [], []
    with torch.no_grad():
        for i, (x_train_batch, y_train_batch) in enumerate(trainloader):
            x_train_batch = torch.Tensor(x_train_batch).type(torch.FloatTensor).cuda()
            y_train_batch = torch.Tensor(y_train_batch).type(torch.long).cuda()
            ip1, y_hat = model(x_train_batch)
            x_train.extend(ip1.detach().cpu().numpy().tolist())
            y_train.extend(y_train_batch.detach().cpu().numpy().tolist())

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_test_n, y_test_n = [], []
        for i, (x_test_batch, y_test_batch) in enumerate(testloader):
            x_test_batch = torch.Tensor(x_test_batch).type(torch.FloatTensor).cuda()
            y_test_batch = torch.Tensor(y_test_batch).type(torch.long).cuda()
            ip1, _ = model(x_test_batch)
            x_test_n.extend(ip1.detach().cpu().numpy().tolist())
            y_test_n.extend(y_test_batch.detach().cpu().numpy().tolist())
        x_test_n = np.array(x_test_n)
        y_test_n = np.array(y_test_n)

        x_test_ab, y_test_ab = [], []
        for i, (x_test_batch, y_test_batch) in enumerate(abtestloader):
            x_test_batch = torch.Tensor(x_test_batch).type(torch.FloatTensor).cuda()
            y_test_batch = torch.Tensor(y_test_batch).type(torch.long).cuda()
            ip1, _ = model(x_test_batch)
            x_test_ab.extend(ip1.detach().cpu().numpy().tolist())
            y_test_ab.extend(y_test_batch.detach().cpu().numpy().tolist())
        x_test_ab = np.array(x_test_ab)
        y_test_ab = np.array(y_test_ab)
        embedding_dimension = x_train.shape[1]

    if embedding_dimension == 2:
        plt.scatter(x_train[:,0], x_train[:,1], label='x_train')
        plt.scatter(x_test_n[:,0], x_test_n[:,1], label='normal')
        plt.scatter(x_test_ab[:,0], x_test_ab[:,1], label='abnormal')
        plt.legend()
        plt.savefig('./images/emb_%s%d%s.pdf' % (type, id, str(epoch)))  # , bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()

    # bag of frames
    k = int(10 // length_s)
    x_train = np.reshape(x_train, (k,  x_train.shape[0]//k, embedding_dimension)).swapaxes(0, 1).reshape(x_train.shape[0]//k, embedding_dimension*k)
    x_test_n = np.reshape(x_test_n, (k,  x_test_n.shape[0]//k, embedding_dimension)).swapaxes(0, 1).reshape(x_test_n.shape[0]//k, embedding_dimension*k)
    x_test_ab = np.reshape(x_test_ab, (k,  x_test_ab.shape[0]//k, embedding_dimension)).swapaxes(0, 1).reshape(x_test_ab.shape[0]//k, embedding_dimension*k)

    if embedding_dimension == 2:
        for i in range(x_train.shape[1]):
            for j in range(x_train.shape[1]):
                if not i > j:
                    plt.scatter(x_test_ab[:, i], x_test_ab[:,j],label='test_abnormal')
                    plt.scatter(x_test_n[:, i], x_test_n[:,j], label='test_normal')
                    plt.scatter(x_train[:, i], x_train[:,j], label='train')
                    plt.legend()
                    plt.savefig('./images/emb_%s%d%s_%d%d.pdf' % (type, id, str(epoch),i,j),dpi=600)  # , bbox_inches='tight')
                    plt.clf()
                    plt.cla()
                    plt.close()

    return x_train, x_test_n, x_test_ab

def save_neural_embeddings(x_train, x_test_n, x_test_ab, type, id, snr, epoch, folder = './neural_embeddings/'):
    if epoch is None: epoch = ''
    if not os.path.exists(folder): create_folder(folder)
    np.save(folder+'x_train_%s_%d_%ddB_%d' % (type, id, snr, epoch), x_train)
    np.save(folder+'x_test_n_%s_%d_%ddB_%d' % (type, id, snr, epoch), x_test_n)
    np.save(folder+'x_test_ab_%s_%d_%sdB_%d' % (type, id, snr, epoch), x_test_ab)

    if x_train.shape[-1] == 2:
        plt.scatter(x_train[:,0], x_train[:,1], label='x_train')
        plt.scatter(x_test_n[:,0], x_test_n[:,1], label='normal')
        plt.scatter(x_test_ab[:,0], x_test_ab[:,1], label='abnormal')
        plt.legend()
        plt.savefig('./images/embed_%s%d%d%s.pdf' % (type, id, snr, str(epoch)), dpi=1000)  # , bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()


def load_neural_embeddings(type, id, snr, epoch, folder='./neural_embeddings/'):
    if epoch is None: epoch = ''
    x_train = np.load(folder+'x_train_%s_%d_%ddB_%d.npy' % (type, id, snr, epoch))
    x_test_n = np.load(folder+'x_test_n_%s_%d_%ddB_%d.npy' % (type, id, snr, epoch))
    x_test_ab = np.load(folder+'x_test_ab_%s_%d_%ddB_%d.npy' % (type, id, snr, epoch))
    return x_train, x_test_n, x_test_ab

def deepOneClass(x_train, x_test_n, x_test_ab):
    t0 = time.time()
    model = DOC(x_train.shape[1]).cuda()
    out_dim = model.out_features

    x_train = torch.Tensor(x_train).type(torch.FloatTensor).cuda()
    x_test_n = torch.Tensor(x_test_n).type(torch.FloatTensor).cuda()
    x_test_ab = torch.Tensor(x_test_ab).type(torch.FloatTensor).cuda()

    # Training Loop #
    batch_size = 128
    n_epochs = 200
    val_ratio = 0.10
    early_stopping = EarlyStopping(patience=10, path=os.path.join(FOLDER, 'ad_model_checkpoint.pt'), verbose=False)

    indices = np.arange(x_train.shape[0])
    n = np.int(np.floor(val_ratio * indices.shape[0]))
    val_indices = indices[:n]
    train_indices = indices[n:]

    N = train_indices.shape[0]
    svdd_loss = SVDD_loss(out_dim, 1.0).cuda()
    criterion = svdd_loss
    optimizer4nn = torch.optim.Adam(model.parameters(), lr=model.lr)
    optimzer4svdd = torch.optim.SGD(svdd_loss.parameters(), lr=model.lr_svdd)
    optimizer = [optimizer4nn, optimzer4svdd]
    c_per_epoch = []
    for epoch in range(1, n_epochs+1):
        # Shuffle Training Indices #
        np.random.shuffle(train_indices)
        running_loss = 0.0
        model.train()
        for i in range( N // batch_size):
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()

            X = x_train[train_indices[i * batch_size: (i + 1) * batch_size]]
            Y = model(X)

            var_reg = 0.1 * Y.var(dim=0).mean() 
            loss = criterion(torch.tensor(0).repeat(Y.shape[0]).cuda(), Y) + var_reg

            running_loss += loss.item()

            # Update Model Parameters #
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()

        c = svdd_loss.centers
        c_per_epoch.append(c)

        # Compute Validation Loss #
        model.eval()
        with torch.no_grad():
            Y_val = model(x_train[val_indices])
            var_reg = 0.1 * Y_val.var(dim=0).mean()
            val_loss = criterion(torch.tensor(0).repeat(Y_val.shape[0]).cuda(), Y_val) + var_reg  # + c * val_loss_reg

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
    
    # load the last checkpoint with the best model
    model = torch.load(FOLDER + '\\ad_model_checkpoint.pt')
    model.eval()
    c = c_per_epoch[early_stopping.best_epoch-1]
    with torch.no_grad():
        Y_train = model(x_train)
        Y_pred_n = model(x_test_n)
        Y_pred_ab = model(x_test_ab)
        anomaly_scores = ((Y_train - c) ** 2).sum(dim=1).cpu().numpy()
        anomaly_scores_n = ((Y_pred_n - c) ** 2).sum(dim=1).cpu().numpy()
        anomaly_scores_ab = ((Y_pred_ab - c) ** 2).sum(dim=1).cpu().numpy()
  
    y_score = anomaly_scores_n.tolist() + anomaly_scores_ab.tolist()
    y_true = [0] * len(anomaly_scores_n) + [1] * len(anomaly_scores_ab)
    auc = roc_auc_score(y_true, y_score)
    pauc = roc_auc_score(y_true, y_score, max_fpr=0.1)

    return auc, pauc


def oneClass(x_train, x_test_n, x_test_ab, epoch=None):
    if epoch is None: epoch = ''
    t0 = time.time()

    svm = OneClassSVM(nu=0.01)
    svm.fit(x_train)
    y_hat_test_n = svm.score_samples(x_test_n)
    y_hat_test_ab = svm.score_samples(x_test_ab)
    y_score = y_hat_test_n.tolist() + y_hat_test_ab.tolist()
    y_true = [1] * len(y_hat_test_n) + [0] * len(y_hat_test_ab)
    auc = roc_auc_score(y_true, y_score)
    pauc = roc_auc_score(y_true, y_score, max_fpr=0.1)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    print('AUC: %.3f - pAUC: %.3f (%s)' % (auc, pauc, 'OCSVM'))
    return auc



def anomaly_detection(model, f_test, type, id, snr, length_s=2., channels=None,
                  epoch=None, batch_size=128):
    t0 = time.time()
    if os.path.exists('./neural_embeddings/x_train_%s_%d_%ddB_%d.npy' % (type, id, snr, epoch)):
        # print('loading embeddings...')
        x_train, x_test_n, x_test_ab = load_neural_embeddings(type, id, snr, epoch)
    else:
        x_train, x_test_n, x_test_ab = predictNeuralEmbeddings(model, batch_size, f_test, length_s=length_s, type=type,
                                                               id=id, snr=snr, channels=channels, epoch=epoch)
        save_neural_embeddings(x_train, x_test_n, x_test_ab, type, id, snr, epoch)

    print("%ds - Anomaly detection:" % int(time.time() - t0), type, id, snr, 'dB -', x_train.shape, x_test_n.shape, x_test_ab.shape)

    auc, pauc = deepOneClass(x_train, x_test_n, x_test_ab)
    auc_svm = oneClass(x_train, x_test_n, x_test_ab)
    return auc, auc_svm

fs = 16000
FOLDER = ''
dataset = MIMII
if __name__ == '__main__':
    f_train, f_val, f_test = dataset.get_train_val_test_filepaths(test_ratio=0.25, val_ratio=0.2, shuffle=False, roll=False)
    FOLDER = create_folder('models/' + time.strftime("%m-%d") + '_CNN1D')
    shutil.copy('main.py', FOLDER + '/main.txt')
    shutil.copy('models.py', FOLDER + '/models.txt')
    sys.stdout = Logger(FOLDER + '/console.txt')

	model = train_neural_embeddings(f_train, f_val, f_test)

    total_aucs = np.zeros((2, 4, 4, 3))
    for t, type in enumerate(dataset.all_types):
        for i, id in enumerate(dataset.get_ids):
            for s, snr in enumerate(dataset.all_snrs):
                total_aucs[:, t, i, s] = anomaly_detection(model, length_s=2.,
                                                           f_test=f_test, type=type, id=id, snr=snr, epoch=12)
                np.savetxt(FOLDER + "/aucs_deep.csv", total_aucs[0].reshape(16, 3), delimiter=";")
                np.savetxt(FOLDER + "/aucs_ocsvm.csv", total_aucs[1].reshape(16, 3), delimiter=";")


