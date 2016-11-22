# LSTM's memory ability test

import numpy as np
import chainer
from chainer import iterators, optimizers, training, datasets
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
import argparse

def complete_sequence(seq_len, type):
    n_vocabs = seq_len
    x = list(range(0, n_vocabs-2))
    n = n_vocabs + type - 3
    x.insert(0, n)
    x.append(n)
    return x

class MyModel(chainer.Chain):
    def __init__(self, n_vocabs, n_units, n_layers, train=True):
        super(MyModel, self).__init__(
                emb = L.EmbedID(n_vocabs, n_units),
                lstm = chainer.ChainList(),
                out = L.Linear(n_units, n_vocabs))

        for i in range(n_layers):
            self.lstm.add_link(L.LSTM(n_units, n_units))

        for param in self.params():
            param.data[...] = np.random.uniform(-0.2, 0.2, param.data.shape)

        self.train = train
        self.n_layers = n_layers

    def __call__(self, x):
        y = self.emb(x)
        h = y
        for i in range(self.n_layers):
            h = self.lstm[i](h)
        z = self.out(h)
        return z

    def reset_state(self):
        for i in range(self.n_layers):
            self.lstm[i].reset_state()

class MyIterator(chainer.dataset.Iterator):
    def __init__(self, seq_len):
        self.epoch = 0
        self.iteration = 0
        self.is_new_epoch = False
        samples = [complete_sequence(seq_len, 1), complete_sequence(seq_len, 2)]
        self.samples = samples

    def __next__(self):
        sample = self.samples[np.random.randint(0,2)]
        self.iteration += 1
        epoch = self.iteration
        if epoch > self.epoch:
            self.is_new_epoch = True
            self.epoch = epoch
        else:
            self.is_new_epoch = False

        return sample

    @property
    def epoch_detail(self):
        return float(self.iteration)

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)

def compute_loss(target, sample_seq, converter):
    data = zip(sample_seq[0:-1], sample_seq[1:])
    loss = 0
    for x, t in data:
        x0, t0 = converter((np.array([x], dtype=np.int32), np.array([t], dtype=np.int32)))
        loss += target(x0, t0)
    return loss

class BPTTUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.device = device

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        sample_seq = next(train_iter)

        optimizer.target.predictor.reset_state()
        converter = lambda x: self.converter(x, device=self.device)
        loss = compute_loss(optimizer.target, sample_seq, converter)
        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()

def predict(predictor, arr):
    predictor.reset_state()
    result = []
    for e in arr:
        t = predictor(chainer.Variable(np.array([e], dtype=np.int32)))
        r = F.softmax(t).data.argmax(axis=1)[0]
        result.append(r)
    return result[-1]

def prediction_test(predictor, seq_len):
    test_sequences = [complete_sequence(seq_len, 1), complete_sequence(seq_len, 2)]
    map(lambda x: x.pop(), test_sequences)
    print("input:{}".format(test_sequences))
    expected = map(lambda x: x[0], test_sequences)
    predicted = map(lambda x: predict(predictor, x), test_sequences)
    print("expected:{}".format(expected))
    print("predicted:{}".format(predicted))
    if predicted == expected:
        result = "SUCCESS"
    else:
        result = "FAILURE"
    print("result:{}".format(result))

def main():
    parser = argparse.ArgumentParser(description="A test of LSTM's memory ability")
    parser.add_argument('--seq_len', type=int, default=5, metavar='N',
            help='length of a sequence used by learning')
    parser.add_argument('--n_layers', type=int, default=1, metavar='N',
            help='number of LSTM layers')
    parser.add_argument('--n_units', type=int, default=50, metavar='N',
            help='number of units a LSTM hidden layer use')
    parser.add_argument('--iter', type=int, default=1000, metavar='N',
            help='number of learning iteration')
    parser.add_argument('--opt', type=str, choices=['SGD', 'ADAM'], default='ADAM',
            help='optimization method')
    parser.add_argument('--silent', default=False, action='store_true',
            help='whether prints out progress status and estimation results')
    parser.add_argument('--gpu', type=int, default=-1, metavar='Z',
            help='GPU id')
    args = parser.parse_args()
    args.verbose = not args.silent

    if args.verbose:
        print("args:{}".format(vars(args)))

    n_vocabs = args.seq_len
    train_iter = MyIterator(args.seq_len)
    model = L.Classifier(MyModel(n_vocabs, args.n_units, args.n_layers, train=True))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    if args.opt == 'SGD':
        optimizer = optimizers.SGD()
    elif args.opt == 'ADAM':
        optimizer = optimizers.Adam()
    else:
        raise Exception('unknown optimizer')

    optimizer.setup(model)

    updater = BPTTUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.iter, 'iteration'), out='result')

    if args.verbose:
        report_params = [
                'iteration',
                'main/loss',
                'main/accuracy',
                ]
        trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
        trainer.extend(extensions.PrintReport(report_params))
        trainer.extend(extensions.ProgressBar())

    trainer.run()

    if args.verbose:
        eval_model = model.copy()
        eval_rnn = eval_model.predictor
        eval_rnn.train = False
        prediction_test(eval_rnn, args.seq_len)

main()
