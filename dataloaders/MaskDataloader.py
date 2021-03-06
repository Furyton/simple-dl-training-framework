import logging
from dataloaders.base import AbstractDataloader
from dataloaders.negative_samplers import negative_sampler_factory

import torch
import torch.utils.data as data_utils
from copy import deepcopy


class MaskDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.train = dataset[0]
        self.val = dataset[1]
        self.test = dataset[2]
        self.user_count = dataset[3]
        self.item_count = dataset[4]
        self.rating_train = dataset[5]
        self.rating_eval = dataset[6]
        self.rating_test = dataset[7]
        # args.num_items = self.item_count

        logging.info("there are {} items in this dataset, {} data".format(args.num_items, self.user_count))

        code = args.train_negative_sampler_code
        # train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test, self.user_count, self.item_count, args.train_negative_sample_size, args.train_negative_sampling_seed, self.save_folder, args.dataset_name)


        code = args.test_negative_sampler_code

        if args.test_negative_sample_size != 0:
            test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test, self.user_count, self.item_count, args.test_negative_sample_size, args.rand_seed, self.save_folder, args.dataset_name)
            # self.train_negative_samples = train_negative_sampler.get_negative_samples()
            self.test_negative_samples = test_negative_sampler.get_negative_samples()

            self.enable_negative_sample = True
        else:
            self.enable_negative_sample = False

        self.max_len = args.max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

    @classmethod
    def code(cls):
        return 'mask'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size, drop_last=True, shuffle=True, pin_memory=False, num_workers=self.worker_num)

        return dataloader

    def _get_train_dataset(self):
        dataset = NormalTrainDataset(self.train, self.rating_train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=self.worker_num)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        # train_dataset = None
        rating = None
        if mode == 'val':
            rating = deepcopy(self.rating_eval)
            train_dataset = deepcopy(self.train)
        else:
            rating = deepcopy(self.rating_test)
            train_dataset = deepcopy(self.train)
            for index, seq in enumerate(train_dataset):
                seq.append(self.val[index][0])

        if self.enable_negative_sample:
            dataset = NormalEvalDataset(train_dataset, rating, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        else:
            dataset = NormalEvalDataset_Without_Neg(train_dataset, rating, answers, self.max_len, self.CLOZE_MASK_TOKEN)

        return dataset


class NormalTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2rating, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.u2rating = u2rating
        self.users = range(len(u2seq))
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq, rating = self._getseq(user)

        tokens = []
        labels = []

        masked_num = 0

        for s in seq:
            prob = self.rng.rand()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items + 1))
                else:
                    tokens.append(s)

                labels.append(s)
                masked_num += 1
            else:
                tokens.append(s)
                labels.append(0)
                # padding is 0

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        rating = rating[-self.max_len:]

        padding_len = self.max_len - len(tokens)

        # padding
        tokens = [0] * padding_len + tokens
        labels = [0] * padding_len + labels
        rating = [0] * padding_len + rating

        return torch.LongTensor(tokens), torch.LongTensor(labels), torch.Tensor(rating)

    def _getseq(self, user):
        return deepcopy(self.u2seq[user]), deepcopy(self.u2rating[user])


class NormalEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2rating, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.users = range(len(u2seq))
        self.u2rating = u2rating
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = deepcopy(self.u2seq[user])
        rating = deepcopy(self.u2rating[user])
        answer = deepcopy(self.u2answer[user])
        negs = deepcopy(self.negative_samples[user])

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels), torch.LongTensor([rating])


class NormalEvalDataset_Without_Neg(data_utils.Dataset):
    def __init__(self, u2seq, u2rating, u2answer, max_len, mask_token):
        self.u2seq = u2seq
        self.users = range(len(u2seq))
        self.u2rating = u2rating
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = deepcopy(self.u2seq[user])
        rating = deepcopy(self.u2rating[user])
        answer = deepcopy(self.u2answer[user])

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(answer), torch.LongTensor([rating])
