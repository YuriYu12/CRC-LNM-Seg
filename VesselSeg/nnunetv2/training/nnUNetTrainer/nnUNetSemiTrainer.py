import itertools
from functools import reduce
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import *


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class MultiStreamBatchSampler(object):
    def __init__(self, index_groups, batch_sizes, iterate_on=0, **kwargs):
        self.batch_sizes = batch_sizes
        self.index_groups = index_groups
        self.iterate_on = iterate_on
        self.batch_size = sum(self.batch_sizes)

        assert all([len(index_group) >= self.batch_sizes[i] >= 0 for i, index_group in enumerate(self.index_groups)]),\
            f"not all lengths of index group is larger than that of its corresponding batch size"
        assert self.batch_sizes[self.iterate_on] > 0, f"iterate axis {self.iterate_on} must be of finite length" 
        
    def __call__(self, indices, sampling_probabilities=None):
        streams = []
        for index_group, batch_size in zip(self.index_groups, self.batch_sizes):
            streams.extend(np.random.choice(index_group, batch_size, replace=True, p=sampling_probabilities).tolist())
            
        return indices[streams]

    def __iter__(self):
        iters = [((iterate_once if i == self.iterate_on else iterate_eternally)(self.index_groups[i]), self.batch_sizes[i])
                 for i in range(len(self.index_groups)) if len(self.batch_sizes[i]) > 0]
        return (
            reduce(lambda x, y: x + y, batches)
            for batches
            in zip(*[grouper(itr, bs) for itr, bs in iters])
        )
        
    def __len__(self):
        return len(self.index_groups[self.iterate_on]) // self.batch_sizes[self.iterate_on]


class nnUNetSemiTrainer(nnUNetTrainer):
    def __init__(self, **nnunet_kw):
        super().__init__(**nnunet_kw)
        
        self.index_groups = (tuple(_ for _ in range(10)), tuple(_ for _ in range(10, 500)))
        self.group_batch_sizes = (1, self.batch_size - 1)
        self.sampler = MultiStreamBatchSampler(self.index_groups, self.group_batch_sizes, iterate_on=1)
        
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, sampler=self.sampler)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, sampler=self.sampler)
        else:
            dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, sampler=self.sampler)
            dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, sampler=self.sampler)
        return dl_tr, dl_val
        
    def train_step(self, batch: dict) -> dict:
        data_all = batch['data']
        target_all = batch['target']
        data, data_soft = torch.split(data_all, self.group_batch_sizes)
        target, target_soft = torch.split(target_all, self.group_batch_sizes)

        data = data.to(self.device, non_blocking=True)
        count_fn = lambda volume: torch.bincount(volume.flatten().to(torch.long),
                                                 minlength=len(self.label_manager.all_labels)).to(torch.float32) / torch.numel(volume)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            self.perceived_prior_carry = self.perceived_prior_carry * 0.5 + np.mean([count_fn(t).data.cpu().numpy() for t in target], 0) * 0.5
        else:
            target = target.to(self.device, non_blocking=True)
            self.perceived_prior_carry = self.perceived_prior_carry * 0.5 + (count_fn(target).data.cpu().numpy()) * 0.5

        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)
            
            output_soft = self.network(data_soft)
            output_confidence_mask = output_soft.softmax(1).max(1) > 0.7
            target_soft = output_soft.argmax(1).data
            target_soft[~output_confidence_mask] = 255
            l_soft = self.loss(output_soft, target_soft)
            
            l += l_soft * 0.5

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy(), "perceived_prior": self.perceived_prior_carry}