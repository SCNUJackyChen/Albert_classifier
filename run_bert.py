""" Finetuning the library models for sequence classification ."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import glob
import numpy as np
import torch
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from argparse import ArgumentParser

from model.modeling_albert import AlbertConfig, AlbertForSequenceClassification
# from model.modeling_albert_bright import AlbertConfig, AlbertForSequenceClassification # chinese version
from model import tokenization_albert
from model.file_utils import WEIGHTS_NAME
from callback.optimization.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup

from metrics.glue_compute_metrics import compute_metrics
from processors import glue_output_modes as output_modes
from processors import glue_processors as processors
from processors import glue_convert_examples_to_features as convert_examples_to_features
from processors import collate_fn
from tools.common import seed_everything
from tools.common import init_logger, logger
from callback.progressbar import ProgressBar
from pybert.io.bert_processor import BertProcessor
from pybert.configs.basic_config import config
from pybert.train.trainer import Trainer
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport
from pybert.train.losses import BCEWithLogLoss
from pybert.preprocessing.preprocessor import EnglishPreProcessor


def run_train(args, model, tokenizer):
    # --------- data
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    train_data = processor.get_train(config['data_dir'] / f"{args.data_name}.train.pkl")
    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_examples_file=config[
                                                                        'data_dir'] / f"cached_train_examples_{args.arch}")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=config[
                                                                        'data_dir'] / "cached_train_features_{}_{}".format(
                                                   args.train_max_seq_len, args.arch
                                               ))
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_data = processor.get_dev(config['data_dir'] / f"{args.data_name}.valid.pkl")
    valid_examples = processor.create_examples(lines=valid_data,
                                               example_type='valid',
                                               cached_examples_file=config[
                                                                        'data_dir'] / f"cached_valid_examples_{args.arch}")

    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=args.eval_max_seq_len,
                                               cached_features_file=config[
                                                                        'data_dir'] / "cached_valid_features_{}_{}".format(
                                                   args.eval_max_seq_len, args.arch
                                               ))
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)


    # --------- model

    logger.info("initializing model")
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(num_training_steps * args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # ---- callbacks
    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'], mode=args.mode,
                                       monitor=args.monitor, arch=args.arch,
                                       save_best_only=args.save_best)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_training_steps)


    trainer = Trainer(n_gpu=args.num_gpu,
                      model=model,
                      epochs=args.epochs,
                      logger=logger,
                      criterion=BCEWithLogLoss(),
                      optimizer=optimizer,
                      lr_scheduler=scheduler,
                      early_stopping=None,
                      training_monitor=train_monitor,
                      fp16=args.fp16,
                      resume_path=args.resume_path,
                      grad_clip=args.grad_clip,
                      model_checkpoint=model_checkpoint,
                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                      batch_metrics=[AccuracyThresh(thresh=0.5)],
                      epoch_metrics=[AUC(average='micro', task_type='binary'),
                      MultiLabelReport(id2label=id2label)])

    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader, seed=args.seed)

def run_test(args):
    pass


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument('--data_name', default='kaggle', type=str)
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)

    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--predict_checkpoints", type=int, default=0)
    parser.add_argument("--valid_size", default=0.2, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int, help='1 : True  0:False ')
    parser.add_argument("--num_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument("--train_max_seq_len", default=256, type=int)
    parser.add_argument("--eval_max_seq_len", default=256, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: ")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--vocab_file",default='', type=str)
    parser.add_argument("--spm_model_file",default='',type=str)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")   
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
      
    args = parser.parse_args()
 
    from pybert.configs.basic_config import config
    init_logger(log_file=config['log_dir'] / f'{args.arch}-{time.strftime("%Y-%m-%d-H:%M:%S", time.localtime())}.log')
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch
    config['checkpoint_dir'].mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    seed_everything(args.seed)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_data:
        from pybert.io.task_data import TaskData
        data = TaskData()
        targets, sentences = data.read_data(raw_data_path=config['raw_data_path'],
                                            preprocessor=EnglishPreProcessor(),
                                            is_train=True)
        data.train_val_split(X=sentences, y=targets, shuffle=True, stratify=False,
                             valid_size=args.valid_size, data_dir=config['data_dir'],
                             data_name=args.data_name)
    if args.do_train:

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        args.output_dir = args.output_dir + '{}'.format(args.model_type)
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        init_logger(log_file=args.output_dir + '/{}-{}.log'.format(args.model_type, args.task_name))
        if os.path.exists(args.output_dir) and os.listdir(
                args.output_dir) and args.do_train and not args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args.output_dir))

        # Setup distant debugging if needed
        # if args.server_ip and args.server_port:
        #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        #     import ptvsd
        #     print("Waiting for debugger attach")
        #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        #     ptvsd.wait_for_attach()

        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            args.n_gpu = 1
        args.device = device
        # Setup logging
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                       args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
        # Set seed
        seed_everything(args.seed)
        # Prepare GLUE task
        # args.task_name = args.task_name.lower()
        # if args.task_name not in processors:
        #     raise ValueError("Task not found: %s" % (args.task_name))
        processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
        
        # args.output_mode = output_modes[args.task_name]
        label_list = processor.get_labels()
        num_labels = len(label_list)

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.model_type = args.model_type.lower()
        config = AlbertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                              num_labels=num_labels,
                                              finetuning_task=args.task_name)
        tokenizer = tokenization_albert.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case,
                                                      spm_model_file=None)
        model = AlbertForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                from_tf=bool('.ckpt' in args.model_name_or_path),
                                                                config=config)

        run_train(args,model,tokenizer)

    if args.do_test:
        run_test(args)

if __name__ == "__main__":
    main()
