#!/usr/bin/env python
try:
    from OpenGL import GLU
except:
    print("no OpenGL.GLU")
import functools
import os
import os.path as osp
from functools import partial

import gym
import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
from mpi4py import MPI

from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dynamics import Dynamics, UNet
from utils import random_agent_ob_mean_std
from wrappers import MontezumaInfoWrapper, make_mario_env, make_robo_pong, make_robo_hockey, \
    make_multi_pong, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit


def start_experiment(**args):
    make_env = partial(make_env_all_params, add_monitor=True, args=args)

    log, tf_sess = get_experiment_environment(**args)
    with log:
        trainer = Trainer(make_env=make_env,
                        num_timesteps=args['num_timesteps'], hps=args,
                        envs_per_process=args['envs_per_process'])

        with tf_sess:
            logdir = logger.get_dir()
            log_run_details(logdir)
            print("results will be saved to ", logdir)
            trainer.train()

def log_run_details(logdir):
    with open(osp.join(logdir, 'run_command.txt'), 'x') as fcmd:
        import sys
        import subprocess
        fcmd.write(' '.join(sys.argv))
        fcmd.write('\n')
        fcmd.write(repr(sys.argv))
        try:
            cwd = osp.dirname(osp.realpath(__file__))
        except NameError:
            cwd = None
        env = os.environ.copy()
        env['LC_ALL'] = 'C'
        def store_command_output(cmd):
            try:
                fcmd.write('\n')
                fcmd.write(subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=cwd, env=env).decode('ascii'))
            except Exception as e:
                print('On logging run details exception occured: {!r}'.format(e))

        store_command_output(['git', 'remote', 'get-url', 'origin'])
        store_command_output(['git', 'rev-parse', 'HEAD'])
        store_command_output(['git', 'status'])

class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()

        self.policy = CnnPolicy(
            scope='pol',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            hidsize=512,
            feat_dim=512,
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nl=tf.nn.leaky_relu)

        self.feature_extractor = {"none": FeatureExtractor,
                                  "idf": InverseDynamics,
                                  "vaesph": partial(VAE, spherical_obs=True),
                                  "vaenonsph": partial(VAE, spherical_obs=False),
                                  "pix2pix": JustPixels}[hps['feat_learning']]
        self.feature_extractor = self.feature_extractor(policy=self.policy,
                                                        features_shared_with_policy=False,
                                                        feat_dim=512,
                                                        layernormalize=hps['layernorm'])

        self.dynamics = Dynamics if hps['feat_learning'] != 'pix2pix' else UNet
        self.dynamics = self.dynamics(auxiliary_task=self.feature_extractor,
                                      predict_from_pixels=hps['dyn_from_pixels'],
                                      feat_dim=512)

        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            dynamics=self.dynamics
        )

        self.agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['aux']
        self.agent.to_report['dyn_loss'] = tf.reduce_mean(self.dynamics.loss)
        self.agent.total_loss += self.agent.to_report['dyn_loss']
        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])
        self.ckpt_update = hps['ckpt_update']
        if hps['ckpt_path'] is None:
            self.ckpt_base_path = osp.join(logger.get_dir(), 'models')
        else:
            ver = 0
            while True:
                base_path = osp.join(hps['ckpt_path'], 'models-{}{}'.format(hps['exp_name'], ver))
                if not os.path.exists(base_path):
                    break
                ver += 1
            self.ckpt_base_path = base_path
        os.makedirs(self.ckpt_base_path)


    def _set_env_vars(self):
        env = self.make_env(0, add_monitor=False, disable_rec=True)
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

    def store_checkpoint(self, global_step=None, write_meta_graph=False):
      self.saver.save(tf.compat.v1.get_default_session(),
          self.models_path,
          global_step=global_step,
          write_meta_graph=write_meta_graph
      )

    def train(self):
        self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamics=self.dynamics)
        self.models_path = osp.join(self.ckpt_base_path, 'model.ckpt')
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep=10,
                                              keep_checkpoint_every_n_hours=4, save_relative_paths=True)
        while True:
            info = self.agent.step()
            last_update = 0
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
                last_update = info['update']['n_updates']
            if self.agent.rollout.stats['tcount'] > self.num_timesteps:
                break
            if self.ckpt_update != 0 and last_update % self.ckpt_update == 1:
                self.store_checkpoint(global_step=last_update)
        self.store_checkpoint()
        self.agent.stop_interaction()


def make_env_all_params(rank, add_monitor, args, disable_rec=False):
    if args["env_kind"] == 'atari':
        env = gym.make(args['env'])
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=args['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        env = ExtraTimeLimit(env, args['max_episode_steps'])
        if 'Montezuma' in args['env']:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == 'mario':
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":
        if not disable_rec and rank == 0 and args["retro_record"]:
            rec_path = osp.join(logger.get_dir(), 'retro-rec')
            os.makedirs(rec_path, exist_ok=True)
        else:
            rec_path = False
        env = make_multi_pong(record_path=rec_path)
    elif args["env_kind"] == 'robopong':
        if args["env"] == "pong":
            env = make_robo_pong()
        elif args["env"] == "hockey":
            env = make_robo_hockey()

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
    return env


def get_experiment_environment(**args):
    from utils import setup_mpi_gpus, setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed
    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    setup_mpi_gpus()

    logger_context = logger.scoped_configure(dir=None,
                                             format_strs=['stdout', 'log',
                                                          'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else ['log'])
    tf_context = setup_tensorflow_session()
    return logger_context, tf_context


def add_environments_params(parser):
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4',
                        type=str)
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=4500, type=int)
    parser.add_argument('--env_kind', type=str, default="atari")
    parser.add_argument('--noop_max', type=int, default=30)
    parser.add_argument('--retro_record', action='store_true', help='in multipong (Atari, gym-retro) store .bk2 recordings on worker 0')


def add_optimization_params(parser):
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--norm_adv', type=int, default=1)
    parser.add_argument('--norm_rew', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ent_coeff', type=float, default=0.001)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--num_timesteps', type=int, default=int(1e8))


def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=128)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    parser.add_argument('--envs_per_process', type=int, default=128)
    parser.add_argument('--nlumps', type=int, default=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser)
    add_optimization_params(parser)
    add_rollout_params(parser)

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=0.)
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--layernorm', type=int, default=0)
    parser.add_argument('--feat_learning', type=str, default="none",
                        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"])
    parser.add_argument('--ckpt_update', type=int, default=5, help='Save checkpoint after each K updates (0 disables)', metavar='K')
    parser.add_argument('--ckpt_path', default=None, help='path to directory in which checkpoints will be stored (by default logger dir)')

    args = parser.parse_args()

    start_experiment(**args.__dict__)
