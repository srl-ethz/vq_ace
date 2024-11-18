from ...common.autoinit_class import AutoInit
import hydra
import numpy as np
import os
from collections import OrderedDict
import imageio
import time
import json
from tqdm import tqdm

class SimulationEnvMixin(AutoInit, cfgname_and_funcs=(("sim_env_cfg", "_init_sim_env"),)):
    """
    Base simulation environment mixin class that configures corresponding simulation environments to the pipeline.
    """
    
    def _init_sim_env(self, **env_dict):
        """
        Initialize the simulation environment configuration.
        A container for objects in srl_il/simulators
        """
        self.sim_env_dict = {
            k: hydra.utils.instantiate(v)
            for k, v in env_dict.items()
        }


    def _sim_env_run_rollout(
        self,
        env, 
        horizon,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
    ):

        self.algo.reset_policy()

        ob_dict = env.reset()

        results = {}
        video_count = 0  # video frame counter

        total_reward = 0.
        success = { k: False for k in env.is_success() } # success metrics

        try:
            policy_times = []
            for step_i in range(horizon):

                # get action from policy
                policy_start_time = time.time()
                ac = self.algo.predict_action (obs_dict=ob_dict).detach()
                policy_times.append(time.time() - policy_start_time)

                # play action
                ob_dict, r, done, _ = env.step(ac)

                # render to screen
                if render:
                    env.render(mode="human")

                # compute reward
                total_reward += r

                cur_success_metrics = env.is_success()
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]

                # visualization
                if video_writer is not None:
                    if video_count % video_skip == 0:
                        video_img = env.render(mode="rgb_array", height=512, width=512)
                        video_writer.append_data(video_img)

                    video_count += 1

                # break if done
                if done or (terminate_on_success and success["task"]):
                    break

        except env.rollout_exceptions as e:
            print("WARNING: got rollout exception {}".format(e))

        results["Return"] = float(total_reward)
        results["Horizon"] = step_i + 1
        results["Success_Rate"] = float(success["task"])
        results["policy_time_max"] = np.max(policy_times)
        results["policy_time_mean"] = np.mean(policy_times)

        # log additional success metrics
        for k in success:
            if k != "task":
                results["{}_Success_Rate".format(k)] = float(success[k])
        return results
    


    def rollout_in_sim_env(
            self,
            horizon,
            num_episodes=None,
            render=False,
            video_dir=None,
            video_path=None,
            epoch=None,
            video_skip=5,
            terminate_on_success=False,
            verbose=False
        ):
        """
        A helper function used in the train loop to conduct evaluation rollouts per environment
        and summarize the results.

        Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
        for all environments).

        Args:
            horizon (int): maximum number of steps to roll the agent out for

            num_episodes (int): number of rollout episodes per environment

            render (bool): if True, render the rollout to the screen

            video_dir (str): if not None, dump rollout videos to this directory (one per environment)

            video_path (str): if not None, dump a single rollout video for all environments

            epoch (int): epoch number (used for video naming)

            video_skip (int): how often to write video frame

            terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

            verbose (bool): if True, print results of each rollout

            drop_successrate_lim (float): Before finish all num_episodes, break out if the success_rate cannot be higher than this lim
        
        Returns:
            all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...) 
                averaged across all rollouts 

            video_paths (dict): path to rollout videos for each environment
        """

        all_rollout_logs = OrderedDict()
        envs = self.sim_env_dict
        # handle paths and create writers for video writing
        assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
        write_video = (video_path is not None) or (video_dir is not None)
        video_paths = OrderedDict()
        video_writers = OrderedDict()
        if video_path is not None:
            # a single video is written for all envs
            video_paths = { k : video_path for k in envs }
            video_writer = imageio.get_writer(video_path, fps=20)
            video_writers = { k : video_writer for k in envs }
        if video_dir is not None:
            # video is written per env
            video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4" 
            video_paths = { k : os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs }
            video_writers = { k : imageio.get_writer(video_paths[k], fps=20) for k in envs }

        for env_name, env in envs.items():
            env_video_writer = None
            if write_video:
                print("video writes to " + video_paths[env_name])
                env_video_writer = video_writers[env_name]

            print("rollout: env={}, horizon={}, num_episodes={}".format(
                env.name, horizon, num_episodes,
            ))
            rollout_logs = []
            iterator = range(num_episodes)
            if not verbose:
                iterator = tqdm(iterator, total=num_episodes)

            num_success = 0
            runned_episodes = 0
            for ep_i in iterator:
                rollout_timestamp = time.time()
                rollout_info = self._sim_env_run_rollout(
                    env=env,
                    horizon=horizon,
                    render=render,
                    video_writer=env_video_writer,
                    video_skip=video_skip,
                    terminate_on_success=terminate_on_success,
                )
                rollout_info["time"] = time.time() - rollout_timestamp
                rollout_logs.append(rollout_info)
                num_success += rollout_info["Success_Rate"]
                runned_episodes += 1
                if verbose:
                    print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                    print(json.dumps(rollout_info, sort_keys=True, indent=4))

            if video_dir is not None:
                # close this env's video writer (next env has it's own)
                env_video_writer.close()

            # average metric across all episodes
            rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
            rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
            rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
            rollout_logs_mean["Runned_Episode"] = runned_episodes
            all_rollout_logs[env_name] = rollout_logs_mean

        if video_path is not None:
            # close video writer that was used for all envs
            video_writer.close()

        return all_rollout_logs, video_paths