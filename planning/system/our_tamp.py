import datetime
import numpy as np
import functools
import torch
import cv2

from stap.utils.spaces import null_tensor
from stap.agents.constant import ConstantAgent
from stap.planners.utils import evaluate_trajectory
from stap.planners.base import PlanningResult
from planning.utils.env_utils import fix_high_level_action
from p_estimator.trainers.trainer import PEstimotarTrainer
from planning.system.base_system import BaseSystemPlanner
from planning.bandits.catalog import BANDITS_CATALOG
from utils.general_utils import save_pickle
from utils.network_utils import load_checkpoint_lightning
import os

def save_action(data:list, output_folder:str, sub_folder_name:str , name:str, images:dict=None):
    #find index
    not_found = True
    index = 0
    os.makedirs(os.path.join(output_folder, sub_folder_name), exist_ok=True)
    while not_found:
        temp_path = os.path.join(output_folder, sub_folder_name, f'{name}_{str(index)}.pickle')
        if not os.path.exists(temp_path):
            not_found = False
        index+=1
    save_pickle(path=temp_path, object_to_save=data)
    if images is not None:
        image_folder_path = os.path.join(output_folder, sub_folder_name, name, f'{str(index)}')
        os.makedirs(image_folder_path, exist_ok=True)
        image_index = 0
        cv2.imwrite(os.path.join(image_folder_path, f'start_image.png'), images["start_image"])
        for image_index, image in enumerate(images["all_images"]):
            success = "success" if data[image_index]["reward"] == 1 else "fail"
            cv2.imwrite(os.path.join(image_folder_path, f'{success}_image_{str(image_index)}.png'), image)

class TAMPPlanner(BaseSystemPlanner):
    def __init__(self, cfg:dict):
        super().__init__(cfg=cfg)
        
        self.bandit = BANDITS_CATALOG["vanilla"](params={})
        self.plans = []
        self.open_high_level_str = "open_high_level"
        self.open_low_level_str = "open_low_level"
        self.num_samples = 2
        self.num_iterations = self.cfg.GENERAL_PARMAS.num_iterations
        self.num_elites_to_keep = 10
        self.num_elites = 10
        self.momentum = 0
        self.population_decay = 1
        self.high_level_reset = True
        self.low_level_reset = True
        self.last_high_level_score = None
        self.low_level_scores = {}
        self.saved_low_level_options = []
        self.turn_off_high_level = self.cfg.GENERAL_PARMAS.turn_off_high_level
        self.stochastic_actions = self.cfg.GENERAL_PARMAS.stochastic_actions if "stochastic_actions" in  self.cfg.GENERAL_PARMAS else False
        if self.cfg.GENERAL_PARMAS.state_estimator.type == "p_value":
            self._load_p_value_model()
        self.index = 0  

    def _load_p_value_model(self):
        self.state_estimator = {}
        for key in self.cfg.GENERAL_PARMAS.state_estimator.params.checkpoints:
            self.state_estimator[key] = load_checkpoint_lightning(
                model=PEstimotarTrainer,
                checkpoint_path=self.cfg.GENERAL_PARMAS.state_estimator.params.checkpoints[key],
                device=self.cfg.GENERAL_PARMAS.device
            )

    def _execute_low_level_action(self, start_high_level_state_id:int,
        end_high_level_state_id:int, low_level_state_id:int, first_action_score:np.array=None):
        
        start_high_level_state =\
            self.graph_manager.high_level_graph.graph.nodes._nodes[start_high_level_state_id]['state']
        #end_high_level_state =\
        #    self.graph_manager.high_level_graph.graph.nodes._nodes[end_high_level_state_id]['state']
        goal_reached = self.graph_manager.high_level_graph.graph.nodes._nodes[end_high_level_state_id]['goal_reached']
        
        high_level_action_id = end_high_level_state_id
        high_level_action =\
            self.graph_manager.high_level_graph.graph.edges._adjdict[start_high_level_state_id][end_high_level_state_id]['high_level_action']
        high_level_action = fix_high_level_action(
            high_level_action=high_level_action
        )
        
        #select low level state
        observation = self.graph_manager.low_level_graph.graph.nodes._nodes[low_level_state_id]['low_level_state']
        robot_state = self.graph_manager.low_level_graph.graph.nodes._nodes[low_level_state_id]['robot_state']
        info = self.graph_manager.low_level_graph.graph.nodes._nodes[low_level_state_id]['info']
        if observation is None:
            return

        self._increase_node_count(
            node_id=low_level_state_id,
            action_id=high_level_action_id,
            )
        self.step +=1 
        
        action_is_ok = self._action_validator(high_level_action)
        if not action_is_ok:
            return {
                "type": "low_level",
                "goal_reached": False,
            }
        if self.stochastic_actions:
            actions_list = []
            primitive = self._get_primitive(
                action=high_level_action,
                env=self.env
            )
            skill_name = primitive.__class__.__name__
            img = self.env.render()
            images = {
                "start_image": img,
                "all_images": [],
            }
            for _ in range(100):
                self.env.set_observation(observation)
                self.env.set_state(state=robot_state)
                primitive = self._get_primitive(
                    action=high_level_action,
                    env=self.env
                )
                skill_name = primitive.__class__.__name__
                #get action from low level plannes
                self.env.set_primitive(primitive=primitive)
                low_level_action = self.low_level_planner.get_action(
                    obs0=observation,
                    observation_indices= info['policy_args']['observation_indices'],
                    raw_high_level_action=high_level_action,
                    device=self.cfg.GENERAL_PARMAS.device,
                    primitive=primitive
                )
                #execute low level action
                try:
                    end_observation, reward, terminated, truncated, info = self.env.step(low_level_action)
                    sample = {
                        "reward": reward,
                        "end_observation": end_observation,
                        "skill_name": skill_name,
                        "high_level_action": high_level_action,
                        "observation": observation,
                        "start_high_level_state": start_high_level_state,
                        "low_level_action": low_level_action,
                        "first_action_score": first_action_score.item() if first_action_score else first_action_score,
                    }
                    img = self.env.render()
                    images["all_images"].append(img)
                    actions_list.append(sample)
                except Exception as e:
                    raise
            save_action(
                data=actions_list,
                output_folder=self.cfg['GENERAL_PARMAS'].output_path,
                sub_folder_name="actions",
                name=skill_name,
                images=images
            )
            
        else:
            self.env.set_observation(observation)
            self.env.set_state(state=robot_state)
            primitive = self._get_primitive(
                action=high_level_action,
                env=self.env
            )
            
            #get action from low level plannes
            self.env.set_primitive(primitive=primitive)
            
            low_level_action = self.low_level_planner.get_action(
                    obs0=observation,
                    observation_indices= info['policy_args']['observation_indices'],
                    raw_high_level_action=high_level_action,
                    device=self.cfg.GENERAL_PARMAS.device,
                    primitive=primitive
                )

            #execute low level action
            try:
                save_image = True
                if save_image:
                    start_image = self.env.render()
                    self.env.record_start()
                end_observation, reward, terminated, truncated, info = self.env.step(
                    low_level_action
                )
                if save_image:
                    end_image = self.env.render()
                    self.env.record_stop()
                    os.makedirs(os.path.join(self.cfg['GENERAL_PARMAS'].output_path, "images"), exist_ok=True)
                    path = os.path.join(self.cfg['GENERAL_PARMAS'].output_path, "images", f'images_{self.index}.gif')
                    self.env.record_save(
                        path=path
                    )
                    cv2.imwrite(os.path.join(self.cfg['GENERAL_PARMAS'].output_path, "images", f'start_image_{self.index}.png'), start_image)
                    cv2.imwrite(os.path.join(self.cfg['GENERAL_PARMAS'].output_path, "images", f'end_image_{self.index}.png'), end_image)
                    self.index += 1
            except Exception as e:
                raise
        
        if reward > 0.0:
            self.graph_manager.add_low_level_node(
                initial_high_level_state=start_high_level_state,
                high_level_state =\
                    self.graph_manager.high_level_graph.graph.nodes._nodes[high_level_action_id]["state"],
                high_level_action=high_level_action,
                low_level_action=low_level_action,
                initial_low_level_state=observation,
                low_level_state=end_observation,
                goal_reached=goal_reached,
                info=info,
                robot_state=self.env.get_state()
            )
        
        results = {
            "type": "low_level",
            "end_observation": end_observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "goal_reached": goal_reached and reward > 0.0,
            "first_action_score": first_action_score if first_action_score is None else first_action_score.item(),
            "observation": observation,
        }
        
        return results
        
    def _get_high_level_plan(self):
        cnt = 0
        while True and cnt < self.max_high_level_steps:
            start_time = datetime.datetime.now()
            results = self._execute_high_level_action()
            self.step +=1 
            cnt += 1
            #logging
            results["selected_step"] = "high"
            time = datetime.datetime.now()-start_time
            results["time"] = time.seconds + time.microseconds/1000000
            self._log(key="results", value=results)
            self.logger_cnt += 1
            if len(results['plan']) > 0:
                return results
            
            if self.step > self.max_steps:
                return None
        
        return None
        
    def _convert_plan(self, plan:list):
        current_id = 0
        new_plan = [current_id]
        for action in plan:
            edges = self.graph_manager.high_level_graph.graph.edges._adjdict[current_id]
            for edge in edges:
                if edges[edge]['high_level_action'] == action.name:
                    current_id = edge
                    new_plan.append(current_id)
                
        assert len(new_plan) == len(plan) + 1
        
        return new_plan
        
    def _get_option_schema(self):
        option = {
            "name": None,
            "score": None
        }
        
        return option
        
    def _get_high_level_option_score(self):
        score = 1/(self.high_level_planner.get_estimate_how_much_nodes_left()) * self.cfg.GENERAL_PARMAS.high_level_balance
        
        return score
        
    def _get_plan_score(self, plan):
        #select low level state
        selected_high_level_state = \
            self.graph_manager.high_level_graph.graph.nodes._nodes[plan[0]]["state"]
        observation, info, robot_state, low_level_state_id =\
            self._select_low_level_state(high_level_state=selected_high_level_state)
            
            
        if (low_level_state_id, tuple(plan)) in self.low_level_scores:
            score, first_action_score = self.low_level_scores[(low_level_state_id, tuple(plan))]
        else:
            #get plan score
            best_plan = self._get_best_plan(
                plan=plan,
                observation=observation
                )
            score = max(best_plan.p_success.item(), 0.1)
            first_action_score = best_plan.first_action_score
            self.low_level_scores[(low_level_state_id, tuple(plan))] = (score, first_action_score)

        #get node score with bandits
        tries = self.graph_manager.low_level_graph.graph.nodes._nodes[low_level_state_id]['tries']
        visits = tries[plan[1]] if plan[1] in tries else sum(tries)
        ratio = self.bandit.score(visits=visits)
        
        return score * ratio, low_level_state_id, first_action_score
        
    def _get_all_options(self) -> dict:
        options = []
        
        #get high level option
        if not self.turn_off_high_level:
            option = self._get_option_schema()
            option["name"] = self.open_high_level_str
            if self.high_level_reset:
                option["score"] = self._get_high_level_option_score()
                option["score"] = max(self.cfg.GENERAL_PARMAS.high_level_min_score, option["score"])
                self.last_high_level_score = option["score"]
            else:
                option["score"] = self.last_high_level_score
            self.high_level_reset = False
            options.append(option)
        
        
        if self.low_level_reset:
            self.saved_low_level_options = []
            #get low level options
            for index, plan in enumerate(self.plans):
                score, low_level_state_id, first_action_score = self._get_plan_score(plan=plan)
                option = self._get_option_schema()
                option["score"] = score
                option["plan"] = plan
                option["name"] =  self.open_low_level_str +"_"+ str(index)
                option["low_level_state_id"] = low_level_state_id
                option["first_action_score"] = first_action_score
                options.append(option)
                self.saved_low_level_options.append(option)
        else:
            for option in self.saved_low_level_options:
                options.append(option)
        
        if len(self.saved_low_level_options) > 0:
            self.low_level_reset = False
        #for item in options:
        #    if item["name"] == 'open_high_level':
        #        print(item["score"])
        
        return options
        
    def _select_option(self, options:list) -> dict:
        """_summary_

        Args:
            options (list): _description_

        Returns:
            dict: _description_
        """
        ids = [i for i in range(len(options))]
        probs = np.array([option["score"] for option in options])
        probs /= sum(probs)
        choice_id = np.random.choice(ids, 1, p=probs).item()
        
        if options[choice_id]["name"] == 'open_high_level':
            self.high_level_reset = True
        else:
            self.low_level_reset = True
        
        return options[choice_id]
    
    def _get_best_plan(self, plan:list, observation:np.ndarray) -> PlanningResult:
        if self.cfg.GENERAL_PARMAS.state_estimator.type == "q_value":
            results = self._get_best_plan_q_value(
                plan=plan,
                observation=observation
            )
        elif self.cfg.GENERAL_PARMAS.state_estimator.type == "p_value":
            results = self._get_best_plan_p_value(
                plan=plan,
                observation=observation
            )
        else:
            raise

        return results

    def _get_best_plan_p_value(self, plan:list, observation:np.ndarray) -> PlanningResult:
        for state_index in range(1):
            action = self.graph_manager.high_level_graph.graph.edges._adjdict[plan[state_index]][plan[state_index+1]]['high_level_action']
            words = action.split(" ")
            words[0] = words[0][1:]
            words[1] = "("+ words[1]
            final = words[0] + words[1] + ", " + words[2]
            action_skeleton = final
        primitive = action_skeleton.split("(")[0]
        batch = {}
        batch["state"] = torch.tensor(observation, device=self.cfg.GENERAL_PARMAS.device)
        batch["state"] = torch.unsqueeze(batch["state"], 0)
        batch["success_rate"] = torch.ones([1], device=self.cfg.GENERAL_PARMAS.device)
        batch['success_rate'] = torch.unsqueeze(batch["success_rate"], 0)
        batch['high_level_action'] = [primitive]
        output, loss = self.state_estimator[primitive].run_step(
            batch=batch,
            batch_idx=0
        )
        temp = np.zeros((1,5))
        first_action_score = output[0].detach().cpu().numpy()
        results = PlanningResult(
            actions=temp,
            states=temp,
            p_success=first_action_score,
            values=np.zeros(5),
            first_action_score=first_action_score,
        )

        return results

    def _get_best_plan_q_value(self, plan:list, observation:np.ndarray) -> PlanningResult:
        p_best_success: float = -float("inf")
        action_skeleton = []
        for state_index in range(len(plan)-1):
            action = self.graph_manager.high_level_graph.graph.edges._adjdict[plan[state_index]][plan[state_index+1]]['high_level_action']
            words = action.split(" ")
            words[0] = words[0][1:]
            words[1] = "("+ words[1]
            final = words[0] + words[1] + ", " + words[2]
            action_skeleton.append(self.low_level_planner.env.get_primitive_info(action_call=final))
            
        value_fns = [
            self.low_level_planner.planner.policies[self.low_level_planner.policy_order_type_to_number[type(primitive).__name__.lower()]].critic for primitive in action_skeleton
        ]
        decode_fns = [
            functools.partial(self.low_level_planner.planner.dynamics.decode, primitive=primitive)
            for primitive in action_skeleton
        ]
        
        with torch.no_grad():
            # Prepare action spaces.
            T = len(action_skeleton)
            actions_low = null_tensor(self.low_level_planner.planner.dynamics.action_space, (T,))
            actions_high = actions_low.clone()
            task_dimensionality = 0
            for t, primitive in enumerate(action_skeleton):
                action_space = self.low_level_planner.planner.policies[self.low_level_planner.policy_order_type_to_number[type(primitive).__name__.lower()]].action_space
                action_shape = action_space.shape[0]
                actions_low[t, :action_shape] = torch.from_numpy(action_space.low)
                actions_high[t, :action_shape] = torch.from_numpy(action_space.high)
                task_dimensionality += action_shape
            actions_low = actions_low.to(self.cfg['GENERAL_PARMAS'].device)
            actions_high = actions_high.to(self.cfg['GENERAL_PARMAS'].device)
            
            # Scale number of samples to task size
            num_samples = self.num_samples * task_dimensionality

            # Get initial state.
            t_observation = torch.from_numpy(observation).to(self.low_level_planner.planner.dynamics.device)

            # Initialize distribution.
            mean, std = self.low_level_planner.planner._compute_initial_distribution(
                t_observation, action_skeleton
            )
            elites = torch.empty(
                (0, *mean.shape), dtype=torch.float32, device=self.cfg['GENERAL_PARMAS'].device
            )
            
            # Prepare constant agents for rollouts.
            policies = [
                ConstantAgent(
                    action=null_tensor(
                        self.low_level_planner.planner.policies[self.low_level_planner.policy_order_type_to_number[type(primitive).__name__.lower()]].action_space,
                        num_samples,
                        device=self.cfg['GENERAL_PARMAS'].device,
                    ),
                    policy=self.low_level_planner.planner.policies[self.low_level_planner.policy_order_type_to_number[type(primitive).__name__.lower()]],
                )
                for t, primitive in enumerate(action_skeleton)
            ]

            for idx_iter in range(self.num_iterations):
                # Sample from distribution.
                samples = torch.distributions.Normal(mean, std).sample((num_samples,))
                samples = torch.clip(samples, actions_low, actions_high)

                # Include the best elites from the previous iteration.
                if idx_iter > 0:
                    range_elites = min(self.num_elites_to_keep, elites.shape[0]-1)
                    samples[: range_elites] = elites[:range_elites]

                # Also include the mean.
                index = min(self.num_elites_to_keep, samples.real.shape[0]-1)
                samples[index] = mean

                # Roll out trajectories.
                for t, policy in enumerate(policies):
                    network = policy.actor.network
                    network.constant = samples[:, t, : policy.action_space.shape[0]]
                states, _ = self.low_level_planner.planner.dynamics.rollout(
                    observation = t_observation,
                    action_skeleton=action_skeleton,
                    policies=policies,
                    batch_size=num_samples,
                    time_index=True,
                )
                

                # Evaluate trajectories.
                p_success, values, values_unc = evaluate_trajectory(
                    value_fns, decode_fns, states, actions=samples, unc_metric = "stddev"
                )

                # Select the top trajectories.
                top_k = min(self.num_elites, p_success.shape[0])
                idx_elites = p_success.topk(top_k).indices
                elites = samples[idx_elites]
                idx_best = idx_elites[0]

                # Track best action.
                _p_best_success = p_success[idx_best].cpu().numpy()
                if _p_best_success > p_best_success:
                    p_best_success = _p_best_success
                    best_actions = samples[idx_best].cpu().numpy()
                    best_states = states[idx_best].cpu().numpy()
                    first_action_score = values[idx_best].cpu().numpy()
                    best_values_unc = values_unc[idx_best].cpu().numpy()

                # Update distribution.
                mean = self.momentum * mean + (1 - self.momentum) * elites.mean(dim=0)
                std = self.momentum * std + (1 - self.momentum) * elites.std(dim=0)
                std = torch.clip(std, 1e-4)

                # Decay population size.
                num_samples = int(self.population_decay * num_samples + 0.5)
                num_samples = max(num_samples, 2 * self.num_elites)

        results = PlanningResult(
            actions=best_actions,
            states=best_states,
            p_success=p_best_success,
            values=first_action_score,
            values_unc=best_values_unc,
            visited_actions=None,
            visited_states=None,
            p_visited_success=None,
            visited_values=None,
            first_action_score=first_action_score,
        )
        return results
    
    def _roll_out_high_level(self):
        results = self._get_high_level_plan()
        if results is None:
            return results
        converted_plan = self._convert_plan(plan=results['plan'])
        self.plans.append(converted_plan)
        
        results["type"] = "high_level"
        
        return results
    
    def _roll_out_low_level(self, task):
        plan = task['plan']
        low_level_state_id = task['low_level_state_id']
        for index in range(len(plan)-1):
            start_high_level_state_id = plan[index]
            end_high_level_state_id = plan[index+1]
            start_time = datetime.datetime.now()
            
            results = self._execute_low_level_action(
                start_high_level_state_id=start_high_level_state_id,
                end_high_level_state_id=end_high_level_state_id,
                low_level_state_id=low_level_state_id,
                first_action_score=task['first_action_score'],
            )
            task['first_action_score'] = None
            if results is not None:
                #logging
                results["selected_step"] = "low"
                time = datetime.datetime.now()-start_time
                results["time"] = time.seconds + time.microseconds/1000000
                self._log(key="results", value=results)
                self.logger_cnt += 1
                if results['reward'] > 0:
                    if plan[index+1:] not in self.plans:
                        self.plans.append(plan[index+1:])
                    low_level_state_id = \
                        self.graph_manager.low_level_graph.get_id_from_state(
                            results['end_observation']
                        )
                else:
                    return results
            else:
                return {}
        return results
        
    def _roll_out(self, task:dict) -> dict:
        if task["name"] == self.open_high_level_str:
            #print("high level was selected")
            results = self._roll_out_high_level()
        elif self.open_low_level_str in task["name"]:
            results = self._roll_out_low_level(task=task)
        else:
            raise
        
        return results
        
    def run(self):
        #total_time = 0
        start_time = datetime.datetime.now()
        while self.step < self.max_steps:
            #print("total_time =", total_time)
            
            #get all options to select from
            options = self._get_all_options()
            
            #select which option to try
            option = self._select_option(options=options)
            
            #try an option
            results = self._roll_out(task=option)
            
            time = datetime.datetime.now()-start_time
            total_time = time.seconds + time.microseconds/1000000
            
            if total_time > self.cfg.GENERAL_PARMAS.max_time:
                break
            
            if results is not None and "goal_reached" in results and results["goal_reached"]:
                print("---problems solved---")
                break
            
        return total_time
