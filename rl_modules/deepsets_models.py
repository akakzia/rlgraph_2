import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations
import numpy as np
from rl_modules.networks import PhiCriticDeepSet, PhiActorDeepSet, RhoActorDeepSet, RhoCriticDeepSet, SelfAttention
from utils import get_idxs_per_object

epsilon = 1e-6


class DsCritic(nn.Module):
    def __init__(self, nb_objects, semantic_ids, dim_body, dim_object, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, 
                 dim_rho_critic_output):
        super(DsCritic, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        self.n_permutations = self.nb_objects * (self.nb_objects - 1)

        self.phi_critic = PhiCriticDeepSet(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.node_self_attention = SelfAttention(dim_phi_critic_output, 1)  # test 1 attention heads
        self.rho_critic = RhoCriticDeepSet(dim_rho_critic_input, dim_rho_critic_output)

        self.semantic_ids = semantic_ids

    def forward(self, obs, act, ag, g):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]
        
        delta_g = g - ag

        inp = torch.stack([torch.cat([act, obs_body, obj, delta_g[:, self.semantic_ids[i]]], dim=1) for i, obj in enumerate(obs_objects)])

        output_phi_critic_1, output_phi_critic_2 = self.phi_critic(inp)
        output_phi_critic_1 = output_phi_critic_1.permute(1, 0, 2)
        output_self_attention_1 = self.node_self_attention(output_phi_critic_1)
        output_self_attention_1 = output_self_attention_1.sum(dim=1)

        output_phi_critic_2 = output_phi_critic_2.permute(1, 0, 2)
        output_self_attention_2 = self.node_self_attention(output_phi_critic_2)
        output_self_attention_2 = output_self_attention_2.sum(dim=1)

        q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_self_attention_1, output_self_attention_2)
        return q1_pi_tensor, q2_pi_tensor

class DsActor(nn.Module):
    def __init__(self, nb_objects, semantic_ids, dim_body, dim_object, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input,
                 dim_rho_actor_output):
        super(DsActor, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        self.n_permutations = self.nb_objects * (self.nb_objects - 1)

        self.phi_actor = PhiActorDeepSet(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.self_attention = SelfAttention(dim_phi_actor_output, 1) # test 1 attention heads
        self.rho_actor = RhoActorDeepSet(dim_rho_actor_input, dim_rho_actor_output)

        self.semantic_ids = semantic_ids

    def forward(self, obs, ag, g):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]
        
        delta_g = g - ag

        inp = torch.stack([torch.cat([obs_body, obj, delta_g[:, self.semantic_ids[i]]], dim=1) for i, obj in enumerate(obs_objects)])

        output_phi_actor = self.phi_actor(inp)
        output_phi_actor = output_phi_actor.permute(1, 0, 2)
        output_self_attention = self.self_attention(output_phi_actor)
        output_self_attention = output_self_attention.sum(dim=1)

        mean, logstd = self.rho_actor(output_self_attention)
        return mean, logstd

    def sample(self, obs, ag, g):
        mean, log_std = self.forward(obs, ag, g)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

class DsSemantic:
    def __init__(self, env_params, args):
        self.dim_body = 10
        self.dim_object = 15
        self.dim_goal = env_params['goal']
        self.dim_act = env_params['action']
        self.nb_objects = args.n_blocks

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # Process indexes for graph construction
        self.semantic_ids = [np.arange(i * 3, (i + 1) * 3) for i in range(args.n_blocks)]
        dim_predicates_per_object = len(self.semantic_ids[0])


        dim_phi_actor_input = self.dim_body + self.dim_object + dim_predicates_per_object
        dim_phi_actor_output = 3 * dim_phi_actor_input
        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = self.dim_act

        dim_phi_critic_input = self.dim_body + self.dim_object + dim_predicates_per_object + self.dim_act
        dim_phi_critic_output = 3 * dim_phi_critic_input
        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1

        self.critic = DsCritic(self.nb_objects, self.semantic_ids, self.dim_body, self.dim_object,
                                dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output)
        self.critic_target = DsCritic(self.nb_objects, self.semantic_ids,self.dim_body, self.dim_object,
                                       dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output)
        self.actor = DsActor(self.nb_objects, self.semantic_ids, self.dim_body, self.dim_object, dim_phi_actor_input, dim_phi_actor_output,
                              dim_rho_actor_input, dim_rho_actor_output)

    def policy_forward_pass(self, obs, ag, g, no_noise=False):
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, ag, g)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, ag, g)

    def forward_pass(self, obs, ag, g, actions=None):
        # edge_features = self.critic.message_passing(obs, ag, g)

        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, ag, g)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(obs, self.pi_tensor, ag, g)
            return self.critic.forward(obs, actions, ag, g)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic_target.forward(obs, self.pi_tensor, ag, g)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None