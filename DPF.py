from turtle import heading
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
class DPF():
	def __init__(self, particle_dim = 4, action_dim = 2, observation_dim = 5*4, particle_num = 100, env = None):
		self.particle_dim = particle_dim
		self.state_dim = 4 # augmented state dim
		self.action_dim = action_dim
		self.observation_dim = observation_dim
		self.particle_num = particle_num

		self.learning_rate = 0.0001
		self.propose_ratio = 0.0

		self.observation_scale = 4.0

		self.env = env

		self.build_model()

	def build_model(self):
		# observation model
		self.encoder = nn.Sequential(nn.Linear(self.observation_dim, 256),
									 nn.PReLU(),
									 nn.Linear(256, 128),
									 nn.PReLU(),
									 nn.Linear(128, 64)).to(device)
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = self.learning_rate)

		self.state_encoder = nn.Sequential(nn.Linear(self.state_dim, 32),
										   nn.PReLU(),
										   nn.Linear(32, 64)).to(device)
		self.state_encoder_optimizer = torch.optim.Adam(self.state_encoder.parameters(), lr = self.learning_rate)

		# observation likelihood estimator that maps states and observation encodings to probabilities
		self.obs_like_estimator = nn.Sequential(nn.Linear(64+64, 128),
												nn.PReLU(),
												nn.Linear(128, 128),
												nn.PReLU(),
												nn.Linear(128, 64),
												nn.PReLU(),
												nn.Linear(64, 1),
												nn.Sigmoid()).to(device)
		self.obs_like_estimator_optimizer = torch.optim.Adam(self.obs_like_estimator.parameters(), lr = self.learning_rate)




	def measurement_update(self, encoding, particles):
		'''
		Compute the likelihood of the encoded observation for each particle.
		'''
		particle_input = particles

		state_encoding = self.state_encoder(particle_input)
		encoding_input = encoding.repeat((particle_input.shape[1], 1))
		encoding_input = encoding_input.unsqueeze(0)

		inputs = torch.cat((state_encoding, encoding_input), axis = -1)
		obs_likelihood = self.obs_like_estimator(inputs)
		return obs_likelihood 

	def motion_update(self, particles, action):
		x, y, heading = action[:3]
		
		for batch in particles: # n,4 x,y,h,scale
			for p in batch:
				cos = torch.cos(p[2])
				sin = torch.sin(p[2])
				dx = x*cos - y*sin
				dy = x*sin + y*cos
				p[0] = p[0] + dx * p[3]
				p[1] = p[1] + dy * p[3]
				p[2] = p[2] + heading
		

	def propose_particles(self, observation, num_particles,heading): #using gaussian
		proposed_particles = []
		
		for i in range(num_particles):
			p0 = observation[0] + np.random.normal(0, 1)
			p1 = observation[1] + np.random.normal(0, 1)
			p2 = heading + np.random.normal(0, 1)
			p3 = 1
			proposed_particles.append([p0,p1,p2,p3])
		return torch.tensor(proposed_particles).to(device).to(torch.float32)

	def resample(self, particles, particle_probs, alpha, num_resampled):
		'''
		particle_probs in log space, unnormalized
		'''
		assert 0.0 < alpha <= 1.0
		

		# normalize
		particle_probs = particle_probs / particle_probs.sum(dim = -1, keepdim = True)
		uniform_probs = torch.ones((1,particles.shape[1])).to(device) / particles.shape[0]

		# bulid up sampling distribution q(s)
		if alpha < 1.0:
			# soft resampling
			q_probs = torch.stack((particle_probs*alpha, uniform_probs*(1.0-alpha)), dim = -1).to(device)
			q_probs = q_probs.sum(dim = -1)
			q_probs = q_probs / q_probs.sum(dim = -1, keepdim = True)
			particle_probs = particle_probs / q_probs
		else:
			# hard resampling
			q_probs = particle_probs
			particle_probs = uniform_probs

		# sample particle indices according to q(s)

		basic_markers = torch.linspace(0.0, (num_resampled-1.0)/num_resampled, num_resampled)
		random_offset = torch.FloatTensor(1).uniform_(0.0, 1.0/num_resampled)
		markers = random_offset[:, None] + basic_markers[None, :] # shape: batch_size * num_resampled
		cum_probs = torch.cumsum(q_probs, axis = 1)
		markers = markers.to(device)
		marker_matching = markers[:, :, None] > cum_probs[:, None, :]
		samples = marker_matching.sum(axis = 2).int()

		idx = samples + self.particle_num*torch.arange(1)[:, None].repeat((1, num_resampled)).to(device)
		particles_resampled = particles.view((1 * self.particle_num, -1))[idx, :]
		particle_probs_resampled = particle_probs.view((1 * self.particle_num, ))[idx]
		particle_probs_resampled = particle_probs_resampled / particle_probs_resampled.sum(dim = -1, keepdim = True)


		return particles_resampled, particle_probs_resampled



	def loop(self, particles, particle_probs, actions, observation, training = False):
		encoding = self.encoder(observation)

		# motion update
		deltas = self.motion_update(particles, actions)
		particles = particles + deltas * self.delta_scale / self.state_scale

		# observation update
		likelihood = (self.measurement_update(encoding, particles).squeeze()+1e-16)
		particle_probs = particle_probs * likelihood # unnormalized

		if likelihood.max() < 0.9:
			propose_ratio = 0.2
		else:
			propose_ratio = 0.00

		num_proposed = int(self.particle_num * propose_ratio)
		num_resampled = self.particle_num - num_proposed

		# resample
		alpha = 0.8
		particles_resampled, particle_probs_resampled = self.resample(particles, particle_probs, alpha, num_resampled)

		# propose
		if num_proposed > 0:
			particles_proposed = []
			scale = self.state_scale.cpu()
			while True:
				prop = self.propose_particles(encoding, num_proposed*2)
				for k in range(prop.shape[0]):
					if self.env.state_validity_checker((prop[:, k] * scale).numpy().T):
						particles_proposed.append(prop[0, k])
					if len(particles_proposed) >= num_proposed:
						break
				if len(particles_proposed) >= num_proposed:
					break
			particles_proposed = torch.stack(particles_proposed)[None, ...].to(device)

			particle_probs_proposed = torch.ones([particles_proposed.shape[0], particles_proposed.shape[1]]) / particles_proposed.shape[1]
			particle_probs_proposed = particle_probs_proposed.to(device)

			# combine
			particles = torch.cat((particles_resampled, particles_proposed), axis = 1)
			particle_probs = torch.cat((particle_probs_resampled, particle_probs_proposed), axis = -1)
			particle_probs = particle_probs / particle_probs.sum(dim = -1, keepdim = True)
			return particles, particle_probs
		else:
			return particles_resampled, particle_probs_resampled

	def init_particles(self,num_particles,gt0):
		proposed_particles = torch.zeros((num_particles,4))
		_,x,y,heading = gt0
		for p in proposed_particles:
			p[0] = x 
			p[1] = y 
			p[2] = heading + np.random.normal(0, 0.1)
			p[3] = 1
		self.particle_probs = torch.ones((1, num_particles)) / num_particles
		self.particle_probs = self.particle_probs.to(device)
		return proposed_particles

	def train_particles_data(self, n_particles,n_particles_probs, n_observations,gt):
		assert(len(n_particles) == len(gt))
		for i in range(len(n_particles)):
			print(i)
			observation = n_observations[i].unsqueeze(0) # batch 1
			particles = n_particles[i].unsqueeze(0)
			particle_probs = n_particles_probs[i].unsqueeze(0)
			encoding = self.encoder(observation)
			likelihood = (self.measurement_update(encoding, particles).squeeze()+1e-16)
			particle_probs = particle_probs * likelihood
			if likelihood.max() < 0: #0.9
				propose_ratio = 0.2
			else:
				propose_ratio = 0.00
	
			num_proposed = int(self.particle_num * propose_ratio)
			num_resampled = self.particle_num - num_proposed
			alpha = 0.8
			particles_resampled, particle_probs_resampled = self.resample(particles, particle_probs, alpha, num_resampled)
			heading = torch.mean(particles_resampled[:,2])
			# repropose particles?  Fig. 3a)
			if num_proposed > 0:
				particles_proposed = []
	
	
				prop = self.propose_particles(observation, num_proposed,heading)
	
				particles_proposed = torch.unsqueeze(prop,0)
	
				particle_probs_proposed = torch.ones([particles_proposed.shape[0], particles_proposed.shape[1]]) / particles_proposed.shape[1]
				particle_probs_proposed = particle_probs_proposed.to(device)
	
	
				# combine
				particles = torch.cat((particles_resampled, particles_proposed), axis = 1)
				particle_probs = torch.cat((particle_probs_resampled, particle_probs_proposed), axis = -1)
				particle_probs = particle_probs / particle_probs.sum(dim = -1, keepdim = True)
				particles_resampled, particle_probs_resampled = particles, particle_probs
			self.particles = particles_resampled
			self.particle_probs = particle_probs_resampled
			
	
			gt_index = i
			x,y = gt[gt_index] #x,y
			target = torch.tensor([x,y]).to(device).to(torch.float32)
	
			target = target.repeat(particles_resampled.shape[1],1).unsqueeze(0)
			sq_distance = (particles_resampled[:,:,:2] - target).pow(2).sum(axis = -1)
			mseloss = (self.particle_probs  * sq_distance).sum(axis=-1).mean()
			print(mseloss)
	
			# target_heading = torch.tensor([heading]).to(device).to(torch.float32)
			# heading_sq_distance = (particles_resampled[:,:,3] - target_heading).pow(2).sum(axis = -1)
			# heading_mseloss = (particle_probs_resampled * heading_sq_distance).sum(axis=-1).mean()
			
			self.state_encoder_optimizer.zero_grad()
			self.encoder_optimizer.zero_grad()
			self.obs_like_estimator_optimizer.zero_grad()
				
			mseloss.backward()
			self.encoder_optimizer.step()
			self.state_encoder_optimizer.step()
			self.obs_like_estimator_optimizer.step()

	def save(self,path='./'):
		torch.save(self.encoder.state_dict(), path+"encoder.pth")
		torch.save(self.obs_like_estimator.state_dict(), path+"estimator.pth")
		torch.save(self.state_encoder.state_dict(),path+'./state_encoder.pth')

	def load(self,path='./'):
		self.encoder.load_state_dict(torch.load(path+"encoder.pth"))
		self.obs_like_estimator.load_state_dict(torch.load(path+"estimator.pth"))
		self.state_encoder.load_state_dict(torch.load(path+'./state_encoder.pth'))


	def train(self, controls, observations, observation_time, gt):
		loss = 0
		observation_time = np.array(observation_time,dtype=float) / 10
		# particles = self.propose_particles(100,observations)
		for i in reversed(range(len(observation_time))):
			t0 = observation_time[i]
			location_index = i
			gt_index = find_nearest(gt[:,0],t0)
			gt0 = gt[gt_index]
			particles = self.init_particles(self.particle_num,gt0)
			self.particles = torch.tensor(particles).to(device).to(torch.float32).unsqueeze(0)
			self.particle_probs = torch.tensor(self.particle_probs).to(device).to(torch.float32)
			print("new")
			for ctrl in controls:
				t = ctrl[3]
				if t < t0:
					continue
				self.motion_update(self.particles,ctrl)
				if location_index >= len(observations):
					location_index = 0
				t_location = observation_time[location_index]
				if abs(t - t_location) <= 60:

					particles = self.particles
					particle_probs = self.particle_probs.clone().detach()
					observation = observations[location_index]
					location_index += 1
					observation = torch.tensor(observation).to(device).to(torch.float32)
					observation = observation.view(-1)
	
					encoding = self.encoder(observation)
					likelihood = (self.measurement_update(encoding, particles).squeeze()+1e-16)
					
					particle_probs = particle_probs * likelihood # unnormalized
					
					if likelihood.max() < 0.9:
						propose_ratio = 0.2
					else:
						propose_ratio = 0.00
	
					num_proposed = int(self.particle_num * propose_ratio)
					num_resampled = self.particle_num - num_proposed
					alpha = 0.8
					particles_resampled, particle_probs_resampled = self.resample(particles, particle_probs, alpha, num_resampled)
					heading = torch.mean(particles_resampled[:,2])
					# repropose particles?  Fig. 3a)
					if num_proposed > 0:
						particles_proposed = []
	
	
						prop = self.propose_particles(observation, num_proposed,heading)
	
						particles_proposed = torch.unsqueeze(prop,0)
	
						particle_probs_proposed = torch.ones([particles_proposed.shape[0], particles_proposed.shape[1]]) / particles_proposed.shape[1]
						particle_probs_proposed = particle_probs_proposed.to(device)
	
	
						# combine
						particles = torch.cat((particles_resampled, particles_proposed), axis = 1)
						particle_probs = torch.cat((particle_probs_resampled, particle_probs_proposed), axis = -1)
						particle_probs = particle_probs / particle_probs.sum(dim = -1, keepdim = True)
						particles_resampled, particle_probs_resampled = particles, particle_probs
					self.particles = particles_resampled
					self.particle_probs = particle_probs_resampled
					
	
					gt_index = find_nearest(gt[:,0],t_location)
					_,x,y,heading = gt[gt_index] #x,y,heading
					target = torch.tensor([x,y]).to(device).to(torch.float32)
	
					target = target.repeat(particles_resampled.shape[1],1).unsqueeze(0)
					sq_distance = (particles_resampled[:,:,:2] - target).pow(2).sum(axis = -1)
					mseloss = (self.particle_probs  * sq_distance).sum(axis=-1).mean()
					
	
					# target_heading = torch.tensor([heading]).to(device).to(torch.float32)
					# heading_sq_distance = (particles_resampled[:,:,3] - target_heading).pow(2).sum(axis = -1)
					# heading_mseloss = (particle_probs_resampled * heading_sq_distance).sum(axis=-1).mean()
			
					self.state_encoder_optimizer.zero_grad()
					self.encoder_optimizer.zero_grad()
					self.obs_like_estimator_optimizer.zero_grad()
				
					mseloss.backward()
					self.encoder_optimizer.step()
					self.state_encoder_optimizer.step()
					self.obs_like_estimator_optimizer.step()
					print(mseloss)
	
		

import math
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x1 - x2, y1 - y2)

def angle_between(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.atan2(y2 - y1, x2 - x1)

def normalize_angle(a):
    while a > math.pi:
        a -= 2*math.pi

    while a <= -math.pi:
        a += 2*math.pi
    
    return a
def extract_control(xs, ys, ts):
    # TODO: option to skip some indices?
    angle = angle_between((xs[0], ys[0]), (xs[1], ys[1]))
    # TODO: it is horrible to save the first pose here
    controls = [(xs[0], ys[0], angle, ts[0])]

    for i in range(len(xs) - 1):
        x, y, t = xs[i], ys[i], ts[i]
        next_x, next_y, next_t = xs[i+1], ys[i+1], ts[i+1]
        next_angle = angle_between((x, y), (next_x, next_y))
        angle_diff = normalize_angle(next_angle - angle)
        angle = next_angle
        d = distance((x, y), (next_x, next_y))
        ctrl = d, 0, angle_diff, next_t
        controls.append(ctrl)

    return controls		

if __name__ == "__main__":
	DATAPATH = './data/'
	gt = np.load(os.path.join(DATAPATH,"gt_1.npy"))
	imu = np.loadtxt(os.path.join(DATAPATH,"seq_1.txt"))
	wifi = pd.read_pickle(os.path.join(DATAPATH,"seq_1.pickle"))
	imu_x = [0]
	imu_y = [0]
	for i in imu:
		imu_x.append(imu_x[-1]+i[3])
		imu_y.append(imu_y[-1]+i[4])
		controls = extract_control(xs=imu_x[1:],ys=imu_y[1:],ts=imu[:,0])
	observations = []
	observations_time = []
	for i in wifi:
		observations.append(wifi.get(i))
		observations_time.append(i)
	heading = 0
	new_gt =[]
	for i in range(1,len(gt)):
		dy = gt[i,2] - gt[i-1,2]
		dx = gt[i,1] - gt[i-1,1]
		heading = math.atan2(dy, dx)
		new_gt.append(np.array([gt[i,0],gt[i,1],gt[i,2],heading]))
	gt = np.array(new_gt)
	