#!/usr/bin/python

# This file is part of the 'Esthera' bayesian estimation software toolkit.
# Copyright (C) 2011-2012  Mehdi Chitchian and Alexander S. van Amesfoort
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys, random, math, copy

NUM_ANGLES=5

ARM_LENGTHS = [0, 3, 3, 3, 3]

NOISE_ANGLE_MEASUREMENTS = 0.1
NOISE_CAMERA_X           = 0.1
NOISE_CAMERA_Y           = 0.1

A  = 10

class State:
	def __init__(self, angles, x, y, vX, vY, weight):
		assert len(angles) == NUM_ANGLES

		self.angles = angles
		self.x      = x
		self.y      = y
		self.vX     = vX
		self.vY     = vY
		self.weight = weight
		self.t      = 0.0

#	def __repr__(self):
#		return "%.16f %.16f %.16f | %.16f" % (self.x, self.y, self.theta, self.weight)

class ControlInput:
	def __init__(self, angles):
		assert len(angles) == NUM_ANGLES

		self.angles = angles

	def __repr__(self):
		return " ".join(map(str, self.angles))

class SensorData:
	def __init__(self, x, y, angles):
		assert len(angles) == NUM_ANGLES

		self.x = x
		self.y = y
		self.angles = angles
	
	def __repr__(self):
		return str(self.x) + " " + str(self.y) + " " + " ".join(map(str, self.angles))


def modelUpdate(particle, control, dt):
	for i in range(NUM_ANGLES):
		particle.angles[i] += dt * control.angles[i]
	
	particle.t += dt

	newX = (A*math.sqrt(2)*math.cos(particle.t))/(math.sin(particle.t)**2+1)
	newY = (A*math.sqrt(2)*math.cos(particle.t)*math.sin(particle.t))/(math.sin(particle.t)**2+1)

	particle.x += particle.vX * dt
	particle.y += particle.vY * dt

	particle.vX = (newX - particle.x) / dt
	particle.vY = (newY - particle.y) / dt



def translateAngles(particle):

	a =  particle.x * math.cos(particle.angles[0]) + particle.y * math.sin(particle.angles[0])
	b = -particle.x * math.sin(particle.angles[0]) + particle.y * math.cos(particle.angles[0])
	c = 0.0

	for i in range(1, NUM_ANGLES):
		newA = a# a unchanged
		newB = math.cos(particle.angles[i]) * b - math.sin(particle.angles[i]) * c
		newC = math.sin(particle.angles[i]) * b + math.cos(particle.angles[i]) * c

		a = newA
		b = newB
		c = newC - ARM_LENGTHS[i]
	
	return a, b

random.seed(0)

if len(sys.argv) != 2:
	sys.exit("usage: " + sys.argv[0] + " num_samples")

num_samples = int(sys.argv[1])

initAngles = []
for i in range(NUM_ANGLES):
	initAngles.append(0.75 * math.pi)

state = State(initAngles, A*math.sqrt(2), 0, 0.0, 0.0, 0)

controlJoints = []
for i in range(NUM_ANGLES):
	controlJoints.append(0.0)

control = ControlInput(controlJoints)
DT = (15.0*math.pi/8.0) / num_samples



# generate
for sample in range(num_samples):

	# random control
	for i in range(NUM_ANGLES):
		control.angles[i] = random.gauss(0, 0.05)

	modelUpdate(state, control, DT)

	cameraX, cameraY = translateAngles(state)

	sensorAngles = copy.copy(state.angles)
	for i in range(NUM_ANGLES):
		sensorAngles[i] += random.gauss(0, NOISE_ANGLE_MEASUREMENTS)

	sensorData = SensorData(
			cameraX+random.gauss(0, NOISE_CAMERA_X),
			cameraY+random.gauss(0, NOISE_CAMERA_Y),
			sensorAngles)

	print sensorData, control, DT, state.x, state.y, " ".join(map(str, state.angles))


