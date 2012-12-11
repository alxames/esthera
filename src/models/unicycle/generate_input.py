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

import sys, random, math

if len(sys.argv) != 2:
	print "give count!"
	exit(1)

count = int(sys.argv[1])
num_sensors = 10

sensors_x = [ 7, 12, 10, 24, 24,  6, 17,  6, 14, 16];
sensors_y = [34, 26, 11, 37, 31, 23, 36, 15, 14, 25];
sensors_z = [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3];

x=20.0
y=40.0
theta=-math.pi/2;
dt=1.0

for i in range(count):
	v = random.gauss(0.3, 0.1);
	if v < 0:
		v = -v;
	omg = random.gauss(0.0, 0.3);

	x = x + v / omg * (math.sin(theta+omg*dt) - math.sin(theta));
	y = y - v / omg * (math.cos(theta+omg*dt) - math.cos(theta));
	theta = theta + omg*dt;

	for j in range(num_sensors):
		d = math.sqrt(math.pow(x-sensors_x[j],2) + math.pow(y-sensors_y[j],2) + math.pow(sensors_z[j],2));
		d_err = d + 1.0*random.gauss(0,1);
		print d_err,
	
	#v_err = v + random.gauss(0, 0.5);
	#omg_err = omg + random.gauss(0, 0.1)

#	for j in range(num_sensors):
#		print random.uniform(0,50),
	print v, omg, dt, x, y, theta
	#print random.uniform(0,1),random.uniform(0,math.pi),1.0,x,y,theta,v, omg
	#print '% 5f % 5f   % 5f % 5f % 5f' % (v, omg, x, y, theta);

