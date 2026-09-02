#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include "pf_model.h"

const float fc_x = 795.97207f;
const float fc_y = 796.20615f;
const float cc_x = 321.78808f;
//const float cc_x = 320;
const float cc_y = 230.25022f;
//const float cc_y = 240;
//const float alpha_c = 0;
const float kc_1 = 0.06353f;
const float kc_2 = -0.15335f;
const float kc_3 = -0.00296f;
const float kc_4 = 0.00087f;
//const float kc_5 = 0.0;

const float NOISE_ANGLES = 0.015f;
const float NOISE_Y      = 0.001f;
const float NOISE_Z      = 0.001f;
const float NOISE_V_Y    = 0.05f;
const float NOISE_V_Z    = 0.05f;

const float NOISE_ANGLE_MEASUREMENTS = 1000.0f;
const float NOISE_CAMERA_X           = 0.0001f;
const float NOISE_CAMERA_Y           = 0.0001f;

const float PLANE_X = 0.45f;

const float d_arm_lenghts[NUM_ANGLES] = {0.065f, 0.25f, 0.25f, 0.125f};

const particle_state init_state = {{0.0f, 0.0f, M_PI_2, 0.0f}, 0.0f, 0.315f, 0.0f, 0.0f, 0.0f};

void sampling_importance(
	particle_state *particle_data,
	const control control_data,
	const measurement measurement_data,
	const float* const d_random,
	const float dt)
{
	float cameraX,cameraY,cameraZ;
	float value=0;

	particle_data->vY += d_random[0] * NOISE_V_Y;
	particle_data->vZ += d_random[1] * NOISE_V_Z;

	particle_data->y += (particle_data->vY * dt) + (d_random[2] * NOISE_Y);
	particle_data->z += (particle_data->vZ * dt) + (d_random[3] * NOISE_Z);

	// i=0
	{
		particle_data->angles[0] += (d_random[4] * NOISE_ANGLES);
		const float angle = particle_data->angles[0];

		float sf, cf;
		sincosf(angle, &sf, &cf);
		cameraX = (PLANE_X * cf) + (particle_data->y * sf);
		cameraY = - (PLANE_X * sf) + (particle_data->y * cf);
		cameraZ = particle_data->z - d_arm_lenghts[0];

		const float e = angle - measurement_data.angles[0];
		value += e * e * NOISE_ANGLE_MEASUREMENTS;
	}

	for (int i=1; i < NUM_ANGLES; ++i)
	{
		particle_data->angles[i] += (d_random[4+i] * NOISE_ANGLES);
		const float angle = particle_data->angles[i];

		float sf, cf;
		sincosf(angle, &sf, &cf);
		const float nX = (cameraX * cf) - (cameraZ * sf);
		const float nZ = (cameraX * sf) + (cameraZ * cf);

		cameraX = nX;
		cameraZ = nZ - d_arm_lenghts[i];

		const float e = angle - measurement_data.angles[i];
		value += e * e * NOISE_ANGLE_MEASUREMENTS;
	}

	//normalised
	const float x_n = cameraX/cameraZ;
	const float y_n = cameraY/cameraZ;

	const float dxx = 2 * kc_3 * x_n * y_n + kc_4 * (3 * x_n * x_n + y_n * y_n);
	const float dxy = kc_3 * (3 * y_n * y_n + x_n * x_n) + 2 * kc_4 * x_n * y_n;

	const float r2 = (x_n * x_n) + (y_n * y_n);
	const float xdx = (1 + kc_1 * r2 + kc_2 * r2 * r2 /*+ kc_5 * r2 * r2 * r2*/) * x_n + dxx;
	const float xdy = (1 + kc_1 * r2 + kc_2 * r2 * r2 /*+ kc_5 * r2 * r2 * r2*/) * y_n + dxy;

	const float xp = -fc_y * xdy/* + cc_y*/;
	const float yp = fc_x * xdx/* + cc_x*/;


	value += (xp-measurement_data.x) * (xp-measurement_data.x) * NOISE_CAMERA_X;
	value += (yp-measurement_data.y) * (yp-measurement_data.y) * NOISE_CAMERA_Y;

	//const float norm_factor = 1.0f;//0.0f / powf(2.0f*((float)M_PI), NUM_SENSORS/2);

	particle_data->weight = expf(-value);
}

int read_trace(FILE* const input_file, control* const control_data, measurement* const measurement_data, particle_state* const actual_state, float* const dt)
{
	float tmp;

	for (int i=0; i < NUM_ANGLES; ++i)
	{
		if (1 != fscanf(input_file, "%f", &measurement_data->angles[i]))
			return -1;
	}

	if (2 != fscanf(input_file, "%f %f", &measurement_data->x, &measurement_data->y))
		return -1;

	if (3 != fscanf(input_file, "%f %f %f", &tmp, &actual_state->y, &actual_state->z))
		return -1;

	if (4 != fscanf(input_file, "%f %f %f %f", &tmp, &tmp, &tmp, &tmp))
		return -1;

	*dt = 0.01;
	actual_state->vY = 0;
	actual_state->vZ = 0;
	return 0;
}

float estimate_error(const particle_state estimate, const particle_state actual)
{
	return sqrtf(((estimate.y-actual.y)*(estimate.y-actual.y)) + ((estimate.z-actual.z)*(estimate.z-actual.z)));
}

void print_particle(const particle_state particle_data)
{
	printf(">>>(%.8f) %.8f %.8f %.8f %.8f [%.8f %.8f %.8f %.8f]\n",
					particle_data.weight,
					particle_data.y,
					particle_data.z,
					particle_data.vY,
					particle_data.vZ,
					particle_data.angles[0],
					particle_data.angles[1],
					particle_data.angles[2],
					particle_data.angles[3]);

}

