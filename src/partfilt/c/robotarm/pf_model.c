#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include "pf_model.h"

const float NOISE_ANGLES = 0.2f;
const float NOISE_X      = 0.2f;
const float NOISE_Y      = 0.2f;
const float NOISE_V_X    = 2.0f;
const float NOISE_V_Y    = 2.0f;

const float NOISE_ANGLE_MEASUREMENTS = 0.1f;
const float NOISE_CAMERA_X           = 0.1f;
const float NOISE_CAMERA_Y           = 0.1f;

const float d_arm_lenghts[NUM_ANGLES] = {0.0f, 3.0f, 3.0f, 3.0f, 3.0f};

const particle_state init_state = {{0.75f * M_PI, 0.75f * M_PI, 0.75f * M_PI, 0.75f * M_PI, 0.75f * M_PI}, 7.0f, 0.0f, 0.0f, 0.0f, 0.0f};

void sampling_importance(
	particle_state *particle_data,
	const control control_data,
	const measurement measurement_data,
	const float* const d_random,
	const float dt)
{
	float cameraX,cameraY,cameraZ;
	float value=0;

	particle_data->x += (particle_data->vX * dt) + (d_random[0] * NOISE_X);
	particle_data->y += (particle_data->vY * dt) + (d_random[1] * NOISE_Y);

	particle_data->vX += d_random[2] * NOISE_V_X;
	particle_data->vY += d_random[3] * NOISE_V_Y;

	// i=0
	{
		particle_data->angles[0] += (control_data.angles[0] * dt) + (d_random[4] * NOISE_ANGLES);
		const float angle = particle_data->angles[0];

		float sf, cf;
		sincosf(angle, &sf, &cf);
		cameraX = ( particle_data->x * cf) + (particle_data->y * sf);
		cameraY = (-particle_data->x * sf) + (particle_data->y * cf);
		cameraZ = 0.0f;

		const float e = angle - measurement_data.angles[0];
		value += e * e * NOISE_ANGLE_MEASUREMENTS;
	}

	for (int i=1; i < NUM_ANGLES; ++i)
	{
		particle_data->angles[i] += (control_data.angles[i] * dt) + (d_random[4+i] * NOISE_ANGLES);
		const float angle = particle_data->angles[i];

		float sf, cf;
		sincosf(angle, &sf, &cf);
		const float nB = (cameraY * cf) - (cameraZ * sf);
		const float nC = (cameraY * sf) + (cameraZ * cf);

		cameraY = nB;
		cameraZ = nC - d_arm_lenghts[i];

		const float e = angle - measurement_data.angles[i];
		value += e * e * NOISE_ANGLE_MEASUREMENTS;
	}

	value += (cameraX-measurement_data.x) * (cameraX-measurement_data.x) * NOISE_CAMERA_X;
	value += (cameraY-measurement_data.y) * (cameraY-measurement_data.y) * NOISE_CAMERA_Y;

	//const float norm_factor = 1.0f;//0.0f / powf(2.0f*((float)M_PI), NUM_SENSORS/2);

	particle_data->weight = expf(-value);
}

int read_trace(FILE* const input_file, control* const control_data, measurement* const measurement_data, particle_state* const actual_state, float* const dt)
{
        if (2 != fscanf(input_file, "%f %f", &measurement_data->x, &measurement_data->y))
                return -1;
        for (int i=0; i < NUM_ANGLES; ++i)
        {
                if (1 != fscanf(input_file, "%f", &measurement_data->angles[i]))
                        return -1;
        }
        for (int i=0; i < NUM_ANGLES; ++i)
        {
                if (1 != fscanf(input_file, "%f", &control_data->angles[i]))
                        return -1;
        }
        if (3 != fscanf(input_file, "%f %f %f", dt, &actual_state->x, &actual_state->y))
                return -1;
        for (int i=0; i < NUM_ANGLES; ++i)
        {
                if (1 != fscanf(input_file, "%f", &actual_state->angles[i]))
                        return -1;
        }
        actual_state->vX = 0;
        actual_state->vY = 0;
        return 0;
        
}

float estimate_error(const particle_state estimate, const particle_state actual)
{
        float error = 0;
        error += (estimate.x-actual.x) * (estimate.x-actual.x) * NOISE_CAMERA_X;
        error += (estimate.y-actual.y) * (estimate.y-actual.y) * NOISE_CAMERA_Y;

        for (int i=0; i<NUM_ANGLES; ++i)
        {
                const float est = estimate.angles[i];
                const float mea = actual.angles[i];
                error += (est - mea) * (est - mea) * NOISE_ANGLE_MEASUREMENTS;
        }
        return error;
}

void print_particle(const particle_state particle_data)
{
	printf(">>>(%.8f) %.8f %.8f %.8f %.8f [%.8f %.8f %.8f %.8f %.8f]\n",
					particle_data.weight,
					particle_data.x,
					particle_data.y,
					particle_data.vX,
					particle_data.vY,
					particle_data.angles[0],
					particle_data.angles[1],
					particle_data.angles[2],
					particle_data.angles[3],
					particle_data.angles[4]);

}

